"use strict";

const crypto = require("node:crypto");
const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret, defineString } = require("firebase-functions/params");

const REGION = "europe-west3";
const CRUDX_DOCS_API = defineString("CRUDX_DOCS_API", {
  default: "https://europe-west3-crudx-e0599.cloudfunctions.net/calendarApi/v1"
});
const LANGGRAPH_RUNTIME_URL = defineString("LANGGRAPH_RUNTIME_URL", { default: "" });
const LANGGRAPH_RUNTIME_TOKEN = defineSecret("LANGGRAPH_RUNTIME_TOKEN");
const LANGGRAPH_RUNTIME_MODE = defineString("LANGGRAPH_RUNTIME_MODE", { default: "standalone" });
const LANGGRAPH_ASSISTANT_ID = defineString("LANGGRAPH_ASSISTANT_ID", { default: "agent" });
const GEMINI_API_KEY = defineSecret("GEMINI_API_KEY");
const GEMINI_TRANSLATION_MODEL = defineString("GEMINI_TRANSLATION_MODEL", { default: "gemini-3.5-flash" });
const GOOGLE_MAPS_API_KEY = defineSecret("GOOGLE_MAPS_API_KEY");
const CRUDX_ID_RE = /^CRUDX-[RDUCX23458]{5}-[RDUCX23458]{5}-[RDUCX23458]{5}$/;
const CRUDX_ALPHABET = "RDUCX23458";
const CRUDX_OWNER = "drueffler@gmail.com";
const LOCATION_ROUTE_TARGETS = {
  gcp: {
    id: "gcp_europe_west3_frankfurt",
    label: "GCP europe-west3 reference point, Frankfurt, Germany",
    latitude: 50.110924,
    longitude: 8.682127,
    note: "GCP europe-west3 is the configured platform region; Frankfurt city center is used as a routing reference point, not a public data-center entrance."
  },
  langsmith: {
    id: "langsmith_reference_point",
    label: "LangSmith cloud reference point, New York, NY, USA",
    latitude: 40.712776,
    longitude: -74.005974,
    note: "LangSmith does not expose a per-request physical processing address in this app; this OKF uses a configured reference point for routing demonstrations."
  }
};

exports.reactAaasInvoke = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 120,
    memory: "512MiB",
    secrets: [LANGGRAPH_RUNTIME_TOKEN]
  },
  async (req, res) => {
    setCors(res);

    if (req.method === "OPTIONS") {
      res.status(204).send("");
      return;
    }

    if (req.method !== "POST") {
      res.status(405).json({ ok: false, error: "METHOD_NOT_ALLOWED" });
      return;
    }

    const startedAt = new Date().toISOString();
    const requestBody = typeof req.body === "object" && req.body ? req.body : {};
    const input = String(requestBody.input || requestBody.query || "").trim();
    const agentId = String(requestBody.agent_id || requestBody.agentId || "").trim();
    const okfKey = String(requestBody.okf_key || requestBody.okfKey || "").trim();
    const dryRun = requestBody.dry_run === true || requestBody.dryRun === true;
    const messages = normalizeMessages(requestBody.messages || requestBody.history || []);
    const runtimeUrl = String(LANGGRAPH_RUNTIME_URL.value() || "").replace(/\/+$/, "");

    if (!input) {
      res.status(400).json({ ok: false, error: "INPUT_REQUIRED" });
      return;
    }

    if (!okfKey || !CRUDX_ID_RE.test(okfKey)) {
      res.status(400).json({
        ok: false,
        error: "VALID_OKF_CRUDX_KEY_REQUIRED",
        expected: "CRUDX-[RDUCX23458]{5}-[RDUCX23458]{5}-[RDUCX23458]{5}"
      });
      return;
    }

    if (!runtimeUrl) {
      res.status(503).json({
        ok: false,
        error: "LANGGRAPH_RUNTIME_URL_REQUIRED",
        message: "LangGraph has been removed from GCF. Configure LANGGRAPH_RUNTIME_URL to a hosted LangGraph runtime endpoint."
      });
      return;
    }

    const runKey = stableCrudxKey("react-aaas-run", `${okfKey}:${input}:${Date.now()}`);
    const trace = [step("system", "middleware_received", { agent_id: agentId || null, okf_key: okfKey })];

    try {
      const okfEnvelope = await readCrudxEnvelope(okfKey);
      trace.push(step("crudx", "okf_loaded", { okf_key: okfKey }));

      const graphResult = await invokeExternalLangGraphRuntime(runtimeUrl, {
        run_key: runKey,
        agent_id: agentId || null,
        okf_key: okfKey,
        input,
        messages,
        dry_run: dryRun,
        okf_envelope: okfEnvelope,
        metadata: {
          ...(typeof requestBody.metadata === "object" && requestBody.metadata ? requestBody.metadata : {}),
          source: "gcf-react-aaas-middleware",
          runtime_boundary: "external-langgraph",
          received_at: startedAt,
          request_context: buildRequestContext(req, requestBody.metadata?.request_context)
        }
      });
      trace.push(step("langgraph", "external_runtime_completed", {
        runtime_url: runtimeUrl,
        nodes: graphResult.graph?.nodes || []
      }));

      const agent = graphResult.agent || unwrapOkfAgent(okfEnvelope, okfKey);
      const finalAnswer = String(graphResult.answer || graphResult.output || "").trim();
      const finishedAt = new Date().toISOString();
      const mergedTrace = [...trace, ...(Array.isArray(graphResult.trace) ? graphResult.trace : [])];
      const runEnvelope = {
        schema: "crudx.react_aaas.run.v1",
        key: runKey,
        type: "react_aaas_run",
        status: "completed",
        created_at: startedAt,
        updated_at: finishedAt,
        okf_key: okfKey,
        agent_id: agent.agent_id || agentId || null,
        input,
        messages,
        output: finalAnswer,
        response_format: graphResult.response_format || "react_sections_v1",
        graph: graphResult.graph || null,
        trace: mergedTrace,
        langsmith: graphResult.langsmith || null,
        llm: graphResult.llm || null,
        runtime: {
          middleware: "gcf-react-aaas",
          graph_runtime: "external",
          graph_runtime_url: runtimeUrl
        }
      };

      await writeCrudxEnvelope(runKey, runEnvelope);
      mergedTrace.push(step("crudx", "run_persisted", { run_key: runKey }));

      res.status(200).json({
        ok: true,
        run_key: runKey,
        okf_key: okfKey,
        agent_id: agent.agent_id || agentId || null,
        answer: finalAnswer,
        response_format: runEnvelope.response_format,
        graph: runEnvelope.graph,
        langsmith: runEnvelope.langsmith,
        trace: mergedTrace,
        runtime: runEnvelope.runtime
      });
    } catch (error) {
      const failedAt = new Date().toISOString();
      trace.push(step("error", "middleware_failed", { message: error.message }));

      const failedEnvelope = {
        schema: "crudx.react_aaas.run.v1",
        key: runKey,
        type: "react_aaas_run",
        status: "failed",
        created_at: startedAt,
        updated_at: failedAt,
        okf_key: okfKey,
        agent_id: agentId || null,
        input,
        error: error.message,
        trace,
        runtime: {
          middleware: "gcf-react-aaas",
          graph_runtime: "external",
          graph_runtime_url: runtimeUrl
        }
      };

      try {
        await writeCrudxEnvelope(runKey, failedEnvelope);
      } catch (persistError) {
        trace.push(step("error", "failed_run_persist_error", { message: persistError.message }));
      }

      res.status(500).json({
        ok: false,
        run_key: runKey,
        error: error.message,
        trace,
        runtime: failedEnvelope.runtime
      });
    }
  }
);

exports.translateAnswer = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 45,
    memory: "256MiB",
    secrets: [GEMINI_API_KEY]
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = typeof req.body === "object" && req.body ? req.body : {};
      const text = String(body.text || body.answer || "").trim();
      const target = normalizeTranslationTarget(body.target || body.language || body.lang);
      if (!text) {
        res.status(400).json({ ok: false, error: "TEXT_REQUIRED" });
        return;
      }
      if (!target) {
        res.status(400).json({ ok: false, error: "SUPPORTED_TARGET_REQUIRED", supported: ["de", "fr"] });
        return;
      }
      const translation = await translateWithGemini(text, target);
      res.status(200).json({
        ok: true,
        target,
        language: target === "de" ? "German" : "French",
        translation
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.searchKnowledgeBase = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 30,
    memory: "256MiB"
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const knowledge = await resolveToolKnowledge(body);
      const snippets = selectKnowledgeSnippets(body.query, knowledge);
      res.status(200).json({
        ok: true,
        tool: "search_knowledge_base",
        okf_key: body.okf_key || null,
        agent_id: body.agent_id || null,
        snippets,
        observation: snippets.length
          ? `OKF knowledge lookup: ${snippets.join(" ")}`
          : "OKF knowledge lookup: no directly matching knowledge snippet was found in the OKF document."
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.academicLiteratureSearch = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 45,
    memory: "512MiB"
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const query = String(body.parameters?.query || body.parameters?.paper_title || body.query || "").trim();
      const doi = String(body.parameters?.doi_or_url || body.parameters?.doi || "").trim();
      const paperTitle = String(body.parameters?.paper_title || inferQuotedTitle(query) || "").trim();
      const maxResults = clampInteger(body.parameters?.max_results || body.parameters?.rows || 5, 1, 10);

      if (!query && !doi) {
        res.status(200).json({
          ok: true,
          tool: "academic_literature_search",
          needs_input: true,
          sources: ["crossref", "arxiv"],
          results: [],
          observation: "Live literature search needs a paper title, DOI, URL, or research question."
        });
        return;
      }

      const [crossref, arxiv] = await Promise.allSettled([
        searchCrossref(query || paperTitle || doi, doi, maxResults, paperTitle),
        searchArxiv(paperTitle || query || doi, maxResults, Boolean(paperTitle))
      ]);
      const crossrefResults = crossref.status === "fulfilled" ? crossref.value : [];
      const arxivResults = arxiv.status === "fulfilled" ? arxiv.value : [];
      const results = mergeLiteratureResults([...crossrefResults, ...arxivResults]).slice(0, maxResults);
      const errors = [
        crossref.status === "rejected" ? `crossref: ${crossref.reason?.message || crossref.reason}` : null,
        arxiv.status === "rejected" ? `arxiv: ${arxiv.reason?.message || arxiv.reason}` : null
      ].filter(Boolean);

      res.status(200).json({
        ok: true,
        tool: "academic_literature_search",
        query: query || doi,
        sources: ["crossref", "arxiv"],
        result_count: results.length,
        results,
        errors,
        observation: buildLiteratureObservation(results, errors)
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.calculateDays = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 30,
    memory: "256MiB"
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const parameters = body.parameters || {};
      const dates = [
        parameters.start_date || parameters.startDate,
        parameters.end_date || parameters.endDate
      ].filter(Boolean);
      const inferredDates = dates.length >= 2 ? dates : extractIsoDates(body.query);

      if (inferredDates.length < 2) {
        res.status(200).json({
          ok: true,
          tool: "calculate_days",
          needs_input: true,
          observation: "Date calculation requested, but no exact start and end dates were supplied in the current turn."
        });
        return;
      }

      const startDate = String(inferredDates[0]).slice(0, 10);
      const endDate = String(inferredDates[1]).slice(0, 10);
      const businessDays = countWeekdays(startDate, endDate);
      res.status(200).json({
        ok: true,
        tool: "calculate_days",
        start_date: startDate,
        end_date: endDate,
        business_days: businessDays,
        assumptions: ["Monday through Friday are counted as business days.", "No holiday calendar override is applied."],
        observation: `Calculated business-day impact for ${startDate} to ${endDate} is ${businessDays} business days, assuming Monday through Friday and no holiday calendar overrides.`
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.peanoAddition = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 30,
    memory: "256MiB"
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const operands = inferPeanoOperands(body);
      if (!operands) {
        res.status(200).json({
          ok: true,
          tool: "peano_addition",
          needs_input: true,
          observation: JSON.stringify({
            error: "Two non-negative integer operands are required."
          })
        });
        return;
      }

      const result = executePeanoAddition(operands.left, operands.right);
      res.status(200).json({
        ok: true,
        tool: "peano_addition",
        ...result,
        observation: JSON.stringify({
          peano_result: result.peano_result,
          integer_value: result.integer_value
        })
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.requestorLocation = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 30,
    memory: "256MiB",
    secrets: [GOOGLE_MAPS_API_KEY]
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const context = buildRequestContext(req, body.parameters?.request_context || body.meta?.request_context);
      const explicitLocation = inferExplicitLocation(body);
      const consentedCoordinates = inferConsentedCoordinates(body, context);
      const reverseGeocoded = !explicitLocation && consentedCoordinates
        ? await reverseGeocodeWithGoogleMaps(consentedCoordinates, GOOGLE_MAPS_API_KEY.value())
        : null;
      const location = explicitLocation || reverseGeocoded || inferLocationFromContext(context);
      const routeRequest = classifyRouteRequest(body.query || body.input || body.parameters?.query);
      const confidence = explicitLocation ? "high" : location ? location.confidence : "none";
      const evidence = [
        explicitLocation ? "explicit user-supplied location" : null,
        consentedCoordinates ? "consented browser geolocation coordinates" : null,
        consentedCoordinates?.accuracy_m ? `browser geolocation accuracy: +/- ${Math.round(Number(consentedCoordinates.accuracy_m))}m` : null,
        reverseGeocoded?.reverse_geocoded ? "Google Maps reverse geocoding succeeded" : null,
        consentedCoordinates && !reverseGeocoded?.reverse_geocoded ? "Google Maps reverse geocoding unavailable; coordinates retained" : null,
        context.time_zone ? `browser time zone: ${context.time_zone}` : null,
        context.language ? `browser language: ${context.language}` : null,
        context.country ? `request country header: ${context.country}` : null,
        context.region ? `request region header: ${context.region}` : null,
        context.city ? `request city header: ${context.city}` : null,
        context.ip ? "forwarded request IP present" : null
      ].filter(Boolean);

      const route = routeRequest ? buildRouteResponse(routeRequest, location, evidence) : null;
      const answer = route
        ? route.observation
        : location
        ? `Requestor location estimate: ${location.label}. Confidence: ${confidence}. Evidence: ${evidence.join("; ")}.`
        : `Requestor location could not be determined reliably. Evidence available: ${evidence.join("; ") || "none"}. Ask the requestor to share location explicitly or pass browser geolocation with consent.`;

      res.status(200).json({
        ok: true,
        tool: "requestor_location",
        location: location || null,
        route: route || null,
        confidence,
        evidence,
        request_context: redactRequestContext(context),
        observation: answer
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.checkServerStatus = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 30,
    memory: "256MiB"
  },
  async (req, res) => {
    setCors(res);
    if (handlePreflightOrReject(req, res)) return;

    try {
      const body = normalizeToolRequest(req.body);
      const knowledge = await resolveToolKnowledge(body);
      const target = String(body.parameters?.target || body.query || "").trim();
      const snippets = selectOperationalStatusSnippets(target, knowledge);
      const observation = snippets.length
        ? `Server status check from OKF: ${snippets.join(" ")}`
        : "Server status check: no matching operational status entry was found in the OKF knowledge.";

      res.status(200).json({
        ok: true,
        tool: "check_server_status",
        okf_key: body.okf_key || null,
        agent_id: body.agent_id || null,
        target,
        snippets,
        observation
      });
    } catch (error) {
      res.status(500).json({ ok: false, error: error.message });
    }
  }
);

exports.crudxApp = onRequest(
  {
    region: REGION,
    cors: true,
    timeoutSeconds: 60,
    memory: "256MiB"
  },
  async (req, res) => {
    setCors(res);

    if (req.method === "OPTIONS") {
      res.status(204).send("");
      return;
    }

    if (req.method !== "GET") {
      res.status(405).json({ ok: false, error: "METHOD_NOT_ALLOWED" });
      return;
    }

    const action = String(req.query.action || "X").toUpperCase();
    const key = String(req.query.key || "").trim();
    const app = String(req.query.app || "").trim();
    const targetKey = app || key;

    if (action !== "X") {
      res.status(400).json({ ok: false, error: "ONLY_ACTION_X_SUPPORTED" });
      return;
    }

    if (!targetKey || !CRUDX_ID_RE.test(targetKey)) {
      res.status(400).json({
        ok: false,
        error: "VALID_APP_CRUDX_KEY_REQUIRED",
        expected: "CRUDX-[RDUCX23458]{5}-[RDUCX23458]{5}-[RDUCX23458]{5}"
      });
      return;
    }

    try {
      const doc = await readCrudxDocRaw(targetKey);
      const value = String(doc.value || "");
      const contentType = String(doc.content_type || "").toLowerCase();
      const mimeType = String(doc.mime_type || "").toLowerCase();

      if (contentType.includes("html") || mimeType.includes("html") || value.trim().startsWith("<!DOCTYPE html")) {
        res.set("Cache-Control", "no-store, max-age=0");
        res.status(200).type("html").send(value);
        return;
      }

      res.set("Cache-Control", "no-store, max-age=0");
      res.status(200).type("json").send({
        ok: true,
        doc,
        launch: {
          key,
          app: targetKey,
          data: req.query.data || null
        }
      });
    } catch (error) {
      res.status(404).json({ ok: false, error: error.message });
    }
  }
);

function setCors(res) {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.set("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS");
}

function handlePreflightOrReject(req, res) {
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return true;
  }
  if (req.method !== "POST") {
    res.status(405).json({ ok: false, error: "METHOD_NOT_ALLOWED" });
    return true;
  }
  return false;
}

function normalizeTranslationTarget(value) {
  const target = String(value || "").trim().toLowerCase();
  if (target === "d" || target === "de" || target === "deutsch" || target === "german") return "de";
  if (target === "f" || target === "fr" || target === "francais" || target === "français" || target === "french") return "fr";
  return "";
}

async function translateWithGemini(text, target) {
  const apiKey = GEMINI_API_KEY.value();
  if (!apiKey) throw new Error("GEMINI_API_KEY is not configured for translation");
  const language = target === "de" ? "German" : "French";
  const model = GEMINI_TRANSLATION_MODEL.value() || "gemini-3.5-flash";
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const prompt = [
    `Translate the following assistant answer into ${language}.`,
    "Preserve meaning, URLs, numbers, names, CRUDX IDs, and formatting.",
    "Do not add commentary, notes, explanations, Markdown fences, or source labels.",
    "",
    text.slice(0, 8000)
  ].join("\n");
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey
    },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.1, maxOutputTokens: 1200 }
    })
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || response.statusText);
  const translated = payload.candidates?.[0]?.content?.parts?.map((part) => part.text || "").join("").trim();
  if (!translated) throw new Error("Gemini translation response did not contain text");
  return translated;
}

function normalizeToolRequest(rawBody) {
  const body = typeof rawBody === "object" && rawBody ? rawBody : {};
  return {
    tool: String(body.tool || "").trim(),
    query: String(body.query || body.input || "").trim(),
    input: String(body.input || body.query || "").trim(),
    agent_id: String(body.agent_id || body.agentId || "").trim(),
    okf_key: String(body.okf_key || body.okfKey || "").trim(),
    parameters: typeof body.parameters === "object" && body.parameters ? body.parameters : {},
    knowledge: String(body.knowledge || ""),
    okf: typeof body.okf === "object" && body.okf ? body.okf : null,
    meta: typeof body.meta === "object" && body.meta ? body.meta : {}
  };
}

async function searchCrossref(query, doiOrUrl, maxResults, paperTitle = "") {
  const doi = extractDoi(doiOrUrl || query);
  const base = doi
    ? `https://api.crossref.org/works/${encodeURIComponent(doi)}`
    : paperTitle
      ? `https://api.crossref.org/works?rows=${maxResults}&query.title=${encodeURIComponent(paperTitle)}&select=DOI,title,author,issued,published-print,published-online,container-title,URL,type,is-referenced-by-count,abstract`
      : `https://api.crossref.org/works?rows=${maxResults}&query.bibliographic=${encodeURIComponent(query)}&select=DOI,title,author,issued,published-print,published-online,container-title,URL,type,is-referenced-by-count,abstract`;
  const response = await fetch(base, {
    headers: {
      "Accept": "application/json",
      "User-Agent": "CRUDX-ReAct-AaaS/0.1 (mailto:drueffler@gmail.com)"
    }
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || response.statusText);
  const message = payload.message || {};
  const items = doi ? [message] : Array.isArray(message.items) ? message.items : [];
  return items.map((item) => ({
    source: "crossref",
    title: firstText(item.title),
    authors: normalizeCrossrefAuthors(item.author),
    year: extractCrossrefYear(item),
    doi: item.DOI || null,
    url: item.URL || (item.DOI ? `https://doi.org/${item.DOI}` : null),
    venue: firstText(item["container-title"]),
    type: item.type || null,
    citation_count_hint: Number.isFinite(item["is-referenced-by-count"]) ? item["is-referenced-by-count"] : null,
    abstract: stripTags(item.abstract || "").slice(0, 900)
  })).filter((item) => item.title || item.doi || item.url);
}

async function searchArxiv(query, maxResults, titleSearch = false) {
  const searchQuery = titleSearch ? `ti:"${query}"` : `all:${query}`;
  const response = await fetch(`https://export.arxiv.org/api/query?search_query=${encodeURIComponent(searchQuery)}&start=0&max_results=${maxResults}&sortBy=relevance&sortOrder=descending`, {
    headers: {
      "Accept": "application/atom+xml",
      "User-Agent": "CRUDX-ReAct-AaaS/0.1 (mailto:drueffler@gmail.com)"
    }
  });
  const xml = await response.text();
  if (!response.ok) throw new Error(response.statusText);
  return parseArxivEntries(xml).map((entry) => ({
    source: "arxiv",
    title: entry.title,
    authors: entry.authors,
    year: entry.published ? Number(entry.published.slice(0, 4)) : null,
    doi: entry.doi || null,
    url: entry.id || null,
    venue: "arXiv",
    type: "preprint",
    citation_count_hint: null,
    abstract: entry.summary.slice(0, 900),
    arxiv_id: extractArxivId(entry.id)
  })).filter((item) => item.title || item.url);
}

function parseArxivEntries(xml) {
  const entries = [];
  const blocks = String(xml || "").match(/<entry\b[\s\S]*?<\/entry>/g) || [];
  for (const block of blocks) {
    entries.push({
      id: decodeXml(firstXmlTag(block, "id")),
      title: normalizeWhitespace(decodeXml(firstXmlTag(block, "title"))),
      summary: normalizeWhitespace(decodeXml(firstXmlTag(block, "summary"))),
      published: decodeXml(firstXmlTag(block, "published")),
      doi: decodeXml(firstXmlTag(block, "arxiv:doi") || firstXmlTag(block, "doi")),
      authors: Array.from(block.matchAll(/<author\b[\s\S]*?<name>([\s\S]*?)<\/name>[\s\S]*?<\/author>/g))
        .map((match) => normalizeWhitespace(decodeXml(match[1])))
        .filter(Boolean)
        .slice(0, 8)
    });
  }
  return entries;
}

function mergeLiteratureResults(items) {
  const byWork = new Map();
  for (const item of items) {
    const key = literatureWorkKey(item);
    if (!key) continue;
    const previous = byWork.get(key);
    if (!previous) {
      byWork.set(key, item);
      continue;
    }
    byWork.set(key, preferLiteratureRecord(previous, item));
  }
  return Array.from(byWork.values()).sort((left, right) => scoreLiteratureRecord(right) - scoreLiteratureRecord(left));
}

function literatureWorkKey(item) {
  const title = normalizeLiteratureTitle(item.title);
  const firstAuthor = normalizeLiteratureTitle(item.authors?.[0] || "");
  if (title && firstAuthor) return `${title}:${firstAuthor}`;
  return String(item.doi || item.arxiv_id || item.url || "").toLowerCase().trim();
}

function preferLiteratureRecord(left, right) {
  const leftScore = scoreLiteratureRecord(left);
  const rightScore = scoreLiteratureRecord(right);
  if (rightScore > leftScore) return mergeLiteratureRecord(right, left);
  return mergeLiteratureRecord(left, right);
}

function mergeLiteratureRecord(primary, secondary) {
  return {
    ...primary,
    source: Array.from(new Set([primary.source, secondary.source].flat().filter(Boolean))).join("+"),
    doi: primary.doi || secondary.doi || null,
    arxiv_id: primary.arxiv_id || secondary.arxiv_id || null,
    url: primary.url || secondary.url || null,
    citation_count_hint: Number.isFinite(primary.citation_count_hint)
      ? primary.citation_count_hint
      : Number.isFinite(secondary.citation_count_hint)
        ? secondary.citation_count_hint
        : null,
    abstract: primary.abstract || secondary.abstract || ""
  };
}

function scoreLiteratureRecord(item) {
  let score = 0;
  if (item.source === "arxiv" || String(item.source || "").includes("arxiv")) score += 10;
  if (item.doi) score += 5;
  if (item.arxiv_id) score += 5;
  if (item.abstract) score += 3;
  if (Number.isFinite(item.citation_count_hint)) score += Math.min(3, item.citation_count_hint / 100);
  if (item.year && item.year < 2024) score += 2;
  return score;
}

function normalizeLiteratureTitle(value) {
  return normalizeWhitespace(value)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function buildLiteratureObservation(results, errors) {
  if (!results.length) {
    return errors.length
      ? `Live literature search returned no usable results. Source errors: ${errors.join("; ")}.`
      : "Live literature search returned no usable Crossref or arXiv results for this query.";
  }
  const lines = results.slice(0, 5).map((item, index) => {
    const authors = item.authors?.length ? item.authors.slice(0, 3).join(", ") : "unknown authors";
    const year = item.year || "n.d.";
    const identifiers = [
      item.arxiv_id ? `arXiv ${item.arxiv_id}` : "",
      item.doi ? `Crossref DOI hint ${item.doi}` : ""
    ].filter(Boolean);
    const id = identifiers.length ? ` ${identifiers.join("; ")}` : "";
    const cites = Number.isFinite(item.citation_count_hint) ? ` Crossref cited-by hint ${item.citation_count_hint}.` : "";
    return `${index + 1}. ${item.title || "Untitled"} (${year}) by ${authors}.${id}.${cites}`;
  });
  const suffix = errors.length ? ` Source warnings: ${errors.join("; ")}.` : "";
  return `Live literature search found ${results.length} result(s) from Crossref/arXiv: ${lines.join(" ")}${suffix}`;
}

function inferQuotedTitle(value) {
  const match = /"([^"]{6,180})"/.exec(String(value || ""))
    || /'([^']{6,180})'/.exec(String(value || ""));
  return match ? match[1].trim() : "";
}

function extractDoi(value) {
  const match = /\b(10\.\d{4,9}\/[-._;()/:A-Z0-9]+)\b/i.exec(String(value || ""));
  return match ? match[1].replace(/[.,;)\]]+$/, "") : "";
}

function extractArxivId(value) {
  const match = /arxiv\.org\/abs\/([^/?#]+)/i.exec(String(value || ""));
  return match ? match[1] : null;
}

function normalizeCrossrefAuthors(authors) {
  if (!Array.isArray(authors)) return [];
  return authors.map((author) => [author.given, author.family].filter(Boolean).join(" ").trim())
    .filter(Boolean)
    .slice(0, 8);
}

function extractCrossrefYear(item) {
  const dateParts = item?.issued?.["date-parts"] || item?.["published-print"]?.["date-parts"] || item?.["published-online"]?.["date-parts"];
  return Array.isArray(dateParts) && Array.isArray(dateParts[0]) ? Number(dateParts[0][0]) || null : null;
}

function firstText(value) {
  return Array.isArray(value) ? String(value[0] || "").trim() : String(value || "").trim();
}

function firstXmlTag(xml, tag) {
  const pattern = new RegExp(`<${escapeRegex(tag)}\\b[^>]*>([\\s\\S]*?)<\\/${escapeRegex(tag)}>`, "i");
  const match = pattern.exec(String(xml || ""));
  return match ? match[1] : "";
}

function decodeXml(value) {
  return String(value || "")
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function normalizeWhitespace(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function stripTags(value) {
  return normalizeWhitespace(decodeXml(String(value || "").replace(/<[^>]+>/g, " ")));
}

function clampInteger(value, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return min;
  return Math.max(min, Math.min(max, Math.trunc(number)));
}

function buildRequestContext(req, clientContext) {
  const forwardedFor = String(req.headers["x-forwarded-for"] || "").split(",").map((item) => item.trim()).filter(Boolean);
  const source = typeof clientContext === "object" && clientContext ? clientContext : {};
  return {
    ip: String(source.ip || forwardedFor[0] || req.headers["x-appengine-user-ip"] || "").trim(),
    country: String(source.country || req.headers["x-appengine-country"] || req.headers["x-vercel-ip-country"] || req.headers["cf-ipcountry"] || "").trim(),
    region: String(source.region || req.headers["x-appengine-region"] || req.headers["x-vercel-ip-country-region"] || "").trim(),
    city: String(source.city || req.headers["x-appengine-city"] || req.headers["x-vercel-ip-city"] || "").trim(),
    latitude: source.latitude ?? source.lat ?? "",
    longitude: source.longitude ?? source.lon ?? source.lng ?? "",
    accuracy_m: source.accuracy_m ?? source.accuracy ?? "",
    time_zone: String(source.time_zone || source.timeZone || "").trim(),
    language: String(source.language || "").trim(),
    languages: Array.isArray(source.languages) ? source.languages.slice(0, 6).map(String) : [],
    browser_geolocation: sanitizeBrowserGeolocation(source.browser_geolocation || source.browserGeolocation),
    user_agent: String(source.user_agent || source.userAgent || req.headers["user-agent"] || "").slice(0, 300),
    url: String(source.url || "").slice(0, 500)
  };
}

function inferExplicitLocation(body) {
  const parameters = body.parameters || {};
  const text = [
    parameters.location,
    parameters.place,
    parameters.address,
    body.query
  ].map((value) => String(value || "")).join("\n");
  const coords = inferCoordinates(parameters);
  if (coords) {
    return {
      label: `${coords.latitude}, ${coords.longitude}`,
      latitude: coords.latitude,
      longitude: coords.longitude,
      accuracy_m: parameters.accuracy_m || parameters.accuracy || null,
      source: "explicit_coordinates",
      confidence: "high"
    };
  }
  const match = /\b(?:i am in|i'm in|my location is|mein standort ist|ich bin in|aufenthaltsort ist)\s+([A-Za-zÄÖÜäöüß '-]{2,80}?)(?:[.?!,;\n]|$)/i.exec(text);
  if (!match) return null;
  return {
    label: match[1].trim().replace(/[.?!,;:]+$/, ""),
    source: "explicit_text",
    confidence: "high"
  };
}

function classifyRouteRequest(query) {
  const text = String(query || "").toLowerCase();
  if (!/\b(route|directions|drive|travel|roundtrip|round trip|maps?)\b/i.test(text)) return null;
  const wantsGcp = /\bgcp\b|google cloud|cloud run|europe-west3|frankfurt/i.test(text);
  const wantsLangSmith = /langsmith|lang chain|langchain|slangsmith/i.test(text);
  const roundtrip = /roundtrip|round trip|there and back|back to (?:me|requestor|origin)|zurück/i.test(text);
  if (roundtrip) {
    return { kind: "roundtrip", targets: ["gcp", "langsmith"] };
  }
  if (wantsGcp && wantsLangSmith) {
    return { kind: "multi_stop", targets: ["gcp", "langsmith"] };
  }
  if (wantsLangSmith) return { kind: "single", targets: ["langsmith"] };
  if (wantsGcp) return { kind: "single", targets: ["gcp"] };
  return null;
}

function buildRouteResponse(routeRequest, location, evidence) {
  const origin = extractLocationCoordinates(location);
  if (!origin) {
    return {
      kind: routeRequest.kind,
      needs_location: true,
      observation: [
        "A route requires an exact requestor origin.",
        "Click the browser location button and allow geolocation, or provide explicit coordinates.",
        `Evidence available: ${evidence.join("; ") || "none"}.`
      ].join(" ")
    };
  }

  const targets = routeRequest.targets.map((key) => LOCATION_ROUTE_TARGETS[key]).filter(Boolean);
  if (!targets.length) return null;

  const originLabel = `${origin.latitude},${origin.longitude}`;
  const finalDestination = routeRequest.kind === "roundtrip"
    ? origin
    : targets[targets.length - 1];
  const waypoints = routeRequest.kind === "roundtrip"
    ? targets
    : targets.slice(0, -1);
  const url = googleMapsDirectionsUrl({
    origin,
    destination: finalDestination,
    waypoints
  });
  const targetLabels = targets.map((target) => target.label).join(" -> ");
  const notes = Array.from(new Set(targets.map((target) => target.note).filter(Boolean)));

  return {
    kind: routeRequest.kind,
    origin: {
      label: location.label || originLabel,
      latitude: origin.latitude,
      longitude: origin.longitude,
      accuracy_m: location.accuracy_m || null
    },
    targets: targets.map((target) => ({
      id: target.id,
      label: target.label,
      latitude: target.latitude,
      longitude: target.longitude,
      note: target.note
    })),
    google_maps_url: url,
    observation: [
      routeRequest.kind === "roundtrip"
        ? `Roundtrip route: requestor -> ${targetLabels} -> requestor.`
        : `Route: requestor -> ${targetLabels}.`,
      `Requestor origin: ${location.label || originLabel}.`,
      `Google Maps route: ${url}`,
      notes.length ? `Notes: ${notes.join(" ")}` : ""
    ].filter(Boolean).join(" ")
  };
}

function extractLocationCoordinates(location) {
  if (!location) return null;
  const latitude = Number(location.latitude);
  const longitude = Number(location.longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return { latitude, longitude };
}

function googleMapsDirectionsUrl({ origin, destination, waypoints = [] }) {
  const url = new URL("https://www.google.com/maps/dir/");
  url.searchParams.set("api", "1");
  url.searchParams.set("origin", `${origin.latitude},${origin.longitude}`);
  url.searchParams.set("destination", `${destination.latitude},${destination.longitude}`);
  if (waypoints.length) {
    url.searchParams.set("waypoints", waypoints.map((point) => `${point.latitude},${point.longitude}`).join("|"));
  }
  url.searchParams.set("travelmode", "driving");
  return url.toString();
}

function inferCoordinates(parameters) {
  const latitude = Number(parameters.latitude ?? parameters.lat);
  const longitude = Number(parameters.longitude ?? parameters.lon ?? parameters.lng);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
  return { latitude, longitude };
}

function sanitizeBrowserGeolocation(value) {
  const source = typeof value === "object" && value ? value : null;
  if (!source) return null;
  const latitude = Number(source.latitude ?? source.lat);
  const longitude = Number(source.longitude ?? source.lon ?? source.lng);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
  return {
    consent: source.consent === true,
    source: "browser_geolocation",
    latitude,
    longitude,
    accuracy_m: source.accuracy_m ?? source.accuracy ?? null,
    captured_at: String(source.captured_at || source.timestamp || "").slice(0, 80)
  };
}

function inferConsentedCoordinates(body, context) {
  const parameters = body.parameters || {};
  const direct = inferCoordinates(parameters);
  if (direct) {
    return {
      ...direct,
      accuracy_m: parameters.accuracy_m || parameters.accuracy || null,
      source: "explicit_coordinates"
    };
  }
  const candidates = [
    parameters.request_context?.browser_geolocation,
    parameters.browser_geolocation,
    body.meta?.request_context?.browser_geolocation,
    context.browser_geolocation
  ];
  for (const candidate of candidates) {
    const geo = sanitizeBrowserGeolocation(candidate);
    if (geo && geo.consent === true) return geo;
  }
  return null;
}

async function reverseGeocodeWithGoogleMaps(coords, apiKey) {
  const fallback = {
    label: `${coords.latitude}, ${coords.longitude}`,
    latitude: coords.latitude,
    longitude: coords.longitude,
    accuracy_m: coords.accuracy_m || null,
    source: coords.source || "browser_geolocation",
    confidence: "high",
    reverse_geocoded: false
  };
  const key = String(apiKey || "").trim();
  if (/^(?:NOT_CONFIGURED|CHANGE_ME|PLACEHOLDER)$/i.test(key)) return fallback;
  if (!key) return fallback;

  const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
  url.searchParams.set("latlng", `${coords.latitude},${coords.longitude}`);
  url.searchParams.set("key", key);
  url.searchParams.set("language", "en");

  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.status !== "OK" || !Array.isArray(payload.results) || !payload.results.length) {
    return {
      ...fallback,
      google_status: payload.status || response.statusText || "UNKNOWN"
    };
  }

  const result = payload.results[0] || {};
  return {
    ...fallback,
    label: result.formatted_address || fallback.label,
    formatted_address: result.formatted_address || null,
    place_id: result.place_id || null,
    location_type: result.geometry?.location_type || null,
    source: "browser_geolocation_google_reverse_geocode",
    reverse_geocoded: true,
    google_status: payload.status
  };
}

function inferLocationFromContext(context) {
  const hasLatitude = context.latitude !== "" && context.latitude !== null && context.latitude !== undefined;
  const hasLongitude = context.longitude !== "" && context.longitude !== null && context.longitude !== undefined;
  const lat = hasLatitude ? Number(context.latitude) : NaN;
  const lon = hasLongitude ? Number(context.longitude) : NaN;
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    return {
      label: `${lat}, ${lon}`,
      latitude: lat,
      longitude: lon,
      accuracy_m: context.accuracy_m || null,
      source: "browser_geolocation",
      confidence: "high"
    };
  }
  if (context.city || context.region || context.country) {
    return {
      label: [context.city, context.region, context.country].filter(Boolean).join(", "),
      country: context.country || null,
      region: context.region || null,
      city: context.city || null,
      source: "request_headers",
      confidence: context.city ? "medium" : "low"
    };
  }
  if (context.time_zone) {
    return {
      label: `time zone ${context.time_zone}`,
      time_zone: context.time_zone,
      source: "browser_time_zone",
      confidence: "low"
    };
  }
  return null;
}

function redactRequestContext(context) {
  return {
    ...context,
    ip: context.ip ? "[redacted]" : ""
  };
}

async function resolveToolKnowledge(body) {
  if (body.knowledge.trim()) return body.knowledge;
  if (body.okf && typeof body.okf.knowledge === "string") return body.okf.knowledge;
  if (!body.okf_key || !CRUDX_ID_RE.test(body.okf_key)) return "";
  const envelope = await readCrudxEnvelope(body.okf_key);
  const data = envelope && envelope.data ? envelope.data : envelope;
  return String(data?.knowledge || "");
}

async function invokeExternalLangGraphRuntime(runtimeUrl, payload) {
  const mode = String(LANGGRAPH_RUNTIME_MODE.value() || "standalone").toLowerCase();
  if (mode === "standalone" || runtimeUrl.endsWith("/invoke")) {
    return await invokeStandaloneLangGraphRuntime(runtimeUrl, payload);
  }
  return await invokeLangGraphApiRuntime(runtimeUrl, payload);
}

async function invokeStandaloneLangGraphRuntime(runtimeUrl, payload) {
  const token = String(LANGGRAPH_RUNTIME_TOKEN.value() || "");
  const invokeUrl = runtimeUrl.endsWith("/invoke") ? runtimeUrl : `${runtimeUrl}/invoke`;
  const response = await fetch(invokeUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify(payload)
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.ok === false) {
    throw new Error(`External LangGraph runtime failed: ${body.error || response.statusText}`);
  }
  return body;
}

async function invokeLangGraphApiRuntime(runtimeUrl, payload) {
  const token = String(LANGGRAPH_RUNTIME_TOKEN.value() || "");
  const assistantId = String(LANGGRAPH_ASSISTANT_ID.value() || "agent");
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
  const graphInput = {
    okfEnvelope: payload.okf_envelope,
    okfKey: payload.okf_key,
    input: payload.input,
    messages: payload.messages || [],
    dryRun: Boolean(payload.dry_run),
    runKey: payload.run_key,
    metadata: payload.metadata || {}
  };

  const threadResponse = await fetch(`${runtimeUrl}/threads`, {
    method: "POST",
    headers,
    body: JSON.stringify({})
  });
  const threadBody = await threadResponse.json().catch(() => ({}));
  if (!threadResponse.ok || !threadBody.thread_id) {
    throw new Error(`LangGraph thread create failed: ${threadBody.error || threadResponse.statusText}`);
  }

  const runResponse = await fetch(`${runtimeUrl}/threads/${encodeURIComponent(threadBody.thread_id)}/runs/wait`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      assistant_id: assistantId,
      input: graphInput,
      metadata: {
        source: "gcf-react-aaas-middleware",
        okf_key: payload.okf_key,
        run_key: payload.run_key
      }
    })
  });
  const runBody = await runResponse.json().catch(() => ({}));
  if (!runResponse.ok || runBody.ok === false) {
    throw new Error(`LangGraph run failed: ${runBody.error || runResponse.statusText}`);
  }

  return normalizeLangGraphState(runBody, {
    threadId: threadBody.thread_id,
    assistantId,
    runtimeUrl,
    payload
  });
}

function normalizeLangGraphState(state, context) {
  const events = Array.isArray(state.graphEvents) ? state.graphEvents : [];
  const nodes = events.map((event) => event.node).filter(Boolean);
  const agent = state.agent || unwrapOkfAgent(context.payload.okf_envelope, context.payload.okf_key);
  return {
    ok: true,
    run_key: context.payload.run_key,
    okf_key: context.payload.okf_key,
    agent_id: agent.agent_id || context.payload.agent_id || null,
    answer: String(state.finalAnswer || state.answer || "").trim(),
    response_format: "react_sections_v1",
    agent,
    graph: {
      runtime: "langgraph-api",
      graph: "crudx_okf_react_discourse",
      assistant_id: context.assistantId,
      thread_id: context.threadId,
      nodes,
      action: state.action || "none",
      history_messages: Array.isArray(context.payload.messages) ? context.payload.messages.length : 0,
      discourse_turns: Math.max(0, events.filter((event) => /^agent_|^tool_/.test(event.node || "")).length)
    },
    trace: [
      step("langgraph", "thread_created", { thread_id: context.threadId, assistant_id: context.assistantId }),
      step("langgraph", "run_completed", { nodes, action: state.action || "none" })
    ],
    langsmith: {
      enabled: true,
      project: "react-aaas-crudx",
      thread_id: context.threadId,
      assistant_id: context.assistantId,
      run_url: null,
      project_url: null,
      note: "LangSmith tracing is owned by the external LangGraph runtime/deployment."
    },
    llm: {
      provider: "gemini",
      model: "gemini-3.5-flash",
      raw: state.llmRaw || null
    },
    runtime: {
      service: "langgraph-api",
      base_url: context.runtimeUrl,
      assistant_id: context.assistantId,
      thread_id: context.threadId
    }
  };
}

async function readCrudxEnvelope(key) {
  const doc = await readCrudxDocRaw(key);
  return unwrapCrudxDoc(doc);
}

async function readCrudxDocRaw(key) {
  const response = await fetch(`${CRUDX_DOCS_API.value()}/docs/${encodeURIComponent(key)}?_=${Date.now()}`, {
    cache: "no-store"
  });
  const body = await response.json().catch(() => ({}));

  if (!response.ok || body.ok === false || !body.doc) {
    throw new Error(`CRUDX document not found: ${key}`);
  }

  return body.doc;
}

async function writeCrudxEnvelope(key, envelope) {
  if (!CRUDX_ID_RE.test(key)) {
    throw new Error(`Invalid CRUDX key: ${key}`);
  }

  const payload = {
    label: `ReAct AaaS Run - ${envelope.agent_id || "unknown"}`,
    title: `ReAct AaaS Run - ${envelope.agent_id || "unknown"}`,
    mime_type: "JSON",
    content_type: "application/json",
    value: JSON.stringify({ ...envelope, key }, null, 2),
    owner: CRUDX_OWNER,
    access_control: [CRUDX_OWNER],
    user_tags: [
      "R+",
      "data",
      "json",
      "aaas",
      "react-aaas-run",
      "agent-run",
      "run",
      "trace",
      "crudx-gcf",
      createdTimelineTag(),
      `agent:${envelope.agent_id || "unknown"}`,
      envelope.okf_key ? `okf:${envelope.okf_key}` : null,
      envelope.langsmith?.trace_id ? "langsmith" : null,
      envelope.langsmith?.trace_id ? "langsmith-trace" : null
    ].filter(Boolean)
  };

  const response = await fetch(`${CRUDX_DOCS_API.value()}/docs/${encodeURIComponent(key)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const body = await response.json().catch(() => ({}));

  if (!response.ok || body.ok === false) {
    throw new Error(`CRUDX write failed for ${key}: ${body.error || response.statusText}`);
  }
}

function createdTimelineTag(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Berlin",
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(date);
  const byType = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `Created>${byType.year}>${byType.month}>${byType.day}`;
}

function unwrapCrudxDoc(doc) {
  if (typeof doc.value === "string" && doc.value.trim()) {
    try {
      return JSON.parse(doc.value);
    } catch (error) {
      throw new Error(`CRUDX document value is not valid JSON: ${doc.key || doc.id || "unknown"}`);
    }
  }
  return doc;
}

function unwrapOkfAgent(envelope, fallbackKey) {
  const data = envelope && envelope.data ? envelope.data : envelope;
  const okf = data;

  if (!okf || !okf.okf) {
    throw new Error(`CRUDX OKF envelope is missing okf payload: ${fallbackKey}`);
  }

  return {
    agent_id: okf.agent_id || okf.meta?.name || fallbackKey,
    meta: okf.meta || {},
    okf: okf.okf,
    knowledge: okf.knowledge || ""
  };
}

function normalizeMessages(rawMessages) {
  if (!Array.isArray(rawMessages)) return [];
  return rawMessages
    .map((message) => {
      const role = String(message.role || message.type || "").toLowerCase();
      const normalizedRole = role === "assistant" || role === "agent" || role === "final"
        ? "assistant"
        : role === "system"
          ? "system"
          : "user";
      const content = String(message.content || message.text || "").trim();
      if (!content) return null;
      return {
        role: normalizedRole,
        content: content.slice(0, 1800)
      };
    })
    .filter(Boolean)
    .slice(-10);
}

function extractIsoDates(input) {
  return Array.from(String(input || "").matchAll(/\b(20\d{2}-\d{2}-\d{2})\b/g))
    .map((match) => match[1])
    .filter((value, index, all) => all.indexOf(value) === index);
}

function countWeekdays(startIso, endIso) {
  const start = parseIsoDate(startIso);
  const end = parseIsoDate(endIso);
  if (!start || !end) return 0;
  const direction = start <= end ? 1 : -1;
  let cursor = new Date(start);
  let count = 0;
  while ((direction === 1 && cursor <= end) || (direction === -1 && cursor >= end)) {
    const day = cursor.getUTCDay();
    if (day !== 0 && day !== 6) count += 1;
    cursor.setUTCDate(cursor.getUTCDate() + direction);
  }
  return count;
}

function parseIsoDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || ""));
  if (!match) return null;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return Number.isNaN(date.getTime()) ? null : date;
}

function selectKnowledgeSnippets(input, knowledge, extraPatterns = []) {
  const text = String(input || "").toLowerCase();
  const terms = Array.from(new Set(text.match(/[a-zäöüß0-9_]{4,}/gi) || []))
    .map((term) => term.toLowerCase());
  const patterns = extraPatterns.length ? extraPatterns : terms.map((term) => new RegExp(escapeRegex(term), "i"));
  return String(knowledge || "")
    .split(/\n+|(?<=\.)\s+/)
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean)
    .filter((line) => isRelevantKnowledgeLine(input, line))
    .filter((line) => !patterns.length || patterns.some((pattern) => pattern.test(line)))
    .slice(0, 5);
}

function selectOperationalStatusSnippets(input, knowledge) {
  const terms = Array.from(new Set(String(input || "").toLowerCase().match(/[a-z0-9._-]{4,}/g) || []));
  const fallbackPattern = /server|node|service|host|status|latency|degraded|online|offline|incident|ops/i;
  return String(knowledge || "")
    .split(/\n+/)
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean)
    .filter((line) => {
      const value = line.toLowerCase();
      return terms.some((term) => value.includes(term)) || fallbackPattern.test(line);
    })
    .slice(0, 5);
}

function inferPeanoOperands(body) {
  const parameters = body.parameters || {};
  const explicitLeft = parameters.left ?? parameters.x ?? parameters.a;
  const explicitRight = parameters.right ?? parameters.y ?? parameters.b;
  if (isNonNegativeIntegerLike(explicitLeft) && isNonNegativeIntegerLike(explicitRight)) {
    return { left: Number(explicitLeft), right: Number(explicitRight) };
  }

  const query = String(body.query || body.input || "");
  const addMatch = /\badd\s+(\d+)\s+(?:and|to)\s+(\d+)\b/i.exec(query)
    || /\b(\d+)\s*(?:\+|plus)\s*(\d+)\b/i.exec(query)
    || /\bA\(\s*(\d+)\s*,\s*(\d+)\s*\)/i.exec(query);
  if (addMatch) return { left: Number(addMatch[1]), right: Number(addMatch[2]) };

  const peanoTerms = Array.from(query.matchAll(/\bS\([^,\n]+?0\)+/g)).map((match) => match[0]);
  if (peanoTerms.length >= 2) {
    const left = peanoToInteger(peanoTerms[0]);
    const right = peanoToInteger(peanoTerms[1]);
    if (Number.isInteger(left) && Number.isInteger(right)) return { left, right };
  }
  return null;
}

function isNonNegativeIntegerLike(value) {
  return /^\d+$/.test(String(value ?? ""));
}

function executePeanoAddition(left, right) {
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right) || left < 0 || right < 0) {
    throw new Error("Peano addition only supports non-negative safe integers.");
  }
  if (left > 30 || right > 30) {
    throw new Error("Peano addition test endpoint is capped at operands <= 30.");
  }
  const trace = [];
  for (let remaining = right; remaining > 0; remaining -= 1) {
    trace.push(`A(${integerToPeano(left)}, ${integerToPeano(remaining)}) -> S(A(${integerToPeano(left)}, ${integerToPeano(remaining - 1)}))`);
  }
  trace.push(`A(${integerToPeano(left)}, 0) = ${integerToPeano(left)}`);
  return {
    input: {
      left,
      right,
      left_peano: integerToPeano(left),
      right_peano: integerToPeano(right)
    },
    peano_result: integerToPeano(left + right),
    integer_value: left + right,
    recursion_trace: trace
  };
}

function integerToPeano(value) {
  let result = "0";
  for (let index = 0; index < value; index += 1) result = `S(${result})`;
  return result;
}

function peanoToInteger(value) {
  let text = String(value || "").replace(/\s+/g, "");
  let count = 0;
  while (text.startsWith("S(") && text.endsWith(")")) {
    count += 1;
    text = text.slice(2, -1);
  }
  return text === "0" ? count : NaN;
}

function isRelevantKnowledgeLine(input, line) {
  const query = String(input || "").toLowerCase();
  const value = String(line || "").toLowerCase();
  const asksMax = /max|mustermann|max_01/.test(query);
  if (asksMax && /sandra|schreiber|sandra_02/.test(value)) return false;
  const asksSandra = /sandra|schreiber|sandra_02/.test(query);
  if (asksSandra && /max|mustermann|max_01/.test(value)) return false;
  return true;
}

function escapeRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function step(source, event, detail = {}) {
  return {
    at: new Date().toISOString(),
    source,
    event,
    detail
  };
}

function stableCrudxKey(namespace, seed) {
  const hash = crypto.createHash("sha256").update(`${namespace}:${seed}`).digest();
  const chars = [];
  for (let i = 0; i < 15; i += 1) {
    chars.push(CRUDX_ALPHABET[hash[i] % CRUDX_ALPHABET.length]);
  }
  return `CRUDX-${chars.slice(0, 5).join("")}-${chars.slice(5, 10).join("")}-${chars.slice(10, 15).join("")}`;
}
