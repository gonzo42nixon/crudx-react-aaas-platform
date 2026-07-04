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
const CRUDX_ID_RE = /^CRUDX-[RDUCX23458]{5}-[RDUCX23458]{5}-[RDUCX23458]{5}$/;
const CRUDX_ALPHABET = "RDUCX23458";

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
          source: "gcf-react-aaas-middleware",
          runtime_boundary: "external-langgraph",
          received_at: startedAt
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
    user_tags: ["R+", "data", "json", "aaas", "react-aaas-run", `agent:${envelope.agent_id || "unknown"}`]
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
