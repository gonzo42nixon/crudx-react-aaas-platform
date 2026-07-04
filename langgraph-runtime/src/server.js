"use strict";

const http = require("node:http");
const crypto = require("node:crypto");
const { URL } = require("node:url");
const { Annotation, END, START, StateGraph } = require("@langchain/langgraph");
const { Client } = require("langsmith");
const { RunTree } = require("langsmith/run_trees");

const PORT = Number(process.env.PORT || 8080);
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";
const LANGSMITH_ENDPOINT = process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com";
const LANGSMITH_PROJECT = process.env.LANGSMITH_PROJECT || "react-aaas-crudx";
const LANGSMITH_WORKSPACE_ID = process.env.LANGSMITH_WORKSPACE_ID || "";
const RUNTIME_TOKEN = process.env.LANGGRAPH_RUNTIME_TOKEN || "";

const server = http.createServer(async (req, res) => {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(res, 200, {
      ok: true,
      service: "crudx-react-aaas-langgraph-runtime",
      runtime: "external-langgraph",
      graph: "crudx_okf_react_discourse"
    });
    return;
  }

  if (req.method !== "POST" || url.pathname !== "/invoke") {
    sendJson(res, 404, { ok: false, error: "NOT_FOUND" });
    return;
  }

  if (RUNTIME_TOKEN && req.headers.authorization !== `Bearer ${RUNTIME_TOKEN}`) {
    sendJson(res, 401, { ok: false, error: "UNAUTHORIZED" });
    return;
  }

  try {
    const body = await readJsonBody(req);
    const result = await invokeGraph(body);
    sendJson(res, 200, result);
  } catch (error) {
    sendJson(res, 500, {
      ok: false,
      error: error.message || String(error)
    });
  }
});

if (require.main === module) {
  server.listen(PORT, () => {
    console.log(`CRUDX ReAct AaaS external LangGraph runtime listening on ${PORT}`);
  });
}

async function invokeGraph(requestBody) {
  const startedAt = new Date().toISOString();
  const input = String(requestBody.input || requestBody.query || "").trim();
  const okfKey = String(requestBody.okf_key || requestBody.okfKey || "").trim();
  const runKey = String(requestBody.run_key || requestBody.runKey || stableRunId(okfKey, input));
  const messages = normalizeMessages(requestBody.messages || requestBody.history || []);
  const dryRun = requestBody.dry_run === true || requestBody.dryRun === true;
  const okfEnvelope = requestBody.okf_envelope || requestBody.okfEnvelope;
  const trace = [step("langgraph_runtime", "invoke_received", { okf_key: okfKey, run_key: runKey })];

  if (!input) throw new Error("INPUT_REQUIRED");
  if (!okfEnvelope) throw new Error("OKF_ENVELOPE_REQUIRED");

  const langsmith = createLangSmithTrace({ runKey, input, okfKey, dryRun });
  await ensureLangSmithProject(langsmith);
  await postLangSmithRun(langsmith);

  try {
    const graphResult = await runExternalLangGraphExecutor({
      okfEnvelope,
      okfKey,
      input,
      messages,
      dryRun,
      langsmith,
      trace
    });
    await postLangGraphSpans(langsmith, graphResult, trace);

    await endLangSmithRun(langsmith, {
      ok: true,
      run_key: runKey,
      answer: graphResult.finalAnswer,
      graph: graphResult.graphSummary
    });
    await resolveLangSmithUrls(langsmith);

    return {
      ok: true,
      run_key: runKey,
      okf_key: okfKey,
      agent_id: graphResult.agent.agent_id,
      answer: graphResult.finalAnswer,
      response_format: "react_sections_v1",
      graph: graphResult.graphSummary,
      langsmith: langsmithStatus(langsmith),
      trace,
      llm: {
        provider: "gemini",
        model: GEMINI_MODEL,
        raw: graphResult.llmRaw || null
      },
      agent: graphResult.agent,
      runtime: {
        service: "external-langgraph",
        graph: "crudx_okf_react_discourse",
        started_at: startedAt,
        completed_at: new Date().toISOString()
      }
    };
  } catch (error) {
    trace.push(step("error", "runtime_failed", { message: error.message }));
    await endLangSmithRun(langsmith, undefined, error);
    await resolveLangSmithUrls(langsmith);
    throw error;
  }
}

function createExternalLangGraph() {
  const GraphState = Annotation.Root({
    okfEnvelope: Annotation(),
    okfKey: Annotation(),
    input: Annotation(),
    dryRun: Annotation(),
    messages: Annotation({
      reducer: (_left, right) => right,
      default: () => []
    }),
    agent: Annotation(),
    plan: Annotation(),
    action: Annotation(),
    observation: Annotation(),
    contextReview: Annotation(),
    reflection: Annotation(),
    followupQuestion: Annotation(),
    policyObservation: Annotation(),
    calendarObservation: Annotation(),
    synthesis: Annotation(),
    finalAnswer: Annotation(),
    llmRaw: Annotation(),
    graphEvents: Annotation({
      reducer: (left, right) => [...(left || []), ...(right || [])],
      default: () => []
    })
  });

  return new StateGraph(GraphState)
    .addNode("load_okf", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const okfEnvelope = await withLangSmithChild(
        langsmith,
        "LangGraph Node: load_okf",
        "retriever",
        { okf_key: state.okfKey },
        { graph_node: "load_okf", crudx_key: state.okfKey },
        async () => state.okfEnvelope,
        (envelope) => ({ okf_key: state.okfKey, document_loaded: Boolean(envelope) })
      );
      const agent = unwrapOkfAgent(okfEnvelope, state.okfKey);
      trace?.push(step("langgraph", "node_load_okf", {
        okf_key: state.okfKey,
        agent_id: agent.agent_id,
        tools: agent.okf.allowed_tools.map((tool) => tool.name)
      }));
      return {
        agent,
        graphEvents: [{ node: "load_okf", okf_key: state.okfKey, agent_id: agent.agent_id }]
      };
    })
    .addNode("context_review", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: context_review",
        "chain",
        {
          input: state.input,
          history_messages: state.messages.length,
          knowledge_chars: String(state.agent.knowledge || "").length
        },
        { graph_node: "context_review", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ contextReview: buildContextReview(state.input, state.messages, state.agent) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_context_review", {
        history_messages: state.messages.length,
        review_chars: result.contextReview.length
      }));
      return {
        contextReview: result.contextReview,
        graphEvents: [{ node: "context_review", history_messages: state.messages.length }]
      };
    })
    .addNode("agent_plan", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_plan",
        "chain",
        {
          input: state.input,
          history_messages: state.messages.length,
          available_tools: state.agent.okf.allowed_tools.map((tool) => tool.name)
        },
        { graph_node: "agent_plan", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => {
          const selectedTool = selectGraphTool(state.input, state.agent);
          return {
            plan: selectedTool
              ? `Start with ${selectedTool.name}, then run follow-up checks before producing the final HR recommendation.`
              : "Answer from the OKF context without invoking a configured tool.",
            action: selectedTool?.name || "none"
          };
        },
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_plan", result));
      return {
        plan: result.plan,
        action: result.action,
        graphEvents: [{ node: "agent_plan", action: result.action }]
      };
    })
    .addNode("tool_observation", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: tool_observation",
        "tool",
        { action: state.action, input: state.input },
        { graph_node: "tool_observation", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ observation: buildGraphObservation(state.input, state.agent, state.action) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_tool_observation", {
        action: state.action,
        observation_chars: result.observation.length
      }));
      return {
        observation: result.observation,
        graphEvents: [{ node: "tool_observation", action: state.action }]
      };
    })
    .addNode("agent_reflection", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_reflection",
        "chain",
        {
          plan: state.plan,
          action: state.action,
          observation: state.observation,
          context_review: state.contextReview
        },
        { graph_node: "agent_reflection", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ reflection: buildAgentReflection(state) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_reflection", { reflection_chars: result.reflection.length }));
      return {
        reflection: result.reflection,
        graphEvents: [{ node: "agent_reflection", text_chars: result.reflection.length }]
      };
    })
    .addNode("agent_followup_question", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_followup_question",
        "chain",
        { reflection: state.reflection, input: state.input },
        { graph_node: "agent_followup_question", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ followupQuestion: buildInternalFollowupQuestion(state) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_followup_question", { question_chars: result.followupQuestion.length }));
      return {
        followupQuestion: result.followupQuestion,
        graphEvents: [{ node: "agent_followup_question", text_chars: result.followupQuestion.length }]
      };
    })
    .addNode("tool_policy_check", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: tool_policy_check",
        "tool",
        { question: state.followupQuestion, tool: "search_knowledge_base" },
        { graph_node: "tool_policy_check", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ policyObservation: buildPolicyObservation(state.input, state.agent) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_tool_policy_check", { observation_chars: result.policyObservation.length }));
      return {
        policyObservation: result.policyObservation,
        graphEvents: [{ node: "tool_policy_check", action: "search_knowledge_base" }]
      };
    })
    .addNode("tool_calendar_check", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: tool_calendar_check",
        "tool",
        { question: state.followupQuestion, tool: "calculate_days" },
        { graph_node: "tool_calendar_check", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ calendarObservation: buildCalendarObservation(state.input) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_tool_calendar_check", { observation_chars: result.calendarObservation.length }));
      return {
        calendarObservation: result.calendarObservation,
        graphEvents: [{ node: "tool_calendar_check", action: "calculate_days" }]
      };
    })
    .addNode("agent_synthesis", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_synthesis",
        "chain",
        {
          observation: state.observation,
          policy_observation: state.policyObservation,
          calendar_observation: state.calendarObservation
        },
        { graph_node: "agent_synthesis", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ synthesis: buildAgentSynthesis(state) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_synthesis", { synthesis_chars: result.synthesis.length }));
      return {
        synthesis: result.synthesis,
        graphEvents: [{ node: "agent_synthesis", text_chars: result.synthesis.length }]
      };
    })
    .addNode("agent_final", async (state, config) => {
      const configurable = config.configurable || {};
      const dryRun = configurable.dryRun === true || state.dryRun === true;
      const geminiKey = configurable.geminiKey || process.env.GEMINI_API_KEY || "";
      const { langsmith, trace } = configurable;
      const prompt = buildGraphFinalPrompt(state);
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_final",
        "llm",
        {
          model: GEMINI_MODEL,
          prompt,
          plan: state.plan,
          action: state.action,
          observation: state.observation
        },
        {
          graph_node: "agent_final",
          ls_provider: "google_genai",
          ls_model_name: GEMINI_MODEL,
          agent_id: state.agent.agent_id,
          okf_key: state.okfKey
        },
        async () => {
          if (dryRun) {
            return { raw: null, text: buildDeterministicGraphSections(state) };
          }
          if (!geminiKey) {
            throw new Error("GEMINI_API_KEY is not available to the external LangGraph runtime");
          }
          return await callGemini(geminiKey, prompt);
        },
        (result) => ({
          model: GEMINI_MODEL,
          text_chars: (result.text || "").length
        })
      );
      const finalAnswer = normalizeGraphFinalAnswer(result.text, state);
      trace?.push(step("langgraph", "node_agent_final", {
        model: GEMINI_MODEL,
        text_chars: finalAnswer.length
      }));
      return {
        finalAnswer,
        llmRaw: result.raw || null,
        graphEvents: [{ node: "agent_final", text_chars: finalAnswer.length }]
      };
    })
    .addEdge(START, "load_okf")
    .addEdge("load_okf", "context_review")
    .addEdge("context_review", "agent_plan")
    .addEdge("agent_plan", "tool_observation")
    .addEdge("tool_observation", "agent_reflection")
    .addEdge("agent_reflection", "agent_followup_question")
    .addEdge("agent_followup_question", "tool_policy_check")
    .addEdge("tool_policy_check", "tool_calendar_check")
    .addEdge("tool_calendar_check", "agent_synthesis")
    .addEdge("agent_synthesis", "agent_final")
    .addEdge("agent_final", END)
    .compile();
}

async function runExternalLangGraphExecutor({ okfEnvelope, okfKey, input, messages, dryRun, langsmith, trace }) {
  const graph = createExternalLangGraph();
  const result = await withLangSmithChild(
    langsmith,
    "LangGraph Executor: external CRUDX OKF ReAct Graph",
    "chain",
    { okf_key: okfKey, input, history_messages: messages.length },
    { graph: "crudx_okf_react_discourse", okf_key: okfKey, runtime_boundary: "external" },
    () => graph.invoke(
      { okfEnvelope, okfKey, input, messages, dryRun },
      { configurable: { dryRun, geminiKey: process.env.GEMINI_API_KEY || "", langsmith, trace } }
    ),
    (state) => ({
      agent_id: state.agent?.agent_id || null,
      final_chars: (state.finalAnswer || "").length,
      nodes: (state.graphEvents || []).map((event) => event.node)
    })
  );

  return {
    agent: result.agent,
    finalAnswer: result.finalAnswer,
    llmRaw: result.llmRaw || null,
    graphEvents: result.graphEvents || [],
    graphSummary: {
      runtime: "external-langgraph-js",
      graph: "crudx_okf_react_discourse",
      nodes: (result.graphEvents || []).map((event) => event.node),
      action: result.action || "none",
      history_messages: messages.length,
      discourse_turns: Math.max(0, (result.graphEvents || []).filter((event) => /^agent_|^tool_/.test(event.node || "")).length)
    }
  };
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
    okf: {
      system_prompt: okf.okf.system_prompt || "You are a careful ReAct agent.",
      llm: {
        provider: okf.okf.llm?.provider || "gemini",
        model: okf.okf.llm?.model || GEMINI_MODEL,
        temperature: Number(okf.okf.llm?.temperature ?? 0.2)
      },
      allowed_tools: Array.isArray(okf.okf.allowed_tools) ? okf.okf.allowed_tools : [],
      mcp_endpoints: Array.isArray(okf.okf.mcp_endpoints) ? okf.okf.mcp_endpoints : []
    },
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
      return { role: normalizedRole, content: content.slice(0, 1800) };
    })
    .filter(Boolean)
    .slice(-10);
}

function buildConversationContext(messages) {
  if (!messages.length) return "No prior conversation turns supplied.";
  return messages.map((message, index) => `${index + 1}. ${message.role.toUpperCase()}: ${message.content}`).join("\n");
}

function selectGraphTool(input, agent) {
  const tools = Array.isArray(agent?.okf?.allowed_tools) ? agent.okf.allowed_tools : [];
  if (!tools.length) return null;
  const text = String(input || "").toLowerCase();
  const dates = extractIsoDates(input);
  if (/vacation|urlaub|days|tage|remaining|balance|approve|approval|period|request/.test(text)) {
    if (dates.length >= 2 || /business-day|business day|from\s+20\d{2}-\d{2}-\d{2}/i.test(input)) {
      return tools.find((tool) => /calculate|days/i.test(tool.name))
        || tools.find((tool) => /search|knowledge/i.test(tool.name))
        || tools[0];
    }
    return tools.find((tool) => /search|knowledge/i.test(tool.name))
      || tools.find((tool) => /calculate|days/i.test(tool.name))
      || tools[0];
  }
  return tools[0];
}

function buildContextReview(input, messages, agent) {
  const text = String(input || "").toLowerCase();
  const history = Array.isArray(messages) ? messages : [];
  const topics = [];
  if (/max|mustermann/i.test(input)) topics.push("employee Max Mustermann");
  if (/vacation|urlaub|days|tage|approval|approve|period|request/.test(text)) topics.push("vacation request");
  if (/2026-\d{2}-\d{2}/.test(text)) topics.push("dated absence period");
  if (!topics.length) topics.push("general OKF-backed request");
  return [
    `The current turn concerns ${topics.join(", ")}.`,
    `Prior conversation turns available: ${history.length}.`,
    `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of local knowledge.`,
    "The executor should avoid a premature final answer and first check balance, policy constraints, and date impact where applicable."
  ].join(" ");
}

function buildGraphObservation(input, agent, action) {
  const knowledge = String(agent?.knowledge || "");
  const text = String(input || "").toLowerCase();
  if (action === "none") return "No configured tool was selected; answer from OKF context and prior conversation.";
  const dates = extractIsoDates(input);
  if (/calculate|days/i.test(action) || dates.length >= 2 || /business-day|business day/.test(text)) {
    if (dates.length >= 2) {
      const days = countWeekdays(dates[0], dates[1]);
      return `Calculated business-day impact for ${dates[0]} to ${dates[1]} is ${days} business days, assuming Monday through Friday and no holiday calendar overrides.`;
    }
    return "Date calculation requested, but no exact start and end dates were supplied in the current turn.";
  }
  if (/search|knowledge/i.test(action)) {
    const snippets = selectKnowledgeSnippets(input, knowledge);
    return snippets.length
      ? `OKF knowledge lookup: ${snippets.join(" ")}`
      : "OKF knowledge lookup: no directly matching knowledge snippet was found in the OKF document.";
  }
  return "Configured tool selected from OKF. Observation is grounded in available OKF knowledge; missing external tool execution is explicitly noted.";
}

function buildAgentReflection(state) {
  const hasDates = extractIsoDates(state.input || "").length >= 2;
  const hasKnowledge = Boolean(String(state.agent?.knowledge || "").trim());
  return [
    "The first observation should be combined with OKF policy knowledge before answering.",
    hasDates ? "The request includes exact dates, so calendar impact can be checked." : "No complete date range was supplied, so date impact may need clarification.",
    hasKnowledge ? "The OKF knowledge base is available for factual grounding." : "No OKF knowledge text is available, so the answer must state that limitation.",
    `Current action was ${state.action || "none"}.`
  ].join(" ");
}

function buildInternalFollowupQuestion(state) {
  if (/2026-\d{2}-\d{2}/.test(String(state.input || ""))) {
    return "Which HR rules and date-calculation facts must be checked before this vacation request can be recommended for approval?";
  }
  return "What missing employee, policy, or date information prevents a fully grounded approval recommendation?";
}

function buildPolicyObservation(input, agent) {
  const knowledge = String(agent?.knowledge || "");
  const snippets = selectKnowledgeSnippets(input, knowledge, [/policy|rule|request|approval|approve|vacation|urlaub|entitled|advance|weeks|days/i]);
  if (snippets.length) {
    return `Policy check from OKF: ${snippets.join(" ")}`;
  }
  return "Policy check: no specific HR policy rule was found for this request in the OKF knowledge.";
}

function buildCalendarObservation(input) {
  const dates = extractIsoDates(input);
  if (dates.length >= 2) {
    const days = countWeekdays(dates[0], dates[1]);
    return `Calendar check: ${dates[0]} through ${dates[1]} spans ${days} weekday business days when weekends are excluded and no holiday override is applied.`;
  }
  if (/next month|period|date|days/i.test(input)) {
    return "Calendar check: no exact start and end dates were supplied in the current turn, so the executor cannot verify the business-day count yet.";
  }
  return "Calendar check: no date calculation was required for this request.";
}

function buildAgentSynthesis(state) {
  return [
    "Synthesis:",
    `1. Initial observation: ${state.observation || "none"}`,
    `2. Policy observation: ${state.policyObservation || "none"}`,
    `3. Calendar observation: ${state.calendarObservation || "none"}`,
    "Recommendation logic: answer only after combining balance, policy timing, and date impact; identify any missing operational approval data separately."
  ].join("\n");
}

function buildGraphFinalPrompt(state) {
  return [
    state.agent.okf.system_prompt,
    "",
    "You are the final response node in a multi-step LangGraph ReAct executor.",
    "Use the supplied graph state, OKF knowledge, and prior conversation.",
    "Return only the final user-facing answer.",
    "Do not output JSON, Markdown code fences, raw graph state, trace data, or hidden chain-of-thought.",
    "Do not use headings named Plan, Dialogue, Evidence, or Final.",
    "Write like a helpful professional assistant speaking to the user.",
    "Mention missing information only when it materially affects the answer.",
    "",
    "Graph state:",
    `Context review: ${state.contextReview || "none"}`,
    `Plan: ${state.plan || "none"}`,
    `Action: ${state.action || "none"}`,
    `Initial observation: ${state.observation || "none"}`,
    `Agent reflection: ${state.reflection || "none"}`,
    `Internal follow-up question: ${state.followupQuestion || "none"}`,
    `Policy observation: ${state.policyObservation || "none"}`,
    `Calendar observation: ${state.calendarObservation || "none"}`,
    `Synthesis: ${state.synthesis || "none"}`,
    "",
    "Available OKF knowledge:",
    state.agent.knowledge || "No extra knowledge supplied.",
    "",
    "Prior conversation:",
    buildConversationContext(state.messages || []),
    "",
    `Current user input: ${state.input}`,
    "",
    "Final response requirements:",
    "- Use prior conversation when relevant.",
    "- Separate known facts from missing information.",
    "- If facts come from OKF, simply say they are in the OKF context when useful.",
    "- LangSmith tracing is handled automatically by the platform. If the user asks for a trace, acknowledge that trace metadata is attached by the platform; do not claim you cannot create traces.",
    "- Keep the final answer concise, practical, and readable.",
    "- Do not reveal internal node names unless the user explicitly asks for debugging."
  ].join("\n");
}

function normalizeGraphFinalAnswer(text, state) {
  const raw = String(text || "").trim();
  const withoutFences = raw.replace(/^```(?:json|markdown|text)?\s*/i, "").replace(/\s*```$/i, "").trim();
  if (looksLikeJson(withoutFences)) return buildDeterministicGraphSections(state);
  const looksIncomplete = !raw
    || withoutFences.length < 180
    || !/[.!?)]$/.test(withoutFences)
    || /(?:\band stating|\band explain|\band recommend|\bwith|,\s*)$/i.test(withoutFences);
  const looksDebuggy = /(^|\n)\s*(Plan|Dialogue|Evidence|Final)\s*:/i.test(withoutFences)
    || /LangGraph Node|Graph state|tool_observation|agent_reflection/i.test(withoutFences);
  return looksIncomplete || looksDebuggy ? buildDeterministicGraphSections(state) : withoutFences;
}

function looksLikeJson(text) {
  const trimmed = String(text || "").trim();
  if (!/^[\[{]/.test(trimmed)) return false;
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
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
  const terms = Array.from(new Set(text.match(/[a-zäöüß0-9]{4,}/gi) || []))
    .map((term) => term.toLowerCase());
  const patterns = extraPatterns.length ? extraPatterns : terms.map((term) => new RegExp(escapeRegex(term), "i"));
  return String(knowledge || "")
    .split(/\n+|(?<=\.)\s+/)
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean)
    .filter((line) => isRelevantKnowledgeLine(input, line))
    .filter((line) => patterns.some((pattern) => pattern.test(line)))
    .slice(0, 5);
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

function buildDeterministicGraphSections(state) {
  const observation = String(state.observation || "No observation available.");
  const policy = String(state.policyObservation || "");
  const calendar = String(state.calendarObservation || "");
  const snippets = uniqueSentences([observation, policy, calendar])
    .filter((part) => part && !/^No observation available/i.test(part) && !/^Policy check: no/i.test(part) && !/^Calendar check: no/i.test(part));
  if (!snippets.length) {
    return "I could not find enough grounded OKF information to answer this safely. Please provide the missing facts or update the agent OKF knowledge so I can give a reliable answer.";
  }
  const dates = extractIsoDates(state.input || "");
  const dateCaveat = dates.length >= 2
    ? ""
    : " I cannot calculate an exact absence period without concrete start and end dates.";
  return `${snippets.join(" ")}${dateCaveat} Based on these OKF-grounded checks, proceed only if the operational approval record and any required coverage confirmation are complete.`;
}

function uniqueSentences(parts) {
  const seen = new Set();
  const result = [];
  for (const sentence of parts.flatMap((part) => String(part || "").split(/(?<=\.)\s+/))) {
    const cleaned = sentence
      .replace(/^OKF knowledge lookup:\s*/i, "")
      .replace(/^Policy check from OKF:\s*/i, "")
      .replace(/^Calendar check:\s*/i, "Calendar check: ")
      .trim();
    const key = cleaned.toLowerCase();
    if (!cleaned || seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
  }
  return result;
}

async function callGemini(apiKey, prompt) {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(GEMINI_MODEL)}:generateContent`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey
    },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 900 }
    })
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`Gemini request failed: ${body.error?.message || response.statusText}`);
  const text = body.candidates?.[0]?.content?.parts?.map((part) => part.text || "").join("").trim();
  if (!text) throw new Error("Gemini response did not contain text");
  return { raw: body, text };
}

function createLangSmithTrace({ runKey, input, okfKey, dryRun }) {
  const apiKey = process.env.LANGSMITH_API_KEY || "";
  if (!apiKey) return { enabled: false, reason: "LANGSMITH_API_KEY not configured", errors: [] };
  try {
    const client = new Client({
      apiKey,
      apiUrl: LANGSMITH_ENDPOINT,
      workspaceId: LANGSMITH_WORKSPACE_ID || undefined
    });
    const root = new RunTree({
      name: "CRUDX ReAct AaaS External LangGraph Invoke",
      run_type: "chain",
      inputs: { input, okf_key: okfKey, run_key: runKey },
      project_name: LANGSMITH_PROJECT,
      client,
      tags: ["crudx", "react-aaas", "external-langgraph", "live"],
      metadata: { okf_key: okfKey, run_key: runKey, dry_run: Boolean(dryRun), runtime: "external-langgraph" }
    });
    return {
      enabled: true,
      root,
      project: LANGSMITH_PROJECT,
      endpoint: LANGSMITH_ENDPOINT,
      workspaceId: LANGSMITH_WORKSPACE_ID || null,
      errors: []
    };
  } catch (error) {
    return { enabled: false, reason: error.message, errors: [error.message] };
  }
}

async function ensureLangSmithProject(langsmith) {
  if (!langsmith?.enabled || !langsmith.root?.client || !langsmith.project) return;
  try {
    const project = await langsmith.root.client.createProject({
      projectName: langsmith.project,
      description: "External LangGraph runtime traces for CRUDX ReAct Agent-as-a-Service.",
      metadata: { app: "crudx-react-aaas", runtime: "external-langgraph" },
      upsert: true
    });
    langsmith.projectId = project?.id || null;
  } catch (error) {
    rememberLangSmithError(langsmith, `project_upsert: ${error.message || error}`);
  }
}

async function postLangSmithRun(langsmith) {
  if (!langsmith?.enabled || !langsmith.root) return;
  try {
    await langsmith.root.postRun();
  } catch (error) {
    rememberLangSmithError(langsmith, error);
  }
}

async function postLangGraphSpans(langsmith, graphResult, trace) {
  const events = Array.isArray(graphResult?.graphEvents) ? graphResult.graphEvents : [];
  const summary = graphResult?.graphSummary || {};
  if (!events.length) return;

  await withLangSmithChild(
    langsmith,
    "LangGraph Executor: external crudx_okf_react_discourse",
    "chain",
    { runtime: summary.runtime, graph: summary.graph, nodes: events.map((event) => event.node), action: summary.action || "none" },
    { graph: summary.graph, runtime: summary.runtime, span_source: "external_graph_event_mirror" },
    async () => ({ ok: true }),
    () => ({ ok: true, nodes: events.map((event) => event.node), action: summary.action || "none" })
  );

  for (const event of events) {
    const node = String(event?.node || "unknown_node");
    const runType = node === "agent_final" ? "llm" : node.startsWith("tool_") ? "tool" : node === "load_okf" ? "retriever" : "chain";
    await withLangSmithChild(
      langsmith,
      `LangGraph Node: ${node}`,
      runType,
      sanitizeLangSmithPayload(event || {}),
      { graph: summary.graph, runtime: summary.runtime, graph_node: node, span_source: "external_graph_event_mirror" },
      async () => ({ ok: true }),
      () => ({ ok: true, ...sanitizeLangSmithPayload(event || {}) })
    );
  }
  trace?.push(step("langsmith", "langgraph_spans_persisted", { nodes: events.map((event) => event.node) }));
}

async function withLangSmithChild(langsmith, name, runType, inputs, metadata, fn, outputMapper) {
  if (!langsmith?.enabled || !langsmith.root) return await fn();
  const child = langsmith.root.createChild({
    name,
    run_type: runType,
    inputs: inputs || {},
    metadata: metadata || {},
    tags: ["crudx", "react-aaas", "external-langgraph"]
  });
  try {
    await child.postRun();
  } catch (error) {
    rememberLangSmithError(langsmith, error);
  }
  try {
    const result = await fn();
    const outputs = outputMapper ? outputMapper(result) : { ok: true };
    try {
      await child.end(outputs);
      await child.patchRun();
    } catch (error) {
      rememberLangSmithError(langsmith, error);
    }
    return result;
  } catch (error) {
    try {
      await child.end(undefined, error.message);
      await child.patchRun();
    } catch (traceError) {
      rememberLangSmithError(langsmith, traceError);
    }
    throw error;
  }
}

async function endLangSmithRun(langsmith, outputs, error) {
  if (!langsmith?.enabled || !langsmith.root) return;
  try {
    if (error) await langsmith.root.end(undefined, error.message);
    else await langsmith.root.end(outputs || { ok: true });
    await langsmith.root.patchRun();
  } catch (traceError) {
    rememberLangSmithError(langsmith, traceError);
  }
}

async function resolveLangSmithUrls(langsmith) {
  if (!langsmith?.enabled || !langsmith.root) return;
  const runId = langsmith.root.id;
  langsmith.runId = runId;
  langsmith.traceId = langsmith.root.trace_id || runId;
  if (langsmith.workspaceId && langsmith.projectId && runId) {
    const base = "https://smith.langchain.com";
    langsmith.projectUrl = `${base}/o/${langsmith.workspaceId}/projects/p/${langsmith.projectId}`;
    langsmith.runUrl = `${langsmith.projectUrl}/r/${runId}?trace_id=${langsmith.traceId}`;
  }
}

function langsmithStatus(langsmith) {
  return {
    enabled: Boolean(langsmith?.enabled),
    project: langsmith?.project || LANGSMITH_PROJECT,
    endpoint: langsmith?.endpoint || LANGSMITH_ENDPOINT,
    run_id: langsmith?.runId || langsmith?.root?.id || null,
    trace_id: langsmith?.traceId || langsmith?.root?.trace_id || null,
    run_url: langsmith?.runUrl || null,
    project_url: langsmith?.projectUrl || null,
    errors: langsmith?.errors || [],
    reason: langsmith?.reason || null
  };
}

function rememberLangSmithError(langsmith, error) {
  if (!langsmith) return;
  const message = typeof error === "string" ? error : error?.message || String(error);
  langsmith.errors = [...(langsmith.errors || []), message];
}

function sanitizeLangSmithPayload(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function step(source, event, detail = {}) {
  return { at: new Date().toISOString(), source, event, detail };
}

function stableRunId(okfKey, input) {
  return crypto.createHash("sha256").update(`${okfKey}:${input}:${Date.now()}`).digest("hex").slice(0, 32);
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 2_000_000) {
        reject(new Error("REQUEST_TOO_LARGE"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(new Error("INVALID_JSON"));
      }
    });
    req.on("error", reject);
  });
}

module.exports = {
  createExternalLangGraph,
  invokeGraph,
  runExternalLangGraphExecutor,
  server
};
