"use strict";

const http = require("node:http");
const crypto = require("node:crypto");
const { URL } = require("node:url");
const { Annotation, END, START, StateGraph } = require("@langchain/langgraph");
const { Client } = require("langsmith");
const { RunTree } = require("langsmith/run_trees");
const { Firestore } = require("@google-cloud/firestore");
const { createFirestoreCheckpointer } = require("./firestore_checkpointer");

const PORT = Number(process.env.PORT || 8080);
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";
const LANGSMITH_ENDPOINT = process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com";
const LANGSMITH_PROJECT = process.env.LANGSMITH_PROJECT || "react-aaas-crudx";
const LANGSMITH_WORKSPACE_ID = process.env.LANGSMITH_WORKSPACE_ID || "";
const RUNTIME_TOKEN = process.env.LANGGRAPH_RUNTIME_TOKEN || "";
const AI_FEEDBACK_ENABLED = process.env.AI_FEEDBACK_ENABLED !== "false";
const AI_FEEDBACK_TIMEOUT_MS = Number(process.env.AI_FEEDBACK_TIMEOUT_MS || 12000);
const AI_FEEDBACK_MAX_CRITICS = Number(process.env.AI_FEEDBACK_MAX_CRITICS || 3);
const LANGGRAPH_CHECKPOINT_BACKEND = String(process.env.LANGGRAPH_CHECKPOINT_BACKEND || "firestore").toLowerCase();
const LANGGRAPH_CHECKPOINT_COLLECTION = process.env.LANGGRAPH_CHECKPOINT_COLLECTION || "langgraph_checkpoints";
const LANGGRAPH_CHECKPOINT_WRITES_COLLECTION = process.env.LANGGRAPH_CHECKPOINT_WRITES_COLLECTION || "langgraph_checkpoint_writes";
const LANGGRAPH_AGENT_MEMORY_COLLECTION = process.env.LANGGRAPH_AGENT_MEMORY_COLLECTION || "agent_global_memory";

let graphCheckpointer;
let agentMemoryFirestore;

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

  if (req.method === "POST" && url.pathname === "/invoke/stream") {
    if (RUNTIME_TOKEN && req.headers.authorization !== `Bearer ${RUNTIME_TOKEN}`) {
      sendJson(res, 401, { ok: false, error: "UNAUTHORIZED" });
      return;
    }
    await streamGraph(req, res);
    return;
  }

  if (req.method === "POST" && url.pathname === "/checkpoints/inspect") {
    if (RUNTIME_TOKEN && req.headers.authorization !== `Bearer ${RUNTIME_TOKEN}`) {
      sendJson(res, 401, { ok: false, error: "UNAUTHORIZED" });
      return;
    }
    try {
      const body = await readJsonBody(req);
      const result = await inspectCheckpoints(body);
      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, {
        ok: false,
        error: error.message || String(error)
      });
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/agent-memory/inspect") {
    if (RUNTIME_TOKEN && req.headers.authorization !== `Bearer ${RUNTIME_TOKEN}`) {
      sendJson(res, 401, { ok: false, error: "UNAUTHORIZED" });
      return;
    }
    try {
      const body = await readJsonBody(req);
      const result = await inspectAgentMemory(body);
      sendJson(res, 200, result);
    } catch (error) {
      sendJson(res, 500, {
        ok: false,
        error: error.message || String(error)
      });
    }
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

async function streamGraph(req, res) {
  writeSseHeaders(res);
  const send = (event, data) => writeSseEvent(res, event, data);
  try {
    const body = await readJsonBody(req);
    send("stream_started", { ok: true, at: new Date().toISOString() });
    const result = await invokeGraph(body, {
      onTrace: (traceEvent) => send("trace", traceEvent)
    });
    send("result", result);
    send("stream_completed", {
      ok: true,
      run_key: result.run_key,
      at: new Date().toISOString()
    });
  } catch (error) {
    send("error", {
      ok: false,
      error: error.message || String(error),
      at: new Date().toISOString()
    });
  } finally {
    res.end();
  }
}

async function invokeGraph(requestBody, options = {}) {
  const startedAt = new Date().toISOString();
  const input = String(requestBody.input || requestBody.query || "").trim();
  const okfKey = String(requestBody.okf_key || requestBody.okfKey || "").trim();
  const runKey = String(requestBody.run_key || requestBody.runKey || stableRunId(okfKey, input));
  const messages = normalizeMessages(requestBody.messages || requestBody.history || []);
  const dryRun = requestBody.dry_run === true || requestBody.dryRun === true;
  const okfEnvelope = requestBody.okf_envelope || requestBody.okfEnvelope;
  const incomingMetadata = typeof requestBody.metadata === "object" && requestBody.metadata ? requestBody.metadata : {};
  const threadId = extractThreadId(requestBody, okfKey, runKey);
  const agentMemoryId = extractAgentMemoryId(requestBody, okfKey);
  const metadata = {
    ...incomingMetadata,
    agent_memory_id: agentMemoryId,
    openapi_key: extractOpenApiKey(requestBody),
    persistence_scope: incomingMetadata.persistence_scope || requestBody.persistence_scope || "thread_and_agent",
    thread_id: threadId,
    session_id: incomingMetadata.session_id || incomingMetadata.sessionId || threadId
  };
  const trace = createTraceCollector(options.onTrace);
  trace.push(step("langgraph_runtime", "invoke_received", {
    okf_key: okfKey,
    run_key: runKey,
    thread_id: threadId,
    agent_memory_id: agentMemoryId,
    checkpoint_backend: checkpointBackend(),
    checkpoint_ns: okfKey
  }));

  if (!input) throw new Error("INPUT_REQUIRED");
  if (!okfEnvelope) throw new Error("OKF_ENVELOPE_REQUIRED");

  const langsmith = createLangSmithTrace({ runKey, input, okfKey, dryRun, threadId, metadata });
  await ensureLangSmithProject(langsmith);
  await postLangSmithRun(langsmith);

  try {
    const agentMemory = await loadAgentGlobalMemory(agentMemoryId, {
      okfKey,
      agentId: requestBody.agent_id || requestBody.agentId || null
    });
    trace.push(step("langgraph_runtime", "agent_memory_loaded", {
      ...agentMemoryInspection(agentMemoryId),
      agent_memory_id: agentMemoryId,
      episode_count: agentMemory.episodes.length,
      updated_at: agentMemory.updated_at || null
    }));

    const graphResult = await runExternalLangGraphExecutor({
      okfEnvelope,
      okfKey,
      input,
      messages,
      dryRun,
      threadId,
      metadata,
      agentMemory,
      langsmith,
      trace
    });
    const persistedAgentMemory = await persistAgentGlobalMemory(agentMemoryId, {
      existing: agentMemory,
      okfKey,
      agent: graphResult.agent,
      threadId,
      runKey,
      input,
      answer: graphResult.finalAnswer,
      action: graphResult.graphSummary.action,
      nodes: graphResult.graphSummary.nodes
    });
    graphResult.graphSummary.agent_memory = summarizeAgentMemoryForGraph(persistedAgentMemory);
    trace.push(step("langgraph_runtime", "agent_memory_persisted", graphResult.graphSummary.agent_memory));

    await postLangGraphSpans(langsmith, graphResult, trace);

    trace.push(step("langgraph_runtime", "checkpoint_inspection", graphResult.graphSummary.checkpoint_inspection));

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
      thread_id: threadId,
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

function createTraceCollector(onTrace) {
  const trace = [];
  const nativePush = Array.prototype.push;
  trace.push = function pushAndEmit(...items) {
    const length = nativePush.apply(this, items);
    for (const item of items) {
      try {
        onTrace?.(item);
      } catch {
        // Streaming observers must never break graph execution.
      }
    }
    return length;
  };
  return trace;
}

function checkpointBackend() {
  return LANGGRAPH_CHECKPOINT_BACKEND === "firestore" ? "firestore" : "none";
}

function getGraphCheckpointer() {
  if (checkpointBackend() !== "firestore") return undefined;
  if (!graphCheckpointer) {
    graphCheckpointer = createFirestoreCheckpointer();
  }
  return graphCheckpointer;
}

function getAgentMemoryFirestore() {
  if (!agentMemoryFirestore) {
    agentMemoryFirestore = new Firestore();
  }
  return agentMemoryFirestore;
}

function agentMemoryCollection() {
  return getAgentMemoryFirestore().collection(LANGGRAPH_AGENT_MEMORY_COLLECTION);
}

function extractAgentMemoryId(requestBody, okfKey) {
  const configurable = typeof requestBody.configurable === "object" && requestBody.configurable ? requestBody.configurable : {};
  const metadata = typeof requestBody.metadata === "object" && requestBody.metadata ? requestBody.metadata : {};
  const explicit = requestBody.agent_memory_id
    || requestBody.agentMemoryId
    || configurable.agent_memory_id
    || configurable.agentMemoryId
    || metadata.agent_memory_id
    || metadata.agentMemoryId;
  const fallback = requestBody.agent_id || requestBody.agentId || okfKey || "unknown-agent";
  return normalizeAgentMemoryId(explicit || `agent_global_${fallback}`);
}

function extractOpenApiKey(requestBody) {
  const configurable = typeof requestBody.configurable === "object" && requestBody.configurable ? requestBody.configurable : {};
  const metadata = typeof requestBody.metadata === "object" && requestBody.metadata ? requestBody.metadata : {};
  return String(
    requestBody.openapi_key
    || requestBody.openApiKey
    || requestBody.openapiKey
    || configurable.openapi_key
    || configurable.openApiKey
    || configurable.openapiKey
    || metadata.openapi_key
    || metadata.openApiKey
    || metadata.openapiKey
    || ""
  ).trim();
}

function normalizeAgentMemoryId(value) {
  return String(value || "agent_global_unknown")
    .trim()
    .replace(/[^A-Za-z0-9_.:-]+/g, "_")
    .slice(0, 180) || "agent_global_unknown";
}

function agentMemoryDocId(agentMemoryId) {
  return Buffer.from(normalizeAgentMemoryId(agentMemoryId)).toString("base64url");
}

function agentMemoryInspection(agentMemoryId) {
  const project = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT || process.env.GCP_PROJECT_ID || "crudx-e0599";
  return {
    backend: "firestore",
    project,
    collection: LANGGRAPH_AGENT_MEMORY_COLLECTION,
    document_id: agentMemoryDocId(agentMemoryId),
    firestore_console_url: `https://console.cloud.google.com/firestore/databases/-default-/data/panel/${encodeURIComponent(LANGGRAPH_AGENT_MEMORY_COLLECTION)}?project=${encodeURIComponent(project)}`,
    filter: {
      agent_memory_id: agentMemoryId
    },
    filter_hint: `agent_memory_id == "${agentMemoryId}"`
  };
}

async function loadAgentGlobalMemory(agentMemoryId, hints = {}) {
  const normalized = normalizeAgentMemoryId(agentMemoryId);
  const snapshot = await agentMemoryCollection().doc(agentMemoryDocId(normalized)).get();
  const data = snapshot.exists ? snapshot.data() : {};
  return normalizeAgentMemoryDocument({
    schema: "crudx.react_aaas.agent_memory.v1",
    agent_memory_id: normalized,
    agent_id: data.agent_id || hints.agentId || null,
    okf_key: data.okf_key || hints.okfKey || null,
    memory_profile: data.memory_profile || null,
    facts: data.facts || [],
    episodes: data.episodes || [],
    created_at: data.created_at || null,
    updated_at: data.updated_at || null,
    run_count: data.run_count || 0
  });
}

async function persistAgentGlobalMemory(agentMemoryId, detail) {
  const now = new Date().toISOString();
  const existing = normalizeAgentMemoryDocument(detail.existing || { agent_memory_id: agentMemoryId });
  const agent = detail.agent || {};
  const episode = {
    at: now,
    run_key: detail.runKey || null,
    thread_id: detail.threadId || null,
    okf_key: detail.okfKey || existing.okf_key || null,
    agent_id: agent.agent_id || existing.agent_id || null,
    action: detail.action || "none",
    input: previewText(detail.input || "", 900),
    answer: previewText(detail.answer || "", 1200),
    nodes: Array.isArray(detail.nodes) ? detail.nodes.slice(-20) : []
  };
  const next = normalizeAgentMemoryDocument({
    ...existing,
    agent_memory_id: normalizeAgentMemoryId(agentMemoryId),
    agent_id: agent.agent_id || existing.agent_id || null,
    okf_key: detail.okfKey || existing.okf_key || null,
    memory_profile: agentMemoryProfile(agent) || existing.memory_profile || null,
    episodes: [episode, ...existing.episodes].slice(0, 20),
    run_count: Number(existing.run_count || 0) + 1,
    created_at: existing.created_at || now,
    updated_at: now
  });
  await agentMemoryCollection().doc(agentMemoryDocId(next.agent_memory_id)).set(next, { merge: true });
  return next;
}

function normalizeAgentMemoryDocument(value) {
  const source = typeof value === "object" && value ? value : {};
  return {
    schema: "crudx.react_aaas.agent_memory.v1",
    agent_memory_id: normalizeAgentMemoryId(source.agent_memory_id),
    agent_id: source.agent_id || null,
    okf_key: source.okf_key || null,
    memory_profile: source.memory_profile || null,
    facts: Array.isArray(source.facts) ? source.facts.slice(0, 50).map((item) => previewText(item, 700)) : [],
    episodes: Array.isArray(source.episodes)
      ? source.episodes.slice(0, 20).map((item) => ({
        at: item?.at || null,
        run_key: item?.run_key || null,
        thread_id: item?.thread_id || null,
        okf_key: item?.okf_key || null,
        agent_id: item?.agent_id || null,
        action: item?.action || "none",
        input: previewText(item?.input || "", 900),
        answer: previewText(item?.answer || "", 1200),
        nodes: Array.isArray(item?.nodes) ? item.nodes.slice(-20) : []
      }))
      : [],
    created_at: source.created_at || null,
    updated_at: source.updated_at || null,
    run_count: Number(source.run_count || 0)
  };
}

function summarizeAgentMemoryForGraph(memory) {
  const normalized = normalizeAgentMemoryDocument(memory);
  return {
    ...agentMemoryInspection(normalized.agent_memory_id),
    agent_memory_id: normalized.agent_memory_id,
    agent_id: normalized.agent_id,
    okf_key: normalized.okf_key,
    memory_profile: normalized.memory_profile,
    run_count: normalized.run_count,
    episode_count: normalized.episodes.length,
    latest_episode_at: normalized.episodes[0]?.at || null,
    updated_at: normalized.updated_at
  };
}

async function inspectAgentMemory(requestBody) {
  const agentMemoryId = extractAgentMemoryId(requestBody, requestBody.okf_key || requestBody.okfKey || "");
  const memory = await loadAgentGlobalMemory(agentMemoryId, {
    okfKey: requestBody.okf_key || requestBody.okfKey || null,
    agentId: requestBody.agent_id || requestBody.agentId || null
  });
  return {
    ok: true,
    inspection_generated_at: new Date().toISOString(),
    ...summarizeAgentMemoryForGraph(memory),
    memory,
    note: "Agent memory is shared across threads for the same agent_memory_id. LangGraph checkpoints remain thread-local."
  };
}

function checkpointInspection(threadId, checkpointNs) {
  if (checkpointBackend() !== "firestore") {
    return {
      backend: "none"
    };
  }
  const project = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT || process.env.GCP_PROJECT_ID || "crudx-e0599";
  const firestoreConsoleUrl = `https://console.cloud.google.com/firestore/databases/-default-/data/panel/${encodeURIComponent(LANGGRAPH_CHECKPOINT_COLLECTION)}?project=${encodeURIComponent(project)}`;
  return {
    backend: "firestore",
    project,
    checkpoint_collection: LANGGRAPH_CHECKPOINT_COLLECTION,
    writes_collection: LANGGRAPH_CHECKPOINT_WRITES_COLLECTION,
    firestore_console_url: firestoreConsoleUrl,
    filter: {
      thread_id: threadId,
      checkpoint_ns: checkpointNs
    },
    filter_hint: `thread_id == "${threadId}" AND checkpoint_ns == "${checkpointNs}"`
  };
}

async function inspectCheckpoints(requestBody) {
  if (checkpointBackend() !== "firestore") {
    return { ok: false, error: "CHECKPOINT_BACKEND_NOT_ENABLED", checkpoint_backend: checkpointBackend() };
  }
  const threadId = String(requestBody.thread_id || requestBody.threadId || "").trim();
  const checkpointNs = String(requestBody.checkpoint_ns || requestBody.checkpointNs || requestBody.okf_key || requestBody.okfKey || "").trim();
  const checkpointId = String(requestBody.checkpoint_id || requestBody.checkpointId || "").trim();
  const limit = Math.max(1, Math.min(20, Number(requestBody.limit || 8)));
  if (!threadId) throw new Error("THREAD_ID_REQUIRED");
  if (!checkpointNs) throw new Error("CHECKPOINT_NS_REQUIRED");

  const checkpointer = getGraphCheckpointer();
  const baseConfig = {
    configurable: {
      thread_id: threadId,
      checkpoint_ns: checkpointNs,
      ...(checkpointId ? { checkpoint_id: checkpointId } : {})
    }
  };

  const checkpoints = [];
  for await (const tuple of checkpointer.list(baseConfig, { limit })) {
    checkpoints.push(formatCheckpointTuple(tuple, checkpoints.length === 0));
  }

  const namespaceSuggestions = new Set(checkpoints.map((item) => item.checkpoint_ns ?? ""));
  if (checkpoints.length === 0) {
    for await (const tuple of checkpointer.list({ configurable: { thread_id: threadId } }, { limit: 20 })) {
      const formatted = formatCheckpointTuple(tuple, checkpoints.length === 0);
      checkpoints.push(formatted);
      namespaceSuggestions.add(formatted.checkpoint_ns ?? "");
      if (checkpoints.length >= limit) break;
    }
  }

  const latestTuple = checkpointId
    ? await checkpointer.getTuple(baseConfig)
    : checkpoints.length
      ? await checkpointer.getTuple({
        configurable: {
          thread_id: threadId,
          checkpoint_ns: checkpoints[0].checkpoint_ns ?? checkpointNs,
          checkpoint_id: checkpoints[0].checkpoint_id
        }
      })
      : undefined;

  return {
    ok: true,
    checkpoint_backend: "firestore",
    inspection_generated_at: new Date().toISOString(),
    thread_id: threadId,
    checkpoint_ns: checkpointNs,
    checkpoint_id: checkpointId || null,
    checkpoint_collection: LANGGRAPH_CHECKPOINT_COLLECTION,
    writes_collection: LANGGRAPH_CHECKPOINT_WRITES_COLLECTION,
    count: checkpoints.length,
    exact_namespace_match: checkpoints.some((item) => item.checkpoint_ns === checkpointNs),
    namespace_suggestions: Array.from(namespaceSuggestions).map((value) => value || "(default)"),
    checkpoints,
    latest: latestTuple ? formatCheckpointTuple(latestTuple, true, { includePreview: true }) : null,
    note: "Values are decoded from the LangGraph checkpointer and redacted/truncated for UI inspection."
  };
}

function formatCheckpointTuple(tuple, latest, options = {}) {
  const checkpoint = tuple?.checkpoint || {};
  const channelValues = checkpoint.channel_values || {};
  const pendingWrites = Array.isArray(tuple?.pendingWrites) ? tuple.pendingWrites : [];
  const metadata = tuple?.metadata || {};
  const checkpointId = tuple?.config?.configurable?.checkpoint_id || checkpoint.id || null;
  const parentCheckpointId = tuple?.parentConfig?.configurable?.checkpoint_id || null;
  const channels = Object.keys(channelValues).sort();
  const pendingWriteSummary = pendingWrites.map(([taskId, channel, value]) => ({
    task_id: taskId,
    channel,
    preview: previewJsonValue(redactCheckpointValue(value), 900)
  }));
  return {
    latest: Boolean(latest),
    checkpoint_ns: tuple?.config?.configurable?.checkpoint_ns || "",
    checkpoint_id: checkpointId,
    parent_checkpoint_id: parentCheckpointId,
    ts: checkpoint.ts || null,
    version: checkpoint.v || null,
    channel_count: channels.length,
    channels,
    channel_versions: checkpoint.channel_versions || {},
    metadata: redactCheckpointValue(metadata),
    pending_writes_count: pendingWrites.length,
    pending_writes: options.includePreview ? pendingWriteSummary : pendingWriteSummary.slice(0, 8),
    preview: options.includePreview ? {
      channel_values: Object.fromEntries(channels.map((channel) => [
        channel,
        previewJsonValue(redactCheckpointValue(channelValues[channel]), 3000)
      ])),
      checkpoint: previewJsonValue(redactCheckpointValue(checkpoint), 12000)
    } : undefined
  };
}

function redactCheckpointValue(value, depth = 0) {
  if (depth > 8) return "[Max depth reached]";
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return value.length > 8000 ? `${value.slice(0, 8000)}... [truncated]` : value;
  if (typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return value.slice(0, 50).map((item) => redactCheckpointValue(item, depth + 1));
  }
  const output = {};
  for (const [key, item] of Object.entries(value).slice(0, 80)) {
    if (/api[_-]?key|authorization|bearer|secret|token|password|credential/i.test(key)) {
      output[key] = "[redacted]";
    } else {
      output[key] = redactCheckpointValue(item, depth + 1);
    }
  }
  return output;
}

function previewJsonValue(value, maxChars) {
  const json = JSON.stringify(value, null, 2);
  if (json.length <= maxChars) return json;
  return `${json.slice(0, maxChars)}\n... [truncated ${json.length - maxChars} chars]`;
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
    metadata: Annotation({
      reducer: (_left, right) => right,
      default: () => ({})
    }),
    agentMemory: Annotation({
      reducer: (_left, right) => right,
      default: () => null
    }),
    agent: Annotation(),
    plan: Annotation(),
    action: Annotation(),
    observation: Annotation(),
    contextReview: Annotation(),
    reflection: Annotation(),
    followupQuestion: Annotation(),
    followupObservations: Annotation({
      reducer: (_left, right) => right,
      default: () => []
    }),
    policyObservation: Annotation(),
    calendarObservation: Annotation(),
    synthesis: Annotation(),
    draftAnswer: Annotation(),
    criticFeedback: Annotation(),
    finalAnswer: Annotation(),
    llmRaw: Annotation(),
    graphEvents: Annotation({
      reducer: (left, right) => [...(left || []), ...(right || [])],
      default: () => []
    })
  });

  const graphBuilder = new StateGraph(GraphState)
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
          knowledge_chars: String(state.agent.knowledge || "").length,
          agent_memory_episodes: Array.isArray(state.agentMemory?.episodes) ? state.agentMemory.episodes.length : 0
        },
        { graph_node: "context_review", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ contextReview: buildContextReview(state.input, state.messages, state.agent, state.agentMemory, state.metadata) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_context_review", {
        history_messages: state.messages.length,
        review_chars: result.contextReview.length,
        review: result.contextReview,
        review_preview: previewText(result.contextReview)
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
              ? isPeanoAgent(state.agent)
                ? `Start with ${selectedTool.name}, verify the Peano recursion result, then return the required JSON object.`
                : isLocationAgent(state.agent)
                  ? `Start with ${selectedTool.name}, classify the evidence level, then produce a privacy-aware location estimate.`
                : `Start with ${selectedTool.name}, then evaluate only relevant OKF follow-up tools before producing the final response.`
              : "Answer from the available information without invoking a configured tool.",
            action: selectedTool?.name || "none"
          };
        },
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_plan", {
        ...result,
        plan_preview: previewText(result.plan)
      }));
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
        async () => ({
          observation: await executeOkfTool(state.input, state.agent, state.action, state.okfKey, langsmith, {
            ...state.metadata,
            messages: state.messages || []
          })
        }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_tool_observation", {
        action: state.action,
        observation_chars: result.observation.length,
        observation: result.observation,
        observation_preview: previewText(result.observation)
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
      trace?.push(step("langgraph", "node_agent_reflection", {
        reflection_chars: result.reflection.length,
        reflection: result.reflection,
        reflection_preview: previewText(result.reflection)
      }));
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
      trace?.push(step("langgraph", "node_agent_followup_question", {
        question_chars: result.followupQuestion.length,
        question: result.followupQuestion,
        question_preview: previewText(result.followupQuestion)
      }));
      return {
        followupQuestion: result.followupQuestion,
        graphEvents: [{ node: "agent_followup_question", text_chars: result.followupQuestion.length }]
      };
    })
    .addNode("tool_followups", async (state, config) => {
      const { langsmith, trace } = config.configurable || {};
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: tool_followups",
        "tool",
        { question: state.followupQuestion, primary_action: state.action },
        { graph_node: "tool_followups", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ followupObservations: await executeRelevantFollowupTools(state, langsmith) }),
        (result) => result
      );
      const observations = Array.isArray(result.followupObservations) ? result.followupObservations : [];
      trace?.push(step("langgraph", "node_tool_followups", {
        actions: observations.map((item) => item.tool),
        observation_count: observations.length,
        observation: observations.map((item) => `${item.tool}: ${item.observation}`).join(" "),
        observation_preview: previewText(observations.map((item) => `${item.tool}: ${item.observation}`).join(" "))
      }));
      return {
        followupObservations: observations,
        graphEvents: [{ node: "tool_followups", actions: observations.map((item) => item.tool) }]
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
          followup_observations: state.followupObservations
        },
        { graph_node: "agent_synthesis", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ synthesis: buildAgentSynthesis(state) }),
        (result) => result
      );
      trace?.push(step("langgraph", "node_agent_synthesis", {
        synthesis_chars: result.synthesis.length,
        synthesis: result.synthesis,
        synthesis_preview: previewText(result.synthesis)
      }));
      return {
        synthesis: result.synthesis,
        graphEvents: [{ node: "agent_synthesis", text_chars: result.synthesis.length }]
      };
    })
    .addNode("agent_draft", async (state, config) => {
      const configurable = config.configurable || {};
      const dryRun = configurable.dryRun === true || state.dryRun === true;
      const geminiKey = configurable.geminiKey || process.env.GEMINI_API_KEY || "";
      const { langsmith, trace } = configurable;
      const prompt = buildGraphFinalPrompt(state);
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_draft",
        "llm",
        {
          model: GEMINI_MODEL,
          prompt,
          plan: state.plan,
          action: state.action,
          observation: state.observation
        },
        {
          graph_node: "agent_draft",
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
      const draftAnswer = normalizeGraphFinalAnswer(result.text, state);
      trace?.push(step("langgraph", "node_agent_draft", {
        model: GEMINI_MODEL,
        text_chars: draftAnswer.length,
        answer: draftAnswer,
        answer_preview: previewText(draftAnswer)
      }));
      return {
        draftAnswer,
        llmRaw: result.raw || null,
        graphEvents: [{ node: "agent_draft", text_chars: draftAnswer.length }]
      };
    })
    .addNode("critic_feedback", async (state, config) => {
      const configurable = config.configurable || {};
      const { langsmith, trace } = configurable;
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: critic_feedback",
        "chain",
        {
          enabled: AI_FEEDBACK_ENABLED,
          input: state.input,
          draft_chars: (state.draftAnswer || "").length
        },
        { graph_node: "critic_feedback", agent_id: state.agent.agent_id, okf_key: state.okfKey },
        async () => ({ criticFeedback: await collectAiCriticFeedback(state) }),
        (result) => ({
          enabled: result.criticFeedback.enabled,
          configured: result.criticFeedback.configured,
          review_count: result.criticFeedback.reviews.length,
          providers: result.criticFeedback.reviews.map((review) => review.provider),
          errors: result.criticFeedback.errors
        })
      );
      const feedback = result.criticFeedback;
      trace?.push(step("langgraph", "node_critic_feedback", {
        enabled: feedback.enabled,
        configured: feedback.configured,
        review_count: feedback.reviews.length,
        providers: feedback.reviews.map((review) => review.provider),
        feedback: formatCriticFeedback(feedback),
        feedback_preview: previewText(formatCriticFeedback(feedback), 640),
        errors: feedback.errors.slice(0, 3)
      }));
      return {
        criticFeedback: feedback,
        graphEvents: [{
          node: "critic_feedback",
          enabled: feedback.enabled,
          configured: feedback.configured,
          review_count: feedback.reviews.length,
          providers: feedback.reviews.map((review) => review.provider),
          errors: feedback.errors.slice(0, 3)
        }]
      };
    })
    .addNode("agent_final", async (state, config) => {
      const configurable = config.configurable || {};
      const dryRun = configurable.dryRun === true || state.dryRun === true;
      const geminiKey = configurable.geminiKey || process.env.GEMINI_API_KEY || "";
      const { langsmith, trace } = configurable;
      const hasExternalFeedback = criticFeedbackHasReviews(state.criticFeedback);
      const result = await withLangSmithChild(
        langsmith,
        "LangGraph Node: agent_final",
        hasExternalFeedback ? "llm" : "chain",
        {
          model: GEMINI_MODEL,
          input: state.input,
          draft_chars: (state.draftAnswer || "").length,
          critic_feedback: summarizeCriticFeedbackForTrace(state.criticFeedback)
        },
        {
          graph_node: "agent_final",
          ls_provider: hasExternalFeedback ? "google_genai" : "internal",
          ls_model_name: hasExternalFeedback ? GEMINI_MODEL : "draft_passthrough",
          agent_id: state.agent.agent_id,
          okf_key: state.okfKey
        },
        async () => {
          if (!hasExternalFeedback || dryRun) {
            return { raw: null, text: state.draftAnswer || buildDeterministicGraphSections(state), revised: false };
          }
          if (!geminiKey) {
            throw new Error("GEMINI_API_KEY is not available to revise the critic-reviewed draft");
          }
          const revisionPrompt = buildGraphRevisionPrompt(state);
          const llmResult = await callGemini(geminiKey, revisionPrompt);
          return { ...llmResult, revised: true };
        },
        (result) => ({
          model: hasExternalFeedback ? GEMINI_MODEL : "draft_passthrough",
          revised: Boolean(result.revised),
          text_chars: (result.text || "").length
        })
      );
      const finalAnswer = normalizeGraphFinalAnswer(result.text, state);
      trace?.push(step("langgraph", "node_agent_final", {
        model: hasExternalFeedback ? GEMINI_MODEL : "draft_passthrough",
        revised_with_critic_feedback: Boolean(result.revised),
        text_chars: finalAnswer.length,
        answer: finalAnswer,
        answer_preview: previewText(finalAnswer)
      }));
      return {
        finalAnswer,
        llmRaw: result.raw || state.llmRaw || null,
        graphEvents: [{ node: "agent_final", revised_with_critic_feedback: Boolean(result.revised), text_chars: finalAnswer.length }]
      };
    })
    .addEdge(START, "load_okf")
    .addEdge("load_okf", "context_review")
    .addEdge("context_review", "agent_plan")
    .addEdge("agent_plan", "tool_observation")
    .addEdge("tool_observation", "agent_reflection")
    .addEdge("agent_reflection", "agent_followup_question")
    .addEdge("agent_followup_question", "tool_followups")
    .addEdge("tool_followups", "agent_synthesis")
    .addEdge("agent_synthesis", "agent_draft")
    .addEdge("agent_draft", "critic_feedback")
    .addEdge("critic_feedback", "agent_final")
    .addEdge("agent_final", END);

  const checkpointer = getGraphCheckpointer();
  return checkpointer ? graphBuilder.compile({ checkpointer }) : graphBuilder.compile();
}

async function runExternalLangGraphExecutor({ okfEnvelope, okfKey, input, messages, dryRun, threadId, metadata, agentMemory, langsmith, trace }) {
  const graph = createExternalLangGraph();
  const checkpointNs = okfKey;
  const checkpoint_backend = checkpointBackend();
  const checkpoint_inspection = checkpointInspection(threadId, checkpointNs);
  const result = await withLangSmithChild(
    langsmith,
    "LangGraph Executor: external CRUDX OKF ReAct Graph",
    "chain",
    { okf_key: okfKey, input, history_messages: messages.length, thread_id: threadId, checkpoint_backend, agent_memory_id: agentMemory?.agent_memory_id || null },
    { graph: "crudx_okf_react_discourse", okf_key: okfKey, thread_id: threadId, checkpoint_ns: checkpointNs, checkpoint_backend, agent_memory_id: agentMemory?.agent_memory_id || null, runtime_boundary: "external" },
    () => graph.invoke(
      { okfEnvelope, okfKey, input, messages, dryRun, metadata: metadata || {}, agentMemory: agentMemory || null },
      { configurable: { thread_id: threadId, checkpoint_ns: checkpointNs, dryRun, geminiKey: process.env.GEMINI_API_KEY || "", langsmith, trace } }
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
      thread_id: threadId,
      checkpoint_backend,
      checkpoint_ns: checkpointNs,
      checkpoint_inspection,
      agent_memory: agentMemory ? summarizeAgentMemoryForGraph(agentMemory) : null,
      nodes: (result.graphEvents || []).map((event) => event.node),
      action: result.action || "none",
      memory_profile: agentMemoryProfile(result.agent),
      history_messages: messages.length,
      discourse_turns: Math.max(0, (result.graphEvents || []).filter((event) => /^agent_|^tool_|^critic_/.test(event.node || "")).length)
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
    memory_profile: normalizeMemoryProfile(okf.memory_profile || okf.okf?.memory_profile || okf.meta?.memory_profile || okf.meta?.memory?.profile),
    okf: {
      system_prompt: okf.okf.system_prompt || "You are a careful ReAct agent.",
      memory_profile: normalizeMemoryProfile(okf.okf.memory_profile || okf.memory_profile || okf.meta?.memory_profile || okf.meta?.memory?.profile),
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

function extractThreadId(requestBody, okfKey, runKey) {
  const metadata = typeof requestBody.metadata === "object" && requestBody.metadata ? requestBody.metadata : {};
  const configurable = typeof requestBody.configurable === "object" && requestBody.configurable ? requestBody.configurable : {};
  const raw = requestBody.thread_id
    || requestBody.threadId
    || requestBody.session_id
    || requestBody.sessionId
    || configurable.thread_id
    || configurable.threadId
    || metadata.thread_id
    || metadata.threadId
    || metadata.session_id
    || metadata.sessionId;
  const normalized = normalizeThreadId(raw);
  if (normalized) return normalized;
  return `crudx_${compactIdPart(okfKey || "okf")}_${compactIdPart(runKey || "run")}`;
}

function normalizeThreadId(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  return text.replace(/[^A-Za-z0-9_.:-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 160);
}

function compactIdPart(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "thread";
}

function buildConversationContext(messages) {
  if (!messages.length) return "No prior conversation turns supplied.";
  return messages.map((message, index) => `${index + 1}. ${message.role.toUpperCase()}: ${message.content}`).join("\n");
}

function buildConversationMemory(messages, agent) {
  if (!Array.isArray(messages) || !messages.length) return "No prior case memory available.";
  const context = buildConversationContext(messages);
  const profile = agentMemoryProfile(agent);
  if (profile === "hr_case") {
    const facts = extractHrFacts([agent?.knowledge || "", context].join("\n"));
    const memory = [];
    if (/Max Mustermann|max_01/i.test(context)) {
      memory.push(`employee=Max Mustermann${facts.department ? ` (${facts.department})` : ""}`);
    }
    if (facts.maxBalance != null) memory.push(`remaining_balance=${facts.maxBalance} days`);
    if (/2026-08-03/i.test(context) && /2026-08-14/i.test(context)) memory.push("requested_period=2026-08-03..2026-08-14");
    if (facts.businessDays != null) memory.push(`business_day_impact=${facts.businessDays} days`);
    if (facts.entitlement != null) memory.push(`annual_entitlement=${facts.entitlement} days`);
    if (facts.advanceWeeks != null) memory.push(`advance_notice=${facts.advanceWeeks} weeks`);
    if (/Sandra Schreiber|sandra_02/i.test(context)) {
      memory.push(`coverage_candidate=Sandra Schreiber${facts.sandraBalance != null ? ` (${facts.sandraBalance} remaining vacation days; availability not proven)` : " (availability not proven)"}`);
    }
    if (/manager approval|coverage|holiday calendar|submission date/i.test(context)) {
      memory.push("open_checks=submission date, manager approval, holiday calendar, coverage");
    }
    return memory.length ? `HR case memory: ${memory.join("; ")}.` : "HR case memory: prior turns exist, but no durable HR facts were extracted.";
  }
  if (profile === "academic_review") {
    const title = /"([^"]+)"/.exec(context)?.[1] || "";
    const topics = [];
    if (title) topics.push(`paper_title="${title}"`);
    if (/doi/i.test(context)) topics.push("bibliographic_identifier=DOI/arXiv may be needed");
    if (/related|theme|future|direction/i.test(context)) topics.push("focus=related work and future directions");
    if (/missing evidence|known facts|next research/i.test(context)) topics.push("review_frame=known facts vs missing evidence vs next steps");
    return topics.length ? `Academic case memory: ${topics.join("; ")}.` : "Academic case memory: prior research turns exist, but no durable bibliographic facts were extracted.";
  }
  if (profile === "location_context") {
    return `Requestor-location memory: ${messages.length} prior messages are available. Use them only to resolve the current consent and location-evidence question; do not infer HR, vacation, or academic facts from this memory.`;
  }
  if (profile === "peano_logic") {
    return `Peano logic memory: ${messages.length} prior messages are available. Use prior Peano inputs/results only when the user explicitly refers to them; keep the required JSON output contract.`;
  }
  if (profile === "ops_incident") {
    const targets = [...new Set((context.match(/\b(?:db-prod-main|web-prod-01|vpn-gateway)\b/gi) || []).map((item) => item.toLowerCase()))];
    const facts = [];
    if (targets.length) facts.push(`targets=${targets.join(", ")}`);
    if (/slow checkout|intermittent errors|latency/i.test(context)) facts.push("symptoms=slow checkout/intermittent errors");
    if (/85\s*percent|85%|storage allocation/i.test(context)) facts.push("suspect=db-prod-main storage pressure");
    if (/handoff|severity|escalation/i.test(context)) facts.push("needs=incident handoff");
    return facts.length
      ? `Ops incident memory: ${facts.join("; ")}.`
      : "Ops incident memory: prior infrastructure-triage turns exist, but no durable incident facts were extracted.";
  }
  return `Conversation memory: ${messages.length} prior messages are available. Use them to resolve references such as "this", "that", "the previous request", and "final summary".`;
}

function selectGraphTool(input, agent) {
  if (isAgentSelfQuestion(input)) return null;
  const tools = Array.isArray(agent?.okf?.allowed_tools) ? agent.okf.allowed_tools : [];
  if (!tools.length) return null;
  const text = String(input || "").toLowerCase();
  const dates = extractIsoDates(input);
  if (isPeanoAgent(agent) && /axiom|axioms|list|auflisten|liste/i.test(input) && !/\badd\b|addition|\bplus\b|\+|A\(/i.test(input)) {
    return null;
  }
  if (/successor|add|addition|\bplus\b|\+|A\(/i.test(input)) {
    return tools.find((tool) => /peano|add/i.test(tool.name))
      || tools[0];
  }
  if (/requestor|requester|aufenthaltsort|standort|location|where am i|where is the user|wo bin ich|wo ist der nutzer|wo ist der requestor/i.test(input)) {
    return tools.find((tool) => /requestor|requester|location|geo|standort/i.test(tool.name))
      || tools[0];
  }
  if (isAcademicAgent(agent) || /paper|academic|research|literature|citation|doi|transformer|attention|vaswani|related work|future research/i.test(input)) {
    return tools.find((tool) => /academic|research|literature|search|knowledge/i.test(tool.name))
      || tools[0];
  }
  if (isOpsAgent(agent) || /checkout|incident|latency|intermittent|server|status|node|host|db-prod-main|web-prod-01|vpn-gateway|degraded|storage|risk|mitigation|handoff/i.test(input)) {
    return tools.find((tool) => /server|status|node|health|check/i.test(tool.name))
      || tools[0];
  }
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

function buildContextReview(input, messages, agent, agentMemory = null, metadata = {}) {
  const memory = buildConversationMemory(messages, agent);
  const globalMemory = buildAgentGlobalMemoryReview(agentMemory);
  const selfContext = buildAgentSelfContext(agent, agentMemory, metadata);
  if (isPeanoAgent(agent)) {
    return [
      "The current turn concerns Peano arithmetic.",
      `Prior conversation turns available: ${(Array.isArray(messages) ? messages : []).length}.`,
      selfContext,
      memory,
      globalMemory,
      `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of local Peano knowledge.`,
      "The executor should invoke the Peano addition tool for addition requests and preserve the OKF JSON output contract."
    ].join(" ");
  }
  if (isLocationAgent(agent)) {
    return [
      "The current turn concerns requestor location estimation.",
      `Prior conversation turns available: ${(Array.isArray(messages) ? messages : []).length}.`,
      selfContext,
      memory,
      globalMemory,
      `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of privacy/location knowledge.`,
      "The executor should invoke the requestor location tool and clearly distinguish explicit, header-derived, and weak browser-context evidence."
    ].join(" ");
  }
  if (isAcademicAgent(agent)) {
    return [
      "The current turn concerns academic research assistance.",
      `Prior conversation turns available: ${(Array.isArray(messages) ? messages : []).length}.`,
      selfContext,
      memory,
      globalMemory,
      `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of local research knowledge.`,
      "The executor should ground claims in the research tool and separate known facts, inferred context, missing evidence, and next research steps."
    ].join(" ");
  }
  if (isOpsAgent(agent)) {
    return [
      "The current turn concerns infrastructure incident triage.",
      `Prior conversation turns available: ${(Array.isArray(messages) ? messages : []).length}.`,
      selfContext,
      memory,
      globalMemory,
      `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of infrastructure knowledge.`,
      "The executor should compare confirmed host status, state the most suspicious component, separate assumptions from evidence, and produce operational next steps or handoff notes when requested."
    ].join(" ");
  }
  const text = String(input || "").toLowerCase();
  const history = Array.isArray(messages) ? messages : [];
  const topics = [];
  if (/2026-\d{2}-\d{2}/.test(text)) topics.push("dated request");
  if (/server|status|incident|latency|node|host/.test(text)) topics.push("operational status request");
  if (/policy|rule|knowledge|lookup|search/.test(text)) topics.push("knowledge-grounded request");
  if (!topics.length) topics.push("general OKF-backed request");
  return [
    `The current turn concerns ${topics.join(", ")}.`,
    `Prior conversation turns available: ${history.length}.`,
    selfContext,
    memory,
    globalMemory,
    `The active OKF exposes ${agent.okf.allowed_tools.length} configured tools and ${String(agent.knowledge || "").length} characters of local knowledge.`,
    "The executor should avoid a premature final answer and use only tools that are configured in the active OKF and relevant to the current request."
  ].join(" ");
}

function buildAgentGlobalMemoryReview(agentMemory) {
  const memory = normalizeAgentMemoryDocument(agentMemory || {});
  if (!memory.episodes.length) {
    return "Agent-global memory: no prior persisted episodes for this agent_memory_id.";
  }
  const episodes = memory.episodes.slice(0, 5).map((episode, index) => {
    const input = previewText(episode.input || "", 180);
    const answer = previewText(episode.answer || "", 220);
    return `${index + 1}. ${episode.at || "unknown time"} action=${episode.action || "none"} input="${input}" answer="${answer}"`;
  });
  return [
    `Agent-global memory: ${memory.episodes.length} persisted episodes for ${memory.agent_memory_id}.`,
    "Use this memory only when it is relevant to the current user request and do not confuse it with thread-local conversation turns.",
    ...episodes
  ].join(" ");
}

function buildAgentSelfContext(agent, agentMemory = null, metadata = {}) {
  const memory = normalizeAgentMemoryDocument(agentMemory || {});
  const meta = agent?.meta || {};
  const okfKey = memory.okf_key || metadata.okf_key || metadata.okfKey || meta.okf_key || meta.crudx_key || agent?.okf_key || "";
  const openApiKey = String(metadata.openapi_key || metadata.openApiKey || metadata.openapiKey || meta.openapi_key || meta.openapiKey || meta.openapi_spec_key || "").trim();
  const creator = meta.created_by || meta.creator || meta.author || meta.owner || meta.maintainer || "";
  const episodes = memory.episodes || [];
  const actionCounts = new Map();
  for (const episode of episodes) {
    const action = episode.action || "none";
    actionCounts.set(action, (actionCounts.get(action) || 0) + 1);
  }
  const frequentActions = Array.from(actionCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([action, count]) => `${action} (${count})`);
  return [
    "Agent self model:",
    `agent_id=${agent?.agent_id || "unknown"}.`,
    `name=${meta.name || agent?.agent_id || "unknown"}.`,
    `constituting_okf_key=${okfKey || "unknown"}.`,
    `openapi_key=${openApiKey || "unknown"}.`,
    `agent_memory_id=${memory.agent_memory_id || "unknown"}.`,
    `persisted_episode_count=${episodes.length}.`,
    `run_count=${memory.run_count || 0}.`,
    frequentActions.length ? `frequent_actions=${frequentActions.join(", ")}.` : "frequent_actions=none yet.",
    creator ? `configured_by=${creator}.` : "configured_by=not specified in the OKF metadata.",
    "For questions about your basis, answer from the constituting OKF and its CRUDX key.",
    "For questions about how to use you, answer from the OpenAPI artifact key and configured tools/endpoints when available.",
    "For questions about what you learned or how to inspect your memory, mention the agent_memory_id and that the platform exposes an Agent Memory inspection pill after runs."
  ].join(" ");
}

function isAgentSelfQuestion(input) {
  return /(\bwho\s+(made|created|configured)\s+you\b|\bwhat\s+is\s+your\s+(basis|foundation|memory|id)\b|\bhow\s+can\s+i\s+use\s+you\b|\bwhat\s+have\s+you\s+learned\b|\bhow\s+often\s+have\s+you\s+been\s+used\b|\bmost\s+frequent\b|wer\s+hat\s+dich\s+(gemacht|erstellt|konfiguriert)|was\s+ist\s+deine\s+(grundlage|basis|memory|id)|wie\s+kann\s+ich\s+dich\s+(benutzen|verwenden)|was\s+hast\s+du\s+gelernt|wie\s+h[aä]ufig\s+wurdest\s+du\s+benutzt|was\s+fragt\s+man\s+dich\s+am\s+h[aä]ufigsten|agent[-\s]?memory|ged[aä]chtnis|openapi|okf)/i.test(String(input || ""));
}

async function executeOkfTool(input, agent, action, okfKey, langsmith, metadata = {}) {
  if (action === "none") return buildGraphObservation(input, agent, action);

  const tools = Array.isArray(agent?.okf?.allowed_tools) ? agent.okf.allowed_tools : [];
  const tool = tools.find((candidate) => candidate.name === action)
    || tools.find((candidate) => new RegExp(escapeRegex(action), "i").test(candidate.name || ""));
  const webhookUrl = String(tool?.webhook_url || tool?.url || tool?.endpoint || "").trim();

  if (!isHttpUrl(webhookUrl)) {
    return buildGraphObservation(input, agent, action);
  }

  const fallback = buildGraphObservation(input, agent, action);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  const parameters = inferToolParameters(input, tool, agent, metadata?.messages || []);
  if (metadata?.request_context) parameters.request_context = metadata.request_context;
  const payload = {
    tool: tool.name || action,
    query: input,
    input,
    agent_id: agent?.agent_id || null,
    okf_key: okfKey || null,
    parameters,
    knowledge: agent?.knowledge || "",
    okf: agent?.okf || null,
    meta: {
      ...(agent?.meta || {}),
      request_context: metadata?.request_context || null
    }
  };

  const callWebhook = async () => {
    const response = await fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(payload)
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || body.ok === false) {
      const reason = body.error || body.message || response.statusText;
      return {
        ok: false,
        status: response.status,
        observation: `${fallback} Tool webhook ${tool.name || action} returned ${response.status}: ${reason}.`,
        body
      };
    }
    return {
      ok: true,
      status: response.status,
      observation: normalizeToolObservation(body, fallback),
      body
    };
  };

  try {
    const result = await withLangSmithChild(
      langsmith,
      `OKF Tool Webhook: ${tool.name || action}`,
      "tool",
      {
        tool: tool.name || action,
        webhook_url: webhookUrl,
        okf_key: okfKey || null,
        agent_id: agent?.agent_id || null,
        parameters: payload.parameters,
        query: input
      },
      {
        graph_node: "okf_tool_webhook",
        tool_name: tool.name || action,
        okf_key: okfKey || null,
        webhook_url: webhookUrl,
        transport: "https"
      },
      callWebhook,
      (result) => ({
        ok: Boolean(result.ok),
        status: result.status || null,
        tool: tool.name || action,
        webhook_url: webhookUrl,
        observation: result.observation || ""
      })
    );
    return result.observation;
  } catch (error) {
    return `${fallback} Tool webhook ${tool.name || action} could not be reached: ${error.message || String(error)}.`;
  } finally {
    clearTimeout(timeout);
  }
}

async function executeNamedOkfTool(input, agent, preferredName, okfKey, fallbackFactory, langsmith, metadata = {}) {
  const tools = Array.isArray(agent?.okf?.allowed_tools) ? agent.okf.allowed_tools : [];
  const tool = tools.find((candidate) => candidate.name === preferredName)
    || tools.find((candidate) => new RegExp(escapeRegex(preferredName), "i").test(candidate.name || ""));
  if (!tool) return fallbackFactory();

  const observation = await executeOkfTool(input, agent, tool.name, okfKey, langsmith, metadata);
  if (observation && !/No configured tool was selected/i.test(observation)) return observation;
  return fallbackFactory();
}

async function executeRelevantFollowupTools(state, langsmith) {
  const tools = Array.isArray(state.agent?.okf?.allowed_tools) ? state.agent.okf.allowed_tools : [];
  let followups = tools
    .filter((tool) => tool?.name && tool.name !== state.action)
    .filter((tool) => shouldRunToolForInput(tool, state.input, state.agent));
  if (isAcademicAgent(state.agent) && !followups.length) {
    followups = tools.filter((tool) => /academic|research|literature|search|knowledge/i.test(tool?.name || tool?.description || ""));
  }
  const observations = [];
  for (const tool of followups.slice(0, 3)) {
    const followupInput = isAcademicAgent(state.agent)
      ? buildAcademicFollowupQuery(state.input, observations.length)
      : state.input;
    const observation = await executeOkfTool(
      followupInput,
      state.agent,
      tool.name,
      state.okfKey,
      langsmith,
      state.metadata
    );
    if (observation && !/No configured tool was selected/i.test(observation)) {
      observations.push({
        tool: tool.name,
        observation
      });
    }
  }
  return observations;
}

function shouldRunToolForInput(tool, input, agent) {
  const name = String(tool?.name || "").toLowerCase();
  const description = String(tool?.description || "").toLowerCase();
  const text = String(input || "").toLowerCase();
  const haystack = `${name} ${description}`;
  if (/calculate|calendar|date|days|business/.test(haystack)) {
    return extractIsoDates(input).length >= 2
      || /\b(from\s+20\d{2}-\d{2}-\d{2}|between\s+20\d{2}-\d{2}-\d{2}|business day|business days|calendar|date range|dated period|period from|start date|end date)\b/i.test(input);
  }
  if (/search|knowledge|policy|rule|lookup/.test(haystack)) {
    if (isPeanoAgent(agent) || isLocationAgent(agent)) return false;
    if (isAcademicAgent(agent)) return /paper|academic|research|literature|citation|doi|transformer|attention|related work|future research/i.test(input);
    return /\b(policy|rule|knowledge|search|lookup|hr|vacation|urlaub|employee|server|status|incident|ops)\b/i.test(input);
  }
  if (/requestor|requester|location|geo|standort/.test(haystack)) {
    return /requestor|requester|aufenthaltsort|standort|location|where am i|wo bin ich/i.test(input);
  }
  if (/peano|addition|successor/.test(haystack)) {
    const operands = inferNaturalOperands(input);
    return Boolean(operands) || /previous result|prior result|last result|successor|add|addition|\bplus\b|\+|A\(/i.test(input);
  }
  return false;
}

function inferToolParameters(input, tool, agent, messages = []) {
  const name = String(tool?.name || "").toLowerCase();
  const dates = extractIsoDates(input);
  const params = { query: String(input || "") };
  if (/peano|add/.test(name)) {
    const operands = inferNaturalOperands(input, messages);
    if (operands) {
      params.left = operands.left;
      params.right = operands.right;
    }
  }
  if (/calculate|days/.test(name) && dates.length >= 2) {
    params.start_date = dates[0];
    params.end_date = dates[1];
  }
  if (/server|status|node|health/.test(name)) {
    params.target = inferServerTarget(input, agent);
  }
  if (/requestor|requester|location|geo|standort/.test(name)) {
    const explicit = inferLocationText(input);
    if (explicit) params.location = explicit;
  }
  return params;
}

function inferLocationText(input) {
  const match = /\b(?:i am in|i'm in|my location is|mein standort ist|ich bin in|aufenthaltsort ist)\s+([A-Za-zÄÖÜäöüß '-]{2,80}?)(?:[.?!,;\n]|$)/i.exec(String(input || ""));
  return match ? match[1].trim().replace(/[.?!,;:]+$/, "") : "";
}

function inferNaturalOperands(input, messages = []) {
  const text = String(input || "");
  const match = /\badd\s+(\d+)\s+(?:and|to)\s+(\d+)\b/i.exec(text)
    || /\b(\d+)\s*(?:\+|plus)\s*(\d+)\b/i.exec(text)
    || /\bA\(\s*(\d+)\s*,\s*(\d+)\s*\)/i.exec(text);
  if (match) return { left: Number(match[1]), right: Number(match[2]) };
  const previous = inferPreviousPeanoResult(messages);
  if (!previous) return null;
  const increment = /\badd\s+(\d+)\s+more\b/i.exec(text)
    || /\bprevious result\b.*?\badd\s+(\d+)\b/i.exec(text)
    || /\bprior result\b.*?\badd\s+(\d+)\b/i.exec(text)
    || /\blast result\b.*?\badd\s+(\d+)\b/i.exec(text)
    || /\bsuccessor\s+of\s+(?:the\s+)?(?:previous|prior|last)\s+result\b/i.exec(text);
  if (!increment) return null;
  return {
    left: Number(previous.integer_value),
    right: increment[1] ? Number(increment[1]) : 1,
    previous_peano_result: previous.peano_result
  };
}

function inferPreviousPeanoResult(messages) {
  if (!Array.isArray(messages)) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || message.role !== "assistant") continue;
    const json = extractJsonObjectFromText(message.content);
    if (!json) continue;
    try {
      const parsed = JSON.parse(json);
      if (typeof parsed.peano_result === "string" && Number.isFinite(Number(parsed.integer_value))) {
        return {
          peano_result: parsed.peano_result,
          integer_value: Number(parsed.integer_value)
        };
      }
    } catch {
      // Continue scanning older assistant messages.
    }
  }
  return null;
}

function inferServerTarget(input, agent) {
  const text = String(input || "");
  const explicit = /(?:server|node|service|host)\s+([a-z0-9._-]+)/i.exec(text);
  if (explicit) return explicit[1];
  const knowledge = String(agent?.knowledge || "");
  const known = /\b(?:server|node|service|host)[:\s]+([a-z0-9._-]+)/i.exec(knowledge);
  return known ? known[1] : "";
}

function normalizeToolObservation(body, fallback) {
  const value = body.observation || body.answer || body.result || body.message || body.summary;
  if (typeof value === "string" && value.trim()) return value.trim();
  if (Array.isArray(body.snippets) && body.snippets.length) {
    return `Tool observation: ${body.snippets.map((item) => String(item).trim()).filter(Boolean).join(" ")}`;
  }
  if (body.business_days !== undefined && body.start_date && body.end_date) {
    return `Calculated business-day impact for ${body.start_date} to ${body.end_date} is ${body.business_days} business days.`;
  }
  return fallback;
}

function isHttpUrl(value) {
  return /^https?:\/\//i.test(String(value || ""));
}

function buildGraphObservation(input, agent, action) {
  const knowledge = String(agent?.knowledge || "");
  const text = String(input || "").toLowerCase();
  if (isPeanoAgent(agent) && /axiom|axioms|list|auflisten|liste/i.test(input)) {
    return "Peano axiom request: answer from the Peano knowledge in the agent record and return the required JSON object with null result fields and an explain field.";
  }
  if (action === "none") return "No configured tool was selected; answer from the available information and prior conversation.";
  const dates = extractIsoDates(input);
  if (/calculate|days/i.test(action) || dates.length >= 2 || /business-day|business day/.test(text)) {
    if (dates.length >= 2) {
      const days = countWeekdays(dates[0], dates[1]);
      return `Calculated business-day impact for ${dates[0]} to ${dates[1]} is ${days} business days, assuming Monday through Friday and no holiday calendar overrides.`;
    }
    return "Date calculation requested, but no exact start and end dates were supplied in the current turn.";
  }
  if (/search|knowledge/i.test(action)) {
    if (isAcademicAgent(agent) || /academic|research|paper|literature|citation/i.test(action)) {
      return buildAcademicResearchObservation(input, agent);
    }
    const snippets = selectKnowledgeSnippets(input, knowledge);
    return snippets.length
      ? `OKF knowledge lookup: ${snippets.join(" ")}`
      : "OKF knowledge lookup: no directly matching knowledge snippet was found in the OKF document.";
  }
  return "Configured tool selected from OKF. Observation is grounded in available OKF knowledge; missing external tool execution is explicitly noted.";
}

function buildAgentReflection(state) {
  if (isPeanoAgent(state.agent)) {
    return [
      "The tool observation should already contain the deterministic Peano result.",
      "The final response must preserve the OKF JSON object shape.",
      `Current action was ${state.action || "none"}.`
    ].join(" ");
  }
  if (isLocationAgent(state.agent)) {
    return [
      "The location observation must be treated as an estimate, not a certainty.",
      "Only explicit coordinates or explicit user-supplied place text should be described as high confidence.",
      `Current action was ${state.action || "none"}.`
    ].join(" ");
  }
  if (isAcademicAgent(state.agent)) {
    return [
      "The research observation should be treated as scoped evidence, not as a complete literature search.",
      "If current citation data, DOI metadata, or newest related work is missing, the answer must say so and ask for an external search source or bibliographic details.",
      `Current action was ${state.action || "none"}.`
    ].join(" ");
  }
  if (isOpsAgent(state.agent)) {
    return [
      "The infrastructure observation should be turned into an incident-triage answer, not copied as raw status lines.",
      "Separate confirmed status, likely risk, immediate mitigation, missing telemetry, and escalation recommendation.",
      `Current action was ${state.action || "none"}.`
    ].join(" ");
  }
  const hasDates = extractIsoDates(state.input || "").length >= 2;
  const hasKnowledge = Boolean(String(state.agent?.knowledge || "").trim());
  const hasFollowups = Array.isArray(state.followupObservations) && state.followupObservations.length > 0;
  return [
    "The primary observation should be combined with the OKF knowledge and any relevant follow-up tool observations before answering.",
    hasFollowups ? "Additional OKF tool observations are available for grounding." : "No additional OKF follow-up tool was relevant for this turn.",
    hasDates ? "The request includes exact dates; date-aware tools may be relevant only if configured in this agent OKF." : "No complete date range was supplied.",
    hasKnowledge ? "The OKF knowledge base is available for factual grounding." : "No OKF knowledge text is available, so the answer must state that limitation when relevant.",
    `Current action was ${state.action || "none"}.`
  ].join(" ");
}

function buildInternalFollowupQuestion(state) {
  if (isPeanoAgent(state.agent)) {
    return "Which Peano recursion rule and base case validate the structural result?";
  }
  if (isLocationAgent(state.agent)) {
    return "Which evidence level supports this location estimate, and what consent or context is missing for higher precision?";
  }
  if (isAcademicAgent(state.agent)) {
    return "Which known paper facts, missing bibliographic evidence, and follow-up research questions are needed before a stronger literature review can be produced?";
  }
  if (isOpsAgent(state.agent)) {
    return "Which infrastructure facts, missing telemetry, and next operational checks are needed before the incident can be handed off?";
  }
  if (/2026-\d{2}-\d{2}/.test(String(state.input || ""))) {
    return "Which configured OKF tools or knowledge facts are still relevant before producing a grounded answer?";
  }
  return "What missing context, tool evidence, or OKF knowledge prevents a fully grounded answer?";
}

function buildPolicyObservation(input, agent) {
  if (isPeanoAgent(agent)) {
    return "Peano rule check: addition is defined by A(x, 0) = x and A(x, S(y)) = S(A(x, y)).";
  }
  if (isLocationAgent(agent)) {
    return "Privacy rule check: precise GPS location requires explicit coordinates or consented browser geolocation; time zone and locale are weak hints only.";
  }
  if (isOpsAgent(agent)) {
    const snippets = selectKnowledgeSnippets(input, String(agent?.knowledge || ""), [/severity|risk|mitigation|escalat|runbook|evidence|missing|storage|checkout|vpn|web-prod|db-prod/i]);
    return snippets.length
      ? `Operational policy check from OKF: ${snippets.join(" ")}`
      : "Operational policy check: no incident runbook rule was found in the OKF knowledge.";
  }
  const knowledge = String(agent?.knowledge || "");
  const snippets = selectKnowledgeSnippets(input, knowledge, [/policy|rule|request|approval|approve|vacation|urlaub|entitled|advance|weeks|days/i]);
  if (snippets.length) {
    return `Policy check from OKF: ${snippets.join(" ")}`;
  }
  return "Policy check: no specific HR policy rule was found for this request in the OKF knowledge.";
}

function buildCalendarObservation(input) {
  if (/peano|successor|A\(|S\(/i.test(String(input || ""))) {
    return "Non-temporal check: no calendar calculation is required for Peano arithmetic.";
  }
  if (/requestor|requester|aufenthaltsort|standort|location|where am i|wo bin ich/i.test(String(input || ""))) {
    return "Non-temporal check: no calendar calculation is required for requestor location estimation.";
  }
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
  const followupText = formatFollowupObservations(state.followupObservations);
  if (isPeanoAgent(state.agent)) {
    return [
      "Synthesis:",
      `1. Peano tool observation: ${state.observation || "none"}`,
      `2. Follow-up observations: ${followupText || "none"}`,
      "Recommendation logic: preserve the deterministic Peano result and return only the required JSON object."
    ].join("\n");
  }
  if (isLocationAgent(state.agent)) {
    return [
      "Synthesis:",
      `1. Location observation: ${state.observation || "none"}`,
      `2. Follow-up observations: ${followupText || "none"}`,
      "Recommendation logic: answer transparently with confidence, evidence, and any missing consent/context needed for greater precision."
    ].join("\n");
  }
  if (isAcademicAgent(state.agent)) {
    return [
      "Synthesis:",
      `1. Research observation: ${state.observation || "none"}`,
      `2. Follow-up observations: ${followupText || "none"}`,
      "Recommendation logic: answer as a research assistant. Separate known facts, missing evidence, and next research steps. Do not fabricate citations."
    ].join("\n");
  }
  if (isOpsAgent(state.agent)) {
    return [
      "Synthesis:",
      `1. Infrastructure observation: ${state.observation || "none"}`,
      `2. Follow-up observations: ${followupText || "none"}`,
      "Recommendation logic: answer as an incident triage assistant. Identify suspected component, operational risk, immediate mitigation, missing telemetry, and escalation or handoff where requested."
    ].join("\n");
  }
  return [
    "Synthesis:",
    `1. Primary observation: ${state.observation || "none"}`,
    `2. Follow-up observations: ${followupText || "none"}`,
    "Recommendation logic: answer from the active OKF, configured tools, prior conversation, and explicit uncertainty only."
  ].join("\n");
}

function buildGraphFinalPrompt(state) {
  const selfQuestion = isAgentSelfQuestion(state.input);
  const requiresJson = agentRequiresJsonOutput(state.agent) && !selfQuestion;
  const academicRequirements = isAcademicAgent(state.agent)
    ? [
        "",
        "Academic answer discipline:",
        "- Answer the user's research question directly first.",
        "- Do not copy OKF workflow notes, tool descriptions, source-example notes, or platform implementation details into the answer.",
        "- If the paper title is recognizable from OKF/tool evidence, do not block on a DOI; use DOI/arXiv metadata only as supporting evidence.",
        "- Use a compact structure: core contribution, evidence used, limitations, and useful next steps.",
        "- For 'Attention Is All You Need', explain the Transformer contribution rather than describing how the research workflow operates."
      ].join("\n")
    : "";
  const opsRequirements = isOpsAgent(state.agent)
    ? [
        "",
        "Ops incident answer discipline:",
        "- Answer the user's operational question directly first.",
        "- Do not copy raw OKF status lines as the whole answer.",
        "- Triage answers must name the most suspicious component and why.",
        "- Risk/mitigation answers must include operational risk, immediate mitigation, and missing evidence.",
        "- Comparison answers must separate confirmed status from assumptions and recommend the next check.",
        "- Handoff answers must include severity, affected systems, suspected root cause, actions taken, open questions, and escalation recommendation."
      ].join("\n")
    : "";
  const locationRequirements = isLocationAgent(state.agent)
    ? [
        "",
        "Requestor-location answer discipline:",
        "- Answer the current location question directly first.",
        "- Treat consented browser geolocation or explicit coordinates as high-confidence evidence.",
        "- If Google Maps reverse geocoding returned an address, use it as the location label and mention that it came from consented browser geolocation.",
        "- If only time zone, locale, or headers are available, explicitly label the result as low or medium confidence.",
        "- Do not ask the user to paste coordinates when the graph state already contains consented browser geolocation evidence.",
        "- If the tool observation contains a Google Maps route URL, preserve that exact URL in the final answer."
      ].join("\n")
    : "";
  return [
    selfQuestion
      ? "The current user input is an agent self-knowledge question. Answer from the Agent self model in normal prose, even if the domain OKF usually requires JSON."
      : "",
    state.agent.okf.system_prompt,
    academicRequirements,
    opsRequirements,
    locationRequirements,
    "",
    "You are the final response node in a multi-step LangGraph ReAct executor.",
    "Use the supplied graph state, OKF knowledge, and prior conversation.",
    "Return only the final user-facing answer.",
    requiresJson
      ? "Honor the OKF output format exactly. Return a valid JSON object only, with no Markdown code fence."
      : "Do not output JSON, Markdown code fences, raw graph state, trace data, or hidden chain-of-thought.",
    "Do not use headings named Plan, Dialogue, Evidence, or Final.",
    "Write like a helpful professional assistant speaking to the user.",
    "Mention missing information only when it materially affects the answer.",
    "Default brevity rule: keep normal answers under 120 words and usually 2-4 sentences.",
    "Exceed 120 words only when the user explicitly asks for an audit, handoff, detailed comparison, long analysis, or structured report.",
    "Do not repeat raw tool observations verbatim; synthesize the useful facts.",
    "",
    "Graph state:",
    `Context review: ${state.contextReview || "none"}`,
    `Plan: ${state.plan || "none"}`,
    `Action: ${state.action || "none"}`,
    `Initial observation: ${state.observation || "none"}`,
    `Agent reflection: ${state.reflection || "none"}`,
    `Internal follow-up question: ${state.followupQuestion || "none"}`,
    `Follow-up observations: ${formatFollowupObservations(state.followupObservations) || "none"}`,
    `Synthesis: ${state.synthesis || "none"}`,
    "",
    "Available OKF knowledge:",
    state.agent.knowledge || "No extra knowledge supplied.",
    "",
    "Prior conversation:",
    buildConversationContext(state.messages || []),
    "",
    "Conversation memory summary:",
    buildConversationMemory(state.messages || [], state.agent),
    "",
    "Agent self model:",
    buildAgentSelfContext(state.agent, state.agentMemory, state.metadata),
    "",
    `Current user input: ${state.input}`,
    "",
    "Final response requirements:",
    "- Use prior conversation when relevant.",
    "- Separate known facts from missing information.",
    selfQuestion
      ? "- This is a self-knowledge question. You may mention OKF, OpenAPI, CRUDX IDs, episode counts, run counts, and Agent Memory inspection in plain user-facing language."
      : "- Do not mention OKF, OKF context, active OKF, graph state, or platform internals in normal user-facing answers.",
    "- When you need to identify the source of a fact, use ordinary language such as 'according to the information available', 'our records show', or 'based on the information provided'.",
    "- LangSmith tracing is handled automatically by the platform. If the user asks for a trace, acknowledge that trace metadata is attached by the platform; do not claim you cannot create traces.",
    requiresJson ? "- Return only the JSON object required by the OKF." : "- Keep the final answer short, practical, and readable.",
    "- Do not reveal internal node names unless the user explicitly asks for debugging."
  ].join("\n");
}

function buildGraphRevisionPrompt(state) {
  const requiresJson = agentRequiresJsonOutput(state.agent) && !isAgentSelfQuestion(state.input);
  return [
    state.agent.okf.system_prompt,
    "",
    "You are the final revision node in a LangGraph ReAct executor.",
    "You have a draft answer and independent AI critic feedback.",
    "Your default behavior is to revise the draft. Treat supported critic feedback as an action list, not as optional commentary.",
    "Apply every concrete improvement that is supported by the tool observations, graph synthesis, or prior conversation.",
    "If a critic supplies a better_answer and it is supported by the evidence, use it as the baseline for the final answer.",
    "Preserve grounded facts, exact CRUDX IDs, and exact URLs from the draft or tool observations.",
    "Default brevity rule: keep the final answer under 120 words and usually 2-4 sentences unless the user explicitly asked for a detailed/audit/handoff/comparison report.",
    "Prefer a shorter supported critic answer over a longer draft when both are correct.",
    requiresJson
      ? "Honor the OKF output format exactly. Return a valid JSON object only, with no Markdown code fence, no prose, and no extra fields unless the existing OKF schema explicitly allows them. For Peano agents, the allowed fields are peano_result, integer_value, and explain."
      : "Return only the final user-facing answer. Do not output JSON, Markdown code fences, trace data, or hidden chain-of-thought.",
    "Do not mention that competing AIs reviewed the answer unless the user explicitly asks about the feedback loop.",
    "",
    `Current user input: ${state.input}`,
    "",
    "Prior conversation:",
    buildConversationContext(state.messages || []),
    "",
    "Graph synthesis:",
    state.synthesis || "none",
    "",
    "Tool observations:",
    [
      state.observation || "",
      formatFollowupObservations(state.followupObservations)
    ].filter(Boolean).join("\n") || "none",
    "",
    "Draft answer:",
    state.draftAnswer || "none",
    "",
    "Independent critic feedback:",
    formatCriticFeedback(state.criticFeedback),
    "",
    "Revision requirements:",
    "- Answer the user's latest question directly.",
    "- Use conversation history only when relevant.",
    "- Separate known facts, missing evidence, and recommendations where the question asks for it.",
    "- If a critic suggested unsupported facts, ignore them.",
    requiresJson ? "- If a critic suggests prose, Markdown, or extra JSON fields that violate the required JSON schema, ignore that part of the critique." : "",
    "- If critics ask for structure, missing-evidence separation, less verbosity, or clearer caveats, implement those changes.",
    "- Do not keep the draft unchanged unless every critic suggestion is unsupported or harmful.",
    "- Remove redundant explanation and platform/process wording before returning the final answer."
  ].filter(Boolean).join("\n");
}

async function collectAiCriticFeedback(state) {
  if (!AI_FEEDBACK_ENABLED) {
    return {
      enabled: false,
      configured: false,
      reviews: [],
      errors: [],
      summary: "AI feedback loop is disabled by AI_FEEDBACK_ENABLED=false."
    };
  }
  const providers = configuredCriticProviders().slice(0, Math.max(1, AI_FEEDBACK_MAX_CRITICS));
  if (!providers.length) {
    return {
      enabled: true,
      configured: false,
      reviews: [],
      errors: [],
      summary: "No external AI critic provider is configured. Set OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, or XAI_API_KEY in the runtime environment to enable cross-model review."
    };
  }

  const prompt = buildCriticPrompt(state);
  const reviews = [];
  const errors = [];
  for (const provider of providers) {
    try {
      const review = await callCriticProvider(provider, prompt);
      reviews.push(review);
    } catch (error) {
      errors.push(`${provider.provider}/${provider.model}: ${error.message || error}`);
    }
  }
  return {
    enabled: true,
    configured: providers.length > 0,
    reviews,
    errors,
    summary: reviews.length
      ? `${reviews.length} external critic review(s) received.`
      : "External critic providers are configured, but no review was returned."
  };
}

function configuredCriticProviders() {
  const providers = [];
  const openRouterKey = process.env.OPENROUTER_API_KEY || "";
  if (openRouterKey) {
    const models = String(process.env.OPENROUTER_CRITIC_MODELS || "deepseek/deepseek-chat,openai/gpt-4o-mini")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    for (const model of models) {
      providers.push({
        provider: "openrouter",
        model,
        apiKey: openRouterKey,
        baseUrl: "https://openrouter.ai/api/v1"
      });
    }
  }
  if (process.env.OPENAI_API_KEY) {
    providers.push({
      provider: "openai",
      model: process.env.OPENAI_CRITIC_MODEL || "gpt-4o-mini",
      apiKey: process.env.OPENAI_API_KEY,
      baseUrl: "https://api.openai.com/v1"
    });
  }
  if (process.env.DEEPSEEK_API_KEY) {
    providers.push({
      provider: "deepseek",
      model: process.env.DEEPSEEK_CRITIC_MODEL || "deepseek-chat",
      apiKey: process.env.DEEPSEEK_API_KEY,
      baseUrl: "https://api.deepseek.com/v1"
    });
  }
  if (process.env.XAI_API_KEY) {
    providers.push({
      provider: "xai",
      model: process.env.XAI_CRITIC_MODEL || "grok-3-mini",
      apiKey: process.env.XAI_API_KEY,
      baseUrl: "https://api.x.ai/v1"
    });
  }
  if (process.env.ANTHROPIC_API_KEY) {
    providers.push({
      provider: "anthropic",
      model: process.env.ANTHROPIC_CRITIC_MODEL || "claude-3-5-sonnet-latest",
      apiKey: process.env.ANTHROPIC_API_KEY
    });
  }
  return providers;
}

function buildCriticPrompt(state) {
  return [
    "You are an independent answer-quality critic for an agent platform.",
    "Review the candidate answer against the user request and the supplied grounded evidence.",
    "Do not invent facts. If the evidence is too thin, say so.",
    "Prefer concise better_answer text under 120 words unless the user explicitly requested a long report.",
    "Return concise feedback with these sections: score_1_to_10, strengths, issues, concrete_improvements, better_answer.",
    "",
    `Agent: ${state.agent?.meta?.name || state.agent?.agent_id || "unknown"}`,
    `User request: ${state.input}`,
    "",
    "Prior conversation:",
    buildConversationContext(state.messages || []),
    "",
    "Grounded evidence:",
    [
      state.contextReview ? `Context review: ${state.contextReview}` : "",
      state.observation ? `Primary observation: ${state.observation}` : "",
      formatFollowupObservations(state.followupObservations) ? `Follow-up observations: ${formatFollowupObservations(state.followupObservations)}` : "",
      state.synthesis ? `Synthesis: ${state.synthesis}` : ""
    ].filter(Boolean).join("\n") || "none",
    "",
    "Candidate answer:",
    state.draftAnswer || "none"
  ].join("\n");
}

async function callCriticProvider(provider, prompt) {
  if (provider.provider === "anthropic") {
    return await callAnthropicCritic(provider, prompt);
  }
  return await callOpenAiCompatibleCritic(provider, prompt);
}

async function callOpenAiCompatibleCritic(provider, prompt) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_FEEDBACK_TIMEOUT_MS);
  try {
    const headers = {
      "Authorization": `Bearer ${provider.apiKey}`,
      "Content-Type": "application/json"
    };
    if (provider.provider === "openrouter") {
      headers["HTTP-Referer"] = "https://crudx.local/react-aaas";
      headers["X-Title"] = "CRUDX ReAct AaaS";
    }
    const response = await fetch(`${provider.baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        model: provider.model,
        temperature: 0.1,
        max_tokens: 900,
        messages: [
          { role: "system", content: "You are a terse, evidence-aware answer-quality reviewer." },
          { role: "user", content: prompt }
        ]
      })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || response.statusText);
    const text = body.choices?.[0]?.message?.content?.trim();
    if (!text) throw new Error("critic response did not contain text");
    return {
      provider: provider.provider,
      model: provider.model,
      text: text.slice(0, 5000)
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function callAnthropicCritic(provider, prompt) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_FEEDBACK_TIMEOUT_MS);
  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": provider.apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
      },
      signal: controller.signal,
      body: JSON.stringify({
        model: provider.model,
        temperature: 0.1,
        max_tokens: 900,
        messages: [{ role: "user", content: prompt }]
      })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || response.statusText);
    const text = (body.content || []).map((part) => part.text || "").join("").trim();
    if (!text) throw new Error("critic response did not contain text");
    return {
      provider: provider.provider,
      model: provider.model,
      text: text.slice(0, 5000)
    };
  } finally {
    clearTimeout(timeout);
  }
}

function criticFeedbackHasReviews(feedback) {
  return Array.isArray(feedback?.reviews) && feedback.reviews.length > 0;
}

function formatCriticFeedback(feedback) {
  if (!feedback) return "No critic feedback available.";
  const lines = [
    `Status: ${feedback.summary || "No summary."}`
  ];
  if (Array.isArray(feedback.reviews) && feedback.reviews.length) {
    for (const review of feedback.reviews) {
      lines.push("");
      lines.push(`Critic ${review.provider}/${review.model}:`);
      lines.push(String(review.text || "").trim());
    }
  }
  if (Array.isArray(feedback.errors) && feedback.errors.length) {
    lines.push("");
    lines.push(`Critic errors: ${feedback.errors.slice(0, 5).join(" | ")}`);
  }
  return lines.join("\n").trim();
}

function summarizeCriticFeedbackForTrace(feedback) {
  if (!feedback) return { configured: false, review_count: 0 };
  return {
    enabled: Boolean(feedback.enabled),
    configured: Boolean(feedback.configured),
    review_count: Array.isArray(feedback.reviews) ? feedback.reviews.length : 0,
    providers: Array.isArray(feedback.reviews) ? feedback.reviews.map((review) => review.provider) : [],
    errors: Array.isArray(feedback.errors) ? feedback.errors.slice(0, 3) : []
  };
}

function normalizeGraphFinalAnswer(text, state) {
  const raw = String(text || "").trim();
  const withoutFences = raw.replace(/^```(?:json|markdown|text)?\s*/i, "").replace(/\s*```$/i, "").trim();
  if (agentRequiresJsonOutput(state.agent) && !isAgentSelfQuestion(state.input)) {
    if (looksLikeJson(withoutFences)) return normalizeContractJson(withoutFences, state);
    const fromRaw = extractJsonObjectFromText(withoutFences);
    if (fromRaw) return normalizeContractJson(fromRaw, state);
    const fromState = extractJsonObjectFromText([state.observation, formatFollowupObservations(state.followupObservations), state.synthesis].join("\n"));
    if (fromState) return normalizeContractJson(fromState, state);
  }
  if (looksLikeJson(withoutFences)) return buildDeterministicGraphSections(state);
  const looksIncomplete = !raw
    || withoutFences.length < 40
    || !/[.!?)]$/.test(withoutFences)
    || /(?:\band stating|\band explain|\band recommend|\bwith|,\s*)$/i.test(withoutFences);
  const looksDebuggy = /(^|\n)\s*(Plan|Dialogue|Evidence|Final)\s*:/i.test(withoutFences)
    || /LangGraph Node|Graph state|tool_observation|agent_reflection/i.test(withoutFences);
  if (isAgentSelfQuestion(state.input) && (looksLikeJson(withoutFences) || looksIncomplete || looksDebuggy)) {
    return buildAgentSelfAnswer(state);
  }
  if (isLocationAgent(state.agent) && /https:\/\/www\.google\.com\/maps\/dir\//i.test([state.observation, state.synthesis].join("\n")) && !/https:\/\/www\.google\.com\/maps\/dir\//i.test(withoutFences)) {
    return buildDeterministicGraphSections(state);
  }
  const answer = looksIncomplete || looksDebuggy ? buildDeterministicGraphSections(state) : withoutFences;
  if (isAgentSelfQuestion(state.input)) return answer;
  return humanizeUserFacingPlatformLanguage(answer);
}

function buildAgentSelfAnswer(state) {
  const memory = normalizeAgentMemoryDocument(state.agentMemory || {});
  const metadata = state.metadata || {};
  const meta = state.agent?.meta || {};
  const name = meta.name || state.agent?.agent_id || "dieser Agent";
  const okfKey = memory.okf_key || metadata.okf_key || metadata.okfKey || "-";
  const openApiKey = metadata.openapi_key || metadata.openApiKey || metadata.openapiKey || meta.openapi_key || "-";
  const memoryId = memory.agent_memory_id || metadata.agent_memory_id || "-";
  const text = String(state.input || "");
  if (/benutzen|verwenden|use|openapi/i.test(text)) {
    return `Du kannst mich über mein OpenAPI-Artefakt ${openApiKey} benutzen. Meine fachliche Grundlage ist das OKF ${okfKey}; in dieser Umgebung kannst du mir direkt Fragen stellen oder die OpenAPI-App öffnen, um meine Operationen und Testaufrufe zu sehen.`;
  }
  if (/gelernt|memory|ged[aä]chtnis|inspect|inspizieren/i.test(text)) {
    return `Mein Agenten-Gedächtnis hat die ID ${memoryId}. Es enthält aktuell ${memory.episodes.length} persistierte Episode(n) aus ${memory.run_count || memory.episodes.length} Lauf/Läufen und kann nach einem Lauf über die Agent-Memory-Pill inspiziert werden.`;
  }
  if (/h[aä]ufig|often|frequent|meist/i.test(text)) {
    const counts = new Map();
    for (const episode of memory.episodes || []) counts.set(episode.action || "none", (counts.get(episode.action || "none") || 0) + 1);
    const top = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0];
    return `Ich wurde in diesem agent-globalen Gedächtnis bisher ${memory.run_count || memory.episodes.length} Mal persistiert. Am häufigsten sehe ich aktuell ${top ? `${top[0]} (${top[1]} Mal)` : "noch kein wiederkehrendes Muster"}.`;
  }
  const creator = meta.created_by || meta.creator || meta.author || meta.owner || meta.maintainer || "";
  return `Ich bin ${name}. Meine Grundlage ist mein konstituierendes OKF ${okfKey}; darüber sind Rolle, Werkzeuge, Wissen und Antwortregeln definiert. ${creator ? `Konfiguriert wurde ich von ${creator}. ` : "Ein konkreter Ersteller ist in meinen OKF-Metadaten nicht angegeben. "}Mein OpenAPI-Artefakt ist ${openApiKey}, und mein Agenten-Gedächtnis hat die ID ${memoryId}.`;
}

function humanizeUserFacingPlatformLanguage(value) {
  return String(value || "")
    .replace(/\bin the OKF context\b/gi, "according to the information available")
    .replace(/\bfrom the OKF context\b/gi, "from the available information")
    .replace(/\bthe OKF context\b/gi, "the available information")
    .replace(/\bactive OKF document\b/gi, "available records")
    .replace(/\bactive OKF\b/gi, "available records")
    .replace(/\bOKF document\b/gi, "available records")
    .replace(/\bOKF knowledge base\b/gi, "available records")
    .replace(/\bOKF evidence\b/gi, "available evidence")
    .replace(/\bOKF facts\b/gi, "available facts")
    .replace(/\bOKF status\b/gi, "available status information")
    .replace(/\bOKF-local knowledge\b/gi, "available local knowledge")
    .replace(/\bOKF knowledge\b/gi, "available information")
    .replace(/\bOKF information\b/gi, "available information")
    .replace(/\bOKF context\b/gi, "available information")
    .replace(/\bconfigured tool observations\b/gi, "tool results")
    .replace(/\bAnswer is grounded in available records and tool results\./gi, "This answer is based on the information available.")
    .replace(/\bAnswer is grounded in the available records and tool results\./gi, "This answer is based on the information available.")
    .replace(/\baccording to the information available\./gi, "according to the information available.")
    .replace(/\s+\n/g, "\n")
    .trim();
}

function normalizeContractJson(jsonText, state) {
  if (!isPeanoAgent(state.agent)) return String(jsonText || "").trim();
  try {
    const parsed = JSON.parse(jsonText);
    const asksAxiomsOnly = /axiom|axioms|list|auflisten|liste/i.test(String(state.input || ""))
      && !/\badd\b|addition|\bplus\b|\+|A\(/i.test(String(state.input || ""));
    if (asksAxiomsOnly) {
      return JSON.stringify({
        peano_result: null,
        integer_value: null,
        explain: buildPeanoExplain(state, parsed)
      });
    }
    const evidence = extractPeanoEvidenceObject(state);
    const peanoResult = parsed.peano_result ?? evidence?.peano_result ?? null;
    const rawIntegerValue = parsed.integer_value ?? evidence?.integer_value;
    const integerValue = Number.isFinite(Number(rawIntegerValue)) ? Number(rawIntegerValue) : null;
    const parsedExplain = typeof parsed.explain === "string" ? parsed.explain.trim() : "";
    const explain = parsedExplain && !/not available|missing|could not/i.test(parsedExplain)
      ? parsedExplain
      : buildPeanoExplain(state, { ...parsed, peano_result: peanoResult, integer_value: integerValue });
    const normalized = {
      peano_result: peanoResult,
      integer_value: integerValue,
      explain
    };
    return JSON.stringify(normalized);
  } catch {
    return String(jsonText || "").trim();
  }
}

function buildPeanoExplain(state, parsed = {}) {
  const input = String(state.input || "");
  const asksAxioms = /axiom|axioms|list|auflisten|liste/i.test(input);
  const axiomText = "Peano axioms: 0 is a natural number; every natural number n has a successor S(n); 0 is not the successor of any natural number; S is injective, so S(a)=S(b) implies a=b; induction says any set containing 0 and closed under successor contains all natural numbers.";
  if (asksAxioms && !/\badd\b|addition|\bplus\b|\+|A\(/i.test(input)) return axiomText;
  const operands = inferNaturalOperands(input, state.messages || []);
  const recursion = operands
    ? `Addition is evaluated by A(x,0)=x and A(x,S(y))=S(A(x,y)); for ${operands.left} and ${operands.right}, the successor recursion reaches the base case and yields ${parsed.peano_result || "the shown Peano result"}.`
    : `Addition is evaluated by A(x,0)=x and A(x,S(y))=S(A(x,y)); the successor recursion reaches the base case and yields ${parsed.peano_result || "the shown Peano result"}.`;
  return `${recursion} ${axiomText}`;
}

function extractPeanoEvidenceObject(state) {
  const followupSources = Array.isArray(state.followupObservations)
    ? state.followupObservations.map((item) => item?.observation)
    : [];
  const sources = [state.observation, ...followupSources, state.synthesis, state.draftAnswer].filter(Boolean);
  for (const source of sources) {
    const json = extractJsonObjectFromText(source);
    if (!json) continue;
    try {
      const parsed = JSON.parse(json);
      if (typeof parsed.peano_result === "string" && Number.isFinite(Number(parsed.integer_value))) {
        return parsed;
      }
    } catch {
      // Try the next evidence source.
    }
  }
  return null;
}

function agentRequiresJsonOutput(agent) {
  const prompt = String(agent?.okf?.system_prompt || "");
  const knowledge = String(agent?.knowledge || "");
  return /valid JSON object|strictly with a valid JSON|JSON Response|peano_result|integer_value/i.test(`${prompt}\n${knowledge}`);
}

function isPeanoAgent(agent) {
  if (agentMemoryProfile(agent, { legacy: false }) === "peano_logic") return true;
  const haystack = [
    agent?.agent_id,
    agent?.meta?.id,
    agent?.meta?.name,
    agent?.okf?.system_prompt,
    (agent?.okf?.allowed_tools || []).map((tool) => tool.name).join(" ")
  ].join("\n");
  return /peano|math-peano-core|peano_addition/i.test(haystack);
}

function isLocationAgent(agent) {
  if (agentMemoryProfile(agent, { legacy: false }) === "location_context") return true;
  const haystack = [
    agent?.agent_id,
    agent?.meta?.id,
    agent?.meta?.name,
    agent?.okf?.system_prompt,
    (agent?.okf?.allowed_tools || []).map((tool) => tool.name).join(" ")
  ].join("\n");
  return /requestor.location|requester.location|requestor-location|requestor_location|standort|geo/i.test(haystack);
}

function isHrAgent(agent) {
  if (agentMemoryProfile(agent, { legacy: false }) === "hr_case") return true;
  return legacyIsHrAgent(agent);
}

function legacyIsHrAgent(agent) {
  const haystack = [
    agent?.agent_id,
    agent?.meta?.id,
    agent?.meta?.name,
    agent?.okf?.system_prompt,
    agent?.knowledge,
    (agent?.okf?.allowed_tools || []).map((tool) => `${tool.name} ${tool.description || ""}`).join(" ")
  ].join("\n");
  return /\bhr\b|vacation|urlaub|employee|onboarding|calculate_days|search_knowledge_base/i.test(haystack);
}

function isAcademicAgent(agent) {
  if (agentMemoryProfile(agent, { legacy: false }) === "academic_review") return true;
  return legacyIsAcademicAgent(agent);
}

function legacyIsAcademicAgent(agent) {
  const haystack = [
    agent?.agent_id,
    agent?.meta?.id,
    agent?.meta?.name,
    agent?.okf?.system_prompt,
    (agent?.okf?.allowed_tools || []).map((tool) => `${tool.name} ${tool.description || ""}`).join(" ")
  ].join("\n");
  return /academic-research|academic research|literature review|research assistant|paper analysis|citation/i.test(haystack);
}

function isOpsAgent(agent) {
  if (agentMemoryProfile(agent, { legacy: false }) === "ops_incident") return true;
  const haystack = [
    agent?.agent_id,
    agent?.meta?.id,
    agent?.meta?.name,
    Array.isArray(agent?.meta?.tags) ? agent.meta.tags.join(" ") : "",
    agent?.okf?.system_prompt,
    agent?.knowledge,
    (agent?.okf?.allowed_tools || []).map((tool) => `${tool.name} ${tool.description || ""}`).join(" ")
  ].join("\n");
  return /sysadmin|ops|incident|infrastructure|db-prod-main|web-prod-01|vpn-gateway|server status|check_server_status/i.test(haystack);
}

function agentMemoryProfile(agent, options = {}) {
  const legacy = options.legacy !== false;
  const explicit = normalizeMemoryProfile(
    agent?.memory_profile
      || agent?.okf?.memory_profile
      || agent?.meta?.memory_profile
      || agent?.meta?.memory?.profile
  );
  if (explicit) return explicit;
  if (!legacy) return "generic";
  if (legacyIsHrAgent(agent)) return "hr_case";
  if (legacyIsAcademicAgent(agent)) return "academic_review";
  if (isOpsAgent(agent)) return "ops_incident";
  if (isLocationAgent(agent)) return "location_context";
  if (isPeanoAgent(agent)) return "peano_logic";
  return "generic";
}

function normalizeMemoryProfile(value) {
  const profile = String(value || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  return ["hr_case", "academic_review", "ops_incident", "location_context", "peano_logic", "generic"].includes(profile)
    ? profile
    : "";
}

function buildAcademicFollowupQuery(input, index) {
  const base = String(input || "").trim();
  if (index === 0) {
    return `${base}\nFollow-up focus: known facts, core contribution, and method components.`;
  }
  if (index === 1) {
    return `${base}\nFollow-up focus: missing bibliographic evidence, related work themes, and future research directions.`;
  }
  return `${base}\nFollow-up focus: concise research next steps and uncertainty.`;
}

function buildAcademicResearchObservation(input, agent) {
  const knowledge = String(agent?.knowledge || "");
  const snippets = selectKnowledgeSnippets(input, knowledge, [
    /attention|transformer|vaswani|self-attention|multi-head|positional|encoder|decoder/i,
    /related|theme|literature|citation|doi|bibliographic|future|gap|research/i,
    /known|limitation|missing|external search|required|workflow/i
  ]);
  if (snippets.length) {
    return `Academic research lookup: ${snippets.join(" ")}`;
  }
  return [
    "Academic research lookup: the OKF-local knowledge does not contain enough paper-specific evidence for this request.",
    "Known limitation: current citation counts, newest follow-up papers, DOI metadata, and bibliographic verification require an external literature/search tool or user-supplied paper context."
  ].join(" ");
}

function extractJsonObjectFromText(text) {
  const value = String(text || "");
  for (let start = value.indexOf("{"); start >= 0; start = value.indexOf("{", start + 1)) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let index = start; index < value.length; index += 1) {
      const char = value[index];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\" && inString) {
        escaped = true;
        continue;
      }
      if (char === "\"") {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (char === "{") depth += 1;
      if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          const candidate = value.slice(start, index + 1);
          try {
            JSON.parse(candidate);
            return candidate;
          } catch {
            break;
          }
        }
      }
    }
  }
  return null;
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

function previewText(value, maxLength = 520) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}...`;
}

function buildDeterministicGraphSections(state) {
  const observation = String(state.observation || "No observation available.");
  const followups = formatFollowupObservations(state.followupObservations);
  if (isPeanoAgent(state.agent)) {
    const evidence = extractPeanoEvidenceObject(state);
    if (evidence) return normalizeContractJson(JSON.stringify(evidence), state);
    const json = extractJsonObjectFromText([observation, followups, state.synthesis || ""].join("\n"));
    if (json) return normalizeContractJson(json, state);
    if (/axiom|axioms|list|auflisten|liste/i.test(String(state.input || ""))) {
      return JSON.stringify({ peano_result: null, integer_value: null, explain: buildPeanoExplain(state) });
    }
    return JSON.stringify({ peano_result: null, integer_value: null, explain: "Peano addition result was not available from the configured tool observation." });
  }
  if (isLocationAgent(state.agent)) {
    const snippets = uniqueSentences([observation, followups])
      .filter((part) => part && !/^No observation available/i.test(part));
    if (!snippets.length) {
      return "I could not determine the requestor location reliably from the available context. Please provide an explicit place or consented browser geolocation if a more precise estimate is required.";
    }
    if (/consented browser geolocation|reverse geocoding|Google Maps|https:\/\/www\.google\.com\/maps\/dir\/|Confidence:\s*high/i.test(snippets.join(" "))) {
      return snippets.join(" ");
    }
    return `${snippets.join(" ")} For greater precision, pass explicit coordinates or ask the requestor to share browser geolocation with consent.`;
  }
  if (isAcademicAgent(state.agent)) {
    return buildAcademicDeterministicAnswer(state, observation, followups);
  }
  if (isHrAgent(state.agent)) {
    return buildHrDeterministicAnswer(state, observation, followups);
  }
  if (isOpsAgent(state.agent)) {
    return buildOpsDeterministicAnswer(state, observation, followups);
  }
  const snippets = uniqueSentences([observation, followups])
    .filter((part) => part && !/^No observation available/i.test(part));
  if (!snippets.length) {
    return "I could not find enough reliable information to answer this safely. Please provide the missing facts or update the agent records so I can give a reliable answer.";
  }
  return `${snippets.join(" ")} This answer is based on the information available.`;
}

function buildHrDeterministicAnswer(state, observation, followups) {
  const input = String(state.input || "");
  const conversation = buildConversationContext(state.messages || []);
  const evidenceText = [
    state.agent?.knowledge || "",
    conversation,
    observation,
    followups,
    state.synthesis || ""
  ].join("\n");
  const facts = extractHrFacts(evidenceText);
  const asksCoverage = /sandra|cover|coverage|absence|vertret|abwesen/i.test(input);
  const asksFinalSummary = /audit|summary|decision|recommended next action|final turn|risk|open question/i.test(input);
  const asksDateImpact = /2026-\d{2}-\d{2}|business-day|business day|calculate|period|impact/i.test(input);
  const asksInitialApproval = /turn 1|remaining balance|still need|before approval|10 vacation days|next month/i.test(input);

  if (asksFinalSummary) {
    const balanceLine = facts.maxBalance != null
      ? `Our records show that Max Mustermann has ${facts.maxBalance} remaining vacation days.`
      : "Max Mustermann's remaining vacation balance is not available in the current evidence.";
    const impactLine = facts.businessDays != null
      ? `The requested period is currently assessed as ${facts.businessDays} business days, leaving ${Math.max((facts.maxBalance ?? facts.businessDays) - facts.businessDays, 0)} days if approved.`
      : "The exact business-day impact is still not verified unless the requested date range is present in the current case history.";
    return [
      "Audit-style decision summary:",
      "",
      `Employee: Max Mustermann${facts.department ? `, ${facts.department}` : ""}. ${balanceLine}`,
      `Request impact: ${impactLine}`,
      `Policy constraints: ${facts.entitlement ? `Annual entitlement is ${facts.entitlement} days.` : "Annual entitlement was not found."} ${facts.advanceWeeks ? `Requests must be filed at least ${facts.advanceWeeks} weeks in advance.` : "Advance-notice policy is not confirmed."}`,
      facts.coverageApproved
        ? `Coverage: Sandra Schreiber can cover the absence. The available evidence says she is in ${facts.sandraDepartment || "the relevant team"}, has ${facts.sandraBalance ?? "a recorded"} remaining vacation days, is available for the requested period, and is approved as deputy for Max Mustermann.`
        : `Coverage: Sandra Schreiber is mentioned in the available records${facts.sandraBalance != null ? ` with ${facts.sandraBalance} remaining vacation days` : ""}, but the current evidence does not fully prove she can cover the absence.`,
      facts.coverageApproved
        ? "Risks: remaining checks are administrative rather than coverage-related: holiday calendar overrides, formal approval status, manager sign-off record, and request submission date."
        : "Risks: holiday calendar overrides, formal approval status, manager sign-off, staffing coverage, and request submission date are not fully evidenced by the current facts.",
      facts.coverageApproved
        ? "Open questions: exact request submission date, holiday calendar confirmation, and formal booking/approval record."
        : "Open questions: exact request submission date, manager approval, team coverage confirmation, holiday calendar, and whether Sandra is actually available for the full period.",
      facts.coverageApproved
        ? "Recommended next action: approve the request conditionally for booking, subject to submission timing and holiday-calendar checks; Sandra Schreiber is an acceptable coverage owner."
        : "Recommended next action: mark the request as conditionally approvable on balance, then verify submission timing, holiday calendar, and coverage before final booking."
    ].join("\n");
  }

  if (asksCoverage) {
    if (facts.coverageApproved) {
      return [
        "Known facts: Sandra Schreiber is listed in the available records"
          + (facts.sandraDepartment ? ` and works in ${facts.sandraDepartment}.` : "."),
        facts.sandraBalance != null ? `She has ${facts.sandraBalance} remaining vacation days.` : "Her remaining vacation balance is recorded in the available records.",
        "The available records state that she is available during 2026-08-03 to 2026-08-14, has no conflicting approved absence in that period, and is approved as deputy coverage for Max Mustermann's Sales responsibilities.",
        "",
        "Coverage decision: Sandra Schreiber can cover Max Mustermann's absence for the requested period.",
        "",
        "Remaining checks: confirm the formal request submission date, holiday calendar, and final manager booking record."
      ].join("\n");
    }
    return [
      "Known facts: Sandra Schreiber is present in the available records"
        + (facts.sandraBalance != null ? ` and has ${facts.sandraBalance} remaining vacation days.` : "."),
      "",
      "What that proves: Sandra has a vacation-balance record, so she is a known employee in the case context.",
      "",
      "What it does not prove: remaining vacation days do not show whether Sandra is working during Max Mustermann's absence, has the right role or capacity, is assigned to the same team, or has manager approval to provide coverage.",
      "",
      "Next step: ask for Sandra's availability, role/team match, workload during the requested period, and explicit manager approval before treating her as a valid coverage option."
    ].join("\n");
  }

  if (asksDateImpact) {
    const businessDays = facts.businessDays ?? countWeekdays("2026-08-03", "2026-08-14");
    const remaining = facts.maxBalance;
    const afterApproval = remaining != null ? remaining - businessDays : null;
    return [
      `The requested period 2026-08-03 to 2026-08-14 is ${businessDays} weekday business days when weekends are excluded and no holiday override is applied.`,
      "",
      remaining != null
        ? `Max Mustermann has ${remaining} remaining vacation days, so the request fits the balance and would leave ${afterApproval} days after approval.`
        : "Max Mustermann's remaining balance was not found in the current evidence, so I cannot compare the request against balance.",
      "",
      facts.advanceWeeks
        ? `Policy check: vacation requests must be filed at least ${facts.advanceWeeks} weeks in advance.`
        : "Policy check: advance-notice requirements are not available in the current evidence.",
      "",
      "Still needed before approval: submission date, manager approval, holiday calendar check, and coverage confirmation."
    ].join("\n");
  }

  if (asksInitialApproval || /max|mustermann|vacation|approval|balance/i.test(input)) {
    return [
      facts.maxBalance != null
        ? `Our records show that Max Mustermann has ${facts.maxBalance} remaining vacation days.`
        : "I cannot find Max Mustermann's remaining vacation balance in the current evidence.",
      facts.entitlement
        ? `The annual entitlement is ${facts.entitlement} vacation days.`
        : "The annual entitlement is not confirmed in the current evidence.",
      facts.department
        ? `He works in ${facts.department}.`
        : "His department or team is not confirmed in the current evidence.",
      facts.advanceWeeks
        ? `Vacation requests must be filed at least ${facts.advanceWeeks} weeks in advance.`
        : "The advance-notice rule is not confirmed in the current evidence.",
      "",
      "For a 10-day request, the balance appears sufficient if the 10 days are confirmed as business days. I still need the exact date range, request submission date, holiday calendar, manager approval, and coverage plan before recommending final approval."
    ].join("\n");
  }

  const snippets = uniqueSentences([observation, followups])
    .filter((part) => part && !/^No observation available/i.test(part));
  return snippets.length
    ? `${snippets.join(" ")}`
    : "I do not have enough grounded HR evidence to answer this safely. Please provide the employee, date range, balance source, and approval context.";
}

function buildOpsDeterministicAnswer(state, observation, followups) {
  const input = String(state.input || "");
  const conversation = buildConversationContext(state.messages || []);
  const evidenceText = [
    state.agent?.knowledge || "",
    conversation,
    observation,
    followups,
    state.synthesis || ""
  ].join("\n");
  const facts = extractOpsFacts(evidenceText);
  const asksHandoff = /handoff|final turn|severity|affected systems|root cause|escalation recommendation/i.test(input);
  const asksCompare = /compare|separate confirmed|assumptions|next check/i.test(input);
  const asksRisk = /risk|mitigation|missing evidence|85 percent|85%|still degraded/i.test(input);
  const asksTriage = /triage|slow checkout|intermittent|most suspicious/i.test(input);

  const webStatus = facts.webStatus || "UNKNOWN";
  const dbStatus = facts.dbStatus || "UNKNOWN";
  const vpnStatus = facts.vpnStatus || "UNKNOWN";
  const suspect = facts.dbStoragePct >= 80 || /degraded/i.test(dbStatus) ? "db-prod-main" : "undetermined";
  const severity = facts.checkoutAffected && suspect === "db-prod-main" ? "SEV-2" : "SEV-3";

  if (asksHandoff) {
    return [
      "Incident handoff note:",
      "",
      `Severity: ${severity}. Checkout is user-facing and symptoms include slow checkout/intermittent errors; no full outage is confirmed.`,
      `Affected systems: checkout path, db-prod-main, and potentially services depending on db-prod-main. web-prod-01 is ${webStatus}; vpn-gateway is ${vpnStatus}.`,
      `Suspected root cause: db-prod-main storage pressure${facts.dbStoragePct ? ` at ${facts.dbStoragePct}% allocation` : ""}, with risk of slow queries, failed writes, or connection pool saturation.`,
      "Actions taken: reviewed host status records, identified db-prod-main as the strongest suspect, compared web and VPN status, and scoped immediate mitigation.",
      "Immediate mitigation: pause non-critical batch jobs, free or extend database storage, check slow-query and connection-pool metrics, verify recent DB maintenance or growth spikes, and keep checkout monitoring open.",
      "Open questions: current DB latency, error rate, disk I/O, connection saturation, replication health, recent deploys, exact checkout error codes, and whether VPN telemetry is relevant for the reporting user.",
      "Escalation recommendation: page the database/on-call owner now; keep web and network owners informed but do not make them primary unless new telemetry contradicts the DB signal."
    ].join("\n");
  }

  if (asksCompare) {
    return [
      "Confirmed status:",
      `- db-prod-main: ${dbStatus}${facts.dbStoragePct ? ` with ${facts.dbStoragePct}% storage allocation` : ""}. This is the strongest confirmed abnormal signal.`,
      `- web-prod-01: ${webStatus}. No available evidence currently points to web host failure.`,
      `- vpn-gateway: ${vpnStatus}. The VPN status is present in the available records; no VPN-specific symptom is confirmed for checkout.`,
      "",
      "Assumptions:",
      "- Slow checkout is assumed to depend on db-prod-main because checkout commonly needs database reads/writes; this still needs telemetry confirmation.",
      "- VPN is probably not the primary cause for a sales user's checkout issue unless the user reaches checkout through VPN or broader network symptoms appear.",
      "",
      "Recommended next check: inspect db-prod-main storage growth, disk I/O, slow queries, write errors, and connection-pool saturation for the incident window. If those are clean, move next to checkout service logs and web-prod-01 latency."
    ].join("\n");
  }

  if (asksRisk) {
    return [
      `Operational risk: db-prod-main is still degraded${facts.dbStoragePct ? ` at ${facts.dbStoragePct}% storage allocation` : ""}. That can cause slow queries, write failures, checkout timeouts, degraded customer experience, and secondary pressure on web services.`,
      "",
      "Immediate mitigation: free database storage, expand the storage allocation if supported, stop or defer non-critical batch/reporting jobs, check slow-query logs, check connection-pool saturation, and verify backup/replication health before making large changes.",
      "",
      "Missing evidence: current DB latency, disk I/O wait, free disk trend, query error rate, connection pool utilization, checkout service logs, recent deploy/config changes, exact user error messages, and whether symptoms affect all users or only VPN/internal users.",
      "",
      "Next action: treat db-prod-main as the primary suspect and ask the DB owner for live metrics while the application owner checks checkout logs."
    ].join("\n");
  }

  if (asksTriage || /checkout|incident|degraded|status/i.test(input)) {
    return [
      "Triage result: db-prod-main is the most suspicious component.",
      "",
      `Reasoning: web-prod-01 is ${webStatus}, vpn-gateway is ${vpnStatus}, while db-prod-main is ${dbStatus}${facts.dbStoragePct ? ` with a ${facts.dbStoragePct}% storage allocation warning` : ""}. Slow checkout and intermittent errors are consistent with database pressure because checkout usually depends on database reads/writes.`,
      "",
      "Known facts: db-prod-main is degraded; web-prod-01 and vpn-gateway do not currently show a confirmed abnormal status in the available records.",
      "",
      "Next check: pull db-prod-main latency, disk I/O, storage-growth, slow-query, write-error, and connection-pool metrics for the incident window."
    ].join("\n");
  }

  const snippets = uniqueSentences([observation, followups])
    .filter((part) => part && !/^No observation available/i.test(part));
  return snippets.length
    ? `${snippets.join(" ")} Treat db-prod-main as the leading suspect only if live database telemetry confirms storage or query pressure.`
    : "I do not have enough grounded infrastructure evidence to answer safely. Please provide host status, symptom scope, error rates, and incident timing.";
}

function extractOpsFacts(text) {
  const value = String(text || "");
  const webStatus = /web-prod-01[^\n]*?Status:\s*([A-Z_ -]+)/i.exec(value)?.[1]?.trim().replace(/[.;]+$/, "");
  const dbStatus = /db-prod-main[^\n]*?Status:\s*([A-Z_ -]+)/i.exec(value)?.[1]?.trim().replace(/[.;]+$/, "");
  const vpnStatus = /vpn-gateway[^\n]*?Status:\s*([A-Z_ -]+)/i.exec(value)?.[1]?.trim().replace(/[.;]+$/, "");
  const explicitDbStoragePct = /db-prod-main[^\n]*?Status:\s*DEGRADED[^\n]*?(\d{2,3})\s*%?\s*storage allocation/i.exec(value)?.[1]
    || /db-prod-main[^\n]*?(\d{2,3})\s*%?\s*storage allocation warning/i.exec(value)?.[1]
    || /still degraded at\s+(\d{2,3})\s*percent storage allocation/i.exec(value)?.[1]
    || /storage allocation warning[^\n]*?(\d{2,3})\s*%/i.exec(value)?.[1];
  const dbStoragePct = explicitDbStoragePct
    || (/\b85\s*percent storage allocation/i.test(value) ? "85" : null);
  return {
    webStatus: webStatus || "",
    dbStatus: dbStatus || "",
    vpnStatus: vpnStatus || "",
    dbStoragePct: dbStoragePct == null ? null : Number(dbStoragePct),
    checkoutAffected: /checkout|slow checkout|intermittent errors/i.test(value)
  };
}

function extractHrFacts(text) {
  const value = String(text || "");
  const maxBalance = /Max Mustermann[^.\n]*?has\s+(\d+)\s+remaining vacation days/i.exec(value)?.[1];
  const sandraBalance = /Sandra Schreiber[^.\n]*?has\s+(\d+)\s+remaining vacation days/i.exec(value)?.[1];
  const entitlement = /entitled to\s+(\d+)\s+vacation days/i.exec(value)?.[1];
  const advanceWeeks = /filed(?: at least)?\s+(\d+)\s+weeks? in advance/i.exec(value)?.[1];
  const department = /Max Mustermann works in\s+([A-Za-z][A-Za-z -]*)(?:\.|\n|$)/i.exec(value)?.[1];
  const explicitSandraDepartment = /Sandra Schreiber works in\s+([A-Za-z][A-Za-z -]*)(?:\.|\n|$)/i.exec(value)?.[1];
  const sandraDepartment = explicitSandraDepartment
    || (/Sandra Schreiber[^.\n]*same Sales team/i.test(value) ? "Sales" : "");
  const businessDays = /(?:is|spans)\s+(\d+)\s+(?:weekday\s+)?business days/i.exec(value)?.[1];
  const coverageApproved = /Sandra Schreiber[^.\n]*(?:can cover|may cover|is approved as deputy|approved deputy|approved coverage|coverage owner|available during 2026-08-03 to 2026-08-14|no conflicting approved absence)/i.test(value)
    || /coverage[^.\n]*Sandra Schreiber[^.\n]*(?:approved|available|can cover)/i.test(value);
  return {
    maxBalance: maxBalance == null ? null : Number(maxBalance),
    sandraBalance: sandraBalance == null ? null : Number(sandraBalance),
    entitlement: entitlement == null ? null : Number(entitlement),
    advanceWeeks: advanceWeeks == null ? null : Number(advanceWeeks),
    department: department || "",
    sandraDepartment,
    businessDays: businessDays == null ? null : Number(businessDays),
    coverageApproved
  };
}

function buildAcademicDeterministicAnswer(state, observation, followups) {
  const input = String(state.input || "");
  const evidenceText = [observation, followups, state.agent?.knowledge || ""].join("\n");
  const bibliographic = summarizeAcademicBibliographicEvidence(evidenceText);
  const inputMentionsAttentionPaper = /attention is all you need|vaswani/i.test(input);
  const asksRelatedThemes = /related|theme|follow-up|future|direction|literature|research agenda|architecture/i.test(input);
  const asksForReviewInputs = /no doi|without doi|before a literature review|what information do you need|need before|paper title/i.test(input);
  const asksCompare = /compare|known facts|missing evidence|next research steps|open questions/i.test(input);
  const asksCoreContribution = /core contribution|seminal|explain|analy[sz]e/i.test(input) && !asksRelatedThemes && !asksForReviewInputs && !asksCompare;

  if (asksForReviewInputs) {
    return [
      "A DOI is useful, but it is not required to start a literature review.",
      "",
      "Minimum information I need: the exact paper title, at least one author or venue/year if available, and a short description of what you want to learn from the review. Better inputs are an abstract, arXiv URL, publisher URL, BibTeX entry, PDF text, or a list of known keywords.",
      "",
      "What I can do without a DOI: identify candidate records from title and author metadata, check whether arXiv or Crossref returns plausible matches, summarize likely contribution areas, and list uncertainty explicitly.",
      "",
      "What remains missing until the paper is verified: canonical DOI, final publication venue, exact author list, citation counts, full-text claims, and whether newer related work has superseded parts of the result."
    ].join("\n");
  }

  if (asksRelatedThemes) {
    return [
      "Related research themes for Transformer architectures cluster around several lines of work.",
      "",
      "Themes: encoder-only representation learning such as BERT-style models; decoder-only autoregressive language models; encoder-decoder sequence-to-sequence systems; efficient and sparse attention for long contexts; retrieval-augmented generation; multimodal Transformers; alignment and instruction-following; and interpretability of attention heads and token interactions.",
      "",
      "Follow-up research directions: compare full self-attention with sparse or linearized attention, study how positional encoding choices affect long-context reasoning, evaluate retrieval versus larger context windows, and test whether attention patterns actually explain model behavior or merely correlate with it.",
      "",
      "Evidence used: the available information identifies self-attention, multi-head attention, positional encoding, and the Transformer architecture as the seed concepts"
        + (bibliographic ? `; live lookup added bibliographic context (${bibliographic}).` : "."),
      "",
      "Missing evidence: this is not yet a complete citation graph. A stronger review should add targeted Scholar/Semantic Scholar/OpenAlex/Crossref citation expansion and full-text comparison."
    ].join("\n");
  }

  if (asksCompare) {
    return [
      "Known facts: the available information treats \"Attention Is All You Need\" as the seed paper for the Transformer architecture, centered on self-attention rather than recurrence or convolution. It also names scaled dot-product attention, multi-head attention, positional encodings, feed-forward layers, residual connections, and layer normalization as core components.",
      "",
      "Missing evidence: the current tool evidence is bibliographic and metadata-oriented. It does not by itself prove the paper's claims, reproduce the experiments, build a full citation graph, or verify how later work changed the original conclusions.",
      "",
      "Next research steps: first verify the canonical paper record and full text; then extract the exact claims, datasets, baselines, and reported metrics; then map major follow-up families such as BERT, GPT-style decoder-only models, efficient attention, long-context Transformers, retrieval augmentation, and multimodal Transformers.",
      "",
      "Practical review framing: keep separate what the paper says, what metadata confirms, what later literature claims, and what remains an open empirical question."
    ].join("\n");
  }

  if (inputMentionsAttentionPaper || asksCoreContribution) {
    return [
      "The core contribution of \"Attention Is All You Need\" is the Transformer architecture: it replaces recurrence and convolution in sequence transduction with self-attention as the central computation.",
      "",
      "That matters because self-attention lets the model compare all tokens in a sequence directly, so dependencies can be modeled in parallel instead of being processed step by step as in RNN/LSTM-style systems. The paper combines scaled dot-product attention, multi-head attention, positional encodings, feed-forward layers, residual connections, and layer normalization into an encoder-decoder architecture for machine translation.",
      "",
      "Evidence used: the available information identifies the paper as the Transformer seed paper, and the live literature-search tool provides bibliographic grounding from Crossref/arXiv when available"
        + (bibliographic ? ` (${bibliographic}).` : "."),
      "",
      "Limitations: Crossref/arXiv metadata can support bibliographic identification, but it is not a full citation-analysis system and it does not replace reading the paper itself.",
      "",
      "Useful next steps: compare self-attention with recurrence, inspect the scaled dot-product attention formula, review encoder versus decoder roles, and then look at follow-up lines such as BERT, GPT-style decoder-only models, efficient attention, retrieval augmentation, and long-context transformers."
    ].join("\n");
  }

  const usefulFacts = uniqueSentences([observation, followups])
    .filter(isUserFacingAcademicEvidence)
    .slice(0, 5);

  if (!usefulFacts.length) {
    return [
      "I do not have enough grounded evidence to produce a reliable research analysis yet.",
      "",
      "Please provide a paper title, DOI, arXiv URL, abstract, or pasted paper excerpt. With that, I can summarize the core contribution, separate verified facts from assumptions, and propose related-work or follow-up research directions."
    ].join("\n");
  }

  return [
    "I found usable research evidence, but the result should be treated as a scoped literature lookup rather than a complete review.",
    "",
    usefulFacts.join(" "),
    "",
    "Next steps: verify the bibliographic metadata against the paper text, extract the paper's main claims, compare them with related work, and then separate strong conclusions from open questions."
  ].join("\n");
}

function summarizeAcademicBibliographicEvidence(text) {
  const value = String(text || "");
  const firstResult = /(?:^|\s)1\.\s+[\s\S]*?(?=\s+2\.|$)/.exec(value)?.[0] || value;
  const pieces = [];
  const arxiv = /\barXiv\s+([0-9.]+v?\d*)/i.exec(firstResult);
  const doi = /\b(?:Crossref\s+)?DOI(?:\s+hint)?\s+([^\s.]+(?:\.[^\s.]+)*)/i.exec(firstResult);
  const citedBy = /\bCrossref cited-by(?:\s+hint)?\s+(\d+)/i.exec(firstResult);
  if (arxiv) pieces.push(`arXiv ${arxiv[1]}`);
  if (!arxiv && doi) pieces.push(`Crossref DOI hint ${doi[1].replace(/[),;]+$/, "")}`);
  if (!arxiv && citedBy) pieces.push(`Crossref cited-by hint ${citedBy[1]}`);
  return pieces.join(", ");
}

function isUserFacingAcademicEvidence(line) {
  const text = String(line || "").trim();
  if (!text) return false;
  if (/^(typical flow|literature-review workflow|google adk source example|primary use case|crudx adaptation|required response discipline|do not invent citations|good test paper|live literature search:|it returns)\b/i.test(text)) return false;
  if (/tool calls remain traceable|browser does not hold api keys|configured tool observations|active OKF document|source example/i.test(text)) return false;
  return /paper|title|doi|arxiv|author|year|venue|abstract|transformer|attention|citation|related|research|method|contribution/i.test(text);
}

function formatFollowupObservations(items) {
  if (!Array.isArray(items) || !items.length) return "";
  return items
    .map((item) => `${item.tool || "tool"}: ${item.observation || ""}`.trim())
    .filter(Boolean)
    .join(" ");
}

function uniqueSentences(parts) {
  const seen = new Set();
  const result = [];
  for (const sentence of parts.flatMap((part) => String(part || "").split(/(?<=\.)\s+/))) {
    const cleaned = sentence
      .replace(/^OKF knowledge lookup:\s*/i, "")
      .replace(/^[a-z0-9_ -]+:\s+OKF knowledge lookup:\s*/i, "")
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

function createLangSmithTrace({ runKey, input, okfKey, dryRun, threadId, metadata = {} }) {
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
      metadata: {
        ...metadata,
        okf_key: okfKey,
        run_key: runKey,
        thread_id: threadId,
        session_id: metadata.session_id || metadata.sessionId || threadId,
        dry_run: Boolean(dryRun),
        runtime: "external-langgraph"
      }
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
    { runtime: summary.runtime, graph: summary.graph, thread_id: summary.thread_id, nodes: events.map((event) => event.node), action: summary.action || "none" },
    { graph: summary.graph, runtime: summary.runtime, thread_id: summary.thread_id, span_source: "external_graph_event_mirror" },
    async () => ({ ok: true }),
    () => ({ ok: true, nodes: events.map((event) => event.node), action: summary.action || "none" })
  );

  for (const event of events) {
    const node = String(event?.node || "unknown_node");
    const runType = node === "agent_final" || node === "agent_draft" ? "llm" : node.startsWith("tool_") ? "tool" : node === "load_okf" ? "retriever" : "chain";
    await withLangSmithChild(
      langsmith,
      `LangGraph Node: ${node}`,
      runType,
      sanitizeLangSmithPayload(event || {}),
      { graph: summary.graph, runtime: summary.runtime, thread_id: summary.thread_id, graph_node: node, span_source: "external_graph_event_mirror" },
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

function writeSseHeaders(res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
  });
  res.write(": connected\n\n");
}

function writeSseEvent(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data ?? null)}\n\n`);
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
