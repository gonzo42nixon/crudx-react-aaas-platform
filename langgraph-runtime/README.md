# CRUDX ReAct AaaS External LangGraph Runtime

This package hosts the LangGraph executor outside Google Cloud Functions.

The intended production flow is:

```text
Browser GUI -> GCF reactAaasInvoke -> External LangGraph Runtime -> Gemini / LangSmith
                                |
                                -> CRUDX run persistence
```

GCF remains the CRUDX and request-validation middleware. The runtime owns graph execution and LLM/tracing calls.

## Generic Agent Model

The runtime is intentionally generic. It does not hard-code Alice, Bob, or any future agent.

Each invocation receives a CRUDX OKF envelope from GCF:

- `okf_key`: the CRUDX document ID for the active agent OKF.
- `okf_envelope`: the Firestore/CRUDX payload containing `agent_id`, `system_prompt`, `allowed_tools`, `mcp_endpoints`, `knowledge`, and model settings.
- `messages`: optional prior conversation turns.
- `input`: the current user query.

That means adding a new agent should be a CRUDX data operation, not a new LangGraph deployment. The graph remains the same; the OKF document selects the agent persona, prompt, tools, knowledge, tags, and runtime metadata.

## Environment

- `GEMINI_API_KEY`: Gemini key for the `agent_final` graph node.
- `LANGSMITH_API_KEY`: LangSmith key for graph traces.
- `LANGSMITH_ENDPOINT`: usually `https://api.smith.langchain.com`.
- `LANGSMITH_PROJECT`: default `react-aaas-crudx`.
- `LANGSMITH_WORKSPACE_ID`: workspace path segment used to build URLs.
- `LANGGRAPH_RUNTIME_TOKEN`: optional bearer token expected from GCF.

## Recommended Deployment: Google Cloud Run

The recommended production target for this project is a small standalone HTTP service on Google Cloud Run. This avoids LangGraph Cloud subscription cost while keeping LangGraph outside GCF.

```text
GCF reactAaasInvoke -> Cloud Run /invoke -> Gemini / LangSmith
```

Build/deploy this folder as a Cloud Run service. The included `Dockerfile` starts:

```text
node src/server.js
```

The service exposes:

- `GET /health`
- `POST /invoke`

After Cloud Run deployment, configure GCF with:

```text
LANGGRAPH_RUNTIME_URL=https://<cloud-run-service-url>
LANGGRAPH_RUNTIME_MODE=standalone
```

## Optional LangGraph Cloud Export

This project also exports the graph through `langgraph.json`:

```json
{
  "graphs": {
    "agent": "./src/graph.mjs:graph"
  }
}
```

LangGraph Cloud deployment is optional and not required for the CRUDX AaaS flow:

```powershell
cd C:\Users\druef\Documents\Codex\2026-07-04\ic\outputs\react-aaas-langgraph-runtime
langgraph deploy
```

The LangGraph assistant ID is `agent`.

## Standalone HTTP API

`POST /invoke`

Payload from GCF:

```json
{
  "run_key": "CRUDX-...",
  "agent_id": "alice-hr",
  "okf_key": "CRUDX-...",
  "input": "User query",
  "messages": [],
  "dry_run": false,
  "okf_envelope": {}
}
```

The response contains `answer`, `graph`, `trace`, and `langsmith`.
