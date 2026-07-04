# CRUDX ReAct AaaS Platform

This repository contains the deployable platform code for the CRUDX ReAct Agent-as-a-Service runtime.

CRUDX remains the data authority. Agent definitions, OKF envelopes, GUI configuration, run documents, and trace references stay in CRUDX / Firestore. GitHub is used only for versioning and CI/CD of the technical platform.

## Architecture

```text
GUI
  -> Google Cloud Function reactAaasInvoke
  -> Cloud Run LangGraph runtime
  -> Gemini
  -> LangSmith tracing
  -> CRUDX / Firestore persistence
```

## Repository Layout

```text
.github/workflows/
  react-aaas-ci.yml
  react-aaas-deploy.yml

langgraph-runtime/
  Cloud Run runtime for the generic CRUDX OKF LangGraph executor

gcf-middleware/
  Firebase / Google Cloud Function middleware for CRUDX and GUI calls
```

## CI/CD

The CI workflow validates both Node packages.

The deploy workflow:

1. Deploys `langgraph-runtime` to Cloud Run.
2. Deploys `reactAaasInvoke` to Firebase Functions.
3. Runs a Cloud Run health check.

## Secrets

Do not store API keys in GitHub.

These values belong in Google Secret Manager:

- `GEMINI_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGGRAPH_RUNTIME_TOKEN`

GitHub Actions authenticates to Google Cloud via Workload Identity Federation.

## Required GitHub Repository Variables

Configure these repository variables before running the deploy workflow:

| Variable | Description |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Google Workload Identity Provider resource name |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | Deploy service account email |

## Current Google Cloud Defaults

| Setting | Value |
| --- | --- |
| Project | `crudx-e0599` |
| Region | `europe-west3` |
| Cloud Run service | `react-aaas-langgraph-runtime` |
| Firebase codebase | `react-aaas` |
| Function | `reactAaasInvoke` |
| LangSmith project | `react-aaas-crudx` |

