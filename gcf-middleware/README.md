# CRUDX ReAct AaaS GCF Middleware

Dieses Artefakt ist die CRUDX/GCF-Middleware vor einer externen LangGraph Runtime. Die Browser-GUI bleibt key-frei. LangGraph, Gemini und LangSmith laufen nicht mehr im aktiven GCF-Codepfad, sondern im separaten Runtime-Service `../react-aaas-langgraph-runtime`.

## Aktueller Architektur-Schnitt

```text
CRUDX GUI
  -> GCF reactAaasInvoke
      -> CRUDX OKF ueber calendarApi/v1 lesen
      -> externe LangGraph Runtime ueber /invoke aufrufen
      -> Run/Trace als CRUDX-Dokument schreiben
      -> Antwort an GUI zurueckgeben

Externe LangGraph Runtime
  -> LangGraph-Knoten ausfuehren
  -> GEMINI_API_KEY serverseitig nutzen
  -> LANGSMITH_API_KEY serverseitig nutzen
  -> Antwort, Graph Summary und LangSmith URLs an GCF zurueckgeben
```

Der Firebase Functions Entry ist `functions/index.external.js`. Ohne gesetztes `LANGGRAPH_RUNTIME_URL` antwortet `reactAaasInvoke` absichtlich mit `LANGGRAPH_RUNTIME_URL_REQUIRED`; es gibt keinen stillen Fallback auf embedded LangGraph.

## Deterministische CAM-Hilfe

Die reservierte Eingabe `?` wird zentral in `reactAaasInvoke` und `reactAaasStream` verarbeitet. Die Middleware liest das aktive CAM und erzeugt daraus eine Markdown-Antwort mit Agentenname, Zweck, konfigurierten Beispielfragen und Help-Link.

Diese Route ist absichtlich nebenwirkungsfrei:

- kein LangGraph-Lauf;
- kein LLM-Aufruf;
- keine Tool-Ausfuehrung;
- kein persistiertes Run-Dokument.

Ein CAM aktiviert die Route durch `help_command.input: "?"`. Seine `example_prompts` sollten `?` als ersten Eintrag enthalten; der Literalwert wird in der ausgegebenen nummerierten Beispielliste nicht wiederholt. Die Implementierung liegt in `functions/help-route.js` und wird auch dann ausgefuehrt, wenn die externe LangGraph Runtime voruebergehend nicht konfiguriert oder erreichbar ist.

Erforderliche GCF-Parameter:

```text
CRUDX_DOCS_API=https://europe-west3-crudx-e0599.cloudfunctions.net/calendarApi/v1
LANGGRAPH_RUNTIME_URL=https://<external-langgraph-runtime-host>
LANGGRAPH_RUNTIME_MODE=standalone
LANGGRAPH_ASSISTANT_ID=agent
LANGGRAPH_RUNTIME_TOKEN=<optional bearer token>
```

## Enthalten

- `functions/index.external.js`: HTTPS Function `reactAaasInvoke` als Middleware fuer externe LangGraph Runtime
- `functions/help-route.js`: deterministische, nebenwirkungsfreie `?`-Route aus dem aktiven CAM
- `functions/package.json`: Node.js 20 Firebase Functions Dependencies inkl. LangSmith SDK
- `firebase.json`: deploybare Firebase-Functions-Konfiguration
- `test-payloads/invoke-dry-run.json`: CRUDX-ID-konformes Testpayload

## Zielbild

```text
CRUDX GUI
  -> reactAaasInvoke
      -> CRUDX OKF ueber calendarApi/v1 lesen
      -> Standalone LangGraph Runtime /invoke mit OKF-State starten
      -> LangGraph Runtime nutzt GEMINI_API_KEY und LANGSMITH_API_KEY serverseitig
      -> Run/Trace als CRUDX-Dokument schreiben
      -> Antwort an GUI zurueckgeben
```

## Installation im bestehenden Projekt

1. Ordnerinhalt in dein Firebase Functions Repo uebernehmen.
2. Falls noch nicht vorhanden, Projekt setzen:

```bash
firebase use crudx-e0599
```

3. Runtime URL setzen:

```bash
LANGGRAPH_RUNTIME_URL=https://<cloud-run-runtime-url>
LANGGRAPH_RUNTIME_MODE=standalone
```

Gemini- und LangSmith-Secrets gehoeren in die externe Standalone Runtime, nicht mehr in diese GCF-Middleware.

4. Dependencies installieren:

```bash
cd functions
npm install
```

5. Function deployen:

```bash
firebase deploy --only functions --project crudx-e0599
```

## Erwarteter Endpoint

```text
https://europe-west3-crudx-e0599.cloudfunctions.net/reactAaasInvoke
```

Dieser Wert gehoert in der GUI in:

```text
Settings -> CRUDX Invoke GCF Endpoint
```

Das Gemini-Key-Feld in der GUI bleibt leer.

## CRUDX GCF Dokument-API

Standardmaessig liest und schreibt die Function ueber:

```text
https://europe-west3-crudx-e0599.cloudfunctions.net/calendarApi/v1
```

Wenn deine CRUDX-Dokument-API anders heisst, setze den Firebase-v2-Parameter `CRUDX_DOCS_API`.
Einfachster Weg: Lege in `functions/.env` an:

```text
CRUDX_DOCS_API=https://europe-west3-crudx-e0599.cloudfunctions.net/calendarApi/v1
```

Optional kannst du dort auch das Modell setzen:

```text
GEMINI_MODEL=gemini-3.5-flash
```

LangGraph Runtime Parameter:

```text
LANGGRAPH_RUNTIME_URL=http://localhost:8080
LANGGRAPH_RUNTIME_MODE=standalone
LANGGRAPH_ASSISTANT_ID=agent
```

`LANGGRAPH_RUNTIME_MODE=standalone` nutzt die guenstige eigene Runtime-Route:

```text
POST /invoke
```

`LANGGRAPH_RUNTIME_MODE=langgraph` bleibt als optionaler Kompatibilitaetsmodus fuer LangGraph Cloud/Studio erhalten, ist aber nicht der empfohlene Standard fuer dieses Projekt.

LangSmith ist serverseitig konfiguriert. Optional kannst du Projekt, Endpoint und Workspace setzen:

```text
LANGSMITH_PROJECT=react-aaas-crudx
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_WORKSPACE_ID=
```

Fuer EU-Workspaces nutze:

```text
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

Ohne `.env` nutzt die Function `docs` als Collection, `gemini-3.5-flash` als Modell und `react-aaas-crudx` als LangSmith-Projekt.

## Test

Dry-run ohne Gemini-Kosten:

```bash
curl -X POST "https://europe-west3-crudx-e0599.cloudfunctions.net/reactAaasInvoke" \
  -H "Content-Type: application/json" \
  -d @test-payloads/invoke-dry-run.json
```

Echter Gemini-Test:

```bash
curl -X POST "https://europe-west3-crudx-e0599.cloudfunctions.net/reactAaasInvoke" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "alice-hr",
    "okf_key": "CRUDX-5432U-D58X8-52X8U",
    "input": "Calculate vacation days for Max Mustermann from France.",
    "trace": true
  }'
```

## Antwortformat

```json
{
  "ok": true,
  "run_key": "CRUDX-.....-.....-.....",
  "okf_key": "CRUDX-5432U-D58X8-52X8U",
  "agent_id": "alice-hr",
  "answer": "...",
  "langsmith": {
    "enabled": true,
    "project": "react-aaas-crudx",
    "endpoint": "https://api.smith.langchain.com",
    "run_id": "...",
    "trace_id": "...",
    "errors": []
  },
  "trace": []
}
```

Der `run_key` folgt den CRUDX-ID-Regeln:

```text
CRUDX-[RDUCX23458]{5}-[RDUCX23458]{5}-[RDUCX23458]{5}
```

## Hinweise

- Der Gemini-Key wird nur serverseitig verwendet.
- Der LangSmith-Key wird nur serverseitig verwendet.
- Die GCF-Middleware selbst braucht keine Gemini- oder LangSmith-Secrets mehr, wenn die externe LangGraph Runtime diese besitzt.
- Die GUI braucht nur den Invoke-Endpoint.
- LangGraph wird ueber `LANGGRAPH_RUNTIME_URL` angebunden.
- Echte LangSmith-Traces entstehen in der externen Standalone Runtime, z.B. auf Google Cloud Run.
