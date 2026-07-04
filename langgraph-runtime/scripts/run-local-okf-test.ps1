$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$payloadPath = Join-Path $root "test-payloads\okf-run.json"

$info = Invoke-RestMethod -Method Get -Uri "http://localhost:2024/info"
Write-Host "LangGraph server:" $info.context $info.langgraph_js_version

$thread = Invoke-RestMethod -Method Post -Uri "http://localhost:2024/threads" -ContentType "application/json" -Body "{}"
Write-Host "Thread:" $thread.thread_id

$payload = Get-Content -Raw -LiteralPath $payloadPath
$result = Invoke-RestMethod -Method Post -Uri "http://localhost:2024/threads/$($thread.thread_id)/runs/wait" -ContentType "application/json" -Body $payload

Write-Host ""
Write-Host "Agent:" $result.agent.agent_id
Write-Host "Action:" $result.action
Write-Host "Nodes:" ($result.graphEvents.node -join " -> ")
Write-Host ""
Write-Host $result.finalAnswer
