# Full pilot pipeline on one Windows machine with an 8 GB GPU.
#   powershell -ExecutionPolicy Bypass -File run_pilot.ps1 [-Run pilot_main] [-Stage all|rollout|belief|replay|analyze]
# The rollout and belief stages need llama-server running (start-server.ps1); replay needs it stopped
# (the GPU cannot hold both). Every stage is resumable or idempotent.
param([string]$Run = "pilot_main", [string]$Stage = "all", [string]$Model = "Qwen/Qwen3.5-9B")
$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot

function Step($name, $cmd) { Write-Host "`n=== $name ===" -ForegroundColor Cyan; & $py @cmd; if ($LASTEXITCODE) { throw "$name failed" } }

if ($Stage -in "all", "rollout") {
    if (-not (Test-Path data\worlds.jsonl)) { Step "build worlds" @("-m", "pilot.worlds.build_worlds") }
    # Resumable, so a crash (or the server staying down past the client's retries) just reruns it.
    for ($i = 1; $i -le 10; $i++) {
        try { Step "rollout (960 episodes), attempt $i" @("-m", "pilot.rollout.run", "--backend", "server", "--model", $Model, "--run", $Run, "--workers", "4"); break }
        catch { if ($i -eq 10) { throw }; Write-Host "$_; retrying in 60 s" -ForegroundColor Yellow; Start-Sleep 60 }
    }
}
if ($Stage -in "all", "belief") {
    Step "belief check" @("-m", "pilot.belief.check", "--run", $Run, "--backend", "server", "--model", $Model)
}
if ($Stage -in "all", "replay") {
    New-Item -ItemType File -Force data\server.stop | Out-Null  # ends start-server.ps1's restart loop
    if (Get-Process llama-server -ErrorAction SilentlyContinue) { Write-Host "Stopping llama-server to free the GPU"; Stop-Process -Name llama-server; Start-Sleep 10 }
    Step "replay equivalence (5 episodes)" @("-m", "pilot.replay.activations", "--run", $Run, "--model", $Model, "--equivalence", "5")
    Step "teacher-forced replay" @("-m", "pilot.replay.activations", "--run", $Run, "--model", $Model)
}
if ($Stage -in "all", "analyze") {
    Step "readouts" @("-m", "pilot.monitors.readout", "--run", $Run)
    Step "analysis" @("-m", "pilot.analysis.score", "--run", $Run)
    Step "figure" @("-m", "pilot.analysis.figure", "--run", $Run)
    Write-Host "`nResults in results\$Run (report.md, headline_figure.png, summary.json)" -ForegroundColor Green
}
