# Serve Qwen3.5-9B (Q4_K_M) with llama.cpp on the local GPU: 4 parallel slots x 8,192 tokens.
#   powershell -ExecutionPolicy Bypass -File start-server.ps1
# Downloads llama.cpp (CUDA 13.4 build) and the GGUF on first use.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$bin = "tools\llama.cpp\bin\llama-server.exe"
$gguf = "tools\models\Qwen3.5-9B-Q4_K_M.gguf"
if (-not (Test-Path $bin)) {
    New-Item -ItemType Directory -Force tools\llama.cpp | Out-Null
    foreach ($f in "llama-b11858-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip") {
        Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/b11858/$f" -OutFile "tools\llama.cpp\$f"
        Expand-Archive "tools\llama.cpp\$f" -DestinationPath tools\llama.cpp\bin -Force
        Remove-Item "tools\llama.cpp\$f"
    }
}
if (-not (Test-Path $gguf)) {
    & .venv\Scripts\hf.exe download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir tools\models
}
& $bin -m $gguf -ngl 99 -c 32768 -np 4 --port 8080 -fa on --jinja --no-webui
