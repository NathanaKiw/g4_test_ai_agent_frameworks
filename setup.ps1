# Instala dependências do baseline vanilla (caso de uso + common).
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
    python -m venv "$Root\.venv"
}

$pip = "$Root\.venv\Scripts\pip.exe"
& $pip install -r "$Root\requirements.txt"
& $pip install -e "$Root\common" -e "$Root\vanilla"

if (-not (Test-Path "$Root\.env")) {
    Copy-Item "$Root\.env.example" "$Root\.env"
    Write-Host "Arquivo .env criado — defina OPENAI_API_KEY antes de executar."
}

Write-Host "Pronto. Ative: .\.venv\Scripts\activate"
