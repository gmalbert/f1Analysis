# Launch the API for direct use on this computer. React continues to use port 5174.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repoDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python is missing: $pythonExecutable. Create the project .venv first."
}

$previousLocalMode = $env:F1_TRUSTED_LOCAL
try {
    $env:F1_TRUSTED_LOCAL = '1'
    Push-Location (Join-Path $PSScriptRoot 'backend')
    try {
        & $pythonExecutable -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers --log-config logging.json
        if ($LASTEXITCODE -ne 0) { throw "The local API exited with code $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    $env:F1_TRUSTED_LOCAL = $previousLocalMode
}
