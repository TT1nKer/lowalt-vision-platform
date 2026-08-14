$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $projectRoot "quality\parking_secondary_sam3\logs"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$entryPoint = Join-Path $projectRoot "parking_secondary_analysis.py"

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Set-Location $projectRoot
& $python -u $entryPoint --project-dir $projectRoot --workers 4 `
    1>> (Join-Path $logRoot "all.stdout.log") `
    2>> (Join-Path $logRoot "all.stderr.log")
exit $LASTEXITCODE
