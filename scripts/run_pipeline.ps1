# Full reproducible pipeline on Windows PowerShell. Stops at the first error.
# Usage (from the repository root, with the virtual environment activated):
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Step($name, [scriptblock]$cmd) {
    Write-Host "== $name" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit $LASTEXITCODE)" }
}
Write-Host "== host: $env:COMPUTERNAME | $(python --version) | $(Get-Date -Format s)"
Step "1/6 acquire data"        { python scripts/download_data.py }
Step "2/6 validate + clean"    { python -m heartrisk.data }
Step "3/6 EDA figures"         { python -m heartrisk.eda }
Step "4/6 train + MLflow"      { python -m heartrisk.train }
Step "5/6 unit tests"          { pytest -q }
Step "6/6 packaged-model check" { python -m heartrisk.predict --input samples/sample_request.json }
Write-Host "== pipeline finished OK" -ForegroundColor Green
