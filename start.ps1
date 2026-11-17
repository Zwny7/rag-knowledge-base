$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_XET = '1'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$existingPython = Join-Path $PSScriptRoot '..\ai-portfolio\.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $projectPython) {
    $runtime = $projectPython
} else {
    Write-Host 'Creating project environment with Python 3.12...'
    if (Test-Path -LiteralPath $existingPython) {
        & $existingPython -m venv .venv
    } else {
        & py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $runtime = $projectPython
}

& $runtime -m pip install --disable-pip-version-check -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runtime launch.py @args
exit $LASTEXITCODE
