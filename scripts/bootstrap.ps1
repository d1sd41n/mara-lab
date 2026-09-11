[CmdletBinding()]
param(
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolDirectory = Join-Path $projectRoot ".tools\uv312"
$toolPython = Join-Path $toolDirectory "Scripts\python.exe"
$uv = Join-Path $toolDirectory "Scripts\uv.exe"

Push-Location $projectRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python Launcher (py.exe) is required. Install Python 3.12 first."
    }

    & py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is required."
    }

    if (-not (Test-Path -LiteralPath $toolPython)) {
        & py -3.12 -m venv $toolDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the local uv environment."
        }
    }

    if (-not (Test-Path -LiteralPath $uv)) {
        & $toolPython -m pip install --disable-pip-version-check "uv==0.12.13"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install uv into the project-local tool environment."
        }
    }

    $syncArguments = @("--cache-dir", ".cache/uv", "sync", "--frozen")
    if ($Gpu) {
        $syncArguments += @("--extra", "gpu")
    }

    & $uv @syncArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency synchronization failed."
    }

    Write-Host "Mara Lab is ready in $projectRoot\.venv"
}
finally {
    Pop-Location
}
