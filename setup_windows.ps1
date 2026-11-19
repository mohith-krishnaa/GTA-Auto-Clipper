$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python was not found. Install Python 3.10 or newer, enable 'Add Python to PATH', then reopen PowerShell."
}

$versionOutput = & python --version
if ($LASTEXITCODE -ne 0) {
    throw "Could not run Python. Check your Python installation and PATH."
}
$versionText = ($versionOutput -replace '^Python\s+', '').Trim()
$pythonVersion = [version]$versionText
if ($pythonVersion.Major -lt 3 -or ($pythonVersion.Major -eq 3 -and $pythonVersion.Minor -lt 10)) {
    throw "Python 3.10 or newer is required. Detected: $versionText"
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating project virtual environment..."
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
}

Write-Host "Installing Python dependencies into .venv..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Review the pip error above." }

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Warning "FFmpeg is missing from PATH. Install FFmpeg, reopen PowerShell, then run: .\.venv\Scripts\python.exe preflight.py --strict"
}
$runtime = Get-Command deno -ErrorAction SilentlyContinue
if (-not $runtime) { $runtime = Get-Command node -ErrorAction SilentlyContinue }
if (-not $runtime) { Write-Warning "Node.js or Deno was not found. Install one for YouTube extraction." }

Write-Host "Running environment check..."
if ($ffmpeg) {
    & $venvPython preflight.py --strict
} else {
    & $venvPython preflight.py
}
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }
if (-not $ffmpeg) { throw "Python dependencies are installed, but FFmpeg is still required. Install it and rerun the strict preflight check." }
Write-Host "Setup complete. Run the project with: .\.venv\Scripts\python.exe clip.py --help"
