# Setup script for venv and dependencies
$venvDir = ".\.venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvDir..."
    python -m venv $venvDir
}

Write-Host "Upgrading pip and ensuring setuptools/wheel are installed..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --upgrade setuptools wheel
Write-Host "Installing requirements (preferring binary wheels to avoid building from source)..."
# Prefer pre-built wheels when available; fallback to source if absolutely necessary
& $venvPython -m pip install --prefer-binary -r requirements.txt
Write-Host "Virtual environment created and dependencies installed. Activate with: . .\.venv\Scripts\Activate"