$ErrorActionPreference = "Stop"

$candidatePythons = @(
    ".\fypenv\Scripts\python.exe",
    "$env:USERPROFILE\anaconda3\python.exe",
    "python"
)

$selectedPython = $null

foreach ($candidate in $candidatePythons) {
    try {
        & $candidate -m streamlit --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = $candidate
            break
        }
    } catch {
    }
}

if (-not $selectedPython) {
    Write-Host "Streamlit was not found in fypenv, Anaconda, or default Python."
    Write-Host "Install it with: pip install -r streamlit_requirements.txt"
    exit 1
}

Write-Host "Starting Streamlit with: $selectedPython"
& $selectedPython -m streamlit run streamlit_app.py
