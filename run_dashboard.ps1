$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
$Streamlit = ".\.venv\Scripts\streamlit.exe"
$Port = 11000

if (-not (Test-Path $Python)) {
    & "C:\Users\yago.martins.SOOW\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
    & $Python -m pip install -r requirements.txt
}

& $Python -m pip install -r requirements.txt

$CanUsePort = & $Python -c "import socket, sys; s=socket.socket(); rc=0;`ntry: s.bind(('127.0.0.1', $Port))`nexcept OSError: rc=1`nfinally: s.close()`nsys.exit(rc)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Porta 11000 indisponivel neste Windows; usando fallback 11001."
    $Port = 11001
}

& $Streamlit run dashboard_app.py --server.port $Port --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
