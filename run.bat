@echo off
cd /d "%~dp0"
echo Starting Satark (100%% Offline On-Device Scam Detector)...

set "PY_EXE=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
)

start /b "" powershell -NoProfile -Command "for ($i=0; $i -lt 60; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8000'; break } } catch {}; Start-Sleep -Milliseconds 300 }"

"%PY_EXE%" -m uvicorn app:app --host 127.0.0.1 --port 8000
