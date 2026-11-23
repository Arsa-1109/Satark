@echo off
cd /d "%~dp0"
echo Starting Satark (100%% Offline On-Device Scam Detector)...
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
