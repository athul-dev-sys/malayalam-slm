@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8010"
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8010
