@echo off
cd /d "%~dp0"
uv run uvicorn zai_adapter.app:app --host 0.0.0.0 --port 8100 --reload
pause
