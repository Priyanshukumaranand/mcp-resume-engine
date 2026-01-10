@echo off
REM Run script that sets up correct Python path
REM The Windows Store Python has all dependencies installed

set PATH=C:\Users\priya\AppData\Local\Microsoft\WindowsApps;%PATH%

REM Run the provided command or start uvicorn by default
if "%~1"=="" (
    echo Starting Resume Discovery API...
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
) else (
    python %*
)
