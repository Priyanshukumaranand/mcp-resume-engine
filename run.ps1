#!/usr/bin/env pwsh
# Run script that sets up correct Python path
# The Windows Store Python has all dependencies installed

$env:PATH = "C:\Users\priya\AppData\Local\Microsoft\WindowsApps;$env:PATH"

if ($args.Count -eq 0) {
    Write-Host "Starting Resume Discovery API..." -ForegroundColor Green
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
} else {
    python @args
}
