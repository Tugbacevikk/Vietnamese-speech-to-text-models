@echo off
chcp 65001 > nul
rem ================================================================
rem  TAM MOD: API + dil tespiti + Vietnamca + Türkçe worker'ları + bildirim + bakım.
rem  PostgreSQL ve Redis çalışıyor olmalı (DATABASE_URL, REDIS_URL: .env).
rem  Docker Desktop varsa ikisini şöyle açabilirsiniz:
rem    docker run -d --name asr-pg -e POSTGRES_USER=asr -e POSTGRES_PASSWORD=asr -e POSTGRES_DB=asr -p 5432:5432 postgres:16-alpine
rem    docker run -d --name asr-redis -p 6379:6379 redis:7-alpine redis-server --appendonly yes
rem ================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
if not exist ".env" copy ".env.example" ".env" > nul

set PORT=8001
set HOST=127.0.0.1
for /f "tokens=1,2" %%A in ('%PY% -c "from viet_asr.interface.settings import Settings; s = Settings(); print(s.host, s.port)"') do (
    set HOST=%%A
    set PORT=%%B
)

start "Ses Analizi - API"        cmd /k "%PY% -m uvicorn viet_asr.interface.api:app --host %HOST% --port %PORT%"
timeout /t 5 /nobreak > nul
start "Ses Analizi - Dil"        cmd /k "%PY% -m viet_asr.interface.stream_worker --kuyruk lid"
start "Ses Analizi - Vietnamca"  cmd /k "%PY% -m viet_asr.interface.stream_worker --kuyruk vi"
start "Ses Analizi - Turkce"     cmd /k "%PY% -m viet_asr.interface.stream_worker --kuyruk tr --surec 1"
start "Ses Analizi - Bildirim"   cmd /k "%PY% -m viet_asr.interface.dispatcher"
start "Ses Analizi - Bakim"      cmd /k "%PY% -m viet_asr.interface.bakim"

echo.
echo   Tum servisler ayri pencerelerde baslatildi: http://%HOST%:%PORT%/docs
echo   Durdurmak icin: durdur.bat
