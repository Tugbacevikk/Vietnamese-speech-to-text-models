@echo off
chcp 65001 > nul
rem ================================================================
rem  SQLite'tan PostgreSQL'e gecis (yerel mod). Once:
rem    - PostgreSQL kurulu ve calisiyor, bos bir veritabani olusturulmus olmali
rem    - .env icinde: LOCAL_DATABASE_URL=postgresql://kullanici:parola@localhost:5432/asr
rem  Islerin, satirlarin, duzeltmelerin hepsi tasinir; eski jobs.db yedek olarak kalir.
rem ================================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
echo [1/3] Sunucu durduruluyor (tasima sirasinda yeni is yazilmasin)...
call durdur.bat sessiz
echo [2/3] Kontrol (hicbir sey yazilmaz)...
%PY% scripts\postgrese_tasi.py --dene || goto :hata
echo [3/3] Tasiniyor...
%PY% scripts\postgrese_tasi.py || goto :hata
echo.
echo   Bitti. baslat.bat ile acin: veriler artik PostgreSQL'de.
pause
goto :eof

:hata
echo.
echo   Tasima yapilmadi. Yukaridaki mesaja bakin (PostgreSQL calisiyor mu, .env adresi dogru mu).
pause
