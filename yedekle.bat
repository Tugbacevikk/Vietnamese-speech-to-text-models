@echo off
chcp 65001 > nul
rem ================================================================
rem  Veritabani yedegi (isler, satirlar, duzeltmeler) -> data\yedek
rem  Sunucu acikken bu zaten gunde bir kendiliginden alinir; bu dosya elle yedek icindir.
rem  Listelemek:     yedekle.bat listele
rem  Geri yuklemek:  durdur.bat, sonra  yedekle.bat geri-yukle data\yedek\yedek_....dump
rem ================================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
%PY% scripts\yedekle.py %*
pause
