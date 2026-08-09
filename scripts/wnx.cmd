@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0wnx.ps1" %*
exit /b %ERRORLEVEL%
