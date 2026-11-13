@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_release.ps1"
set "release_exit_code=%errorlevel%"
popd
if not "%release_exit_code%"=="0" pause
exit /b %release_exit_code%
