@echo off
cd /d %~dp0
cd ..

echo Checking for prerequisites...

where py >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Python not found. Administrator privileges are required to install it.
    goto install_python_prompt
)
echo  - Python: Found.

goto main

:install_python_prompt
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_python'"
    exit /b
)
echo [INFO] Now running as Administrator to install Python...
set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
set "INSTALLER_PATH=%TEMP%\python_installer.exe"
echo Downloading Python 3.11.5 installer...
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%PYTHON_INSTALLER_URL%', '%INSTALLER_PATH%')"
if %errorlevel% neq 0 ( goto end_error )
start /wait %INSTALLER_PATH% /quiet InstallAllUsers=1 PrependPath=1
del "%INSTALLER_PATH%"
echo [SUCCESS] Python has been installed. Please re-run this script.
goto end_success

:main
pip install -e ".[dev]"
goto end_success

:end_error
echo.
echo [ERROR]
echo.
pause
exit /b 1

:end_success
pause
exit /b 0
