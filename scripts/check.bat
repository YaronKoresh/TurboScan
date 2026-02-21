@echo off
cd /d %~dp0
cd ..

call poe check

pause
exit /b 0
