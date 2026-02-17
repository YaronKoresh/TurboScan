@echo off
cd /d %~dp0
cd ..

pytest --cov=src/turboscan --cov-report=html --cov-report=term-missing

pause
exit /b 0
