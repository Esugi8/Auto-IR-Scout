@echo off
setlocal
cd /d %~dp0

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] Installing requirements...
pip install -r requirements.txt

echo [3/3] Checking installed packages...
pip list

echo ==========================================
echo  Install completed.
echo ==========================================
pause