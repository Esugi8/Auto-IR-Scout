@echo off
setlocal
cd /d %~dp0

echo ==========================================
echo  Python virtual environment setup...
echo ==========================================

python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

call .venv\Scripts\activate.bat

echo Updating pip...
python -m pip install --upgrade pip

if exist requirements.txt (
    echo Installing requirements.txt...
    pip install -r requirements.txt
)

echo Finished!
pause