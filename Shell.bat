@echo off
setlocal
cd /d %~dp0

echo ==========================================
echo  Opening terminal with .venv activated...
echo ==========================================

rem Check if .venv exists
if not exist .venv (
    echo [ERROR] .venv folder not found.
    pause
    exit /b
)

rem Open a new cmd window, activate .venv, and keep it open (/k)
cmd /k ".venv\Scripts\activate.bat"

rem & python app.py"