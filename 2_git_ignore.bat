@echo off
setlocal
cd /d %~dp0

echo ==========================================
echo  Git Setup: Updating .gitignore
echo ==========================================

set GITIGNORE=.gitignore

rem Create .gitignore if it does not exist
if not exist %GITIGNORE% (
    echo [INFO] Creating new .gitignore file.
    type nul > %GITIGNORE%
)

rem Add .venv/ to .gitignore if not already present
findstr /C:".venv/" %GITIGNORE% >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Adding .venv/ to .gitignore...
    echo.>> %GITIGNORE%
    echo # Python Virtual Environment >> %GITIGNORE%
    echo .venv/ >> %GITIGNORE%
    echo venv/ >> %GITIGNORE%
	echo .streamlit/ >> %GITIGNORE%
    echo __pycache__/ >> %GITIGNORE%
) else (
    echo [OK] .venv/ is already in .gitignore.
)

rem Remove .venv from git index (just in case it was added)
echo [INFO] Removing .venv from git tracking...
git rm -r --cached .venv --ignore-unmatch >nul 2>&1
git rm -r --cached __pycache__ --ignore-unmatch >nul 2>&1

echo ==========================================
echo  Setup Complete!
echo  Your .venv folder is now ignored by Git.
echo ==========================================
pause