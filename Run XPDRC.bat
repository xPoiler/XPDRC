@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo    XPDRC 1.2 - Automated Launcher
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Error: Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

:: Define virtual environment directory
set VENV_DIR=.venv

:: Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo [*] Creating virtual environment...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [!] Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and install requirements
echo [*] Activating environment and checking dependencies...
call %VENV_DIR%\Scripts\activate

:: Check if requirements are already installed (basic check)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo [*] Starting XPDRC...
echo [i] The application will open in your default browser shortly.
echo [i] Keep this window open while using the app.
echo ------------------------------------------

python app.py

if %errorlevel% neq 0 (
    echo.
    echo [!] XPDRC closed with an error (Code: %errorlevel%).
    pause
)

deactivate
