@echo off
echo =======================================
echo  Python Analytics Environment Setup
echo =======================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
echo This may take a few minutes depending on your internet connection...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup Complete! 
echo ========================================
echo.
echo To start the Python Analytics server:
echo 1. Run: venv\Scripts\activate.bat
echo 2. Run: python python_api_server.py
echo.
echo The server will be available at: http://localhost:5000
echo.
pause
