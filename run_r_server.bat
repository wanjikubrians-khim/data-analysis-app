@echo off
echo ======================================
echo      R-Integrated Data Analysis App
echo ======================================

REM Check if R is installed
where R >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: R is not installed or not in PATH
    echo Please install R from https://cran.r-project.org/
    echo Or run r-analytics\setup_r_env.bat first
    echo.
    echo Continuing without R integration...
) else (
    echo ✓ R installation found
)

REM Check if Python virtual environment exists
if not exist python-analytics\venv (
    echo Creating Python virtual environment...
    cd python-analytics
    python -m venv venv
    cd ..
)

REM Activate virtual environment and install dependencies
echo Activating Python environment and installing dependencies...
cd python-analytics
call venv\Scripts\activate.bat
pip install -q -r requirements.txt

REM Check if R analytics script exists
if not exist "..\r-analytics\data_analyzer.R" (
    echo WARNING: R analytics script not found at r-analytics\data_analyzer.R
    echo Some R features may not work properly
)

REM Start the integrated server
echo.
echo Starting Python + R Data Analysis API Server...
echo Server will be available at http://localhost:5000
echo.
python python_api_server.py

pause
