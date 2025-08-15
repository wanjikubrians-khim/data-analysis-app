@echo off
echo =======================================
echo  Starting Python Analytics Server
echo =======================================
echo.

echo Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup_python_env.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting Python Analytics API server...
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python python_api_server.py

pause
