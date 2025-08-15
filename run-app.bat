@echo off
echo ================================
echo  Data Analysis App Launcher
echo ================================
echo.

echo Checking Java installation...
java -version
if errorlevel 1 (
    echo ERROR: Java is not installed or not in PATH
    echo Please install Java 8 or higher and try again
    pause
    exit /b 1
)
echo.

echo Checking Maven installation...
mvn -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Maven is not installed or not in PATH
    echo.
    echo Please install Maven to run the application:
    echo 1. Download Maven from: https://maven.apache.org/download.cgi
    echo 2. Extract to a folder (e.g., C:\apache-maven-3.9.4)
    echo 3. Add the bin folder to your system PATH
    echo 4. Restart command prompt and run this script again
    echo.
    echo Alternative: Open the project in an IDE like IntelliJ IDEA or Eclipse
    echo and run the DataAnalysisApplication.java file directly
    pause
    exit /b 1
)
echo.

echo Building the application...
call mvn clean compile
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)
echo.

echo Starting the Data Analysis Application...
echo.
echo The application will be available at: http://localhost:8080
echo.
echo Press Ctrl+C to stop the application
echo.
call mvn spring-boot:run

pause
