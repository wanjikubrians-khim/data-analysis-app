@echo off
echo ==========================================
echo      R Analytics Environment Setup
echo ==========================================
echo.

REM Check if R is installed
where R >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ R is not installed or not in PATH
    echo.
    echo To install R:
    echo 1. Download R from: https://cran.r-project.org/bin/windows/base/
    echo 2. Run the installer as administrator
    echo 3. Make sure to check 'Add R to system PATH' during installation
    echo 4. Restart your command prompt after installation
    echo 5. Re-run this script
    echo.
    
    REM Try to detect R in common locations
    if exist "C:\Program Files\R" (
        echo Found R installation at: C:\Program Files\R
        echo Please add R to your system PATH manually or reinstall with PATH option
    )
    
    if exist "C:\Program Files (x86)\R" (
        echo Found R installation at: C:\Program Files ^(x86^)\R
        echo Please add R to your system PATH manually or reinstall with PATH option
    )
    
    pause
    exit /b 1
) else (
    echo ✅ R installation found
    R --version
)

echo.
echo Installing required R packages...
echo This may take several minutes...
echo.

REM Create R script to install packages
echo # R Package Installation Script > temp_r_setup.R
echo cat("Installing R packages for data analysis...\n") >> temp_r_setup.R
echo. >> temp_r_setup.R
echo # Define required packages >> temp_r_setup.R
echo packages ^<- c( >> temp_r_setup.R
echo   "readr",        # Data reading >> temp_r_setup.R
echo   "dplyr",        # Data manipulation >> temp_r_setup.R
echo   "jsonlite",     # JSON handling >> temp_r_setup.R
echo   "corrplot",     # Correlation plots >> temp_r_setup.R
echo   "cluster",      # Clustering >> temp_r_setup.R
echo   "survival",     # Survival analysis >> temp_r_setup.R
echo   "forecast",     # Time series >> temp_r_setup.R
echo   "randomForest", # Machine learning >> temp_r_setup.R
echo   "e1071",        # SVM and statistical tests >> temp_r_setup.R
echo   "psych",        # Psychological statistics >> temp_r_setup.R
echo   "nortest",      # Normality tests >> temp_r_setup.R
echo   "tseries",      # Time series analysis >> temp_r_setup.R
echo   "VIM",          # Visualization and Imputation of Missing values >> temp_r_setup.R
echo   "Hmisc",        # Harrell Miscellaneous >> temp_r_setup.R
echo   "moments"       # Statistical moments >> temp_r_setup.R
echo ^) >> temp_r_setup.R
echo. >> temp_r_setup.R
echo # Function to install packages with error handling >> temp_r_setup.R
echo install_package ^<- function^(pkg^) { >> temp_r_setup.R
echo   tryCatch^({ >> temp_r_setup.R
echo     if ^(!require^(pkg, character.only = TRUE^)^) { >> temp_r_setup.R
echo       cat^("Installing", pkg, "...\n"^) >> temp_r_setup.R
echo       install.packages^(pkg, dependencies = TRUE, quiet = FALSE^) >> temp_r_setup.R
echo       library^(pkg, character.only = TRUE^) >> temp_r_setup.R
echo       cat^("✅", pkg, "installed successfully\n"^) >> temp_r_setup.R
echo     } else { >> temp_r_setup.R
echo       cat^("✅", pkg, "already installed\n"^) >> temp_r_setup.R
echo     } >> temp_r_setup.R
echo   }, error = function^(e^) { >> temp_r_setup.R
echo     cat^("❌ Failed to install", pkg, ":", conditionMessage^(e^), "\n"^) >> temp_r_setup.R
echo   }^) >> temp_r_setup.R
echo } >> temp_r_setup.R
echo. >> temp_r_setup.R
echo # Install all packages >> temp_r_setup.R
echo cat^("Starting package installation...\n"^) >> temp_r_setup.R
echo for ^(pkg in packages^) { >> temp_r_setup.R
echo   install_package^(pkg^) >> temp_r_setup.R
echo } >> temp_r_setup.R
echo. >> temp_r_setup.R
echo cat^("Package installation completed!\n"^) >> temp_r_setup.R
echo cat^("Testing data_analyzer.R script...\n"^) >> temp_r_setup.R
echo. >> temp_r_setup.R
echo # Test if our script works >> temp_r_setup.R
echo if ^(file.exists^("data_analyzer.R"^)^) { >> temp_r_setup.R
echo   cat^("Found data_analyzer.R script\n"^) >> temp_r_setup.R
echo } else { >> temp_r_setup.R
echo   cat^("WARNING: data_analyzer.R script not found in current directory\n"^) >> temp_r_setup.R
echo } >> temp_r_setup.R

REM Run the R installation script
Rscript temp_r_setup.R

REM Clean up temporary file
del temp_r_setup.R

echo.
echo ==========================================
echo       R Environment Setup Complete
echo ==========================================
echo.
echo Your R environment is now configured for:
echo ✅ Statistical Analysis
echo ✅ Machine Learning
echo ✅ Time Series Analysis
echo ✅ Survival Analysis
echo ✅ Advanced Clustering
echo ✅ Correlation Analysis
echo.
echo Next steps:
echo 1. Test the R integration: run_r_server.bat
echo 2. Start the Python+R server for full analytics
echo.

pause
