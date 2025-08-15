@echo off
echo =======================================
echo    R Analytics Environment Setup
echo =======================================
echo.

echo Checking R installation...
R --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: R is not installed or not in PATH
    echo.
    echo Please install R from: https://cran.r-project.org/bin/windows/base/
    echo.
    echo Installation steps:
    echo 1. Download R for Windows from the CRAN website
    echo 2. Run the installer as Administrator
    echo 3. Add R to your system PATH (usually C:\Program Files\R\R-x.x.x\bin)
    echo 4. Restart command prompt and run this script again
    echo.
    pause
    exit /b 1
)

echo R is installed! Checking version...
R --version

echo.
echo Installing required R packages...
echo This may take several minutes depending on your internet connection...
echo.

R -e "
# Set CRAN mirror
options(repos = c(CRAN = 'https://cloud.r-project.org/'))

# List of required packages
required_packages <- c(
  'jsonlite', 'dplyr', 'ggplot2', 'plotly', 'corrplot',
  'VIM', 'mice', 'car', 'lmtest', 'tseries', 'forecast',
  'cluster', 'factoextra', 'psych', 'GPArotation', 'MASS',
  'survival', 'randomForest', 'rpart', 'e1071', 'caret',
  'nortest', 'moments', 'changepoint', 'pracma'
)

# Function to install packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat('Installing package:', pkg, '\n')
      install.packages(pkg, dependencies = TRUE, quiet = FALSE)
    } else {
      cat('Package', pkg, 'already installed\n')
    }
  }
}

# Install packages
cat('Installing R packages for analytics...\n')
install_if_missing(required_packages)

# Test installations
cat('\nTesting package installations...\n')
failed_packages <- c()
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    failed_packages <- c(failed_packages, pkg)
    cat('FAILED:', pkg, '\n')
  } else {
    cat('SUCCESS:', pkg, '\n')
  }
}

if (length(failed_packages) > 0) {
  cat('\nSome packages failed to install:\n')
  cat(paste(failed_packages, collapse = ', '), '\n')
  cat('Please install them manually using: install.packages(c(\"', paste(failed_packages, collapse = '\", \"'), '\"))\n')
  quit(status = 1)
} else {
  cat('\nAll packages installed successfully!\n')
  cat('R Analytics environment is ready!\n')
}
"

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed
    echo Please check your internet connection and R installation
    pause
    exit /b 1
)

echo.
echo ========================================
echo   R Analytics Setup Complete! 
echo ========================================
echo.
echo R Analytics engine is ready to use!
echo.
echo To test the R analytics:
echo 1. Run: Rscript data_analyzer.R sample-data.csv basic
echo 2. Or run: Rscript data_analyzer.R sample-data.csv comprehensive
echo.
echo The R engine will be available at the command line and
echo through the Python Flask API integration.
echo.
pause
