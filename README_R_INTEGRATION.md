# 📊 Advanced Data Analysis Web Application

A comprehensive web-based data analysis application built with **Java Spring Boot**, **Python**, and **R**, designed to provide enterprise-level statistical analysis and machine learning capabilities through an intuitive web interface.

## ✨ What's New in v2.0

🎉 **Major Update**: Full R Analytics Integration!

- 🔬 **Dual Analytics Engines**: Python for ML and data processing, R for advanced statistics
- 📈 **30+ Statistical Methods**: From basic stats to survival analysis and time series forecasting
- 🤝 **Combined Analysis**: Leverage both Python and R in a single workflow
- 🌐 **RESTful API**: 25+ endpoints for comprehensive data analysis
- 📊 **Advanced Visualizations**: Statistical plots, survival curves, time series decompositions

## 🚀 Quick Start

### Option 1: Python-Only Analytics
1. **Clone the repository**
2. **Set up Python environment**: Run `python-analytics/setup_python_env.bat`
3. **Start Python API**: Run `start_python_api.bat`
4. **Run the Java application**: Run `run.bat`
5. **Access the web interface**: Open http://localhost:8080

### Option 2: Full Python + R Analytics (Recommended)
1. **Clone the repository**
2. **Set up R environment**: Run `r-analytics/setup_r_complete.bat`
3. **Start integrated server**: Run `run_r_server.bat`
4. **Run the Java application**: Run `run.bat`
5. **Access the web interface**: Open http://localhost:8080

## 📊 Features

### Java Spring Boot Backend
- RESTful API endpoints for data analysis requests
- File upload handling for CSV data
- Integration with Python + R analytics engines
- Responsive Bootstrap-based web interface
- Advanced statistical analysis integration

### Python Analytics Engine 🐍
- **Statistical Analysis**: Comprehensive descriptive statistics, correlation analysis
- **Machine Learning**: Classification and regression with multiple algorithms (Random Forest, SVM, etc.)
- **Data Quality Assessment**: Missing value analysis, outlier detection, data profiling
- **Advanced Analytics**: 
  - Outlier detection using IQR and Z-score methods
  - Statistical tests (Shapiro-Wilk, Kolmogorov-Smirnov)
  - Time series analysis with trend decomposition
  - Clustering analysis with multiple algorithms
- **Visualization**: Automated chart generation (histograms, scatter plots, correlation heatmaps)
- **Flask API**: RESTful endpoints for seamless Java integration

### R Analytics Engine 📈
- **Advanced Statistics**: Comprehensive statistical modeling and hypothesis testing
- **Regression Analysis**: Linear, logistic, polynomial, and robust regression models
- **Survival Analysis**: Kaplan-Meier estimation, Cox proportional hazards models
- **Time Series**: ARIMA modeling, seasonal decomposition, forecasting
- **Advanced Clustering**: Hierarchical, k-means, and model-based clustering
- **Statistical Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, and more
- **Correlation Analysis**: Pearson, Spearman, and partial correlations
- **JSON Integration**: All R analyses return structured JSON for web integration

## 🛠️ Technology Stack

- **Backend**: Java 17, Spring Boot 3.0
- **Python Analytics**: Python 3.8+, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
- **R Analytics**: R 4.0+, tidyverse, survival, forecast, cluster, randomForest
- **API Integration**: Flask for Python-Java bridge, subprocess for R execution
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Build**: Maven
- **Development**: Compatible with IntelliJ IDEA, VS Code

## 📁 Project Structure

```
data-analysis-app/
├── src/main/java/           # Java Spring Boot application
├── src/main/resources/      # Configuration and static files
├── python-analytics/        # Python analysis engine
│   ├── data_analyzer.py     # Core Python analysis functionality
│   ├── python_api_server.py # Flask API server with R integration
│   ├── requirements.txt     # Python dependencies
│   └── setup_python_env.bat # Python environment setup
├── r-analytics/            # R analysis engine
│   ├── data_analyzer.R      # Comprehensive R analysis script
│   ├── setup_r_env.bat      # Basic R package installation
│   └── setup_r_complete.bat # Full R environment setup
├── temp_uploads/           # Temporary file storage
├── visualizations/         # Generated charts and plots
├── test_r_integration.py   # R integration testing script
├── run_r_server.bat        # Start Python+R integrated server
└── *.bat                   # Windows batch scripts
```

## 🔧 Setup Instructions

### Prerequisites
- Java 17 or higher
- Python 3.8 or higher
- R 4.0 or higher (for R analytics)
- Maven 3.6 or higher

### Installation

#### Basic Setup (Python Only)
1. **Java Setup**: Ensure Java 17+ is installed and JAVA_HOME is set
2. **Python Setup**: Install Python 3.8+ and ensure it's in your PATH
3. **Maven Setup**: Install Maven 3.6+ for building the Java application
4. **Python Dependencies**: Run `python-analytics/setup_python_env.bat`

#### Full Setup (Python + R - Recommended)
1. **Java & Python**: Follow steps 1-3 above
2. **R Installation**: Download and install R from https://cran.r-project.org/
   - **Important**: Check "Add R to system PATH" during installation
3. **R Environment**: Run `r-analytics/setup_r_complete.bat`
4. **Test Integration**: Run `python test_r_integration.py`

### Configuration

The application uses default ports:
- Java Spring Boot: `8080`
- Python Flask API: `5000`

To modify ports, edit:
- `src/main/resources/application.properties` (Java)
- `python-analytics/python_api_server.py` (Python)

## 🚀 Usage

### Web Interface
1. Upload CSV files through the web interface
2. Select analysis type (Python, R, or Combined)
3. Configure analysis parameters
4. View results and download visualizations

### API Endpoints

#### Python Analytics API (Port 5000)
**Python-based Analysis:**
- `POST /api/analyze/upload` - Upload and basic analysis
- `POST /api/analyze/correlation` - Correlation analysis
- `POST /api/analyze/machine_learning` - ML analysis (requires target column)
- `POST /api/analyze/clustering` - Clustering analysis
- `POST /api/analyze/outliers` - Outlier detection
- `POST /api/analyze/quality` - Data quality assessment
- `POST /api/analyze/statistical_tests` - Statistical hypothesis testing
- `POST /api/analyze/timeseries` - Time series analysis
- `POST /api/analyze/comprehensive` - Complete Python analysis suite

**R-based Analysis:**
- `POST /api/r/analyze/basic` - R advanced statistics
- `POST /api/r/analyze/correlation` - R correlation analysis with significance tests
- `POST /api/r/analyze/tests` - R comprehensive statistical tests
- `POST /api/r/analyze/regression` - R advanced regression modeling
- `POST /api/r/analyze/survival` - R survival analysis
- `POST /api/r/analyze/timeseries` - R time series with ARIMA
- `POST /api/r/analyze/clustering` - R advanced clustering
- `POST /api/r/analyze/comprehensive` - Complete R analysis suite

**Combined Analysis:**
- `POST /api/combined/comprehensive` - Python + R combined analysis

**Utility Endpoints:**
- `GET /api/r/health` - Check R installation status
- `GET /api/models/available` - Available models and analysis types
- `GET /api/sample/generate` - Generate sample data for testing

#### Java Spring Boot API (Port 8080)
- `GET /` - Web interface homepage
- `POST /api/analyze` - Main analysis endpoint (forwards to Python/R)
- `POST /api/upload` - File upload endpoint

## 🧪 Testing

### R Integration Testing
```bash
# Test R integration components
python test_r_integration.py
```

### Python Analytics Testing
```bash
cd python-analytics
call venv\Scripts\activate.bat
python -c "from data_analyzer import DataAnalyzer; print('Python analytics ready!')"
```

### R Analytics Testing (after R installation)
```bash
# Test R script directly
cd r-analytics
Rscript data_analyzer.R sample_data.csv basic
```

### API Testing
```bash
# Test Python analysis
curl -X GET http://localhost:5000/api/sample/generate

# Test R health check
curl -X GET http://localhost:5000/api/r/health

# Test combined analysis
curl -X POST -F "file=@your_data.csv" http://localhost:5000/api/combined/comprehensive
```

## 📊 Analysis Capabilities

### Python Analytics Engine 🐍

**Statistical Analysis**
- Descriptive statistics (mean, median, std, quartiles)
- Distribution analysis and normality testing
- Correlation analysis (Pearson, Spearman)
- Hypothesis testing (t-tests, chi-square)

**Machine Learning**
- **Classification**: Random Forest, Logistic Regression, SVM, Gradient Boosting
- **Regression**: Linear, Ridge, Lasso, Random Forest Regression
- **Clustering**: K-Means, Hierarchical, DBSCAN
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Data Quality Assessment**
- Missing value analysis and imputation suggestions
- Outlier detection using multiple methods (IQR, Z-score)
- Data type inference and validation
- Duplicate detection and handling

### R Analytics Engine 📊

**Advanced Statistical Analysis**
- Comprehensive descriptive statistics with confidence intervals
- Advanced normality tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)
- Robust statistical measures and trimmed statistics
- Distribution fitting and goodness-of-fit tests

**Advanced Regression Modeling**
- Linear, logistic, and polynomial regression
- Robust regression methods
- Model diagnostics and residual analysis
- Variable selection and model comparison

**Survival Analysis**
- Kaplan-Meier survival curves
- Cox proportional hazards models
- Log-rank tests for group comparisons
- Hazard ratio estimation

**Time Series Analysis**
- ARIMA model fitting and forecasting
- Seasonal decomposition (STL, classical)
- Stationarity testing (ADF, KPSS)
- ACF/PACF analysis and model selection

**Advanced Clustering**
- Hierarchical clustering with various linkage methods
- Model-based clustering (mixture models)
- Cluster validation and optimal cluster selection
- Silhouette analysis and gap statistic

**Enhanced Correlation Analysis**
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Partial and semi-partial correlations
- Correlation significance testing
- Correlation network analysis

### Combined Analytics (Python + R) 🤝
- Seamless integration of both engines
- Comprehensive analysis using best-of-breed methods
- Cross-validation between Python and R results
- Unified JSON output for web integration

### Visualization
- Distribution plots (histograms, box plots, Q-Q plots)
- Advanced correlation heatmaps and networks
- Survival curves and hazard plots
- Time series decomposition plots
- Cluster dendrograms and validation plots
- Regression diagnostic plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional R statistical methods
- Enhanced visualizations
- Performance optimizations
- Additional file format support
- Advanced ML algorithms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spring Boot for the robust backend framework
- Python ecosystem (pandas, scikit-learn, matplotlib) for ML and analytics
- R ecosystem (tidyverse, survival, forecast) for advanced statistics
- Flask for seamless Python-Java integration
- Bootstrap for the responsive web interface

---

**Built with ❤️ combining the power of Java, Python, and R for comprehensive data analysis**

🔬 Ready for enterprise-level analytics! 🎉
