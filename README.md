# 🚀 Advanced Multi-Engine Data Analysis Platform

A comprehensive enterprise-grade data analysis platform integrating **Java Spring Boot**, **Python**, and **R** to deliver powerful statistical analysis, machine learning capabilities, and interactive visualizations through an intuitive web interface.

![Data Analysis Platform](https://img.shields.io/badge/Platform-Data%20Analysis-blue)
![Languages](https://img.shields.io/badge/Languages-Java%20%7C%20Python%20%7C%20R-green)
![Version](https://img.shields.io/badge/Version-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ✨ Key Features

- 🔄 **Multi-Engine Architecture**: Seamlessly combine Java, Python, and R for comprehensive analytics
- 🧠 **Advanced Machine Learning**: Classification, regression, clustering using scikit-learn algorithms
- 📊 **Statistical Excellence**: 30+ statistical methods from basic stats to survival analysis and time series
- 📈 **Interactive Visualizations**: Automated chart generation with customizable parameters
- 🔍 **Data Quality Tools**: Missing value analysis, outlier detection, and data validation
- 🌐 **RESTful API**: 25+ endpoints for comprehensive data analysis services
- 📱 **Responsive Interface**: Modern, mobile-friendly web interface using Bootstrap 5
- 📄 **Comprehensive Reports**: Markdown and JSON reports with visualizations and interpretation

## 🏗️ Architecture

The platform employs a modular, microservices-inspired architecture:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│                     │     │                      │     │                    │
│  Web Interface      │     │  Java Spring Boot    │     │  Python Analytics  │
│  (Bootstrap 5 UI)   │◄───►│  (RESTful Backend)   │◄───►│  (ML & Data Proc.) │
│                     │     │                      │     │                    │
└─────────────────────┘     └──────────────────────┘     └────────┬───────────┘
                                                                   │
                                                                   ▼
                                                         ┌────────────────────┐
                                                         │                    │
                                                         │   R Analytics      │
                                                         │   (Adv. Statistics)│
                                                         │                    │
                                                         └────────────────────┘
```

- **Frontend Layer**: Bootstrap 5 + JavaScript for responsive user experience
- **Application Layer**: Java Spring Boot with RESTful API endpoints
- **Analytics Layer**: 
  - Python engine for ML, data processing, and visualization
  - R engine for advanced statistical analysis and specialized methods
- **Integration Layer**: Flask API server with R subprocess integration

## 🚀 Quick Start

### Prerequisites

- Java 8+ (JDK or JRE)
- Python 3.8+
- R 4.0+ (for R analytics features)
- Maven 3.6+

### Option 1: Python-Only Analytics

1. **Clone the repository**
2. **Set up Python environment**:
   ```bash
   cd python-analytics
   setup_python_env.bat
   ```
3. **Start Python API**:
   ```bash
   start_python_server.bat
   ```
4. **Run the Java application**:
   ```bash
   run-app.bat
   ```
5. **Access the application**: Open [http://localhost:8080](http://localhost:8080)

### Option 2: Full Python + R Analytics (Recommended)

1. **Clone the repository**
2. **Set up R environment**:
   ```bash
   cd r-analytics
   setup_r_complete.bat
   ```
3. **Start integrated server**:
   ```bash
   run_r_server.bat
   ```
4. **Run the Java application**:
   ```bash
   run-app.bat
   ```
5. **Access the application**: Open [http://localhost:8080](http://localhost:8080)

## 📊 Analytics Capabilities

### Python Analytics Engine 🐍

- **Statistical Analysis**
  - Descriptive statistics (mean, median, std, quartiles)
  - Distribution analysis and normality testing
  - Correlation analysis (Pearson, Spearman)
  - Hypothesis testing (t-tests, chi-square)

- **Machine Learning**
  - **Classification**: Random Forest, Logistic Regression, SVM, Gradient Boosting
  - **Regression**: Linear, Ridge, Lasso, Random Forest Regression
  - **Clustering**: K-Means, Hierarchical, DBSCAN
  - **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

- **Data Quality Assessment**
  - Missing value analysis and imputation suggestions
  - Outlier detection using multiple methods (IQR, Z-score)
  - Data type inference and validation
  - Duplicate detection and handling

### R Analytics Engine 📈

- **Advanced Statistical Analysis**
  - Comprehensive descriptive statistics with confidence intervals
  - Advanced normality tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)
  - Robust statistical measures and trimmed statistics
  - Distribution fitting and goodness-of-fit tests

- **Advanced Regression Modeling**
  - Linear, logistic, and polynomial regression
  - Robust regression methods
  - Model diagnostics and residual analysis
  - Variable selection and model comparison

- **Specialized Analyses**
  - **Survival Analysis**: Kaplan-Meier survival curves, Cox proportional hazards models
  - **Time Series Analysis**: ARIMA modeling, seasonal decomposition, forecasting
  - **Advanced Clustering**: Hierarchical clustering, model-based clustering
  - **Correlation Analysis**: Partial correlations, correlation significance testing

## 📁 Project Structure

```
data-analysis-app/
├── src/main/java/           # Java Spring Boot application
├── src/main/resources/      # Configuration and static files
├── python-analytics/        # Python analysis engine
│   ├── data_analyzer.py     # Core Python analysis functionality
│   ├── python_api_server.py # Flask API server with R integration
│   ├── report_generator.py  # Report generation functionality
│   ├── requirements.txt     # Python dependencies
│   └── setup_python_env.bat # Python environment setup
├── r-analytics/             # R analysis engine
│   ├── data_analyzer.R      # Comprehensive R analysis script
│   ├── setup_r_env.bat      # Basic R package installation
│   └── setup_r_complete.bat # Full R environment setup
├── temp_uploads/            # Temporary file storage
├── visualizations/          # Generated charts and plots
├── sample-data.csv          # Sample dataset for testing
├── test_r_integration.py    # R integration testing script
├── run_r_server.bat         # Start Python+R integrated server
├── run-app.bat              # Start the Java application
└── *.md                     # Documentation files
```

## 🌐 API Endpoints

### Python Analytics API (Port 5000)

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
- `POST /api/analyze/visualizations` - Generate visualizations

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

**Report Generation:**
- `POST /api/reports/generate` - Generate comprehensive markdown report
- `POST /api/reports/combined` - Generate Python+R combined report
- `GET /api/reports/list` - List all generated reports
- `GET /api/reports/download/<id>/<type>` - Download report files

**Utility Endpoints:**
- `GET /api/r/health` - Check R installation status
- `GET /api/models/available` - Available models and analysis types
- `GET /api/sample/generate` - Generate sample data for testing
- `GET /health` - API health check

### Java Spring Boot API (Port 8080)
- `GET /` - Web interface homepage
- `POST /api/analyze` - Main analysis endpoint (forwards to Python/R)
- `POST /api/upload` - File upload endpoint

## 🛠️ Technology Stack

### Backend
- **Java 8+** - Core application platform
- **Spring Boot 2.7.18** - Web framework
- **Spring Web** - RESTful web services
- **Thymeleaf** - Template engine
- **Maven** - Dependency management

### Python Analytics
- **Python 3.8+** - Programming language
- **pandas 2.1.4** - Data manipulation
- **scikit-learn 1.3.2** - Machine learning
- **matplotlib 3.8.2** - Visualization
- **Flask 3.0.0** - API server
- **numpy 1.26.2** - Numerical computing
- **scipy 1.11.4** - Scientific computing

### R Analytics
- **R 4.0+** - Statistical computing
- **dplyr** - Data manipulation
- **ggplot2** - Visualization
- **survival** - Survival analysis
- **forecast** - Time series forecasting
- **cluster** - Clustering algorithms
- **lmtest** - Regression testing

### Frontend
- **HTML5 & CSS3** - Markup and styling
- **Bootstrap 5** - Responsive UI framework
- **JavaScript ES6** - Client-side functionality
- **Chart.js** - Data visualization

## 📊 How to Use

### 1. Upload Data
- Select a CSV file from your computer
- File requirements:
  - CSV format (.csv)
  - First row should contain column headers
  - Supported data types: numeric, string, date

### 2. Choose Analysis Type
- **Python Analysis**: For machine learning and basic statistics
- **R Analysis**: For advanced statistical methods
- **Combined Analysis**: For comprehensive analysis using both engines

### 3. Configure Analysis Parameters
- Select target variable for predictive modeling
- Choose number of clusters for clustering analysis
- Specify date and value columns for time series analysis

### 4. View Results
- Interactive visualizations
- Statistical summaries
- Model performance metrics
- Data quality assessments

### 5. Generate Reports
- Download comprehensive markdown reports
- Export visualizations and tables
- Access raw JSON data for further processing

## 🧪 Testing and Validation

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

### R Analytics Testing
```bash
# Test R script directly
cd r-analytics
Rscript data_analyzer.R sample-data.csv basic
```

## 🔍 Features in Detail

### Statistical Analysis
- **Descriptive Statistics**: Count, Mean, Median, Min, Max
- **Variability Measures**: Standard deviation, Variance, IQR
- **Distribution Analysis**: Skewness, Kurtosis, Normality tests
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations

### Machine Learning
- **Supervised Learning**: Classification and regression models
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Model Evaluation**: Cross-validation, confusion matrices, ROC curves
- **Feature Importance**: Variable significance analysis

### Data Quality Assessment
- **Missing Data Analysis**: Detection and visualization of missing values
- **Outlier Detection**: Multiple methods (IQR, Z-score, modified Z-score)
- **Data Validation**: Type checking, range validation, consistency checks
- **Recommendation Engine**: Automated data cleaning suggestions

### Time Series Analysis
- **Trend Analysis**: Linear trend detection and significance testing
- **Seasonality Detection**: Seasonal pattern identification
- **Decomposition**: Trend, seasonal, and residual components
- **Forecasting**: ARIMA models and predictions

### Specialized Analyses
- **Survival Analysis**: Time-to-event modeling and hazard estimation
- **Clustering**: Segment discovery and cluster validation
- **Statistical Tests**: Hypothesis testing for various data scenarios
- **Advanced Regression**: Multiple model types with diagnostics

## 🔧 Configuration

### Environment Variables
- `JAVA_HOME` - Path to Java installation
- `PYTHON_PATH` - Path to Python executable
- `R_HOME` - Path to R installation

### Application Properties
- `server.port` - Java application port (default: 8080)
- `python.api.url` - Python API URL (default: http://localhost:5000)
- `file.upload.max-size` - Maximum upload file size (default: 10MB)

### Python Server Configuration
- Edit `python-analytics/python_api_server.py` to modify:
  - API port (default: 5000)
  - Upload directory
  - Visualization settings

### R Integration Configuration
- Edit `r-analytics/data_analyzer.R` to modify:
  - R packages to use
  - Statistical method parameters
  - Output formatting

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- Additional statistical methods
- Enhanced visualizations
- Performance optimizations
- Additional file format support
- Advanced ML algorithms
- Improved UI/UX features

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Spring Boot for the robust backend framework
- Python ecosystem (pandas, scikit-learn, matplotlib) for ML and analytics
- R ecosystem (tidyverse, survival, forecast) for advanced statistics
- Flask for seamless Python-Java integration
- Bootstrap for the responsive web interface

---

**Built with ❤️ combining the power of Java, Python, and R for comprehensive data analysis ~ KHIMLabs 👌**

🔬 Ready for enterprise-level analytics! 🎉
