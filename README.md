# Data Analysis Web Application 📊

A comprehensive web-based data analysis application built with **Java Spring Boot**, designed to provide statistical analysis and data visualization capabilities through an intuitive web interface.

## 🚀 Features

- **File Upload**: Upload CSV files for instant analysis
- **Statistical Analysis**: Comprehensive statistics including mean, median, standard deviation, and more
- **Data Visualization**: Interactive charts and histograms using Chart.js
- **Sample Data**: Built-in sample dataset for testing and demonstration
- **Responsive Design**: Modern, mobile-friendly interface using Bootstrap 5
- **Real-time Processing**: Instant analysis with loading indicators
- **Cross-platform Support**: Foundation ready for mobile and desktop app expansion

## 🛠️ Tech Stack

### Backend
- **Java 8** - Programming language
- **Spring Boot 2.7.18** - Web framework
- **Spring Web** - RESTful web services
- **Thymeleaf** - Template engine
- **Apache Commons Math** - Statistical calculations
- **Apache Commons CSV** - CSV file processing
- **Maven** - Dependency management

### Frontend
- **HTML5 & CSS3** - Markup and styling
- **Bootstrap 5** - Responsive UI framework
- **JavaScript ES6** - Client-side functionality
- **Chart.js** - Data visualization
- **Font Awesome** - Icons

## 📁 Project Structure

```
data-analysis-app/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/dataanalysis/app/
│   │   │       ├── DataAnalysisApplication.java    # Main application class
│   │   │       ├── controller/
│   │   │       │   └── DataAnalysisController.java # Web controller
│   │   │       └── service/
│   │   │           └── DataAnalysisService.java    # Business logic
│   │   └── resources/
│   │       ├── application.properties              # Configuration
│   │       ├── static/
│   │       │   ├── css/
│   │       │   │   └── style.css                  # Custom styles
│   │       │   └── js/
│   │       │       └── app.js                     # JavaScript functionality
│   │       └── templates/
│   │           ├── index.html                     # Home page
│   │           ├── upload.html                    # Upload page
│   │           └── results.html                   # Results page
│   └── test/
└── pom.xml                                        # Maven configuration
```

## 🔧 Prerequisites

- **Java 8 or higher** ✅ (You have Java 8 installed)
- **Maven 3.6+** (for building the project)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## ⚡ Quick Start

### Option 1: Using Maven (Recommended)

1. **Install Maven** (if not already installed):
   - Download from [Maven Official Site](https://maven.apache.org/download.cgi)
   - Add Maven to your system PATH

2. **Build and run the application**:
   ```bash
   cd data-analysis-app
   mvn clean compile
   mvn spring-boot:run
   ```

3. **Access the application**:
   - Open your browser and navigate to: `http://localhost:8080`

### Option 2: Using IDE (IntelliJ IDEA / Eclipse)

1. **Import the project**:
   - Open your IDE
   - Import as Maven project
   - Point to the `data-analysis-app` folder

2. **Run the application**:
   - Navigate to `DataAnalysisApplication.java`
   - Right-click and select "Run"

3. **Access the application**:
   - Open your browser and navigate to: `http://localhost:8080`

## 📊 How to Use

### 1. Home Page
- Welcome screen with feature overview
- Navigation to upload or sample data sections

### 2. Upload Data
- Click "Upload Data" in the navigation
- Select a CSV file from your computer
- Drag and drop support available
- File format requirements:
  - CSV format (.csv)
  - First row should contain column headers
  - Numeric columns for statistical analysis

### 3. View Results
- Statistical summary for numeric columns
- Interactive histograms and charts
- Data preview table
- Export and download options

### 4. Sample Data
- Test the application with built-in sample data
- Employee dataset with sales and demographic information

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/upload` | Upload page |
| POST | `/upload` | Process uploaded file |
| GET | `/sample-data` | Generate sample analysis |
| GET | `/api/analysis/{type}` | Get analysis by type (JSON) |

## 🔍 Features in Detail

### Statistical Analysis
- **Descriptive Statistics**: Count, Mean, Median, Min, Max
- **Variability Measures**: Standard deviation, Variance
- **Data Quality**: Missing value detection
- **Distribution Analysis**: Histograms and frequency distributions

### Data Visualization
- **Interactive Charts**: Built with Chart.js
- **Histogram Generation**: Automatic binning for numeric data
- **Responsive Design**: Charts adapt to screen size
- **Color Coding**: Consistent color scheme throughout

### File Processing
- **CSV Parser**: Robust CSV file handling
- **Large File Support**: Up to 10MB file size
- **Error Handling**: Comprehensive error messages
- **Data Validation**: Input validation and sanitization

## 🎨 Customization

### Styling
- Modify `src/main/resources/static/css/style.css` for custom styles
- Bootstrap 5 classes available throughout
- CSS custom properties for consistent theming

### Configuration
- Update `src/main/resources/application.properties` for:
  - Server port configuration
  - File upload limits
  - Logging levels
  - Cache settings

### Adding New Analysis Types
1. Extend `DataAnalysisService.java`
2. Add new methods for specific analysis
3. Update the controller endpoints
4. Create corresponding frontend components

## 🚀 Future Enhancements (Mobile & Desktop Apps)

### Phase 2: Mobile App
- **Technology**: React Native or Flutter
- **API Integration**: RESTful services from Spring Boot backend
- **Features**: 
  - Camera integration for document scanning
  - Offline analysis capabilities
  - Push notifications for analysis completion

### Phase 3: Desktop App
- **Technology**: JavaFX or Electron
- **Features**:
  - Advanced data manipulation tools
  - Local file system integration
  - Export to multiple formats (PDF, Excel, etc.)
  - Batch processing capabilities

## 🐛 Troubleshooting

### Common Issues

1. **Port 8080 already in use**:
   ```bash
   # Change port in application.properties
   server.port=8081
   ```

2. **File upload fails**:
   - Check file size (max 10MB)
   - Ensure CSV format
   - Verify first row contains headers

3. **Maven build errors**:
   - Verify Java version: `java -version`
   - Update Maven: `mvn -version`
   - Clear Maven cache: `mvn clean`

4. **Browser compatibility**:
   - Use modern browsers (Chrome 80+, Firefox 75+, Safari 13+)
   - Enable JavaScript
   - Clear browser cache

## 📈 Performance Tips

- **File Size**: Keep CSV files under 10MB for optimal performance
- **Data Quality**: Clean data provides better analysis results
- **Browser**: Use Chrome or Firefox for best Chart.js performance
- **Memory**: Ensure sufficient RAM for large datasets

## 🤝 Contributing

This is a foundation project ready for expansion. Areas for contribution:
- Additional statistical methods
- More chart types
- Database integration
- User authentication
- Data export features

## 📄 License

This project is created for educational and development purposes. Feel free to modify and extend as needed.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check application logs for detailed error messages
4. Ensure CSV files follow the specified format

---

**Built with ❤️ using Java Spring Boot**

Ready to analyze your data! 🎉
