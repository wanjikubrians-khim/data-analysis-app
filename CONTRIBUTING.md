# Contributing to Data Analysis App

Thank you for your interest in contributing to the Data Analysis App! 🎉

## 🚀 Getting Started

### Prerequisites
- Java 8 or higher
- Maven 3.6+
- NetBeans IDE (recommended) or any Java IDE
- Git for version control

### Setting up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/wanjikubrians-khim/data-analysis-app.git
   cd data-analysis-app
   ```

2. **Open in NetBeans**
   - File → Open Project
   - Select the `data-analysis-app` folder
   - NetBeans will automatically detect it as a Maven project

3. **Run the application**
   - Right-click on `DataAnalysisApplication.java`
   - Select "Run File" or press `Shift+F6`
   - Access at `http://localhost:8080`

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   - Run the application locally
   - Test all existing functionality
   - Upload sample CSV files to verify analysis works

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Areas for Contribution

### High Priority
- [ ] **Database Integration** - Add database support for storing analysis history
- [ ] **User Authentication** - Implement user accounts and session management
- [ ] **Export Features** - Add PDF, Excel export functionality
- [ ] **Advanced Statistics** - Correlation analysis, regression, etc.
- [ ] **More Chart Types** - Scatter plots, box plots, line charts

### Medium Priority
- [ ] **File Format Support** - Excel, JSON, XML file processing
- [ ] **Batch Processing** - Multiple file uploads
- [ ] **Data Cleaning** - Handle missing values, outliers
- [ ] **API Documentation** - Swagger/OpenAPI integration
- [ ] **Performance Optimization** - Large file handling

### Future Enhancements
- [ ] **Mobile App Integration** - REST API for mobile clients
- [ ] **Real-time Analysis** - WebSocket support for live data
- [ ] **Machine Learning** - Predictive analytics features
- [ ] **Collaborative Features** - Share analysis results
- [ ] **Dashboard Creation** - Custom analytics dashboards

## 🎨 Code Style Guidelines

### Java Code Style
- Use 4 spaces for indentation
- Follow Java naming conventions (camelCase for variables, PascalCase for classes)
- Add JavaDoc comments for public methods
- Keep methods focused and under 50 lines when possible

### Frontend Code Style
- Use 2 spaces for HTML/CSS/JavaScript indentation
- Follow Bootstrap conventions for styling
- Use meaningful CSS class names
- Keep JavaScript functions small and focused

### Commit Message Format
```
Type: Brief description

Detailed explanation if needed

- Add new statistical method
- Update documentation
- Fix bug in CSV parser
```

**Types:** Add, Update, Fix, Remove, Refactor, Test, Doc

## 🧪 Testing Guidelines

### Manual Testing Checklist
- [ ] Home page loads correctly
- [ ] File upload works with sample CSV
- [ ] Statistical analysis displays correctly
- [ ] Charts render properly
- [ ] Sample data functionality works
- [ ] Responsive design on mobile devices

### Future Automated Testing
- Unit tests for service layer methods
- Integration tests for REST endpoints
- Frontend testing with Selenium

## 📁 Project Structure

```
src/main/
├── java/com/dataanalysis/app/
│   ├── DataAnalysisApplication.java    # Main Spring Boot application
│   ├── controller/                     # REST controllers
│   └── service/                        # Business logic
├── resources/
│   ├── static/                         # CSS, JS, images
│   ├── templates/                      # Thymeleaf HTML templates
│   └── application.properties          # Configuration
```

## 🐛 Reporting Issues

When reporting bugs, please include:
1. **Description** - What happened vs. what you expected
2. **Steps to Reproduce** - Detailed steps to recreate the issue
3. **Environment** - OS, Java version, browser
4. **Files** - Sample CSV that caused the issue (if applicable)
5. **Screenshots** - Visual evidence of the problem

## 💡 Feature Requests

For new features, please provide:
1. **Use Case** - Why is this feature needed?
2. **Description** - What should the feature do?
3. **Mockups** - Visual representation if applicable
4. **Priority** - How important is this feature?

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Special thanks in documentation

## 📞 Getting Help

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Code Questions:** Comment on specific lines in Pull Requests

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Happy Contributing! 🚀**

Let's build an amazing data analysis platform together!
