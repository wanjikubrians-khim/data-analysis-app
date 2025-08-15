#!/usr/bin/env python3
"""
Python Analysis API Server
Flask-based API server for advanced data analysis integration with Java Spring Boot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import tempfile
import uuid
import subprocess
from pathlib import Path
import logging
from data_analyzer import DataAnalyzer
from report_generator import AnalysisReportGenerator, generate_complete_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Java Spring Boot integration

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
VISUALIZATION_FOLDER = 'visualizations'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(VISUALIZATION_FOLDER).mkdir(exist_ok=True)

# Global analyzer instance
analyzer = DataAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Python Data Analysis API',
        'version': '1.0.0'
    })

@app.route('/api/analyze/upload', methods=['POST'])
def upload_and_analyze():
    """Upload CSV file and perform basic analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            return jsonify(load_result), 400
        
        # Perform basic analysis
        basic_stats = analyzer.basic_statistics()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'file_info': load_result,
            'analysis': basic_stats,
            'analysis_type': 'basic_statistics'
        })
        
    except Exception as e:
        logger.error(f"Error in upload_and_analyze: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/correlation', methods=['POST'])
def correlation_analysis():
    """Perform correlation analysis on uploaded data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform correlation analysis
        analyzer.load_data(file_path)
        correlation_result = analyzer.correlation_analysis()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'correlation',
            'result': correlation_result
        })
        
    except Exception as e:
        logger.error(f"Error in correlation_analysis: {str(e)}")
        return jsonify({'error': f'Correlation analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/machine_learning', methods=['POST'])
def machine_learning_analysis():
    """Perform machine learning analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'Target column is required for ML analysis'}), 400
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform ML analysis
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        ml_result = analyzer.machine_learning_analysis(target_column)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'machine_learning',
            'target_column': target_column,
            'result': ml_result
        })
        
    except Exception as e:
        logger.error(f"Error in machine_learning_analysis: {str(e)}")
        return jsonify({'error': f'ML analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/clustering', methods=['POST'])
def clustering_analysis():
    """Perform clustering analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        n_clusters = int(request.form.get('n_clusters', 3))
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform clustering
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        clustering_result = analyzer.clustering_analysis(n_clusters)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'clustering',
            'n_clusters': n_clusters,
            'result': clustering_result
        })
        
    except Exception as e:
        logger.error(f"Error in clustering_analysis: {str(e)}")
        return jsonify({'error': f'Clustering analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/visualizations', methods=['POST'])
def generate_visualizations():
    """Generate visualizations for uploaded data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Create visualization directory for this analysis
        viz_dir = os.path.join(VISUALIZATION_FOLDER, file_id)
        
        # Load data and generate visualizations
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        viz_result = analyzer.generate_visualizations(viz_dir)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'visualizations',
            'result': viz_result,
            'visualization_id': file_id
        })
        
    except Exception as e:
        logger.error(f"Error in generate_visualizations: {str(e)}")
        return jsonify({'error': f'Visualization generation failed: {str(e)}'}), 500

@app.route('/api/analyze/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """Perform comprehensive analysis including all available methods"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')  # Optional for ML
        n_clusters = int(request.form.get('n_clusters', 3))
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        # Perform all analyses
        results = {
            'file_info': load_result,
            'basic_statistics': analyzer.basic_statistics(),
            'correlation_analysis': analyzer.correlation_analysis(),
            'outlier_detection': analyzer.outlier_detection(),
            'data_quality': analyzer.data_quality_assessment(),
            'statistical_tests': analyzer.statistical_tests(),
            'clustering_analysis': analyzer.clustering_analysis(n_clusters)
        }
        
        # Add ML analysis if target column provided
        if target_column and target_column in analyzer.data.columns:
            results['machine_learning'] = analyzer.machine_learning_analysis(target_column)
        
        # Generate visualizations
        viz_dir = os.path.join(VISUALIZATION_FOLDER, file_id)
        results['visualizations'] = analyzer.generate_visualizations(viz_dir)
        results['visualization_id'] = file_id
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'comprehensive',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive_analysis: {str(e)}")
        return jsonify({'error': f'Comprehensive analysis failed: {str(e)}'}), 500

@app.route('/api/visualizations/<viz_id>/<filename>', methods=['GET'])
def serve_visualization(viz_id, filename):
    """Serve generated visualization files"""
    try:
        viz_path = os.path.join(VISUALIZATION_FOLDER, viz_id, filename)
        if os.path.exists(viz_path):
            from flask import send_file
            return send_file(viz_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        logger.error(f"Error serving visualization: {str(e)}")
        return jsonify({'error': 'Failed to serve visualization'}), 500

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available ML models and analysis types"""
    return jsonify({
        'classification_models': [
            'Random Forest',
            'Logistic Regression',
            'Support Vector Machine',
            'Gradient Boosting'
        ],
        'regression_models': [
            'Random Forest',
            'Linear Regression',
            'Ridge Regression',
            'Lasso Regression'
        ],
        'clustering_algorithms': [
            'K-Means',
            'Hierarchical Clustering',
            'DBSCAN'
        ],
        'analysis_types': [
            'basic_statistics',
            'correlation_analysis',
            'machine_learning',
            'clustering_analysis',
            'outlier_detection',
            'data_quality',
            'statistical_tests',
            'time_series',
            'visualizations',
            'comprehensive'
        ]
    })

@app.route('/api/analyze/outliers', methods=['POST'])
def outlier_detection():
    """Perform comprehensive outlier detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform outlier detection
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        outlier_result = analyzer.outlier_detection()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'outlier_detection',
            'result': outlier_result
        })
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        return jsonify({'error': f'Outlier detection failed: {str(e)}'}), 500

@app.route('/api/analyze/quality', methods=['POST'])
def data_quality_assessment():
    """Perform comprehensive data quality assessment"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform quality assessment
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        quality_result = analyzer.data_quality_assessment()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'data_quality',
            'result': quality_result
        })
        
    except Exception as e:
        logger.error(f"Error in data quality assessment: {str(e)}")
        return jsonify({'error': f'Data quality assessment failed: {str(e)}'}), 500

@app.route('/api/analyze/statistical_tests', methods=['POST'])
def statistical_tests():
    """Perform statistical tests"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform statistical tests
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        tests_result = analyzer.statistical_tests()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'statistical_tests',
            'result': tests_result
        })
        
    except Exception as e:
        logger.error(f"Error in statistical tests: {str(e)}")
        return jsonify({'error': f'Statistical tests failed: {str(e)}'}), 500

@app.route('/api/analyze/timeseries', methods=['POST'])
def time_series_analysis():
    """Perform time series analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        date_column = request.form.get('date_column')
        value_column = request.form.get('value_column')
        
        if not date_column or not value_column:
            return jsonify({'error': 'Date and value columns are required for time series analysis'}), 400
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load data and perform time series analysis
        load_result = analyzer.load_data(file_path)
        if not load_result['success']:
            os.remove(file_path)
            return jsonify(load_result), 400
        
        ts_result = analyzer.time_series_analysis(date_column, value_column)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'time_series',
            'date_column': date_column,
            'value_column': value_column,
            'result': ts_result
        })
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        return jsonify({'error': f'Time series analysis failed: {str(e)}'}), 500

# R Analytics Integration Functions
def check_r_installation():
    """Check if R is installed and accessible"""
    try:
        result = subprocess.run(['R', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def execute_r_script(csv_file_path, analysis_type, *args):
    """Execute R analytics script and return JSON result"""
    try:
        # Get the R script path
        r_script_path = os.path.join('..', 'r-analytics', 'data_analyzer.R')
        
        if not os.path.exists(r_script_path):
            return {'error': 'R analytics script not found'}
        
        # Prepare R command
        cmd = ['Rscript', r_script_path, csv_file_path, analysis_type] + list(args)
        
        # Execute R script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {
                'error': f'R script execution failed: {result.stderr}',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        
        # Parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {
                'error': f'Failed to parse R output as JSON: {str(e)}',
                'raw_output': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {'error': 'R script execution timed out'}
    except Exception as e:
        return {'error': f'R execution error: {str(e)}'}

# R Analytics API Endpoints

@app.route('/api/r/health', methods=['GET'])
def r_health_check():
    """Check R installation and availability"""
    r_available = check_r_installation()
    r_script_path = os.path.join('..', 'r-analytics', 'data_analyzer.R')
    script_exists = os.path.exists(r_script_path)
    
    return jsonify({
        'r_installed': r_available,
        'r_script_available': script_exists,
        'status': 'healthy' if r_available and script_exists else 'unavailable',
        'message': 'R Analytics ready' if r_available and script_exists else 'R Analytics not available'
    })

@app.route('/api/r/analyze/basic', methods=['POST'])
def r_basic_analysis():
    """Perform R-based advanced statistical analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'basic')
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_basic_statistics',
            'engine': 'R',
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R basic analysis: {str(e)}")
        return jsonify({'error': f'R basic analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/correlation', methods=['POST'])
def r_correlation_analysis():
    """Perform R-based advanced correlation analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'correlation')
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_correlation_analysis',
            'engine': 'R',
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R correlation analysis: {str(e)}")
        return jsonify({'error': f'R correlation analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/tests', methods=['POST'])
def r_statistical_tests():
    """Perform R-based comprehensive statistical tests"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'tests')
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_statistical_tests',
            'engine': 'R',
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R statistical tests: {str(e)}")
        return jsonify({'error': f'R statistical tests failed: {str(e)}'}), 500

@app.route('/api/r/analyze/regression', methods=['POST'])
def r_regression_analysis():
    """Perform R-based advanced regression modeling"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        method = request.form.get('method', 'linear')
        
        if not target_column:
            return jsonify({'error': 'Target column is required for regression analysis'}), 400
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'regression', target_column, method)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_regression_analysis',
            'engine': 'R',
            'target_column': target_column,
            'method': method,
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R regression analysis: {str(e)}")
        return jsonify({'error': f'R regression analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/survival', methods=['POST'])
def r_survival_analysis():
    """Perform R-based survival analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        time_column = request.form.get('time_column')
        event_column = request.form.get('event_column')
        group_column = request.form.get('group_column', '')
        
        if not time_column or not event_column:
            return jsonify({'error': 'Time and event columns are required for survival analysis'}), 400
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        args = [time_column, event_column]
        if group_column:
            args.append(group_column)
            
        r_result = execute_r_script(file_path, 'survival', *args)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_survival_analysis',
            'engine': 'R',
            'time_column': time_column,
            'event_column': event_column,
            'group_column': group_column if group_column else None,
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R survival analysis: {str(e)}")
        return jsonify({'error': f'R survival analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/timeseries', methods=['POST'])
def r_time_series_analysis():
    """Perform R-based advanced time series analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        date_column = request.form.get('date_column')
        value_column = request.form.get('value_column')
        
        if not date_column or not value_column:
            return jsonify({'error': 'Date and value columns are required for time series analysis'}), 400
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'timeseries', date_column, value_column)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_time_series_analysis',
            'engine': 'R',
            'date_column': date_column,
            'value_column': value_column,
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R time series analysis: {str(e)}")
        return jsonify({'error': f'R time series analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/clustering', methods=['POST'])
def r_clustering_analysis():
    """Perform R-based advanced clustering analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        n_clusters = request.form.get('n_clusters', '3')
        method = request.form.get('method', 'kmeans')
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R analysis
        r_result = execute_r_script(file_path, 'clustering', n_clusters, method)
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_clustering_analysis',
            'engine': 'R',
            'n_clusters': int(n_clusters),
            'method': method,
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R clustering analysis: {str(e)}")
        return jsonify({'error': f'R clustering analysis failed: {str(e)}'}), 500

@app.route('/api/r/analyze/comprehensive', methods=['POST'])
def r_comprehensive_analysis():
    """Perform comprehensive R-based analysis"""
    try:
        if not check_r_installation():
            return jsonify({'error': 'R is not installed or not accessible'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Execute R comprehensive analysis
        r_result = execute_r_script(file_path, 'comprehensive')
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'analysis_type': 'r_comprehensive_analysis',
            'engine': 'R',
            'result': r_result
        })
        
    except Exception as e:
        logger.error(f"Error in R comprehensive analysis: {str(e)}")
        return jsonify({'error': f'R comprehensive analysis failed: {str(e)}'}), 500

# Combined Python + R Analysis
@app.route('/api/combined/comprehensive', methods=['POST'])
def combined_comprehensive_analysis():
    """Perform comprehensive analysis using both Python and R engines"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')  # Optional for ML
        n_clusters = int(request.form.get('n_clusters', 3))
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        results = {
            'python_analysis': {},
            'r_analysis': {},
            'analysis_type': 'combined_comprehensive'
        }
        
        # Python analysis
        try:
            load_result = analyzer.load_data(file_path)
            if load_result['success']:
                results['python_analysis'] = {
                    'file_info': load_result,
                    'basic_statistics': analyzer.basic_statistics(),
                    'correlation_analysis': analyzer.correlation_analysis(),
                    'outlier_detection': analyzer.outlier_detection(),
                    'data_quality': analyzer.data_quality_assessment(),
                    'statistical_tests': analyzer.statistical_tests(),
                    'clustering_analysis': analyzer.clustering_analysis(n_clusters)
                }
                
                # Add ML analysis if target column provided
                if target_column and target_column in analyzer.data.columns:
                    results['python_analysis']['machine_learning'] = analyzer.machine_learning_analysis(target_column)
                    
        except Exception as e:
            results['python_analysis'] = {'error': f'Python analysis failed: {str(e)}'}
        
        # R analysis
        if check_r_installation():
            try:
                r_result = execute_r_script(file_path, 'comprehensive')
                results['r_analysis'] = r_result
            except Exception as e:
                results['r_analysis'] = {'error': f'R analysis failed: {str(e)}'}
        else:
            results['r_analysis'] = {'error': 'R is not installed or not accessible'}
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in combined comprehensive analysis: {str(e)}")
        return jsonify({'error': f'Combined analysis failed: {str(e)}'}), 500

# Report Generation Endpoints

@app.route('/api/reports/generate', methods=['POST'])
def generate_comprehensive_report():
    """Generate comprehensive analysis report with visualizations and tables"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        include_code = request.form.get('include_code', 'false').lower() == 'true'
        include_raw_results = request.form.get('include_raw_results', 'false').lower() == 'true'
        report_format = request.form.get('format', 'markdown')  # markdown, json, or both
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Generate comprehensive report
            report_info = generate_complete_report(
                file_path, 
                output_dir=f"reports/{file_id}",
                include_code=include_code,
                include_raw_results=include_raw_results
            )
            
            # Clean up uploaded file
            os.remove(file_path)
            
            if 'error' in report_info:
                return jsonify(report_info), 400
            
            # Read the generated markdown report
            with open(report_info['report_path'], 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            response_data = {
                'status': 'success',
                'report_id': file_id,
                'report_info': report_info['summary'],
                'download_links': {
                    'markdown_report': f"/api/reports/download/{file_id}/markdown",
                    'summary_json': f"/api/reports/download/{file_id}/summary"
                }
            }
            
            # Include content based on format request
            if report_format in ['markdown', 'both']:
                response_data['markdown_content'] = markdown_content
            
            if report_format in ['json', 'both']:
                response_data['json_summary'] = report_info['summary']
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/reports/download/<report_id>/<file_type>', methods=['GET'])
def download_report_file(report_id, file_type):
    """Download generated report files"""
    try:
        from flask import send_file
        
        if file_type == 'markdown':
            # Find the markdown file
            report_dir = Path(f"reports/{report_id}")
            markdown_files = list(report_dir.glob("*.md"))
            if not markdown_files:
                return jsonify({'error': 'Markdown report not found'}), 404
            return send_file(markdown_files[0], as_attachment=True)
            
        elif file_type == 'summary':
            # Find the summary JSON file
            report_dir = Path(f"reports/{report_id}")
            summary_files = list(report_dir.glob("*_summary.json"))
            if not summary_files:
                return jsonify({'error': 'Summary file not found'}), 404
            return send_file(summary_files[0], as_attachment=True)
            
        elif file_type == 'plots':
            # Create zip file of all plots
            import zipfile
            import tempfile
            
            report_dir = Path(f"reports/{report_id}")
            plots_dir = report_dir / "plots"
            
            if not plots_dir.exists():
                return jsonify({'error': 'Plots directory not found'}), 404
            
            # Create temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for plot_file in plots_dir.glob("*.png"):
                    zipf.write(plot_file, plot_file.name)
            
            return send_file(temp_zip.name, as_attachment=True, 
                           download_name=f"plots_{report_id}.zip")
            
        elif file_type == 'tables':
            # Create zip file of all tables
            import zipfile
            import tempfile
            
            report_dir = Path(f"reports/{report_id}")
            tables_dir = report_dir / "tables"
            
            if not tables_dir.exists():
                return jsonify({'error': 'Tables directory not found'}), 404
            
            # Create temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                for table_file in tables_dir.glob("*.csv"):
                    zipf.write(table_file, table_file.name)
            
            return send_file(temp_zip.name, as_attachment=True,
                           download_name=f"tables_{report_id}.zip")
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Error downloading report file: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/reports/list', methods=['GET'])
def list_generated_reports():
    """List all generated reports"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return jsonify({'reports': []})
        
        reports = []
        for report_dir in reports_dir.iterdir():
            if report_dir.is_dir():
                # Look for summary file
                summary_files = list(report_dir.glob("*_summary.json"))
                if summary_files:
                    try:
                        with open(summary_files[0], 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        reports.append({
                            'report_id': report_dir.name,
                            'generated_at': summary.get('generated_at'),
                            'plots_count': summary.get('plots_generated', 0),
                            'tables_count': summary.get('tables_generated', 0),
                            'download_links': {
                                'markdown': f"/api/reports/download/{report_dir.name}/markdown",
                                'summary': f"/api/reports/download/{report_dir.name}/summary",
                                'plots': f"/api/reports/download/{report_dir.name}/plots",
                                'tables': f"/api/reports/download/{report_dir.name}/tables"
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Error reading report summary {report_dir.name}: {e}")
        
        # Sort by generation time (newest first)
        reports.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
        
        return jsonify({
            'reports': reports,
            'total_count': len(reports)
        })
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        return jsonify({'error': f'Failed to list reports: {str(e)}'}), 500

@app.route('/api/reports/combined', methods=['POST'])
def generate_combined_report():
    """Generate comprehensive report using both Python and R analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        include_code = request.form.get('include_code', 'false').lower() == 'true'
        target_column = request.form.get('target_column')
        n_clusters = int(request.form.get('n_clusters', 3))
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Initialize report generator
            reporter = AnalysisReportGenerator(f"reports/{file_id}")
            
            # Perform combined Python + R analysis
            analyzer.load_data(file_path)
            
            # Python analysis
            python_results = {
                'basic_statistics': analyzer.basic_statistics(),
                'correlation_analysis': analyzer.correlation_analysis(),
                'outlier_detection': analyzer.outlier_detection(),
                'data_quality': analyzer.data_quality_assessment(),
                'clustering_analysis': analyzer.clustering_analysis(n_clusters)
            }
            
            if target_column and target_column in analyzer.data.columns:
                python_results['machine_learning'] = analyzer.machine_learning_analysis(target_column)
            
            # R analysis (if available)
            r_results = {}
            if check_r_installation():
                try:
                    r_results = execute_r_script(file_path, 'comprehensive')
                except Exception as e:
                    r_results = {'error': f'R analysis failed: {str(e)}'}
            else:
                r_results = {'error': 'R not available'}
            
            # Combined results
            combined_results = {
                'python_analysis': python_results,
                'r_analysis': r_results,
                'data_quality': python_results.get('data_quality', {}),
                'basic_statistics': python_results.get('basic_statistics', {}),
                'correlation_analysis': python_results.get('correlation_analysis', {})
            }
            
            # Generate comprehensive report
            data_filename = os.path.basename(file_path).replace('.csv', '')
            title = f"Combined Python+R Analysis - {data_filename}"
            
            report_info = reporter.generate_comprehensive_report(
                analyzer.data,
                combined_results,
                title=title,
                include_code=include_code,
                include_raw_results=True
            )
            
            # Clean up
            os.remove(file_path)
            
            # Read the generated report
            with open(report_info['report_path'], 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            return jsonify({
                'status': 'success',
                'report_id': file_id,
                'report_info': report_info['summary'],
                'markdown_content': markdown_content,
                'python_available': True,
                'r_available': check_r_installation(),
                'download_links': {
                    'markdown_report': f"/api/reports/download/{file_id}/markdown",
                    'plots': f"/api/reports/download/{file_id}/plots",
                    'tables': f"/api/reports/download/{file_id}/tables"
                }
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error generating combined report: {str(e)}")
        return jsonify({'error': f'Combined report generation failed: {str(e)}'}), 500

@app.route('/api/sample/generate', methods=['GET'])
def generate_sample_data():
    """Generate sample data for testing"""
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'age': np.random.randint(20, 65, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education_years': np.random.randint(12, 20, n_samples),
            'experience': np.random.randint(0, 30, n_samples),
            'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n_samples),
            'performance_score': np.random.uniform(1, 5, n_samples)
        })
        
        # Save sample data
        sample_file = os.path.join(UPLOAD_FOLDER, 'sample_data.csv')
        sample_data.to_csv(sample_file, index=False)
        
        # Analyze sample data
        analyzer.load_data(sample_file)
        basic_stats = analyzer.basic_statistics()
        correlation = analyzer.correlation_analysis()
        
        # Clean up
        os.remove(sample_file)
        
        return jsonify({
            'message': 'Sample data generated and analyzed',
            'data_shape': sample_data.shape,
            'basic_statistics': basic_stats,
            'correlation_analysis': correlation
        })
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return jsonify({'error': f'Sample data generation failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🐍🔬 Starting Enhanced Python + R Data Analysis API Server...")
    print("📊 Python Analytics Endpoints:")
    print("   POST /api/analyze/upload - Upload and basic analysis")
    print("   POST /api/analyze/correlation - Correlation analysis")
    print("   POST /api/analyze/machine_learning - ML analysis")
    print("   POST /api/analyze/clustering - Clustering analysis")
    print("   POST /api/analyze/outliers - Outlier detection")
    print("   POST /api/analyze/quality - Data quality assessment")
    print("   POST /api/analyze/statistical_tests - Statistical tests")
    print("   POST /api/analyze/timeseries - Time series analysis")
    print("   POST /api/analyze/comprehensive - Complete Python analysis")
    print()
    print("📈 R Analytics Endpoints:")
    print("   POST /api/r/analyze/basic - R advanced statistics")
    print("   POST /api/r/analyze/correlation - R advanced correlation")
    print("   POST /api/r/analyze/tests - R comprehensive statistical tests")
    print("   POST /api/r/analyze/regression - R advanced regression modeling")
    print("   POST /api/r/analyze/survival - R survival analysis")
    print("   POST /api/r/analyze/timeseries - R advanced time series")
    print("   POST /api/r/analyze/clustering - R advanced clustering")
    print("   POST /api/r/analyze/comprehensive - Complete R analysis")
    print()
    print("🤝 Combined Analytics:")
    print("   POST /api/combined/comprehensive - Python + R combined analysis")
    print()
    print("📄 Report Generation Endpoints:")
    print("   POST /api/reports/generate - Generate comprehensive markdown report")
    print("   POST /api/reports/combined - Generate Python+R combined report")
    print("   GET  /api/reports/list - List all generated reports")
    print("   GET  /api/reports/download/<id>/<type> - Download report files")
    print("        Types: markdown, summary, plots, tables")
    print()
    print("🛠️ Utility Endpoints:")
    print("   GET  /api/r/health - Check R installation")
    print("   GET  /api/models/available - Available models")
    print("   GET  /api/sample/generate - Generate sample data")
    print("   GET  /health - Health check")
    print()
    print("🚀 Server starting on http://localhost:5000")
    print("📊 Complete analytics with markdown reports, plots, and tables!")
    print("💡 Use both Python ML power and R statistical excellence!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
