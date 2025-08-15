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
from pathlib import Path
import logging
from data_analyzer import DataAnalyzer

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
            'comprehensive'
        ]
    })

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
    print("🐍 Starting Python Data Analysis API Server...")
    print("📊 Available endpoints:")
    print("   POST /api/analyze/upload - Upload and analyze CSV")
    print("   POST /api/analyze/correlation - Correlation analysis")
    print("   POST /api/analyze/machine_learning - ML analysis")
    print("   POST /api/analyze/clustering - Clustering analysis")
    print("   POST /api/analyze/comprehensive - Complete analysis")
    print("   GET  /api/models/available - Available models")
    print("   GET  /health - Health check")
    print()
    print("🚀 Server starting on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
