#!/usr/bin/env python3
"""
Advanced Data Analysis Engine
Provides comprehensive data analysis, machine learning, and visualization capabilities
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Core analysis libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly

# Statistical analysis
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Advanced Data Analysis Engine with ML capabilities"""
    
    def __init__(self):
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.results = {}
        
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            self._identify_column_types()
            
            return {
                'success': True,
                'message': f'Data loaded successfully. Shape: {self.data.shape}',
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error loading data: {str(e)}'
            }
    
    def _identify_column_types(self):
        """Identify numeric and categorical columns"""
        self.numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.data.select_dtypes(include=['object']).columns)
    
    def basic_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive basic statistics"""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        stats_dict = {
            'dataset_info': {
                'total_rows': int(self.data.shape[0]),
                'total_columns': int(self.data.shape[1]),
                'memory_usage': f"{self.data.memory_usage().sum() / 1024:.2f} KB",
                'missing_values': int(self.data.isnull().sum().sum()),
                'duplicate_rows': int(self.data.duplicated().sum())
            },
            'numeric_analysis': {},
            'categorical_analysis': {}
        }
        
        # Numeric columns analysis
        if self.numeric_columns:
            numeric_desc = self.data[self.numeric_columns].describe()
            for col in self.numeric_columns:
                col_data = self.data[col].dropna()
                stats_dict['numeric_analysis'][col] = {
                    'count': int(numeric_desc.loc['count', col]),
                    'mean': float(numeric_desc.loc['mean', col]),
                    'std': float(numeric_desc.loc['std', col]),
                    'min': float(numeric_desc.loc['min', col]),
                    'q1': float(numeric_desc.loc['25%', col]),
                    'median': float(numeric_desc.loc['50%', col]),
                    'q3': float(numeric_desc.loc['75%', col]),
                    'max': float(numeric_desc.loc['max', col]),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis()),
                    'variance': float(col_data.var()),
                    'range': float(col_data.max() - col_data.min()),
                    'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0,
                    'outliers_iqr': int(self._count_outliers_iqr(col_data)),
                    'missing_count': int(self.data[col].isnull().sum())
                }
        
        # Categorical columns analysis
        if self.categorical_columns:
            for col in self.categorical_columns:
                value_counts = self.data[col].value_counts()
                stats_dict['categorical_analysis'][col] = {
                    'unique_values': int(self.data[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'missing_count': int(self.data[col].isnull().sum()),
                    'value_counts': {str(k): int(v) for k, v in value_counts.head(10).items()}
                }
        
        return stats_dict
    
    def _count_outliers_iqr(self, data: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return len(data[(data < lower_bound) | (data > upper_bound)])
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis"""
        if self.data is None or len(self.numeric_columns) < 2:
            return {'error': 'Insufficient numeric data for correlation analysis'}
        
        numeric_data = self.data[self.numeric_columns]
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]
                
                if abs(pearson_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'pearson': float(pearson_val),
                        'spearman': float(spearman_val),
                        'strength': 'strong' if abs(pearson_val) > 0.7 else 'moderate'
                    })
        
        return {
            'pearson_matrix': pearson_corr.round(3).to_dict(),
            'spearman_matrix': spearman_corr.round(3).to_dict(),
            'strong_correlations': strong_correlations,
            'summary': {
                'total_variables': len(self.numeric_columns),
                'strong_correlations_count': len([c for c in strong_correlations if c['strength'] == 'strong']),
                'moderate_correlations_count': len([c for c in strong_correlations if c['strength'] == 'moderate'])
            }
        }
    
    def machine_learning_analysis(self, target_column: str) -> Dict[str, Any]:
        """Perform machine learning analysis"""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        if target_column not in self.data.columns:
            return {'error': f'Target column {target_column} not found'}
        
        try:
            # Prepare data
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.mean() if len(X.select_dtypes(include=[np.number]).columns) > 0 else 0)
            y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])
            
            # Determine if regression or classification
            is_classification = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() < 10
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = {
                'task_type': 'classification' if is_classification else 'regression',
                'feature_count': len(X.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'models': {}
            }
            
            if is_classification:
                # Classification models
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
                }
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = model.score(X_test, y_test)
                        
                        results['models'][name] = {
                            'accuracy': float(accuracy),
                            'predictions_sample': y_pred[:5].tolist() if len(y_pred) > 0 else [],
                            'feature_importance': dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                        }
                    except Exception as e:
                        results['models'][name] = {'error': str(e)}
            
            else:
                # Regression models
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Linear Regression': LinearRegression()
                }
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results['models'][name] = {
                            'mse': float(mse),
                            'rmse': float(np.sqrt(mse)),
                            'r2_score': float(r2),
                            'predictions_sample': y_pred[:5].tolist() if len(y_pred) > 0 else [],
                            'feature_importance': dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
                        }
                    except Exception as e:
                        results['models'][name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            return {'error': f'Machine learning analysis failed: {str(e)}'}
    
    def clustering_analysis(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis"""
        if self.data is None or len(self.numeric_columns) < 2:
            return {'error': 'Insufficient numeric data for clustering'}
        
        try:
            # Prepare data
            numeric_data = self.data[self.numeric_columns].fillna(self.data[self.numeric_columns].mean())
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = numeric_data[cluster_labels == i]
                cluster_stats[f'Cluster {i}'] = {
                    'size': int(len(cluster_data)),
                    'percentage': float(len(cluster_data) / len(numeric_data) * 100),
                    'means': cluster_data.mean().round(3).to_dict()
                }
            
            # Inertia and silhouette analysis
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            
            return {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_),
                'silhouette_score': float(silhouette_avg),
                'cluster_statistics': cluster_stats,
                'features_used': self.numeric_columns
            }
            
        except Exception as e:
            return {'error': f'Clustering analysis failed: {str(e)}'}
    
    def generate_visualizations(self, output_dir: str = "visualizations") -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        try:
            Path(output_dir).mkdir(exist_ok=True)
            generated_plots = []
            
            # 1. Distribution plots for numeric variables
            if self.numeric_columns:
                fig, axes = plt.subplots(len(self.numeric_columns), 2, 
                                       figsize=(12, 4 * len(self.numeric_columns)))
                if len(self.numeric_columns) == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(self.numeric_columns):
                    # Histogram
                    axes[i, 0].hist(self.data[col].dropna(), bins=30, alpha=0.7, color='skyblue')
                    axes[i, 0].set_title(f'Distribution of {col}')
                    axes[i, 0].set_xlabel(col)
                    axes[i, 0].set_ylabel('Frequency')
                    
                    # Box plot
                    axes[i, 1].boxplot(self.data[col].dropna())
                    axes[i, 1].set_title(f'Box Plot of {col}')
                    axes[i, 1].set_ylabel(col)
                
                plt.tight_layout()
                dist_plot_path = f"{output_dir}/distributions.png"
                plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(dist_plot_path)
            
            # 2. Correlation heatmap
            if len(self.numeric_columns) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = self.data[self.numeric_columns].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                corr_plot_path = f"{output_dir}/correlation_matrix.png"
                plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(corr_plot_path)
            
            # 3. Categorical variable plots
            if self.categorical_columns:
                for col in self.categorical_columns[:3]:  # Limit to first 3 categorical columns
                    plt.figure(figsize=(10, 6))
                    value_counts = self.data[col].value_counts().head(10)
                    value_counts.plot(kind='bar', color='lightcoral')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    cat_plot_path = f"{output_dir}/categorical_{col}.png"
                    plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    generated_plots.append(cat_plot_path)
            
            return {
                'success': True,
                'generated_plots': generated_plots,
                'plot_count': len(generated_plots),
                'output_directory': output_dir
            }
            
        except Exception as e:
            return {'error': f'Visualization generation failed: {str(e)}'}

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 3:
        print("Usage: python data_analyzer.py <csv_file> <analysis_type>")
        print("Analysis types: basic, correlation, ml, clustering, visualize")
        return
    
    csv_file = sys.argv[1]
    analysis_type = sys.argv[2].lower()
    
    analyzer = DataAnalyzer()
    
    # Load data
    load_result = analyzer.load_data(csv_file)
    if not load_result['success']:
        print(json.dumps(load_result, indent=2))
        return
    
    # Perform analysis based on type
    if analysis_type == 'basic':
        result = analyzer.basic_statistics()
    elif analysis_type == 'correlation':
        result = analyzer.correlation_analysis()
    elif analysis_type == 'ml':
        target = sys.argv[3] if len(sys.argv) > 3 else analyzer.numeric_columns[0]
        result = analyzer.machine_learning_analysis(target)
    elif analysis_type == 'clustering':
        n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        result = analyzer.clustering_analysis(n_clusters)
    elif analysis_type == 'visualize':
        result = analyzer.generate_visualizations()
    else:
        result = {'error': f'Unknown analysis type: {analysis_type}'}
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
