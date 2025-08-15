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
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.offline import plot
# import plotly

# Statistical analysis
# import statsmodels.api as sm  # Temporarily disabled for Windows compatibility
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Advanced Data Analysis Engine with ML capabilities"""
    
    def __init__(self):
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.results = {}
    
    def _make_json_serializable(self, obj):
        """Convert numpy/pandas objects to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj) or (hasattr(obj, 'isna') and obj.isna()):
            return None
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        elif str(type(obj)).startswith('<class \'numpy.'):
            # Catch any remaining numpy types
            try:
                return obj.item() if hasattr(obj, 'item') else str(obj)
            except:
                return str(obj)
        else:
            return obj
        
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load data from various file formats (CSV, Excel, JSON, TSV, etc.)"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            # Determine file type and load accordingly
            if file_extension == '.csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try with error handling
                    self.data = pd.read_csv(file_path, encoding='utf-8', errors='replace')
                    
            elif file_extension in ['.xlsx', '.xls']:
                # Excel files
                try:
                    # Try to read the first sheet
                    self.data = pd.read_excel(file_path, engine='openpyxl' if file_extension == '.xlsx' else 'xlrd')
                except ImportError:
                    # Fallback if openpyxl is not available
                    self.data = pd.read_excel(file_path)
                    
            elif file_extension == '.json':
                # JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, list):
                    self.data = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Handle different JSON structures
                    if 'data' in json_data:
                        self.data = pd.DataFrame(json_data['data'])
                    else:
                        # Try to create DataFrame from dict
                        self.data = pd.DataFrame([json_data]) if not any(isinstance(v, list) for v in json_data.values()) else pd.DataFrame(json_data)
                else:
                    raise ValueError("JSON format not supported")
                    
            elif file_extension in ['.tsv', '.txt']:
                # Tab-separated files
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, sep='\t', encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8', errors='replace')
                    
            elif file_extension == '.parquet':
                # Parquet files
                try:
                    self.data = pd.read_parquet(file_path)
                except ImportError:
                    raise ImportError("pyarrow or fastparquet required for Parquet files")
                    
            elif file_extension in ['.pkl', '.pickle']:
                # Pickle files
                self.data = pd.read_pickle(file_path)
                
            elif file_extension == '.feather':
                # Feather files
                try:
                    self.data = pd.read_feather(file_path)
                except ImportError:
                    raise ImportError("pyarrow required for Feather files")
                    
            elif file_extension == '.h5' or file_extension == '.hdf5':
                # HDF5 files
                try:
                    # Try to read the first key
                    with pd.HDFStore(file_path, 'r') as store:
                        keys = store.keys()
                        if keys:
                            self.data = store[keys[0]]
                        else:
                            raise ValueError("No data found in HDF5 file")
                except ImportError:
                    raise ImportError("tables required for HDF5 files")
                    
            elif file_extension in ['.orc']:
                # ORC files
                try:
                    self.data = pd.read_orc(file_path)
                except ImportError:
                    raise ImportError("pyarrow required for ORC files")
                    
            else:
                # Default: try CSV with different separators
                separators = [',', ';', '\t', '|']
                for sep in separators:
                    try:
                        self.data = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                        # Check if we got meaningful columns
                        if len(self.data.columns) > 1 or self.data.shape[0] > 1:
                            break
                    except:
                        continue
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Validate data
            if self.data is None or self.data.empty:
                raise ValueError("No data could be loaded from the file")
            
            # Clean column names (remove special characters, spaces)
            self.data.columns = [str(col).strip() for col in self.data.columns]
            
            self._identify_column_types()
            
            return {
                'success': True,
                'message': f'Data loaded successfully from {file_extension} file. Shape: {self.data.shape}',
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'file_format': file_extension,
                'data_types': {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error loading data from {Path(file_path).suffix} file: {str(e)}',
                'supported_formats': ['.csv', '.xlsx', '.xls', '.json', '.tsv', '.txt', '.parquet', '.pkl', '.pickle', '.feather', '.h5', '.hdf5', '.orc']
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
    
    def outlier_detection(self) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        if self.data is None or not self.numeric_columns:
            return {'error': 'No numeric data available for outlier detection'}
        
        try:
            outlier_results = {}
            overall_outliers = set()
            
            for col in self.numeric_columns:
                col_data = self.data[col].dropna()
                if len(col_data) < 4:  # Need minimum data points
                    continue
                    
                outliers = {
                    'iqr_outliers': [],
                    'zscore_outliers': [],
                    'modified_zscore_outliers': []
                }
                
                # IQR Method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outlier_indices = col_data[(col_data < lower_bound) | (col_data > upper_bound)].index.tolist()
                outliers['iqr_outliers'] = iqr_outlier_indices
                
                # Z-Score Method
                z_scores = np.abs(stats.zscore(col_data))
                zscore_outlier_indices = col_data[z_scores > 3].index.tolist()
                outliers['zscore_outliers'] = zscore_outlier_indices
                
                # Modified Z-Score Method
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))
                if mad != 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    mod_zscore_outlier_indices = col_data[np.abs(modified_z_scores) > 3.5].index.tolist()
                    outliers['modified_zscore_outliers'] = mod_zscore_outlier_indices
                
                # Combine all outliers for this column
                all_col_outliers = set(iqr_outlier_indices + zscore_outlier_indices + mod_zscore_outlier_indices)
                overall_outliers.update(all_col_outliers)
                
                outlier_results[col] = {
                    'total_outliers': len(all_col_outliers),
                    'outlier_percentage': float(len(all_col_outliers) / len(col_data) * 100),
                    'methods': outliers,
                    'outlier_values': col_data.loc[list(all_col_outliers)].tolist() if all_col_outliers else []
                }
            
            return {
                'total_rows_with_outliers': len(overall_outliers),
                'overall_outlier_percentage': float(len(overall_outliers) / len(self.data) * 100),
                'column_analysis': outlier_results,
                'outlier_row_indices': sorted(list(overall_outliers)),
                'methods_used': ['IQR', 'Z-Score', 'Modified Z-Score'],
                'recommendations': self._get_outlier_recommendations(outlier_results)
            }
            
        except Exception as e:
            return {'error': f'Outlier detection failed: {str(e)}'}
    
    def _get_outlier_recommendations(self, outlier_results: Dict) -> List[str]:
        """Generate recommendations based on outlier analysis"""
        recommendations = []
        
        total_outliers = sum(result['total_outliers'] for result in outlier_results.values())
        if total_outliers == 0:
            recommendations.append("✅ No significant outliers detected in the dataset")
        elif total_outliers < len(self.data) * 0.05:  # Less than 5%
            recommendations.append("✅ Low outlier count - dataset appears clean")
        elif total_outliers < len(self.data) * 0.10:  # Less than 10%
            recommendations.append("⚠️ Moderate outlier count - consider investigating suspicious values")
        else:
            recommendations.append("🚨 High outlier count - data quality issues may be present")
            
        # Column-specific recommendations
        for col, result in outlier_results.items():
            if result['outlier_percentage'] > 15:
                recommendations.append(f"🔍 Column '{col}' has {result['outlier_percentage']:.1f}% outliers - requires attention")
                
        return recommendations
    
    def data_quality_assessment(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        try:
            quality_report = {
                'overview': {
                    'total_rows': int(self.data.shape[0]),
                    'total_columns': int(self.data.shape[1]),
                    'numeric_columns': len(self.numeric_columns),
                    'categorical_columns': len(self.categorical_columns)
                },
                'completeness': {},
                'consistency': {},
                'validity': {},
                'uniqueness': {},
                'quality_score': 0,
                'recommendations': []
            }
            
            # Completeness Analysis
            missing_data = self.data.isnull().sum()
            total_cells = len(self.data) * len(self.data.columns)
            total_missing = missing_data.sum()
            completeness_score = (1 - total_missing / total_cells) * 100
            
            quality_report['completeness'] = {
                'overall_completeness': float(completeness_score),
                'total_missing_values': int(total_missing),
                'missing_percentage': float(total_missing / total_cells * 100),
                'columns_with_missing': {col: int(count) for col, count in missing_data.items() if count > 0}
            }
            
            # Consistency Analysis (for numeric columns)
            consistency_issues = {}
            for col in self.numeric_columns:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    # Check for extreme values that might indicate data entry errors
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    extreme_values = col_data[(col_data > mean_val + 4 * std_val) | (col_data < mean_val - 4 * std_val)]
                    
                    if len(extreme_values) > 0:
                        consistency_issues[col] = {
                            'extreme_values_count': len(extreme_values),
                            'extreme_values': extreme_values.tolist()[:5]  # Show first 5
                        }
            
            quality_report['consistency'] = consistency_issues
            
            # Validity Analysis (basic data type validation)
            validity_issues = {}
            for col in self.categorical_columns:
                # Check for very long strings that might be data entry errors
                col_data = self.data[col].dropna().astype(str)
                long_strings = col_data[col_data.str.len() > 100]
                if len(long_strings) > 0:
                    validity_issues[col] = {
                        'long_strings_count': len(long_strings),
                        'max_length': int(col_data.str.len().max())
                    }
            
            quality_report['validity'] = validity_issues
            
            # Uniqueness Analysis
            uniqueness_stats = {}
            for col in self.data.columns:
                unique_count = self.data[col].nunique()
                total_count = self.data[col].count()  # Exclude nulls
                uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
                
                uniqueness_stats[col] = {
                    'unique_values': int(unique_count),
                    'total_values': int(total_count),
                    'uniqueness_ratio': float(uniqueness_ratio),
                    'duplicate_count': int(total_count - unique_count)
                }
            
            quality_report['uniqueness'] = uniqueness_stats
            
            # Calculate overall quality score
            consistency_score = 100 - (len(consistency_issues) / len(self.numeric_columns) * 100 if self.numeric_columns else 0)
            validity_score = 100 - (len(validity_issues) / len(self.categorical_columns) * 100 if self.categorical_columns else 0)
            
            overall_score = (completeness_score + consistency_score + validity_score) / 3
            quality_report['quality_score'] = float(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(quality_report)
            quality_report['recommendations'] = recommendations
            
            return quality_report
            
        except Exception as e:
            return {'error': f'Data quality assessment failed: {str(e)}'}
    
    def _generate_quality_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Completeness recommendations
        completeness = quality_report['completeness']['overall_completeness']
        if completeness < 80:
            recommendations.append("🚨 Poor data completeness - consider data imputation or collection review")
        elif completeness < 95:
            recommendations.append("⚠️ Moderate missing data - evaluate impact on analysis")
        else:
            recommendations.append("✅ Good data completeness")
        
        # Consistency recommendations
        if quality_report['consistency']:
            recommendations.append("⚠️ Potential consistency issues detected - review extreme values")
        
        # Overall quality recommendations
        if quality_report['quality_score'] >= 90:
            recommendations.append("🌟 Excellent data quality - ready for analysis")
        elif quality_report['quality_score'] >= 70:
            recommendations.append("✅ Good data quality with minor issues")
        else:
            recommendations.append("🔧 Data quality needs improvement before analysis")
        
        return recommendations
    
    def statistical_tests(self) -> Dict[str, Any]:
        """Perform various statistical tests"""
        if self.data is None or len(self.numeric_columns) < 2:
            return {'error': 'Insufficient data for statistical tests'}
        
        try:
            test_results = {
                'normality_tests': {},
                'correlation_tests': {},
                'independence_tests': {},
                'summary': {}
            }
            
            # Normality tests for numeric columns
            for col in self.numeric_columns:
                col_data = self.data[col].dropna()
                if len(col_data) < 8:  # Minimum sample size for tests
                    continue
                    
                # Shapiro-Wilk test (best for small samples)
                if len(col_data) <= 5000:  # Shapiro-Wilk has sample size limitations
                    shapiro_stat, shapiro_p = stats.shapiro(col_data)
                    shapiro_result = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                else:
                    shapiro_result = {'note': 'Sample too large for Shapiro-Wilk test'}
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                ks_result = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > 0.05
                }
                
                test_results['normality_tests'][col] = {
                    'shapiro_wilk': shapiro_result,
                    'kolmogorov_smirnov': ks_result,
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
            
            # Correlation significance tests
            for i, col1 in enumerate(self.numeric_columns):
                for col2 in self.numeric_columns[i+1:]:
                    data1 = self.data[col1].dropna()
                    data2 = self.data[col2].dropna()
                    
                    # Ensure we have matching indices
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) < 3:
                        continue
                    
                    data1_matched = data1.loc[common_idx]
                    data2_matched = data2.loc[common_idx]
                    
                    # Pearson correlation test
                    pearson_r, pearson_p = stats.pearsonr(data1_matched, data2_matched)
                    
                    # Spearman correlation test
                    spearman_r, spearman_p = stats.spearmanr(data1_matched, data2_matched)
                    
                    test_results['correlation_tests'][f'{col1}_vs_{col2}'] = {
                        'pearson': {
                            'correlation': float(pearson_r),
                            'p_value': float(pearson_p),
                            'significant': pearson_p < 0.05
                        },
                        'spearman': {
                            'correlation': float(spearman_r),
                            'p_value': float(spearman_p),
                            'significant': spearman_p < 0.05
                        }
                    }
            
            # Summary statistics
            normal_columns = [col for col, test in test_results['normality_tests'].items() 
                            if test.get('shapiro_wilk', {}).get('is_normal', False)]
            significant_correlations = [pair for pair, test in test_results['correlation_tests'].items()
                                     if test['pearson']['significant']]
            
            test_results['summary'] = {
                'normal_distributions_count': len(normal_columns),
                'normal_distributions': normal_columns,
                'significant_correlations_count': len(significant_correlations),
                'significant_correlations': significant_correlations
            }
            
            return test_results
            
        except Exception as e:
            return {'error': f'Statistical tests failed: {str(e)}'}
    
    def time_series_analysis(self, date_column: str, value_column: str) -> Dict[str, Any]:
        """Basic time series analysis"""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        if date_column not in self.data.columns or value_column not in self.data.columns:
            return {'error': f'Columns {date_column} or {value_column} not found'}
        
        try:
            # Convert date column
            ts_data = self.data[[date_column, value_column]].copy()
            ts_data[date_column] = pd.to_datetime(ts_data[date_column], errors='coerce')
            ts_data = ts_data.dropna().sort_values(date_column)
            
            if len(ts_data) < 3:
                return {'error': 'Insufficient data points for time series analysis'}
            
            # Basic time series statistics
            ts_data.set_index(date_column, inplace=True)
            values = ts_data[value_column]
            
            # Calculate trends and patterns
            result = {
                'data_points': len(values),
                'time_range': {
                    'start': str(values.index.min()),
                    'end': str(values.index.max()),
                    'duration_days': (values.index.max() - values.index.min()).days
                },
                'basic_stats': {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'trend': 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing'
                },
                'seasonality': {},
                'stationarity': {}
            }
            
            # Simple trend analysis
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                result['trend_analysis'] = {
                    'slope': float(slope),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'significant_trend': p_value < 0.05
                }
            
            # Basic seasonality detection (if enough data)
            if len(values) >= 24:  # Need sufficient data for seasonality
                try:
                    # Extract frequency components
                    values_diff = values.diff().dropna()
                    if len(values_diff) > 0:
                        autocorr_1 = values_diff.autocorr(lag=1)
                        result['seasonality'] = {
                            'autocorrelation_lag1': float(autocorr_1),
                            'potential_seasonality': abs(autocorr_1) > 0.3
                        }
                except:
                    result['seasonality'] = {'note': 'Could not compute seasonality metrics'}
            
            return result
            
        except Exception as e:
            return {'error': f'Time series analysis failed: {str(e)}'}
    
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
        print("Usage: python data_analyzer.py <csv_file> <analysis_type> [additional_args]")
        print("Analysis types:")
        print("  basic          - Basic statistical analysis")
        print("  correlation    - Correlation analysis")
        print("  ml <target>    - Machine learning analysis (requires target column)")
        print("  clustering <n> - Clustering analysis (optional: number of clusters)")
        print("  outliers       - Comprehensive outlier detection")
        print("  quality        - Data quality assessment")
        print("  tests          - Statistical tests")
        print("  timeseries <date_col> <value_col> - Time series analysis")
        print("  visualize      - Generate visualizations")
        print("  comprehensive  - All analyses combined")
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
        if len(sys.argv) < 4:
            print("Error: ML analysis requires target column")
            print(f"Available columns: {', '.join(analyzer.data.columns)}")
            return
        target = sys.argv[3]
        result = analyzer.machine_learning_analysis(target)
    elif analysis_type == 'clustering':
        n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        result = analyzer.clustering_analysis(n_clusters)
    elif analysis_type == 'outliers':
        result = analyzer.outlier_detection()
    elif analysis_type == 'quality':
        result = analyzer.data_quality_assessment()
    elif analysis_type == 'tests':
        result = analyzer.statistical_tests()
    elif analysis_type == 'timeseries':
        if len(sys.argv) < 5:
            print("Error: Time series analysis requires date and value columns")
            print(f"Available columns: {', '.join(analyzer.data.columns)}")
            return
        date_col = sys.argv[3]
        value_col = sys.argv[4]
        result = analyzer.time_series_analysis(date_col, value_col)
    elif analysis_type == 'visualize':
        result = analyzer.generate_visualizations()
    elif analysis_type == 'comprehensive':
        # Run all analyses
        result = {
            'basic_statistics': analyzer.basic_statistics(),
            'correlation_analysis': analyzer.correlation_analysis(),
            'outlier_detection': analyzer.outlier_detection(),
            'data_quality': analyzer.data_quality_assessment(),
            'statistical_tests': analyzer.statistical_tests(),
            'clustering': analyzer.clustering_analysis(),
            'visualizations': analyzer.generate_visualizations()
        }
        # Add ML analysis if we have numeric columns for target
        if analyzer.numeric_columns:
            result['machine_learning'] = analyzer.machine_learning_analysis(analyzer.numeric_columns[0])
    else:
        result = {'error': f'Unknown analysis type: {analysis_type}'}
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
