#!/usr/bin/env python3
"""
Comprehensive Report Generator
Creates detailed markdown reports with all analysis results, visualizations, and tables
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
from pathlib import Path

class AnalysisReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.report_sections = []
        self.generated_plots = []
        self.generated_tables = []
        
    def add_header(self, title, level=1):
        """Add markdown header to report"""
        header = "#" * level + " " + title
        self.report_sections.append(header + "\n")
        
    def add_text(self, text):
        """Add text content to report"""
        self.report_sections.append(text + "\n")
        
    def add_code_block(self, code, language="python"):
        """Add code block to report"""
        self.report_sections.append(f"```{language}")
        self.report_sections.append(code)
        self.report_sections.append("```\n")
        
    def save_plot_as_image(self, fig, filename, title=""):
        """Save matplotlib figure as image and return markdown reference"""
        filepath = self.plots_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        self.generated_plots.append({
            'filename': filename,
            'filepath': filepath,
            'title': title
        })
        
        # Return markdown image reference
        relative_path = f"plots/{filename}.png"
        return f"![{title}]({relative_path})\n"
        
    def save_table_as_markdown(self, df, filename, title="", max_rows=50):
        """Save dataframe as markdown table and return reference"""
        # Limit rows for display
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        # Save as CSV for full data
        csv_path = self.tables_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Create markdown table
        markdown_table = display_df.to_markdown(index=False, floatfmt=".3f")
        
        self.generated_tables.append({
            'filename': filename,
            'csv_path': csv_path,
            'title': title,
            'rows': len(df),
            'displayed_rows': len(display_df)
        })
        
        table_section = f"### {title}\n\n" if title else ""
        table_section += markdown_table + "\n\n"
        
        if len(df) > max_rows:
            table_section += f"*Showing {max_rows} of {len(df)} rows. Full data available in: `tables/{filename}.csv`*\n\n"
            
        return table_section
        
    def create_basic_statistics_report(self, data, results):
        """Create comprehensive basic statistics report section"""
        self.add_header("📊 Basic Statistical Analysis", 2)
        
        # Data overview
        self.add_header("Data Overview", 3)
        overview_data = {
            'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
            'Value': [
                results.get('total_rows', len(data)),
                results.get('total_columns', len(data.columns)),
                results.get('numeric_columns', len(data.select_dtypes(include=[np.number]).columns)),
                results.get('categorical_columns', len(data.select_dtypes(include=['object']).columns)),
                results.get('total_missing', data.isnull().sum().sum())
            ]
        }
        overview_df = pd.DataFrame(overview_data)
        self.add_text(self.save_table_as_markdown(overview_df, "data_overview", "Data Overview"))
        
        # Descriptive statistics
        if 'descriptive_stats' in results:
            self.add_header("Descriptive Statistics", 3)
            desc_stats = pd.DataFrame(results['descriptive_stats']).T
            self.add_text(self.save_table_as_markdown(desc_stats, "descriptive_stats", "Descriptive Statistics"))
            
        # Column information
        if 'column_info' in results:
            self.add_header("Column Information", 3)
            col_info_df = pd.DataFrame(results['column_info'])
            self.add_text(self.save_table_as_markdown(col_info_df, "column_info", "Column Information"))
            
    def create_correlation_report(self, data, results):
        """Create correlation analysis report section"""
        self.add_header("🔗 Correlation Analysis", 2)
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            # Correlation matrix visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            correlation_matrix = numeric_data.corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, mask=mask, ax=ax, fmt='.3f')
            ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
            
            plot_ref = self.save_plot_as_image(fig, "correlation_heatmap", "Correlation Matrix")
            self.add_text(plot_ref)
            
            # Correlation table
            self.add_text(self.save_table_as_markdown(
                correlation_matrix, "correlation_matrix", "Correlation Matrix"
            ))
            
            # Strong correlations
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strong_corr.append({
                            'Variable 1': correlation_matrix.columns[i],
                            'Variable 2': correlation_matrix.columns[j],
                            'Correlation': corr_val,
                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
                self.add_text(self.save_table_as_markdown(
                    strong_corr_df, "strong_correlations", "Strong Correlations (|r| > 0.5)"
                ))
                
    def create_distribution_plots(self, data):
        """Create distribution plots for numeric variables"""
        self.add_header("📈 Data Distributions", 2)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Multiple distribution plots
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    ax = axes[i] if n_rows > 1 or n_cols > 1 else axes
                    data[col].hist(bins=30, alpha=0.7, ax=ax, color=f'C{i}')
                    ax.set_title(f'Distribution of {col}', fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plot_ref = self.save_plot_as_image(fig, "distributions", "Data Distributions")
            self.add_text(plot_ref)
            
    def create_outlier_report(self, data, results):
        """Create outlier detection report"""
        if 'outlier_detection' in results:
            self.add_header("🎯 Outlier Analysis", 2)
            
            outlier_results = results['outlier_detection']
            
            # Outlier summary
            if 'summary' in outlier_results:
                summary_data = []
                for method, method_results in outlier_results['summary'].items():
                    for col, count in method_results.items():
                        summary_data.append({
                            'Method': method.upper(),
                            'Column': col,
                            'Outliers Found': count,
                            'Percentage': f"{(count/len(data)*100):.2f}%"
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    self.add_text(self.save_table_as_markdown(
                        summary_df, "outlier_summary", "Outlier Detection Summary"
                    ))
                    
            # Box plots for outlier visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        ax = axes[i] if n_rows > 1 or n_cols > 1 else axes
                        data.boxplot(column=col, ax=ax)
                        ax.set_title(f'Box Plot - {col}', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for i in range(len(numeric_cols), len(axes)):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                plot_ref = self.save_plot_as_image(fig, "outlier_boxplots", "Outlier Detection - Box Plots")
                self.add_text(plot_ref)
                
    def create_machine_learning_report(self, results):
        """Create machine learning analysis report"""
        if 'machine_learning' in results:
            self.add_header("🤖 Machine Learning Analysis", 2)
            
            ml_results = results['machine_learning']
            
            if 'model_performance' in ml_results:
                # Model performance table
                perf_data = []
                for model, metrics in ml_results['model_performance'].items():
                    if isinstance(metrics, dict):
                        perf_data.append({
                            'Model': model,
                            **metrics
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    self.add_text(self.save_table_as_markdown(
                        perf_df, "ml_performance", "Model Performance Comparison"
                    ))
                    
            if 'feature_importance' in ml_results:
                # Feature importance plot
                importance_data = ml_results['feature_importance']
                if importance_data:
                    features = list(importance_data.keys())
                    importances = list(importance_data.values())
                    
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    bars = ax.barh(features, importances, color='skyblue', edgecolor='navy', alpha=0.7)
                    ax.set_xlabel('Importance Score', fontweight='bold')
                    ax.set_title('Feature Importance', fontsize=16, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center')
                    
                    plt.tight_layout()
                    plot_ref = self.save_plot_as_image(fig, "feature_importance", "Feature Importance")
                    self.add_text(plot_ref)
                    
    def create_clustering_report(self, results):
        """Create clustering analysis report"""
        if 'clustering_analysis' in results:
            self.add_header("🎯 Clustering Analysis", 2)
            
            clustering_results = results['clustering_analysis']
            
            if 'cluster_summary' in clustering_results:
                cluster_summary = clustering_results['cluster_summary']
                summary_df = pd.DataFrame(cluster_summary)
                self.add_text(self.save_table_as_markdown(
                    summary_df, "cluster_summary", "Cluster Summary"
                ))
                
            # Cluster evaluation metrics
            if 'evaluation_metrics' in clustering_results:
                metrics = clustering_results['evaluation_metrics']
                metrics_data = []
                for metric, value in metrics.items():
                    metrics_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value,
                        'Interpretation': self._interpret_clustering_metric(metric, value)
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                self.add_text(self.save_table_as_markdown(
                    metrics_df, "clustering_metrics", "Clustering Evaluation Metrics"
                ))
                
    def _interpret_clustering_metric(self, metric, value):
        """Interpret clustering metrics"""
        interpretations = {
            'silhouette_score': 'Excellent' if value > 0.7 else 'Good' if value > 0.5 else 'Fair' if value > 0.25 else 'Poor',
            'calinski_harabasz_score': 'Higher is better - measures cluster separation',
            'davies_bouldin_score': 'Lower is better - measures cluster similarity'
        }
        return interpretations.get(metric, 'See documentation for interpretation')
        
    def create_time_series_report(self, results):
        """Create time series analysis report"""
        if 'time_series' in results:
            self.add_header("📈 Time Series Analysis", 2)
            
            ts_results = results['time_series']
            
            # Time series components
            if 'decomposition' in ts_results:
                self.add_header("Time Series Decomposition", 3)
                self.add_text("Time series has been decomposed into trend, seasonal, and residual components.")
                
            # Trend analysis
            if 'trend_analysis' in ts_results:
                trend = ts_results['trend_analysis']
                trend_data = {
                    'Component': ['Direction', 'Strength', 'Start Value', 'End Value', 'Total Change'],
                    'Value': [
                        trend.get('direction', 'N/A'),
                        trend.get('strength', 'N/A'),
                        trend.get('start_value', 'N/A'),
                        trend.get('end_value', 'N/A'),
                        trend.get('total_change', 'N/A')
                    ]
                }
                trend_df = pd.DataFrame(trend_data)
                self.add_text(self.save_table_as_markdown(
                    trend_df, "trend_analysis", "Trend Analysis Results"
                ))
                
    def create_data_quality_report(self, results):
        """Create data quality assessment report"""
        if 'data_quality' in results:
            self.add_header("🔍 Data Quality Assessment", 2)
            
            quality_results = results['data_quality']
            
            # Overall quality score
            if 'overall_score' in quality_results:
                score = quality_results['overall_score']
                self.add_text(f"**Overall Data Quality Score: {score:.2f}/100**\n")
                
            # Quality metrics by column
            if 'column_quality' in quality_results:
                col_quality = quality_results['column_quality']
                quality_data = []
                for col, metrics in col_quality.items():
                    quality_data.append({
                        'Column': col,
                        'Missing %': f"{metrics.get('missing_percentage', 0):.2f}%",
                        'Unique Values': metrics.get('unique_values', 0),
                        'Data Type': metrics.get('inferred_type', 'Unknown'),
                        'Quality Score': f"{metrics.get('quality_score', 0):.2f}"
                    })
                
                quality_df = pd.DataFrame(quality_data)
                self.add_text(self.save_table_as_markdown(
                    quality_df, "data_quality", "Column Quality Assessment"
                ))
                
    def generate_comprehensive_report(self, data, analysis_results, 
                                    title="Data Analysis Report", 
                                    include_code=False,
                                    include_raw_results=False):
        """Generate comprehensive analysis report"""
        
        # Clear previous report
        self.report_sections = []
        self.generated_plots = []
        self.generated_tables = []
        
        # Report header
        self.add_header(title, 1)
        self.add_text(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_text(f"**Data Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        self.add_text("---\n")
        
        # Table of Contents
        self.add_header("📋 Table of Contents", 2)
        toc = """
1. [📊 Basic Statistical Analysis](#-basic-statistical-analysis)
2. [🔗 Correlation Analysis](#-correlation-analysis)
3. [📈 Data Distributions](#-data-distributions)
4. [🎯 Outlier Analysis](#-outlier-analysis)
5. [🤖 Machine Learning Analysis](#-machine-learning-analysis)
6. [🎯 Clustering Analysis](#-clustering-analysis)
7. [📈 Time Series Analysis](#-time-series-analysis)
8. [🔍 Data Quality Assessment](#-data-quality-assessment)
9. [📋 Summary and Recommendations](#-summary-and-recommendations)
"""
        if include_code:
            toc += "10. [💻 Source Code](#-source-code)\n"
        if include_raw_results:
            toc += "11. [📄 Raw Analysis Results](#-raw-analysis-results)\n"
            
        self.add_text(toc)
        
        # Generate report sections
        self.create_basic_statistics_report(data, analysis_results)
        self.create_correlation_report(data, analysis_results)
        self.create_distribution_plots(data)
        self.create_outlier_report(data, analysis_results)
        self.create_machine_learning_report(analysis_results)
        self.create_clustering_report(analysis_results)
        self.create_time_series_report(analysis_results)
        self.create_data_quality_report(analysis_results)
        
        # Summary section
        self.add_header("📋 Summary and Recommendations", 2)
        self.generate_summary_section(data, analysis_results)
        
        # Optional: Include source code
        if include_code:
            self.add_code_section()
            
        # Optional: Include raw results
        if include_raw_results:
            self.add_raw_results_section(analysis_results)
            
        # Generate final report
        return self.save_report(title)
        
    def generate_summary_section(self, data, results):
        """Generate summary and recommendations"""
        summary_points = []
        
        # Data overview
        summary_points.append(f"• **Dataset Size**: {data.shape[0]:,} rows and {data.shape[1]} columns")
        
        # Missing data
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        if missing_pct > 0:
            summary_points.append(f"• **Missing Data**: {missing_pct:.2f}% of total data points")
        
        # Data types
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(data.select_dtypes(include=['object']).columns)
        summary_points.append(f"• **Column Types**: {numeric_cols} numeric, {categorical_cols} categorical")
        
        # Add specific insights based on results
        if 'correlation_analysis' in results:
            summary_points.append("• **Correlations**: Analyzed relationships between numeric variables")
            
        if 'machine_learning' in results:
            summary_points.append("• **Machine Learning**: Applied predictive modeling techniques")
            
        if 'clustering_analysis' in results:
            summary_points.append("• **Clustering**: Identified natural groupings in the data")
            
        self.add_text("### Key Findings\n")
        for point in summary_points:
            self.add_text(point)
            
        # Recommendations
        recommendations = [
            "• Review data quality issues identified in the assessment",
            "• Consider feature engineering based on correlation analysis",
            "• Investigate outliers detected in the analysis",
            "• Use clustering insights for data segmentation strategies"
        ]
        
        self.add_text("\n### Recommendations\n")
        for rec in recommendations:
            self.add_text(rec)
            
    def add_code_section(self):
        """Add source code section to report"""
        self.add_header("💻 Source Code", 2)
        self.add_text("The following code was used to generate this analysis:\n")
        
        # Include key Python files
        code_files = [
            "python-analytics/data_analyzer.py",
            "python-analytics/python_api_server.py",
            "r-analytics/data_analyzer.R"
        ]
        
        for file_path in code_files:
            if os.path.exists(file_path):
                self.add_header(f"File: {file_path}", 3)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    language = "r" if file_path.endswith('.R') else "python"
                    self.add_code_block(code_content, language)
                except Exception as e:
                    self.add_text(f"*Error reading file: {e}*\n")
                    
    def add_raw_results_section(self, results):
        """Add raw analysis results as JSON"""
        self.add_header("📄 Raw Analysis Results", 2)
        self.add_text("Complete analysis results in JSON format:\n")
        
        try:
            results_json = json.dumps(results, indent=2, default=str)
            self.add_code_block(results_json, "json")
        except Exception as e:
            self.add_text(f"*Error serializing results: {e}*\n")
            
    def save_report(self, title="analysis_report"):
        """Save the complete report as markdown file"""
        # Create filename-safe title
        filename = title.lower().replace(' ', '_').replace('/', '_')
        report_path = self.output_dir / f"{filename}.md"
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            for section in self.report_sections:
                f.write(section + "\n")
                
        # Create report summary
        summary = {
            'report_path': str(report_path),
            'generated_at': datetime.now().isoformat(),
            'plots_generated': len(self.generated_plots),
            'tables_generated': len(self.generated_tables),
            'plots': self.generated_plots,
            'tables': self.generated_tables
        }
        
        # Save summary as JSON
        summary_path = self.output_dir / f"{filename}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
            
        return {
            'report_path': report_path,
            'summary_path': summary_path,
            'summary': summary
        }

# Example usage function
def generate_complete_report(data_path, output_dir="reports", include_code=False, include_raw_results=False):
    """Generate complete analysis report for given data"""
    from data_analyzer import DataAnalyzer
    
    # Initialize components
    analyzer = DataAnalyzer()
    reporter = AnalysisReportGenerator(output_dir)
    
    # Load and analyze data
    load_result = analyzer.load_data(data_path)
    if not load_result['success']:
        return {'error': f"Failed to load data: {load_result.get('error', 'Unknown error')}"}
    
    # Perform comprehensive analysis
    analysis_results = {
        'basic_statistics': analyzer.basic_statistics(),
        'correlation_analysis': analyzer.correlation_analysis(),
        'outlier_detection': analyzer.outlier_detection(),
        'data_quality': analyzer.data_quality_assessment(),
        'clustering_analysis': analyzer.clustering_analysis(),
        'statistical_tests': analyzer.statistical_tests()
    }
    
    # Generate report
    data_filename = os.path.basename(data_path).replace('.csv', '')
    title = f"Analysis Report - {data_filename}"
    
    report_info = reporter.generate_comprehensive_report(
        analyzer.data, 
        analysis_results, 
        title=title,
        include_code=include_code,
        include_raw_results=include_raw_results
    )
    
    return report_info

if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'age': np.random.randint(25, 65, 100),
        'income': np.random.normal(50000, 15000, 100),
        'experience': np.random.randint(1, 20, 100),
        'performance': np.random.uniform(1, 5, 100)
    })
    
    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)
    
    # Generate report
    result = generate_complete_report('sample_data.csv', include_code=True)
    print("Report generated:", result)
