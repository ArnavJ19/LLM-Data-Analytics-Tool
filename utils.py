import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
import io
import json

def export_report(df: pd.DataFrame, eda_results: Dict[str, Any], 
                 llm_insights: Dict[str, str], format: str = "markdown") -> str:
    """
    Export a comprehensive analysis report.
    
    Args:
        df: Input DataFrame
        eda_results: EDA results from DataProcessor
        llm_insights: LLM insights from LLMInsightGenerator
        format: Export format ('markdown' or 'html')
        
    Returns:
        Report content as string
    """
    
    if format.lower() == "markdown":
        return _generate_markdown_report(df, eda_results, llm_insights)
    elif format.lower() == "html":
        return _generate_html_report(df, eda_results, llm_insights)
    else:
        raise ValueError(f"Unsupported format: {format}")

def _generate_markdown_report(df: pd.DataFrame, eda_results: Dict[str, Any], 
                            llm_insights: Dict[str, str]) -> str:
    """Generate a comprehensive markdown report."""
    
    report_sections = []
    
    # Header
    report_sections.append(f"""# Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
""")
    
    # Executive Summary
    if llm_insights.get('dataset_summary'):
        report_sections.append(f"""## 📋 Executive Summary

{llm_insights['dataset_summary']}

---
""")
    
    # Dataset Overview
    num_rows, num_cols = df.shape
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    missing_total = df.isnull().sum().sum()
    
    report_sections.append(f"""## 📊 Dataset Overview

### Basic Information
- **Total Rows:** {num_rows:,}
- **Total Columns:** {num_cols}
- **Numerical Columns:** {len(numerical_cols)}
- **Categorical Columns:** {len(categorical_cols)}
- **DateTime Columns:** {len(datetime_cols)}
- **Total Missing Values:** {missing_total:,}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

### Column Information
""")
    
    # Add column details
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        report_sections.append(f"""- **{col}** ({dtype}): {unique_count:,} unique values, {missing_count} missing ({missing_pct:.1f}%)""")
    
    report_sections.append("\n---\n")
    
    # Statistical Summary
    if not eda_results.get('numerical_summary', pd.DataFrame()).empty:
        report_sections.append("""## 📈 Statistical Summary

### Numerical Variables
""")
        numerical_summary = eda_results['numerical_summary']
        
        # Convert to markdown table
        report_sections.append(numerical_summary.to_markdown(index=False, floatfmt=".3f"))
        
        if llm_insights.get('statistical_insights'):
            report_sections.append(f"""
### Key Statistical Insights
{llm_insights['statistical_insights']}
""")
        
        report_sections.append("\n---\n")
    
    # Categorical Summary
    if not eda_results.get('categorical_summary', pd.DataFrame()).empty:
        report_sections.append("""## 📝 Categorical Analysis

### Categorical Variables Summary
""")
        categorical_summary = eda_results['categorical_summary']
        report_sections.append(categorical_summary.to_markdown(index=False))
        report_sections.append("\n---\n")
    
    # Data Quality Assessment
    missing_patterns = eda_results.get('missing_patterns', {})
    report_sections.append(f"""## 🔍 Data Quality Assessment

### Missing Value Analysis
- **Total Missing Values:** {missing_patterns.get('total_missing', 0):,}
- **Rows with Missing Data:** {missing_patterns.get('rows_with_missing', 0):,}
- **Complete Rows:** {missing_patterns.get('complete_rows', 0):,}

### Missing Values by Column
""")
    
    missing_by_col = missing_patterns.get('missing_by_column', {})
    for col, count in missing_by_col.items():
        if count > 0:
            pct = missing_patterns.get('missing_percentage', {}).get(col, 0)
            report_sections.append(f"- **{col}:** {count:,} missing ({pct:.1f}%)")
    
    if llm_insights.get('data_quality'):
        report_sections.append(f"""
### Data Quality Insights
{llm_insights['data_quality']}
""")
    
    report_sections.append("\n---\n")
    
    # Correlation Analysis
    if eda_results.get('correlations') is not None:
        report_sections.append("""## 🔗 Correlation Analysis

### Correlation Matrix
""")
        corr_matrix = eda_results['correlations']
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    strong_correlations.append(f"- **{var1} ↔ {var2}:** {corr_val:.3f}")
        
        if strong_correlations:
            report_sections.append("### Strong Correlations (|r| > 0.5)\n")
            report_sections.append("\n".join(strong_correlations))
        else:
            report_sections.append("No strong correlations (|r| > 0.5) found.")
        
        if llm_insights.get('correlation_insights'):
            report_sections.append(f"""
### Correlation Insights
{llm_insights['correlation_insights']}
""")
        
        report_sections.append("\n---\n")
    
    # Distribution Analysis
    distribution_stats = eda_results.get('distribution_stats', {})
    if distribution_stats:
        report_sections.append("""## 📊 Distribution Analysis

### Distribution Characteristics
""")
        
        for var, stats in distribution_stats.items():
            normality = "Normal" if stats.get('is_normal', False) else "Non-normal"
            zeros_pct = stats.get('zeros_percentage', 0)
            
            report_sections.append(f"""- **{var}:** {normality} distribution, {zeros_pct:.1f}% zeros""")
        
        if llm_insights.get('distribution_insights'):
            report_sections.append(f"""
### Distribution Insights
{llm_insights['distribution_insights']}
""")
        
        report_sections.append("\n---\n")
    
    # Outlier Analysis
    outliers = eda_results.get('outliers', {})
    if outliers:
        report_sections.append("""## 🎯 Outlier Analysis

### Outlier Detection Results
""")
        
        for var, info in outliers.items():
            count = info.get('count', 0)
            percentage = info.get('percentage', 0)
            methods = ', '.join(info.get('methods', []))
            
            report_sections.append(f"""- **{var}:** {count} outliers ({percentage:.1f}%) detected by {methods}""")
        
        if llm_insights.get('outlier_insights'):
            report_sections.append(f"""
### Outlier Insights
{llm_insights['outlier_insights']}
""")
        
        report_sections.append("\n---\n")
    
    # Recommendations
    report_sections.append("""## 💡 Recommendations

### Data Preparation
1. **Missing Values:** Implement appropriate imputation strategies based on data types and missing patterns
2. **Outliers:** Investigate extreme values and consider transformation or removal based on domain knowledge
3. **Data Types:** Ensure all columns have appropriate data types for analysis

### Analysis Suggestions
1. **Feature Engineering:** Consider creating new features based on identified patterns
2. **Modeling Approach:** Select appropriate algorithms based on distribution characteristics
3. **Validation Strategy:** Implement robust cross-validation considering data characteristics

### Further Investigation
1. Explore temporal patterns if datetime columns are present
2. Investigate business context for unusual patterns
3. Consider domain expertise for feature interpretation

---

*Report generated by AI-Powered Data Analytics Tool*
""")
    
    return "".join(report_sections)

def _generate_html_report(df: pd.DataFrame, eda_results: Dict[str, Any], 
                        llm_insights: Dict[str, str]) -> str:
    """Generate an HTML report (placeholder for future implementation)."""
    
    # Convert markdown to HTML (simplified version)
    markdown_content = _generate_markdown_report(df, eda_results, llm_insights)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .insight-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin: 15px 0;
            }}
            .metric {{
                display: inline-block;
                background-color: #e9ecef;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <pre>{markdown_content}</pre>
    </body>
    </html>
    """
    
    return html_template

def create_download_link(content: str, filename: str, content_type: str = "text/plain") -> str:
    """
    Create a download link for content.
    
    Args:
        content: Content to download
        filename: Name of the file
        content_type: MIME type of the content
        
    Returns:
        Base64 encoded download link
    """
    
    b64_content = base64.b64encode(content.encode()).decode()
    href = f'data:{content_type};base64,{b64_content}'
    
    return href

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with appropriate decimal places and thousand separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:,.{decimals}f}"

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect and categorize column types for better analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with categorized column names
    """
    
    column_types = {
        'numerical': [],
        'categorical': [],
        'datetime': [],
        'boolean': [],
        'text': [],
        'id_like': []
    }
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        non_null_count = df[col].count()
        
        # DateTime columns
        if pd.api.types.is_datetime64_any_dtype(dtype):
            column_types['datetime'].append(col)
        
        # Numerical columns
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it's actually categorical (few unique values)
            if unique_count <= 10 and unique_count < non_null_count * 0.1:
                column_types['categorical'].append(col)
            # Check if it's boolean-like
            elif unique_count == 2:
                column_types['boolean'].append(col)
            # Check if it's ID-like (mostly unique)
            elif unique_count > non_null_count * 0.9:
                column_types['id_like'].append(col)
            else:
                column_types['numerical'].append(col)
        
        # Object/string columns
        else:
            # Check if it's ID-like
            if unique_count > non_null_count * 0.9:
                column_types['id_like'].append(col)
            # Check if it's boolean-like
            elif unique_count == 2:
                column_types['boolean'].append(col)
            # Check if it's categorical
            elif unique_count <= 50:
                column_types['categorical'].append(col)
            else:
                column_types['text'].append(col)
    
    return column_types

def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data profile information
    """
    
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'duplicated_rows': df.duplicated().sum(),
        },
        'column_types': detect_column_types(df),
        'missing_summary': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'complete_columns': df.columns[~df.isnull().any()].tolist()
        },
        'data_quality_flags': {
            'high_missing_columns': [col for col in df.columns if (df[col].isnull().sum() / len(df)) > 0.5],
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [col for col in df.columns if df[col].nunique() > len(df) * 0.8],
        }
    }
    
    return profile

def suggest_analysis_approach(df: pd.DataFrame, target_column: str = None) -> Dict[str, List[str]]:
    """
    Suggest analysis approaches based on data characteristics.
    
    Args:
        df: Input DataFrame
        target_column: Optional target column for supervised learning
        
    Returns:
        Dictionary with analysis suggestions
    """
    
    column_types = detect_column_types(df)
    profile = generate_data_profile(df)
    
    suggestions = {
        'descriptive_analysis': [],
        'visualization_recommendations': [],
        'modeling_suggestions': [],
        'preprocessing_steps': []
    }
    
    # Descriptive analysis suggestions
    if column_types['numerical']:
        suggestions['descriptive_analysis'].append("Statistical summary for numerical variables")
        suggestions['descriptive_analysis'].append("Distribution analysis and normality testing")
    
    if column_types['categorical']:
        suggestions['descriptive_analysis'].append("Frequency analysis for categorical variables")
        suggestions['descriptive_analysis'].append("Chi-square tests for independence")
    
    if len(column_types['numerical']) > 1:
        suggestions['descriptive_analysis'].append("Correlation analysis between numerical variables")
    
    # Visualization recommendations
    if column_types['numerical']:
        suggestions['visualization_recommendations'].extend([
            "Histograms and box plots for distributions",
            "Scatter plots for relationships"
        ])
    
    if column_types['categorical']:
        suggestions['visualization_recommendations'].append("Bar charts for categorical distributions")
    
    if column_types['datetime']:
        suggestions['visualization_recommendations'].append("Time series plots for temporal patterns")
    
    # Modeling suggestions
    if target_column:
        if target_column in column_types['categorical']:
            suggestions['modeling_suggestions'].append("Classification algorithms (Random Forest, Logistic Regression)")
        elif target_column in column_types['numerical']:
            suggestions['modeling_suggestions'].append("Regression algorithms (Linear Regression, Random Forest)")
    else:
        suggestions['modeling_suggestions'].extend([
            "Clustering analysis for pattern discovery",
            "Principal Component Analysis for dimensionality reduction"
        ])
    
    # Preprocessing suggestions
    if profile['missing_summary']['total_missing'] > 0:
        suggestions['preprocessing_steps'].append("Handle missing values through imputation or removal")
    
    if profile['basic_info']['duplicated_rows'] > 0:
        suggestions['preprocessing_steps'].append("Remove or investigate duplicate rows")
    
    if column_types['categorical']:
        suggestions['preprocessing_steps'].append("Encode categorical variables for modeling")
    
    if len(column_types['numerical']) > 0:
        suggestions['preprocessing_steps'].append("Consider feature scaling for numerical variables")
    
    return suggestions