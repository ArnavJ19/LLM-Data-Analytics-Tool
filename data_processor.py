import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def safe_round(value, decimals=4):
    """Safely round a value, handling numpy arrays and various data types."""
    if value is None:
        return None
    
    # Handle numpy arrays - take the first element if it's an array
    if isinstance(value, np.ndarray):
        if value.size > 0:
            value = value.item() if value.size == 1 else value[0]
        else:
            return None
    
    # Handle pandas objects
    if hasattr(value, 'iloc'):
        value = value.iloc[0] if len(value) > 0 else None
    
    # Handle NaN values
    if pd.isna(value) or np.isnan(value):
        return None
    
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError, OverflowError):
        return None

class DataProcessor:
    """
    Handles data loading, preprocessing, and exploratory data analysis.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame or None if loading fails
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            elif file_extension == 'json':
                return self._load_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection."""
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                
                # If successful and has reasonable shape, return
                if len(df.columns) > 1 or len(df) > 0:
                    return df
            except:
                continue
        
        # If all encodings fail, try with error handling
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
    
    def _load_excel(self, uploaded_file) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(uploaded_file)
    
    def _load_json(self, uploaded_file) -> pd.DataFrame:
        """Load JSON file."""
        uploaded_file.seek(0)
        data = json.load(uploaded_file)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to normalize nested JSON
            return pd.json_normalize(data)
        else:
            raise ValueError("Invalid JSON structure for DataFrame conversion")
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive column information.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with column information
        """
        info_data = []
        
        for col in df.columns:
            col_info = {
                'Column': col,
                'Data_Type': str(df[col].dtype),
                'Non_Null_Count': df[col].count(),
                'Null_Count': df[col].isnull().sum(),
                'Null_Percentage': safe_round((df[col].isnull().sum() / len(df)) * 100, 2),
                'Unique_Values': df[col].nunique(),
                'Memory_Usage_KB': safe_round(df[col].memory_usage(deep=True) / 1024, 2)
            }
            
            # Add type-specific information
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'Min_Value': safe_round(df[col].min(), 4) if df[col].count() > 0 else None,
                    'Max_Value': safe_round(df[col].max(), 4) if df[col].count() > 0 else None,
                    'Mean_Value': safe_round(df[col].mean(), 2) if df[col].count() > 0 else None
                })
            else:
                col_info.update({
                    'Min_Value': None,
                    'Max_Value': None,
                    'Mean_Value': None
                })
            
            info_data.append(col_info)
        
        return pd.DataFrame(info_data)
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality check results
        """
        quality_checks = {
            'duplicates': df.duplicated().sum(),
            'total_missing': df.isnull().sum().sum(),
            'high_missing_cols': [],
            'constant_cols': [],
            'high_cardinality_cols': []
        }
        
        # Check for columns with high missing percentage
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                quality_checks['high_missing_cols'].append(col)
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                quality_checks['constant_cols'].append(col)
        
        # Check for high cardinality columns
        for col in df.columns:
            if df[col].nunique() > len(df) * 0.8:
                quality_checks['high_cardinality_cols'].append(col)
        
        return quality_checks
    
    def generate_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive exploratory data analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing EDA results
        """
        eda_results = {
            'numerical_summary': pd.DataFrame(),
            'categorical_summary': pd.DataFrame(),
            'outliers': {},
            'correlations': None,
            'missing_patterns': {},
            'distribution_stats': {}
        }
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical summary
        if numerical_cols:
            eda_results['numerical_summary'] = self._generate_numerical_summary(df[numerical_cols])
            eda_results['outliers'] = self._detect_outliers(df[numerical_cols])
            eda_results['correlations'] = df[numerical_cols].corr()
        
        # Categorical summary
        if categorical_cols:
            eda_results['categorical_summary'] = self._generate_categorical_summary(df[categorical_cols])
        
        # Missing value patterns
        eda_results['missing_patterns'] = self._analyze_missing_patterns(df)
        
        # Distribution statistics
        eda_results['distribution_stats'] = self._calculate_distribution_stats(df)
        
        return eda_results
    
    def _generate_numerical_summary(self, df_num: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive numerical summary statistics."""
        summary_stats = []
        
        for col in df_num.columns:
            series = df_num[col].dropna()
            
            if len(series) == 0:
                continue
            
            stats_dict = {
                'Column': col,
                'Count': len(series),
                'Mean': safe_round(series.mean(), 4),
                'Median': safe_round(series.median(), 4),
                'Std': safe_round(series.std(), 4),
                'Min': safe_round(series.min(), 4),
                'Q1': safe_round(series.quantile(0.25), 4),
                'Q3': safe_round(series.quantile(0.75), 4),
                'Max': safe_round(series.max(), 4),
                'IQR': safe_round(series.quantile(0.75) - series.quantile(0.25), 4),
                'Skewness': safe_round(series.skew(), 4),
                'Kurtosis': safe_round(series.kurtosis(), 4),
                'CV': safe_round((series.std() / series.mean()) * 100, 2) if series.mean() != 0 else 0
            }
            
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def _generate_categorical_summary(self, df_cat: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical summary statistics."""
        summary_stats = []
        
        for col in df_cat.columns:
            series = df_cat[col].dropna()
            
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            stats_dict = {
                'Column': col,
                'Count': len(series),
                'Unique_Values': series.nunique(),
                'Most_Frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'Most_Frequent_Count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'Most_Frequent_Pct': safe_round((value_counts.iloc[0] / len(series)) * 100, 2) if len(value_counts) > 0 else None,
                'Least_Frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'Least_Frequent_Count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else None
            }
            
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def _detect_outliers(self, df_num: pd.DataFrame, methods: List[str] = ['iqr', 'zscore']) -> Dict[str, Dict]:
        """
        Detect outliers using multiple methods.
        
        Args:
            df_num: Numerical DataFrame
            methods: List of methods to use ['iqr', 'zscore']
            
        Returns:
            Dictionary with outlier information for each column
        """
        outliers = {}
        
        for col in df_num.columns:
            series = df_num[col].dropna()
            
            if len(series) == 0:
                continue
            
            outlier_indices = set()
            outlier_methods = []
            
            # IQR method
            if 'iqr' in methods:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)].index
                outlier_indices.update(iqr_outliers)
                if len(iqr_outliers) > 0:
                    outlier_methods.append('IQR')
            
            # Z-score method
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = series[z_scores > 3].index
                outlier_indices.update(zscore_outliers)
                if len(zscore_outliers) > 0:
                    outlier_methods.append('Z-Score')
            
            if outlier_indices:
                outliers[col] = {
                    'indices': list(outlier_indices),
                    'values': series.loc[list(outlier_indices)].tolist(),
                    'methods': outlier_methods,
                    'count': len(outlier_indices),
                    'percentage': safe_round((len(outlier_indices) / len(series)) * 100, 2)
                }
        
        return outliers
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'complete_rows': len(df) - df.isnull().any(axis=1).sum()
        }
        
        return missing_info
    
    def _calculate_distribution_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distribution statistics for numerical columns."""
        distribution_stats = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Test for normality (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            if len(series) <= 5000:
                try:
                    stat, p_value = stats.shapiro(series)
                    normality_test = 'Shapiro-Wilk'
                except:
                    stat, p_value = 0, 1
                    normality_test = 'Unable to test'
            else:
                try:
                    result = stats.anderson(series, dist='norm')
                    stat = result.statistic
                    # Anderson-Darling returns critical values, not p-value
                    # Use a simple approximation for p-value
                    p_value = 0.05 if stat > result.critical_values[2] else 0.1
                    normality_test = 'Anderson-Darling'
                except:
                    stat, p_value = 0, 1
                    normality_test = 'Unable to test'
            
            distribution_stats[col] = {
                'normality_test': normality_test,
                'normality_statistic': safe_round(stat, 4),
                'normality_p_value': safe_round(p_value, 4),
                'is_normal': (safe_round(p_value, 4) or 0) > 0.05,
                'zeros_count': int((series == 0).sum()),
                'zeros_percentage': safe_round(((series == 0).sum() / len(series)) * 100, 2)
            }
        
        return distribution_stats
    
    def suggest_data_cleaning(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Suggest data cleaning operations based on EDA results.
        
        Args:
            df: Input DataFrame
            eda_results: EDA results from generate_eda
            
        Returns:
            Dictionary with cleaning suggestions
        """
        suggestions = {
            'missing_values': [],
            'outliers': [],
            'duplicates': [],
            'encoding': [],
            'data_types': []
        }
        
        # Missing value suggestions
        missing_info = eda_results['missing_patterns']
        for col, missing_count in missing_info['missing_by_column'].items():
            if missing_count > 0:
                missing_pct = missing_info['missing_percentage'][col]
                
                if missing_pct > 50:
                    suggestions['missing_values'].append(f"Consider dropping column '{col}' - {missing_pct:.1f}% missing")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    suggestions['missing_values'].append(f"Impute '{col}' with median/mean - {missing_pct:.1f}% missing")
                else:
                    suggestions['missing_values'].append(f"Impute '{col}' with mode/forward fill - {missing_pct:.1f}% missing")
        
        # Outlier suggestions
        if eda_results['outliers']:
            for col, outlier_info in eda_results['outliers'].items():
                if outlier_info['percentage'] > 5:
                    suggestions['outliers'].append(f"Column '{col}' has {outlier_info['percentage']:.1f}% outliers - consider investigation")
                else:
                    suggestions['outliers'].append(f"Column '{col}' has {outlier_info['count']} outliers - consider capping/transformation")
        
        # Duplicate suggestions
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            suggestions['duplicates'].append(f"Remove {duplicate_count} duplicate rows")
        
        # Encoding suggestions
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > 10:
                suggestions['encoding'].append(f"Consider target encoding for high-cardinality column '{col}' ({unique_count} unique values)")
            else:
                suggestions['encoding'].append(f"Consider one-hot encoding for column '{col}' ({unique_count} categories)")
        
        # Data type suggestions
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    suggestions['data_types'].append(f"Convert '{col}' to numeric type")
                except:
                    # Check if it's datetime
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        suggestions['data_types'].append(f"Convert '{col}' to datetime type")
                    except:
                        pass
        
        return suggestions