import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from config import get_config
from prompt_templates import PromptTemplates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """Safely serialize object to JSON, handling numpy types."""
    def convert_types(item):
        if isinstance(item, np.integer):
            return int(item)
        elif isinstance(item, np.floating):
            return float(item)
        elif isinstance(item, dict):
            return {k: convert_types(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_types(i) for i in item]
        return item
    
    return json.dumps(convert_types(obj), cls=NumpyEncoder, **kwargs)

class LLMInsightGenerator:
    """
    Handles LLM-powered insight generation using OLLAMA with configurable models.
    """
    
    def __init__(self, config_override: Dict[str, Any] = None):
        """
        Initialize the LLM Insight Generator.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        self.llm_config = self.config.llm
        
        # Apply any configuration overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.llm_config, key):
                    setattr(self.llm_config, key, value)
        
        self.api_url = f"{self.llm_config.base_url}/api/generate"
        self.prompt_templates = PromptTemplates()
        
        # Test connection
        self.is_available = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if OLLAMA is available and model is loaded."""
        try:
            response = requests.get(f"{self.llm_config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                # Check primary model
                if any(self.llm_config.model_name in model for model in available_models):
                    logger.info(f"✅ OLLAMA connection successful. Model {self.llm_config.model_name} is available.")
                    return True
                
                # Check alternative models
                for alt_model in self.llm_config.alternative_models:
                    if any(alt_model in model for model in available_models):
                        logger.info(f"✅ Using alternative model: {alt_model}")
                        self.llm_config.model_name = alt_model
                        return True
                
                logger.warning(f"⚠️ Model {self.llm_config.model_name} not found. Available models: {available_models}")
                return False
            else:
                logger.error(f"❌ OLLAMA server responded with status {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            logger.error(f"❌ Failed to connect to OLLAMA: Connection timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Failed to connect to OLLAMA: Connection refused or server not running")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to OLLAMA: {str(e)}")
            return False
            
    def _check_server_health(self) -> bool:
        """Check if OLLAMA server is healthy before making a request."""
        try:
            response = requests.get(f"{self.llm_config.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            self.is_available = False
            return False
    
    def _generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate response from OLLAMA model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (uses config default if None)
            
        Returns:
            Generated response text
        """
        if not self.is_available:
            return "❌ OLLAMA is not available. Please ensure OLLAMA is running and the model is loaded."
            
        # Check server health before making request
        if not self._check_server_health():
            return "❌ OLLAMA server is not responding. Please check if the server is running properly."
        
        if max_tokens is None:
            max_tokens = self.llm_config.max_tokens
        
        try:
            # Add system prompt
            full_prompt = f"{self.prompt_templates.get_system_prompt()}\n\n{prompt}"
            
            payload = {
                "model": self.llm_config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": self.llm_config.temperature,
                    "top_p": self.llm_config.top_p
                }
            }
            
            # Implement retry logic with exponential backoff
            max_retries = 3
            retry_delay = 2  # Initial delay in seconds
            
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        json=payload,
                        timeout=self.llm_config.timeout
                    )
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout:
                    if retry < max_retries - 1:  # Don't sleep on the last retry
                        logger.warning(f"Request timed out, retrying ({retry+1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise the last timeout exception
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"OLLAMA API error: {response.status_code}")
                return f"❌ API Error: {response.status_code}"
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error connecting to OLLAMA: {str(e)}")
            return f"❌ Timeout Error: The request to the LLM server timed out after {self.llm_config.timeout} seconds. Please ensure Ollama is running properly and not overloaded."
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to OLLAMA: {str(e)}")
            return f"❌ Connection Error: Could not connect to the LLM server at {self.llm_config.base_url}. Please ensure Ollama is running."
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def generate_comprehensive_insights(self, df: pd.DataFrame, 
                                      eda_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate comprehensive insights about the dataset.
        
        Args:
            df: Input DataFrame
            eda_results: EDA results from DataProcessor
            
        Returns:
            Dictionary with different types of insights
        """
        insights = {}
        
        # Dataset overview
        insights['dataset_summary'] = self._generate_dataset_summary(df, eda_results)
        
        # Statistical insights
        insights['statistical_insights'] = self._generate_statistical_insights(df, eda_results)
        
        # Data quality insights
        insights['data_quality'] = self._generate_data_quality_insights(df, eda_results)
        
        # Correlation insights
        if eda_results.get('correlations') is not None:
            insights['correlation_insights'] = self._generate_correlation_insights(eda_results['correlations'])
        
        # Distribution insights
        insights['distribution_insights'] = self._generate_distribution_insights(df, eda_results)
        
        # Outlier insights
        if eda_results.get('outliers'):
            insights['outlier_insights'] = self._generate_outlier_insights(eda_results['outliers'])
        
        return insights
    
    def _generate_dataset_summary(self, df: pd.DataFrame, 
                                eda_results: Dict[str, Any]) -> str:
        """Generate a high-level summary of the dataset."""
        
        # Prepare dataset metadata
        num_rows, num_cols = df.shape
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_percentage = (df.isnull().sum().sum() / (num_rows * num_cols)) * 100
        
        # Add data sample
        data_sample = df.head(5).to_string()
        
        # Add unique values for categorical columns
        categorical_values = {}
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) <= 20:  # Only include if not too many unique values
                categorical_values[col] = unique_vals
            else:
                # For high cardinality columns, show a sample
                categorical_values[col] = unique_vals[:20] + ["..."] 
        
        categorical_info = "\n".join([f"{col}: {vals}" for col, vals in categorical_values.items()])
        
        prompt = self.prompt_templates.dataset_summary_prompt(
            num_rows=num_rows,
            num_cols=num_cols,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            missing_percentage=missing_percentage,
            column_names=df.columns.tolist()
        )
        
        # Add data sample and categorical values to the prompt
        prompt += f"""

**Data Sample (First 5 rows):**
{data_sample}

**Unique Values in Categorical Columns:**
{categorical_info}

Make sure to reference the ACTUAL data values shown above in your analysis.
"""
        
        return self._generate_response(prompt, max_tokens=400)
    
    def _generate_statistical_insights(self, df: pd.DataFrame, 
                                     eda_results: Dict[str, Any]) -> str:
        """Generate insights about statistical properties."""
        
        numerical_summary = eda_results.get('numerical_summary', pd.DataFrame())
        
        if numerical_summary.empty:
            return "No numerical variables found for statistical analysis."
        
        # Get key statistics
        high_variance_cols = []
        skewed_cols = []
        
        for _, row in numerical_summary.iterrows():
            cv = row.get('CV', 0)
            skewness = row.get('Skewness', 0)
            
            if cv > 100:  # High coefficient of variation
                high_variance_cols.append(f"{row['Column']} (CV: {cv:.1f}%)")
            
            if abs(skewness) > 1:  # Highly skewed
                skewed_cols.append(f"{row['Column']} (Skew: {skewness:.2f})")
        
        prompt = self.prompt_templates.statistical_insights_prompt(
            numerical_summary_df=numerical_summary,
            high_variance_cols=high_variance_cols,
            skewed_cols=skewed_cols
        )
        
        return self._generate_response(prompt, max_tokens=300)
    
    def _generate_data_quality_insights(self, df: pd.DataFrame, 
                                      eda_results: Dict[str, Any]) -> str:
        """Generate insights about data quality."""
        
        missing_patterns = eda_results.get('missing_patterns', {})
        quality_issues = []
        
        # Check for quality issues
        if missing_patterns.get('total_missing', 0) > 0:
            quality_issues.append(f"Missing values: {missing_patterns['total_missing']:,} total")
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"Duplicate rows: {duplicate_count}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_issues.append(f"Constant columns: {', '.join(constant_cols)}")
        
        prompt = f"""
        Assess the data quality of this dataset:

        **Missing Value Patterns:**
        - Total missing values: {missing_patterns.get('total_missing', 0):,}
        - Rows with missing data: {missing_patterns.get('rows_with_missing', 0):,}
        - Complete rows: {missing_patterns.get('complete_rows', 0):,}

        **Missing by Column:**
        {json.dumps(missing_patterns.get('missing_by_column', {}), indent=2)}

        **Quality Issues Identified:**
        {chr(10).join(f"- {issue}" for issue in quality_issues) if quality_issues else "- No major quality issues detected"}

        Please provide:
        1. Overall data quality assessment
        2. Impact of missing data on analysis
        3. Recommended data cleaning strategies
        4. Priority order for addressing issues

        Keep response focused and actionable, around 150-200 words.
        """
        
        return self._generate_response(prompt, max_tokens=300)
    
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame) -> str:
        """Generate insights about variable correlations."""
        
        # Find strong correlations
        strong_positive = []
        strong_negative = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    if corr_val > 0:
                        strong_positive.append(f"{var1} ↔ {var2} ({corr_val:.3f})")
                    else:
                        strong_negative.append(f"{var1} ↔ {var2} ({corr_val:.3f})")
        
        prompt = f"""
        Analyze the correlation patterns in this dataset:

        **Strong Positive Correlations (>0.7):**
        {chr(10).join(f"- {corr}" for corr in strong_positive) if strong_positive else "- None found"}

        **Strong Negative Correlations (<-0.7):**
        {chr(10).join(f"- {corr}" for corr in strong_negative) if strong_negative else "- None found"}

        **Correlation Matrix:**
        {corr_matrix.round(3).to_string()}

        Please provide insights about:
        1. Most significant relationships between variables
        2. Potential multicollinearity concerns
        3. Unexpected or interesting correlations
        4. Implications for feature selection and modeling

        Focus on business/analytical implications in 150-200 words.
        """
        
        return self._generate_response(prompt, max_tokens=300)
    
    def _generate_distribution_insights(self, df: pd.DataFrame, 
                                      eda_results: Dict[str, Any]) -> str:
        """Generate insights about data distributions."""
        
        distribution_stats = eda_results.get('distribution_stats', {})
        
        normal_vars = []
        non_normal_vars = []
        zero_heavy_vars = []
        
        for var, stats in distribution_stats.items():
            if stats.get('is_normal', False):
                normal_vars.append(var)
            else:
                non_normal_vars.append(var)
            
            if stats.get('zeros_percentage', 0) > 20:
                zero_heavy_vars.append(f"{var} ({stats['zeros_percentage']:.1f}% zeros)")
        
        prompt = f"""
        Analyze the distribution characteristics of the numerical variables:

        **Distribution Analysis:**
        - Variables with normal distribution: {', '.join(normal_vars) if normal_vars else 'None'}
        - Variables with non-normal distribution: {', '.join(non_normal_vars) if non_normal_vars else 'None'}
        - Variables with high zero content: {', '.join(zero_heavy_vars) if zero_heavy_vars else 'None'}

        **Detailed Distribution Stats:**
        {safe_json_dumps(distribution_stats, indent=2)}

        Please provide insights about:
        1. Overall distribution patterns and their implications
        2. Variables that may need transformation
        3. Potential data generation processes
        4. Recommended analytical approaches based on distributions

        Keep response analytical and practical, around 150-200 words.
        """
        
        return self._generate_response(prompt, max_tokens=300)
    
    def _generate_outlier_insights(self, outliers_dict: Dict[str, Any]) -> str:
        """Generate insights about outliers."""
        
        outlier_summary = []
        for var, info in outliers_dict.items():
            outlier_summary.append(
                f"- {var}: {info['count']} outliers ({info['percentage']:.1f}%) detected by {', '.join(info['methods'])}"
            )
        
        prompt = f"""
        Analyze the outlier patterns in this dataset:

        **Outlier Summary:**
        {chr(10).join(outlier_summary)}

        **Detailed Outlier Information:**
        {safe_json_dumps({k: {key: val for key, val in v.items() if key != 'values'} 
                    for k, v in outliers_dict.items()}, indent=2)}

        Please provide insights about:
        1. Variables most affected by outliers
        2. Potential causes of these outliers
        3. Impact on analysis and modeling
        4. Recommended outlier treatment strategies

        Focus on practical recommendations in 150-200 words.
        """
        
        return self._generate_response(prompt, max_tokens=1000)
    
    def answer_data_question(self, question: str, df: pd.DataFrame, eda_results: Dict[str, Any]) -> str:
        """
        Answer a user's question about the data using the LLM.
        
        Args:
            question: User's question about the data
            df: Input DataFrame
            eda_results: EDA results
            
        Returns:
            Answer to the question
        """
        
        # Prepare context about the data
        context = self._prepare_data_context(df, eda_results)
        
        prompt = f"""
        You are a data analyst answering questions about a dataset. Use the provided data context to answer the user's question accurately and insightfully.

        **Dataset Context:**
        {context}

        **User Question:**
        {question}

        **Instructions:**
        1. Answer the question directly and specifically using the ACTUAL DATA provided in the context
        2. ALWAYS refer to the specific data points shown in the data sample and unique values sections
        3. Use concrete numbers, exact values, and specific examples from the dataset
        4. NEVER make up or generalize data - only use what is explicitly shown in the context
        5. If asked about specific items (like cities, products, categories), list the ACTUAL values from the dataset
        6. If the question cannot be fully answered with available data, explain what additional data would be needed
        7. Keep the response focused and practical

        Provide a comprehensive but concise answer (150-250 words) that references SPECIFIC data points from the provided context.
        """
        
        return self._generate_response(prompt, max_tokens=500)
    
    def _prepare_data_context(self, df: pd.DataFrame, eda_results: Dict[str, Any]) -> str:
        """Prepare a comprehensive context about the dataset for question answering."""
        
        # Basic info
        num_rows, num_cols = df.shape
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Key statistics
        context_parts = [
            f"Dataset: {num_rows:,} rows × {num_cols} columns",
            f"Numerical columns: {', '.join(numerical_cols)}",
            f"Categorical columns: {', '.join(categorical_cols)}"
        ]
        
        if datetime_cols:
            context_parts.append(f"Datetime columns: {', '.join(datetime_cols)}")
        
        # Add summary statistics for numerical columns
        if not eda_results.get('numerical_summary', pd.DataFrame()).empty:
            num_summary = eda_results['numerical_summary']
            context_parts.append(f"Key statistics available for: {', '.join(num_summary['Column'].tolist())}")
        
        # Add correlation info
        if eda_results.get('correlations') is not None:
            context_parts.append("Correlation analysis available between numerical variables")
        
        # Add missing value info
        missing_info = eda_results.get('missing_patterns', {})
        if missing_info.get('total_missing', 0) > 0:
            context_parts.append(f"Missing values: {missing_info['total_missing']} total")
        
        # Add actual data samples
        context_parts.append("\n\nData Sample (first 10 rows):")
        context_parts.append(df.head(10).to_string())
        
        # Add unique values for categorical columns (limited to avoid context explosion)
        context_parts.append("\n\nUnique Values in Categorical Columns:")
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            unique_values = df[col].dropna().unique().tolist()
            if len(unique_values) <= 50:  # Only include if not too many unique values
                context_parts.append(f"{col}: {unique_values}")
            else:
                # For high cardinality columns, show a sample of unique values
                context_parts.append(f"{col}: {unique_values[:50]} ... (and {len(unique_values)-50} more unique values)")
        
        # Add summary statistics in more detail
        if not eda_results.get('numerical_summary', pd.DataFrame()).empty:
            context_parts.append("\n\nNumerical Summary Statistics:")
            context_parts.append(eda_results['numerical_summary'].to_string())
        
        if not eda_results.get('categorical_summary', pd.DataFrame()).empty:
            context_parts.append("\n\nCategorical Summary Statistics:")
            context_parts.append(eda_results['categorical_summary'].to_string())
        
        return "\n".join(context_parts)
    
    def generate_visualization_insights(self, visualization_type: str, 
                                      column_names: List[str], 
                                      data_summary: str) -> str:
        """
        Generate insights about a specific visualization.
        
        Args:
            visualization_type: Type of visualization
            column_names: Columns used in the visualization
            data_summary: Summary of the data shown
            
        Returns:
            Insights about the visualization
        """
        
        prompt = f"""
        Provide insights about this data visualization:

        **Visualization Type:** {visualization_type}
        **Columns Analyzed:** {', '.join(column_names)}
        **Data Summary:** {data_summary}

        Please explain:
        1. What the visualization reveals about the data
        2. Key patterns or trends visible
        3. Potential business or analytical implications
        4. Any anomalies or interesting observations

        Keep the explanation clear and actionable (100-150 words).
        """
        
        return self._generate_response(prompt, max_tokens=250)