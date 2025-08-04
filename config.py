"""
Configuration settings for the Data Analytics Web Application.

This module contains all configurable parameters for the application,
making it easy to customize behavior without modifying core code.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    # OLLAMA settings
    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout: int = 480
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Alternative models (if available)
    alternative_models: List[str] = None
    
    def __post_init__(self):
        if self.alternative_models is None:
            self.alternative_models = [
                "qwen2.5:7b",
                "llama3.1:8b", 
                "mistral:7b",
                "phi3:mini"
            ]

@dataclass
class DataProcessingConfig:
    """Configuration for data processing and analysis."""
    # File upload limits
    max_file_size_mb: int = 100
    max_rows_preview: int = 100
    max_columns_analysis: int = 50
    
    # EDA settings
    outlier_methods: List[str] = None
    correlation_threshold: float = 0.5
    missing_threshold: float = 0.5  # Flag columns with >50% missing
    high_cardinality_threshold: float = 0.8  # Flag if unique values > 80% of rows
    
    # Statistical settings
    normality_test_sample_size: int = 5000  # Max sample for normality tests
    confidence_level: float = 0.05  # For statistical tests
    
    def __post_init__(self):
        if self.outlier_methods is None:
            self.outlier_methods = ["iqr", "zscore"]

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    # Plotly settings
    default_template: str = "plotly_white"
    color_palette: List[str] = None
    
    # Chart limits (to prevent performance issues)
    max_categories_bar_chart: int = 20
    max_variables_correlation: int = 20
    max_variables_scatter_matrix: int = 5
    
    # Chart dimensions
    default_height: int = 400
    default_width: int = 600
    heatmap_size: int = 600
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]

@dataclass
class UIConfig:
    """Configuration for user interface."""
    # Page settings
    page_title: str = "AI-Powered Data Analytics Tool"
    page_icon: str = "📊"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Theme colors
    primary_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    secondary_color: str = "#f0f2f6"
    
    # Component settings
    enable_chat: bool = True
    enable_export: bool = True
    enable_advanced_settings: bool = True
    
    # Performance settings
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    max_concurrent_users: int = 10

@dataclass
class ExportConfig:
    """Configuration for export functionality."""
    # Export formats
    supported_formats: List[str] = None
    default_format: str = "markdown"
    
    # Report settings
    include_visualizations: bool = True
    include_raw_data: bool = False
    max_report_size_mb: int = 50
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    report_prefix: str = "data_analysis_report"
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["markdown", "html", "json"]

class AppConfig:
    """Main application configuration."""
    
    def __init__(self):
        # Load configurations
        self.llm = LLMConfig()
        self.data_processing = DataProcessingConfig()
        self.visualization = VisualizationConfig()
        self.ui = UIConfig()
        self.export = ExportConfig()
        
        # Environment-specific overrides
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        
        # LLM overrides
        if os.getenv("OLLAMA_MODEL"):
            self.llm.model_name = os.getenv("OLLAMA_MODEL")
        
        if os.getenv("OLLAMA_BASE_URL"):
            self.llm.base_url = os.getenv("OLLAMA_BASE_URL")
        
        if os.getenv("OLLAMA_TIMEOUT"):
            self.llm.timeout = int(os.getenv("OLLAMA_TIMEOUT"))
        
        # Data processing overrides
        if os.getenv("MAX_FILE_SIZE_MB"):
            self.data_processing.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))
        
        if os.getenv("CORRELATION_THRESHOLD"):
            self.data_processing.correlation_threshold = float(os.getenv("CORRELATION_THRESHOLD"))
        
        # UI overrides
        if os.getenv("DISABLE_CHAT"):
            self.ui.enable_chat = os.getenv("DISABLE_CHAT").lower() != "true"
        
        if os.getenv("DISABLE_EXPORT"):
            self.ui.enable_export = os.getenv("DISABLE_EXPORT").lower() != "true"
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types for upload."""
        return ['csv', 'xlsx', 'xls', 'json']
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.data_processing.max_file_size_mb * 1024 * 1024
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_flags = {
            'chat': self.ui.enable_chat,
            'export': self.ui.enable_export,
            'advanced_settings': self.ui.enable_advanced_settings,
            'llm_insights': True,  # Always available if OLLAMA is running
        }
        return feature_flags.get(feature, False)
    
    def get_chart_config(self, chart_type: str) -> Dict[str, Any]:
        """Get configuration for specific chart types."""
        base_config = {
            'template': self.visualization.default_template,
            'height': self.visualization.default_height,
            'width': self.visualization.default_width,
            'color_discrete_sequence': self.visualization.color_palette
        }
        
        # Chart-specific configurations
        chart_configs = {
            'correlation_heatmap': {
                **base_config,
                'height': self.visualization.heatmap_size,
                'width': self.visualization.heatmap_size
            },
            'bar_chart': {
                **base_config,
                'max_categories': self.visualization.max_categories_bar_chart
            },
            'scatter_matrix': {
                **base_config,
                'max_variables': self.visualization.max_variables_scatter_matrix
            }
        }
        
        return chart_configs.get(chart_type, base_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm': self.llm.__dict__,
            'data_processing': self.data_processing.__dict__,
            'visualization': self.visualization.__dict__,
            'ui': self.ui.__dict__,
            'export': self.export.__dict__
        }

# Global configuration instance
config = AppConfig()

# Environment-specific configurations
class DevelopmentConfig(AppConfig):
    """Development environment configuration."""
    def __init__(self):
        super().__init__()
        self.llm.timeout = 60  # Longer timeout for development
        self.data_processing.max_file_size_mb = 50  # Smaller for dev
        self.ui.cache_ttl = 60  # Shorter cache for development

class ProductionConfig(AppConfig):
    """Production environment configuration."""
    def __init__(self):
        super().__init__()
        self.llm.timeout = 30  # Strict timeout for production
        self.data_processing.max_file_size_mb = 100
        self.ui.cache_ttl = 3600  # Longer cache for production
        self.ui.enable_advanced_settings = False  # Hide advanced settings

def get_config() -> AppConfig:
    """Get configuration based on environment."""
    env = os.getenv('APP_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'development':
        return DevelopmentConfig()
    else:
        return AppConfig()

# Configuration validation
def validate_config(config: AppConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Validate LLM config
    if not config.llm.model_name:
        issues.append("LLM model name is required")
    
    if not config.llm.base_url:
        issues.append("LLM base URL is required")
    
    if config.llm.temperature < 0 or config.llm.temperature > 1:
        issues.append("LLM temperature must be between 0 and 1")
    
    # Validate data processing config
    if config.data_processing.max_file_size_mb <= 0:
        issues.append("Maximum file size must be positive")
    
    if config.data_processing.correlation_threshold < 0 or config.data_processing.correlation_threshold > 1:
        issues.append("Correlation threshold must be between 0 and 1")
    
    # Validate visualization config
    if config.visualization.default_height <= 0:
        issues.append("Chart height must be positive")
    
    if config.visualization.default_width <= 0:
        issues.append("Chart width must be positive")
    
    return issues