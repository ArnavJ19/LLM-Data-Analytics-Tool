import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import io
import base64
from pathlib import Path
import plotly.express as px

# Import custom modules
from data_processor import DataProcessor
from visualizer import DataVisualizer
from llm_insights import LLMInsightGenerator
from predictive_analytics import PredictiveAnalytics
from utils import export_report, create_download_link

# Page configuration
st.set_page_config(
    page_title="Grekko: AI-Powered Data Analytics Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI and readability
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Card styling for metrics and insights */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Insight box styling */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e3f2fd;
        border-radius: 1rem 1rem 0.2rem 1rem;
        padding: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .ai-message {
        background-color: #f1f3f4;
        border-radius: 1rem 1rem 1rem 0.2rem;
        padding: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 2px solid #3498db;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-container, .insight-box {
            padding: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyticsApp:
    def __init__(self):
        """Initialize the Data Analytics Application."""
        self.data_processor = DataProcessor()
        self.visualizer = DataVisualizer()
        self.llm_generator = LLMInsightGenerator()
        self.predictive_analytics = PredictiveAnalytics()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        
        
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'file_uploaded' not in st.session_state:
            st.session_state.file_uploaded = False
        if 'eda_results' not in st.session_state:
            st.session_state.eda_results = None
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = {}
        if 'llm_insights' not in st.session_state:
            st.session_state.llm_insights = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'predictive_model' not in st.session_state:
            st.session_state.predictive_model = None
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = None
            
        # Debug after initialization
        if 'model_visualizations' not in st.session_state:
            st.session_state.model_visualizations = {}
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'feature_importance' not in st.session_state:
            st.session_state.feature_importance = None
    
    def run(self):
        """Main application runner."""
        self.render_header()
        self.render_sidebar()
        
        
        # Create tabs regardless of file upload status
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Dataset Overview", 
            "📈 Auto-EDA", 
            "📊 Visualizations",  
            "🔮 Predictive Analytics",
            "🤖 AI Insights",
            "💬 Data Chat"
        ])
        
        # Show file upload section or content based on file_uploaded state
        if not st.session_state.file_uploaded:
            # Show upload section in all tabs with unique keys
            tab_names = ["overview", "eda", "viz","predictive","insights","chat"]
            for i, tab in enumerate([tab1, tab2, tab3, tab4, tab5, tab6]):
                with tab:
                    self.render_upload_section(tab_key=tab_names[i])
        else:
            # Show appropriate content in each tab
            with tab1:
                self.render_dataset_overview()
            with tab2:
                self.render_eda_tab()
            with tab3:
                self.render_visualizations_tab()
            with tab4:
                self.render_predictive_analytics_tab()
            with tab5:
                self.render_insights_tab()
            with tab6:
                self.render_chat_tab()
    
    def render_header(self):
        """Render the application header with improved styling."""
        st.markdown('<h1 class="main-header">Grekko: AI-Powered Data Analytics Tool</h1>', 
                   unsafe_allow_html=True)
        
        # More engaging and descriptive subheader
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div style="font-size: 1.2rem; margin-bottom: 1.5rem; color: #4a5568;">
                Transform your data into actionable insights with AI assistance. 
                Upload your dataset to discover patterns, visualize trends, and get intelligent recommendations.
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # File upload section
            if not st.session_state.file_uploaded:
                st.info("👆 Upload a file to get started")
            else:
                st.success(f"✅ File loaded: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
                
                if st.button("🔄 Upload New File"):
                    self.reset_session_state()
                    st.rerun()
            
            # Analysis settings
            if st.session_state.file_uploaded:
                st.subheader("📊 Analysis Settings")
                
                self.outlier_method = st.selectbox(
                    "Outlier Detection Method",
                    ["IQR", "Z-Score", "Both"],
                    help="Choose method for detecting outliers"
                )
                
                self.correlation_threshold = st.slider(
                    "Correlation Threshold",
                    0.0, 1.0, 0.5,
                    help="Minimum correlation to highlight"
                )
                
                self.enable_llm = st.checkbox(
                    "Enable LLM Insights",
                    value=True,
                    help="Generate AI-powered insights (requires OLLAMA)"
                )
    
    def render_upload_section(self, tab_key="default"):
        """Render the file upload section with improved UI.
        
        Args:
            tab_key (str): A unique key suffix for the file uploader widget to avoid duplicate widget IDs.
        """
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="font-size: 1.8rem; margin-bottom: 1rem;">📁 Upload Your Dataset</h2>
            <p style="color: #6b7280; margin-bottom: 2rem;">Get started by uploading your data file</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a card-like container for file upload
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""<div style="background-color: #f8f9fa; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px dashed #3498db;"></div>""", unsafe_allow_html=True)
            
            # Place the uploader over the styled container
            uploaded_file = st.file_uploader(
                "Drag and drop or click to browse",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Supported formats: CSV, Excel (.xlsx, .xls), JSON",
                key=f"file_uploader_{tab_key}"
            )
            
            # Display supported formats
            st.markdown("""
            <div style="text-align: center; margin-top: 1rem;">
                <p style="color: #6b7280; font-size: 0.9rem;">Supported formats: CSV, Excel (.xlsx, .xls), JSON</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample data option
            st.markdown("<div style='text-align: center; margin: 1.5rem 0 0.5rem 0;'><p>or</p></div>", unsafe_allow_html=True)
            
            if st.button("📊 Use Sample Dataset", help="Load a sample dataset to explore the tool's capabilities", key=f"sample_data_{tab_key}"):
                # Here you would load a sample dataset
                try:
                    from sample_data_generator import SampleDataGenerator
                    generator = SampleDataGenerator()
                    sample_df = generator.generate_sales_data(500)
                    st.session_state.df = sample_df
                    st.session_state.file_uploaded = True
                    st.session_state.eda_results = self.data_processor.generate_eda(sample_df)
                    st.success("✅ Sample dataset loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
            
            if uploaded_file is not None:
                self.process_uploaded_file(uploaded_file)
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file."""
        try:
            with st.spinner("📊 Processing your data..."):
                # Debug information to help troubleshoot the issue
                st.sidebar.markdown("### Processing File Debug Info")
                st.sidebar.write(f"Before processing, file_uploaded: {st.session_state.file_uploaded}")
                
                # Load data
                df = self.data_processor.load_data(uploaded_file)
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.file_uploaded = True
                    
                    # Debug after setting file_uploaded
                    st.sidebar.write(f"After setting, file_uploaded: {st.session_state.file_uploaded}")
                    
                    # Generate initial EDA
                    st.session_state.eda_results = self.data_processor.generate_eda(df)
                    
                    st.success("✅ File processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process the file. Please check the format.")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    def render_main_tabs(self):
        """Legacy method - kept for backward compatibility.
        This method is no longer used as the tab creation has been moved to the run method.
        """
        # Log that this method is deprecated
        st.sidebar.markdown("### ⚠️ Warning: Using deprecated method")
        st.sidebar.write("render_main_tabs is deprecated and should not be called directly.")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Dataset Overview", 
            "📈 Auto-EDA", 
            "📊 Visualizations", 
            "🤖 AI Insights", 
            "🔮 Predictive Analytics",
            "💬 Data Chat"
        ])
        
        # Show appropriate content in each tab
        with tab1:
            self.render_dataset_overview()
        with tab2:
            self.render_eda_tab()
        with tab3:
            self.render_visualizations_tab()
        with tab4:
            self.render_insights_tab()
        with tab5:
            self.render_predictive_analytics_tab()
        with tab6:
            self.render_chat_tab()
    
    def render_dataset_overview(self):
        """Render dataset overview tab with enhanced presentation."""
        st.header("📋 Dataset Overview")
        
        # Introduction to dataset overview
        st.markdown("""<div style="margin-bottom: 1.5rem; color: #4a5568;">
            This overview provides key information about your dataset, including basic statistics, 
            data preview, column details, and quality checks.
        </div>""", unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Basic info with improved styling
        st.markdown("<h3 style='font-size: 1.3rem; margin-bottom: 1rem;'>📊 Key Metrics</h3>", unsafe_allow_html=True)
        
        # Create a card-like container for metrics
        st.markdown("""<style>.metric-row{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;}}.metric-card{{background-color:#f8f9fa;border-radius:8px;padding:15px;box-shadow:0 2px 5px rgba(0,0,0,0.05);flex:1;min-width:150px;transition:transform 0.2s,box-shadow 0.2s;}}.metric-card:hover{{transform:translateY(-2px);box-shadow:0 4px 8px rgba(0,0,0,0.1);}}.metric-title{{font-size:0.9rem;color:#6c757d;margin-bottom:5px;}}.metric-value{{font-size:1.5rem;font-weight:600;color:#3498db;}}</style><div class="metric-row"><div class="metric-card"><div class="metric-title">📊 Rows</div><div class="metric-value">{0:,}</div></div><div class="metric-card"><div class="metric-title">📋 Columns</div><div class="metric-value">{1}</div></div><div class="metric-card"><div class="metric-title">💾 Memory Usage</div><div class="metric-value">{2:.1f} MB</div></div><div class="metric-card"><div class="metric-title">🕳️ Missing Values</div><div class="metric-value">{3:,}</div></div></div>""".format(
            df.shape[0],
            df.shape[1],
            df.memory_usage(deep=True).sum() / 1024**2,
            df.isnull().sum().sum()
        ), unsafe_allow_html=True)
        
        # Data preview with tabs for different views
        st.markdown("<h3 style='font-size: 1.3rem; margin-top: 1.5rem;'>🔍 Data Preview</h3>", unsafe_allow_html=True)
        
        preview_tabs = st.tabs(["📋 First 100 Rows", "📊 Sample Rows", "📉 Last 100 Rows"])
        
        with preview_tabs[0]:
            st.dataframe(df.head(100), use_container_width=True)
        
        with preview_tabs[1]:
            if len(df) > 100:
                st.dataframe(df.sample(min(100, len(df))), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        with preview_tabs[2]:
            st.dataframe(df.tail(100), use_container_width=True)
        
        # Column information with improved presentation
        st.markdown("<h3 style='font-size: 1.3rem; margin-top: 1.5rem;'>📊 Column Information</h3>", unsafe_allow_html=True)
        
        col_info = self.data_processor.get_column_info(df)
        
        # Add search and filter functionality
        col_search = st.text_input("🔍 Search columns", "", help="Filter columns by name or data type")
        
        if col_search:
            try:
                filtered_info = col_info[col_info.index.str.contains(col_search, case=False) | 
                                        col_info['Type'].str.contains(col_search, case=False)]
                st.dataframe(filtered_info, use_container_width=True)
            except AttributeError:
                # Handle case where index is not string type
                st.dataframe(col_info, use_container_width=True)
        else:
            st.dataframe(col_info, use_container_width=True)
        
        # Data quality checks with improved visualization
        st.markdown("<h3 style='font-size: 1.3rem; margin-top: 1.5rem;'>🔍 Data Quality</h3>", unsafe_allow_html=True)
        
        quality_checks = self.data_processor.check_data_quality(df)
        
        # Create quality score
        quality_issues = 0
        max_issues = 2  # Number of potential issues we're checking
        
        if quality_checks['duplicates'] > 0:
            quality_issues += 1
            st.warning(f"⚠️ Found {quality_checks['duplicates']} duplicate rows")
        
        if quality_checks['high_missing_cols']:
            quality_issues += 1
            st.warning(f"⚠️ Columns with >50% missing values: {', '.join(quality_checks['high_missing_cols'])}")
            
            # Add visualization for missing values
            if len(quality_checks['high_missing_cols']) > 0:
                missing_data = pd.DataFrame({
                    'Column': quality_checks['high_missing_cols'],
                    'Missing %': [df[col].isna().mean() * 100 for col in quality_checks['high_missing_cols']]
                })
                
                fig = px.bar(missing_data, x='Column', y='Missing %', 
                           title='Columns with High Missing Values',
                           labels={'Missing %': 'Percentage Missing'},
                           color='Missing %', color_continuous_scale='Reds')
                
                st.plotly_chart(fig, use_container_width=True)
        
        if not quality_checks['duplicates'] and not quality_checks['high_missing_cols']:
            quality_issues = 0
            st.success("✅ No major data quality issues detected")
        
        # Display quality score
        quality_score = 100 - (quality_issues / max_issues * 100)
        
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
            <h4 style="margin-bottom: 0.5rem;">Data Quality Score</h4>
            <div style="font-size: 2rem; font-weight: bold; color: {'#2ecc71' if quality_score > 80 else '#f39c12' if quality_score > 50 else '#e74c3c'};">
                {quality_score:.0f}%
            </div>
            <div style="margin-top: 0.5rem; color: #6c757d;">
                {"Excellent" if quality_score > 80 else "Needs Attention" if quality_score > 50 else "Poor Quality"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_eda_tab(self):
        """Render EDA results tab with enhanced presentation."""
        st.header("📈 Exploratory Data Analysis")
        
        # Introduction to EDA
        st.markdown("""
        <div style="margin-bottom: 1.5rem; color: #4a5568;">
            Explore your data through statistical summaries, distributions, and relationships. 
            This analysis helps you understand patterns, outliers, and correlations in your dataset.
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.eda_results is None:
            with st.spinner("🔍 Generating comprehensive EDA..."):
                st.session_state.eda_results = self.data_processor.generate_eda(
                    st.session_state.df
                )
        
        eda_results = st.session_state.eda_results
        
        # Create tabs for different EDA sections
        eda_tabs = st.tabs(["📊 Summary Statistics", "🎯 Outlier Analysis", "🔗 Correlation Analysis"])
        
        # Tab 1: Summary Statistics
        with eda_tabs[0]:
            # Add filter for numerical/categorical
            summary_type = st.radio(
                "Select variable type:",
                ["Numerical", "Categorical", "Both"],
                horizontal=True,
                key="summary_type"
            )
            
            # Numerical columns summary with improved presentation
            if (summary_type in ["Numerical", "Both"]) and not eda_results['numerical_summary'].empty:
                st.markdown("<h3 style='font-size: 1.2rem;'>🔢 Numerical Variables</h3>", unsafe_allow_html=True)
                
                # Add search functionality
                num_search = st.text_input("🔍 Search numerical variables", "", key="num_search")
                
                if num_search:
                    try:
                        filtered_num = eda_results['numerical_summary'][eda_results['numerical_summary'].index.str.contains(num_search, case=False)]
                        st.dataframe(filtered_num, use_container_width=True)
                    except AttributeError:
                        # Handle case where index is not string type
                        st.dataframe(eda_results['numerical_summary'], use_container_width=True)
                else:
                    st.dataframe(eda_results['numerical_summary'], use_container_width=True)
                
                # Add a summary of key statistics
                if not eda_results['numerical_summary'].empty:
                    num_cols = len(eda_results['numerical_summary'])
                    skewed_cols = sum(abs(eda_results['numerical_summary']['skew']) > 1) if 'skew' in eda_results['numerical_summary'].columns else 0
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                        <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">Quick Summary</h4>
                        <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                            <li>Your dataset has <b>{num_cols}</b> numerical variables</li>
                            <li><b>{skewed_cols}</b> variables show significant skewness (|skew| > 1)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Categorical columns summary with improved presentation
            if (summary_type in ["Categorical", "Both"]) and not eda_results['categorical_summary'].empty:
                st.markdown("<h3 style='font-size: 1.2rem; margin-top: 1.5rem;'>📝 Categorical Variables</h3>", unsafe_allow_html=True)
                
                # Add search functionality
                cat_search = st.text_input("🔍 Search categorical variables", "", key="cat_search")
                
                if cat_search:
                    try:
                        filtered_cat = eda_results['categorical_summary'][eda_results['categorical_summary'].index.str.contains(cat_search, case=False)]
                        st.dataframe(filtered_cat, use_container_width=True)
                    except AttributeError:
                        # Handle case where index is not string type
                        st.dataframe(eda_results['categorical_summary'], use_container_width=True)
                else:
                    st.dataframe(eda_results['categorical_summary'], use_container_width=True)
                
                # Add a summary of key statistics
                if not eda_results['categorical_summary'].empty:
                    cat_cols = len(eda_results['categorical_summary'])
                    high_cardinality = sum(eda_results['categorical_summary']['unique'] > 10) if 'unique' in eda_results['categorical_summary'].columns else 0
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                        <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">Quick Summary</h4>
                        <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                            <li>Your dataset has <b>{cat_cols}</b> categorical variables</li>
                            <li><b>{high_cardinality}</b> variables have high cardinality (>10 unique values)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Tab 2: Outlier Analysis
        with eda_tabs[1]:
            if eda_results['outliers']:
                st.markdown("<h3 style='font-size: 1.2rem;'>🎯 Outlier Detection Results</h3>", unsafe_allow_html=True)
                
                # Summary of outliers
                total_outliers = sum(len(info['indices']) for info in eda_results['outliers'].values())
                cols_with_outliers = len(eda_results['outliers'])
                
                # Create metrics for outliers
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Outliers", f"{total_outliers:,}")
                with col2:
                    st.metric("Columns with Outliers", cols_with_outliers)
                
                # Create a more visual representation of outliers
                outlier_data = {
                    'Column': [],
                    'Outlier Count': [],
                    'Percentage': []
                }
                
                for col, outlier_info in eda_results['outliers'].items():
                    outlier_data['Column'].append(col)
                    outlier_data['Outlier Count'].append(len(outlier_info['indices']))
                    outlier_data['Percentage'].append(len(outlier_info['indices']) / len(st.session_state.df) * 100)
                
                if outlier_data['Column']:
                    outlier_df = pd.DataFrame(outlier_data)
                    
                        # Bar chart for outliers removed
                
                # Detailed outlier information in expandable sections
                st.markdown("<h4 style='font-size: 1rem; margin-top: 1rem;'>Detailed Outlier Information</h4>", unsafe_allow_html=True)
                
                for col, outlier_info in eda_results['outliers'].items():
                    with st.expander(f"📊 {col} ({len(outlier_info['indices'])} outliers)"):
                        st.write(f"**Number of outliers:** {len(outlier_info['indices'])}")
                        st.write(f"**Percentage of data:** {len(outlier_info['indices']) / len(st.session_state.df) * 100:.2f}%")
                        
                        # Show outlier values in a more structured way
                        if outlier_info['values']:
                            st.write("**Sample outlier values:**")
                            outlier_sample = pd.DataFrame({
                                'Value': outlier_info['values'][:10]  # Show first 10
                            })
                            st.dataframe(outlier_sample, use_container_width=True)
                            
                            # Add statistics about outliers
                            if len(outlier_info['values']) > 0:
                                st.write("**Outlier statistics:**")
                                st.write(f"Min: {min(outlier_info['values']):.2f}")
                                st.write(f"Max: {max(outlier_info['values']):.2f}")
                                st.write(f"Mean: {sum(outlier_info['values']) / len(outlier_info['values']):.2f}")
            else:
                st.info("No outliers detected in the dataset based on the selected method.")
        
        # Tab 3: Correlation Analysis
        with eda_tabs[2]:
            if eda_results['correlations'] is not None:
                st.markdown("<h3 style='font-size: 1.2rem;'>🔗 Correlation Analysis</h3>", unsafe_allow_html=True)
                
                # Add correlation threshold slider
                corr_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Show correlations with absolute value above this threshold"
                )
                
                # Filter correlation matrix based on threshold
                corr_matrix = eda_results['correlations'].copy()
                corr_matrix_filtered = corr_matrix.where(abs(corr_matrix) >= corr_threshold, 0)
                
                # Create an enhanced correlation heatmap
                fig = self.visualizer.create_correlation_heatmap(corr_matrix_filtered)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlations in a table
                st.markdown("<h4 style='font-size: 1rem; margin-top: 1rem;'>Top Correlations</h4>", unsafe_allow_html=True)
                
                # Get top correlations
                corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) >= corr_threshold:
                            corrs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if corrs:
                    corrs_df = pd.DataFrame(corrs).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(corrs_df, use_container_width=True)
                else:
                    st.info(f"No correlations above the threshold of {corr_threshold} were found.")
            else:
                st.info("No correlation analysis available for this dataset.")
                st.write("This could be because there are no numerical columns or insufficient data for correlation analysis.")
    
    def render_visualizations_tab(self):
        """Render visualizations tab with enhanced presentation."""
        st.header("📊 Auto-Generated Visualizations")
        
        # Introduction to visualizations
        st.markdown("""
        <div style="margin-bottom: 1.5rem; color: #4a5568;">
            Explore your data visually through automatically generated charts and graphs. 
            These visualizations help you identify patterns, trends, and relationships in your dataset.
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Generate visualizations if not already done
        if not st.session_state.visualizations:
            with st.spinner("🎨 Generating visualizations..."):
                st.session_state.visualizations = self.visualizer.generate_auto_visualizations(df)
        
        visualizations = st.session_state.visualizations
        
        # Create a visualization category selector
        if visualizations:
            viz_types = [viz_type for viz_type, figures in visualizations.items() if figures]
            
            if viz_types:
                # Create tabs for different visualization categories
                viz_tabs = st.tabs([f"📈 {viz_type.replace('_', ' ').title()}" for viz_type in viz_types])
                
                # Display visualizations in tabs
                for i, viz_type in enumerate(viz_types):
                    with viz_tabs[i]:
                        figures = visualizations[viz_type]
                        
                        # Add description based on visualization type
                        if viz_type == 'distribution_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">📊 Distribution Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots show how values are distributed across your variables. They help identify patterns, skewness, and potential outliers.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif viz_type == 'relationship_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">🔗 Relationship Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots show relationships between variables, helping you identify correlations, patterns, and potential insights.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif viz_type == 'categorical_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">📊 Categorical Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots show distributions and relationships for categorical variables, helping you understand frequencies and patterns.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif viz_type == 'time_series_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">📈 Time Series Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots show how variables change over time, helping you identify trends, seasonality, and anomalies.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif viz_type == 'missing_value_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">🕳️ Missing Value Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots show patterns of missing data in your dataset, helping you identify potential issues for data cleaning.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif viz_type == 'outlier_plots':
                            st.markdown("""
                            <div style="margin-bottom: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">⚠️ Outlier Plots</h4>
                                <p style="margin-bottom: 0; color: #4a5568;">These plots highlight outliers in your data, helping you identify unusual values that may require further investigation.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Create a grid layout for visualizations
                        if len(figures) > 1:
                            # Add a selector for specific visualizations
                            selected_viz = st.selectbox(
                                "Select visualization:",
                                list(figures.keys()),
                                key=f"select_{viz_type}"
                            )
                            
                            if selected_viz in figures:
                                # Display the selected visualization
                                st.markdown(f"<h3 style='font-size: 1.2rem;'>{selected_viz}</h3>", unsafe_allow_html=True)
                                st.plotly_chart(figures[selected_viz], use_container_width=True)
                                
                                # Add download button for the visualization
                                st.download_button(
                                    label="💾 Download Chart",
                                    data="",  # This would need to be implemented with actual image data
                                    file_name=f"{selected_viz.replace(' ', '_')}.png",
                                    mime="image/png",
                                    key=f"download_{viz_type}_{selected_viz}"
                                )
                        else:
                            # If there's only one visualization, display it directly
                            for fig_name, fig in figures.items():
                                st.markdown(f"<h3 style='font-size: 1.2rem;'>{fig_name}</h3>", unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add download button for the visualization
                                st.download_button(
                                    label="💾 Download Chart",
                                    data="",  # This would need to be implemented with actual image data
                                    file_name=f"{fig_name.replace(' ', '_')}.png",
                                    mime="image/png",
                                    key=f"download_{viz_type}_{fig_name}"
                                )
            else:
                st.info("No visualizations could be generated for this dataset.")
        else:
            st.info("No visualizations available. This could be due to insufficient data or unsupported data types.")
            st.markdown("""
            <div style="margin-top: 1rem; padding: 0.8rem; background-color: #f8f9fa; border-radius: 8px;">
                <h4 style="margin-bottom: 0.5rem; font-size: 1rem;">Visualization Tips</h4>
                <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                    <li>Make sure your dataset has numerical columns for distribution plots</li>
                    <li>For relationship plots, you need at least two numerical columns</li>
                    <li>For categorical plots, ensure you have categorical or text columns</li>
                    <li>Time series plots require a datetime column</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def render_insights_tab(self):
        """Render AI insights tab with improved presentation."""
        st.header("🤖 AI-Powered Insights")
        
        # Introduction to AI insights
        st.markdown("""
        <div style="margin-bottom: 1.5rem; color: #4a5568;">
            Discover AI-generated insights about your data. These insights are automatically generated 
            based on statistical analysis and patterns detected in your dataset.
        </div>
        """, unsafe_allow_html=True)
        
        if not self.enable_llm:
            st.warning("⚠️ LLM insights are disabled. Enable them in the sidebar to access this feature.")
            return
        
        # Generate insights if not already done
        if not st.session_state.llm_insights:
            with st.spinner("🧠 Generating AI insights..."):
                try:
                    insights = self.llm_generator.generate_comprehensive_insights(
                        st.session_state.df,
                        st.session_state.eda_results
                    )
                    st.session_state.llm_insights = insights
                except Exception as e:
                    st.error(f"❌ Error generating insights: {str(e)}")
                    st.info("💡 Make sure OLLAMA is running with Qwen model loaded")
                    return
        
        insights = st.session_state.llm_insights
        
        # Create tabs for different insight categories
        if insights:
            insight_tabs = st.tabs([
                f"📊 {section.replace('_', ' ').title()}" 
                for section, content in insights.items() 
                if content
            ])
            
            # Display insights in tabs
            tab_index = 0
            for section, content in insights.items():
                if content:
                    with insight_tabs[tab_index]:
                        # Add icons based on insight type
                        icon = "📈"
                        if "summary" in section:
                            icon = "📋"
                        elif "quality" in section:
                            icon = "🔍"
                        elif "correlation" in section:
                            icon = "🔗"
                        elif "distribution" in section:
                            icon = "📊"
                        elif "outlier" in section:
                            icon = "⚠️"
                        
                        # Format the content with better styling
                        formatted_content = content
                        
                        # Highlight key metrics and findings
                        formatted_content = formatted_content.replace(
                            "**", "<span style='color: #3498db; font-weight: 600;'>**"
                        ).replace("**", "**</span>")
                        
                        # Add a card-like container for each insight
                        st.markdown(f'<div class="insight-box">{formatted_content}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Add action buttons for each insight
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            st.button(f"📋 Copy", key=f"copy_{section}")
                        with col2:
                            st.button(f"💾 Save", key=f"save_{section}")
                    
                    tab_index += 1
        else:
            st.info("No insights generated yet. Please wait or check if the LLM service is running properly.")
    
    def render_chat_tab(self):
        """Render enhanced data chat interface."""
        st.header("💬 Chat with Your Data")
        
        # Introduction with examples
        with st.expander("ℹ️ How to use Data Chat", expanded=len(st.session_state.chat_history) == 0):
            st.markdown("""
            Ask questions about your dataset in natural language! The AI assistant will analyze your data and provide insights.
            
            **Example questions you can ask:**
            - What are the main trends in this dataset?
            - Summarize the key statistics for [column_name]
            - Is there a correlation between [column_1] and [column_2]?
            - What insights can you provide about the outliers?
            - How is [column_name] distributed?
            - What recommendations do you have based on this data?
            
            The assistant remembers your conversation context, so you can ask follow-up questions.
            """)
        
        # Chat container with styling
        chat_container = st.container()
        with chat_container:
            # Display chat history with improved styling
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"<div class='user-message'><strong>🙋 You:</strong> {question}</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 6])
                with col2:
                    st.markdown(f"<div class='ai-message'><strong>🤖 AI:</strong> {answer}</div>", unsafe_allow_html=True)
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        
        # Feedback mechanism (if there's chat history)
        if st.session_state.chat_history:
            with st.expander("📝 Provide feedback on the last response"):
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 1])
                with feedback_col1:
                    st.button("👍 Helpful", key="helpful")
                with feedback_col2:
                    st.button("👎 Not Helpful", key="not_helpful")
                with feedback_col3:
                    st.button("🔄 Regenerate Response", key="regenerate")
                
                st.text_area("Additional feedback (optional)", key="feedback_text")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback!")
        
        # Chat input with placeholder suggestions
        question = st.chat_input("Ask a question about your data...")
        
        if question:
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = self.llm_generator.answer_data_question(
                        question, 
                        st.session_state.df,
                        st.session_state.eda_results
                    )
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}\n\nPlease try rephrasing your question or check if OLLAMA is running properly.")
        
        # Export functionality with improved UI
        st.markdown("<hr style='margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
        st.subheader("📥 Export Analysis")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📄 Export as Markdown", help="Export a comprehensive report including data analysis and chat history"):
                # Include chat history in the export
                chat_history_text = "\n\n## Chat History\n\n"
                for q, a in st.session_state.chat_history:
                    chat_history_text += f"**Question:** {q}\n\n**Answer:** {a}\n\n---\n\n"
                
                report = export_report(
                    st.session_state.df,
                    st.session_state.eda_results,
                    st.session_state.llm_insights,
                    format="markdown"
                )
                
                # Add chat history if it exists
                if st.session_state.chat_history:
                    report += chat_history_text
                
                st.download_button(
                    label="⬇️ Download Markdown Report",
                    data=report,
                    file_name="data_analysis_report.md",
                    mime="text/markdown"
                )
        
        with export_col2:
            if st.button("📊 Export Dashboard as HTML", help="Export an interactive HTML dashboard with visualizations"):
                st.info("🚧 HTML export feature coming soon!")
    
    def reset_session_state(self):
        """Reset session state for new file upload."""
        # Debug information to help troubleshoot the issue
        st.sidebar.markdown("### Reset Session State Debug Info")
        st.sidebar.write(f"Before reset, file_uploaded: {st.session_state.get('file_uploaded')}")
        st.sidebar.write(f"Before reset, session state keys: {list(st.session_state.keys())}")
        
        for key in ['df', 'file_uploaded', 'eda_results', 'visualizations', 
                   'llm_insights', 'chat_history', 'predictive_model', 'model_metrics',
                   'model_visualizations', 'forecast_results', 'feature_importance']:
            if key in st.session_state:
                st.sidebar.write(f"Deleting key: {key}")
                del st.session_state[key]
        
        st.sidebar.write(f"After deleting keys, before initialize: {list(st.session_state.keys())}")
        self.initialize_session_state()
        
    def render_predictive_analytics_tab(self):
        """Render predictive analytics tab."""
        st.header("🔮 Predictive Analytics")
        
        # Introduction to predictive analytics
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <p>Build predictive models to forecast future trends or classify data points. Select your analysis type, 
            target variable, and features to train a machine learning model on your dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Create tabs for setup and results
        setup_tab, results_tab = st.tabs(["📝 Model Setup", "📊 Model Results"])
        
        with setup_tab:
            # Create a card-like container for model configuration
            st.markdown("<div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
            
            # Analysis type with icons and descriptions
            st.subheader("📊 Analysis Type")
            analysis_type = st.radio(
                "Select Analysis Type",
                options=["Regression", "Classification", "Time Series Forecasting"],
                horizontal=True,
                help="Choose the type of predictive analysis to perform"
            )
            
            # Show description based on selected analysis type
            if analysis_type == "Regression":
                st.info("📈 **Regression**: Predict continuous values like prices, ratings, or quantities.")
            elif analysis_type == "Classification":
                st.info("🏷️ **Classification**: Categorize data into classes like yes/no, high/medium/low, etc.")
            else:  # Time Series
                st.info("⏱️ **Time Series Forecasting**: Predict future values based on historical time-ordered data.")
            
            # Target and feature selection in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Target Variable")
                target_column = st.selectbox(
                    "Select Target Column",
                    df.columns.tolist(),
                    help="Column you want to predict"
                )
                
                # Show target variable preview
                if target_column:
                    st.markdown("**Target Preview:**")
                    if pd.api.types.is_numeric_dtype(df[target_column]):
                        st.markdown(f"📊 **Type**: Numeric")
                        st.markdown(f"📏 **Range**: {df[target_column].min():.2f} to {df[target_column].max():.2f}")
                    else:
                        st.markdown(f"📊 **Type**: Categorical")
                        st.markdown(f"🔢 **Classes**: {len(df[target_column].unique())}")
            
            with col2:
                st.subheader("🧩 Feature Selection")
                all_columns = [col for col in df.columns if col != target_column]
                
                # Option to select all features or choose specific ones
                use_all_features = st.checkbox("Use all available features", value=True)
                
                if use_all_features:
                    selected_features = all_columns
                    st.success(f"✅ Using all {len(all_columns)} features")
                else:
                    selected_features = st.multiselect(
                        "Select Features",
                        all_columns,
                        default=all_columns[:min(5, len(all_columns))],
                        help="Select columns to use as features for prediction"
                    )
                    if selected_features:
                        st.success(f"✅ Selected {len(selected_features)} features")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Model configuration section
            st.subheader("⚙️ Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Test size slider with visual indicator
                test_size = st.slider(
                    "Test Set Size",
                    0.1, 0.5, 0.2,
                    help="Proportion of data to use for testing"
                )
                
                # Visual representation of train/test split
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                    <div style="background-color: #4CAF50; width: {int((1-test_size)*100)}%; height: 20px; border-radius: 4px 0 0 4px; text-align: center; color: white;">
                        Train {int((1-test_size)*100)}%
                    </div>
                    <div style="background-color: #2196F3; width: {int(test_size*100)}%; height: 20px; border-radius: 0 4px 4px 0; text-align: center; color: white;">
                        Test {int(test_size*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Model selection based on analysis type with descriptions
                if analysis_type == "Regression":
                    model_options = [
                        "auto", "linear_regression", "ridge_regression", 
                        "lasso_regression", "random_forest", "gradient_boosting"
                    ]
                    model_type = "regression"
                elif analysis_type == "Classification":
                    model_options = [
                        "auto", "logistic_regression", "random_forest", "gradient_boosting"
                    ]
                    model_type = "classification"
                else:  # Time Series
                    model_options = [
                        "auto", "linear_regression", "random_forest", "gradient_boosting"
                    ]
                    model_type = "regression"
                
                selected_model = st.selectbox(
                    "Select Model",
                    model_options,
                    index=0,
                    help="'auto' will select the best model based on cross-validation"
                )
                
                # Show model description
                if selected_model == "auto":
                    st.info("🤖 **Auto**: Automatically selects the best model based on cross-validation.")
                elif "linear" in selected_model or "ridge" in selected_model or "lasso" in selected_model:
                    st.info("📏 **Linear Model**: Simple, interpretable model for linear relationships.")
                elif "random_forest" in selected_model:
                    st.info("🌲 **Random Forest**: Ensemble of decision trees, robust to overfitting.")
                elif "gradient_boosting" in selected_model:
                    st.info("🚀 **Gradient Boosting**: Advanced ensemble method with high performance.")
                elif "logistic" in selected_model:
                    st.info("🔄 **Logistic Regression**: Simple classification model for binary/multiclass problems.")
            
            # Additional settings for time series in an expander
            if analysis_type == "Time Series Forecasting":
                with st.expander("⏱️ Time Series Settings", expanded=True):
                    # Select datetime column
                    datetime_cols = [col for col in df.columns if 
                                pd.api.types.is_datetime64_any_dtype(df[col]) or 
                                'date' in col.lower() or 'time' in col.lower()]
                    
                    if datetime_cols:
                        datetime_column = st.selectbox(
                            "Select Datetime Column",
                            datetime_cols,
                            help="Column containing date/time information"
                        )
                    else:
                        st.warning("⚠️ No datetime columns detected. Please ensure your time column is in datetime format.")
                        datetime_column = st.selectbox(
                            "Select Column to Convert to Datetime",
                            df.columns.tolist(),
                            help="Column to convert to datetime format"
                        )
                    
                    # Forecast horizon with visual indicator
                    forecast_horizon = st.slider(
                        "Forecast Horizon",
                        1, 100, 10,
                        help="Number of periods to forecast into the future"
                    )
                    
                    # Visual representation of forecast horizon
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem;">
                        <p>Forecasting {forecast_horizon} periods into the future</p>
                        <div style="display: flex; align-items: center;">
                            <div style="background-color: #9C27B0; width: 70%; height: 20px; border-radius: 4px 0 0 4px; text-align: center; color: white;">
                                Historical Data
                            </div>
                            <div style="background-color: #FF9800; width: 30%; height: 20px; border-radius: 0 4px 4px 0; text-align: center; color: white;">
                                Forecast
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Advanced time series options
                    st.markdown("### Advanced Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        include_seasonal = st.checkbox("Include Seasonal Features", value=True, 
                                                    help="Add weekend, month start/end, and holiday indicators")
                        calculate_intervals = st.checkbox("Show Prediction Intervals", value=True,
                                                        help="Display uncertainty ranges for forecasts")
                    
                    with col2:
                        include_cyclical = st.checkbox("Include Cyclical Encoding", value=True,
                                                     help="Add sine/cosine transformations of time features")
                        show_components = st.checkbox("Show Seasonal Components", value=False,
                                                    help="Display seasonal patterns in visualization")
            else:
                datetime_column = None
                forecast_horizon = 10
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Train model button with enhanced styling
            train_col1, train_col2, train_col3 = st.columns([1, 2, 1])
            with train_col2:
                train_button = st.button(
                    "🚀 Train Model",
                    use_container_width=True,
                    help="Start training the selected model with your data"
                )
            
            if train_button:
                if not selected_features:
                    st.error("❌ Please select at least one feature")
                else:
                    with st.spinner("🔮 Training predictive model..."):
                        try:
                            # Prepare data
                            if analysis_type == "Time Series Forecasting" and datetime_column:
                                # For time series, we need to handle the datetime column
                                if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
                                    # Try to convert to datetime
                                    try:
                                        df[datetime_column] = pd.to_datetime(df[datetime_column])
                                    except Exception as e:
                                        st.error(f"❌ Could not convert {datetime_column} to datetime: {str(e)}")
                                        st.stop()
                                
                                # Generate forecast
                                forecast_df, metrics = self.predictive_analytics.forecast_time_series(
                                    df, target_column, datetime_column, 
                                    horizon=forecast_horizon,
                                    feature_columns=selected_features,
                                    model_name=selected_model,
                                    include_seasonal_features=include_seasonal,
                                    include_cyclical_encoding=include_cyclical,
                                    calculate_intervals=calculate_intervals
                                )
                                
                                # Store results
                                st.session_state.forecast_results = forecast_df
                                st.session_state.model_metrics = metrics
                                
                                # Display information about forecast features
                                with st.expander("📊 Forecast Information", expanded=False):
                                    st.markdown("### Features Used in Forecast")
                                    feature_info = [
                                        f"✅ **Seasonal Features**: {'Enabled' if include_seasonal else 'Disabled'}",
                                        f"✅ **Cyclical Encoding**: {'Enabled' if include_cyclical else 'Disabled'}",
                                        f"✅ **Prediction Intervals**: {'Calculated' if calculate_intervals else 'Not calculated'}",
                                        f"✅ **Forecast Horizon**: {forecast_horizon} periods",
                                        f"✅ **Model Used**: {selected_model.title()}"
                                    ]
                                    st.markdown("\n".join(feature_info))
                                    
                                    # Show metrics if available
                                    if metrics:
                                        st.markdown("### Model Performance Metrics")
                                        # Format metrics for display, converting None to 'N/A'
                                        formatted_metrics = {k: (v if v is not None else 'N/A') for k, v in metrics.items()}
                                        metrics_df = pd.DataFrame([formatted_metrics])
                                        st.dataframe(metrics_df, use_container_width=True)
                                
                                # Create visualization
                                forecast_viz = self.predictive_analytics.create_forecast_visualization(
                                    df, forecast_df, target_column, datetime_column,
                                    show_intervals=calculate_intervals,
                                    show_components=show_components
                                )
                                
                                st.session_state.model_visualizations = {
                                    'forecast': forecast_viz
                                }
                                
                                # Display forecast data in a table
                                with st.expander("📋 Forecast Data", expanded=False):
                                    st.dataframe(forecast_df, use_container_width=True)
                                    
                                    # Add download button for forecast data
                                    csv = forecast_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Forecast Data",
                                        data=csv,
                                        file_name="forecast_results.csv",
                                        mime="text/csv",
                                    )
                                
                            else:
                                # Regular regression or classification
                                X_train, X_test, y_train, y_test = self.predictive_analytics.prepare_data(
                                    df, target_column, selected_features, test_size=test_size
                                )
                                
                                # Train model
                                pipeline = self.predictive_analytics.train_model(
                                    X_train, y_train, model_type=model_type, model_name=selected_model
                                )
                                
                                # Evaluate model
                                metrics = self.predictive_analytics.evaluate_model(pipeline, X_test, y_test)
                                
                                # Store results
                                st.session_state.predictive_model = pipeline
                                st.session_state.model_metrics = metrics
                                
                                # Generate visualizations
                                viz = self.predictive_analytics.create_evaluation_visualizations()
                                st.session_state.model_visualizations = viz
                                
                                # Get feature importance
                                st.session_state.feature_importance = self.predictive_analytics.feature_importance
                            
                            st.success("✅ Model trained successfully! View results in the 'Model Results' tab.")
                            
                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
        
        # Results tab
        with results_tab:
            if st.session_state.model_metrics is not None:
                # Model performance metrics in a card
                st.subheader("📊 Model Performance Metrics")
                
                metrics = st.session_state.model_metrics
                # Format metrics for display, converting None to 'N/A'
                formatted_metrics = {k: (v if v is not None else 'N/A') for k, v in metrics.items()}
                metrics_df = pd.DataFrame({
                    'Metric': list(formatted_metrics.keys()),
                    'Value': list(formatted_metrics.values())
                })
                
                # Display metrics in a more visual way
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Calculate an overall score (average of normalized metrics)
                    # Filter out None values from metrics
                    valid_metrics = {k: v for k, v in metrics.items() if v is not None}
                    overall_score = sum(valid_metrics.values()) / len(valid_metrics) if valid_metrics else 0
                    if 'accuracy' in metrics or 'r2' in metrics:
                        # For metrics where higher is better
                        performance_color = "#4CAF50" if overall_score > 0.7 else "#FF9800" if overall_score > 0.5 else "#F44336"
                    else:
                        # For metrics where lower is better (like RMSE, MAE)
                        performance_color = "#4CAF50" if overall_score < 0.3 else "#FF9800" if overall_score < 0.5 else "#F44336"
                    
                    st.markdown(f"""
                    <div style="background-color: {performance_color}; padding: 1rem; border-radius: 0.5rem; color: white; text-align: center;">
                        <h1 style="margin: 0; font-size: 3rem;">{overall_score:.2f}</h1>
                        <p style="margin: 0;">Overall Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Display metrics as a styled dataframe
                    st.dataframe(metrics_df, use_container_width=True, height=200)
                    
                    # Add download button for metrics
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Metrics",
                        data=csv,
                        file_name="model_metrics.csv",
                        mime="text/csv",
                    )
                
                # Display visualizations in tabs if available
                if st.session_state.model_visualizations:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("📈 Model Visualizations")
                    
                    # Create tabs for different visualization types
                    viz_names = list(st.session_state.model_visualizations.keys())
                    if len(viz_names) > 1:
                        viz_tabs = st.tabs([name.replace('_', ' ').title() for name in viz_names])
                        
                        for i, (viz_name, fig) in enumerate(st.session_state.model_visualizations.items()):
                            with viz_tabs[i]:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add download button for each visualization
                                try:
                                    buf = io.BytesIO()
                                    fig.write_image(buf, format="png")
                                    st.download_button(
                                        label=f"📥 Download {viz_name.replace('_', ' ').title()} Plot",
                                        data=buf.getvalue(),
                                        file_name=f"{viz_name}.png",
                                        mime="image/png",
                                    )
                                except Exception as e:
                                    st.warning("📝 Image download requires additional dependencies. Install 'kaleido' package for this feature.")
                                    st.info(f"💡 Run: pip install kaleido")
                    else:
                        # If only one visualization, display it directly
                        viz_name = viz_names[0]
                        fig = st.session_state.model_visualizations[viz_name]
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add download button
                        try:
                            buf = io.BytesIO()
                            fig.write_image(buf, format="png")
                            st.download_button(
                                label=f"📥 Download {viz_name.replace('_', ' ').title()} Plot",
                                data=buf.getvalue(),
                                file_name=f"{viz_name}.png",
                                mime="image/png",
                            )
                        except Exception as e:
                            st.warning("📝 Image download requires additional dependencies. Install 'kaleido' package for this feature.")
                            st.info(f"💡 Run: pip install kaleido")
                
                # Display feature importance if available
                if st.session_state.feature_importance is not None:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("🔍 Feature Importance")
                    
                    # Create a bar chart for feature importance
                    feature_imp = st.session_state.feature_importance.head(10).copy()
                    fig = px.bar(
                        feature_imp,
                        x=feature_imp.columns[1],
                        y=feature_imp.columns[0],
                        orientation='h',
                        title="Top 10 Most Important Features",
                        labels={feature_imp.columns[1]: "Importance", feature_imp.columns[0]: "Feature"},
                        color=feature_imp.columns[1],
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display full feature importance table in an expander
                    with st.expander("View Full Feature Importance Table"):
                        st.dataframe(st.session_state.feature_importance, use_container_width=True)
                        
                        # Add download button for feature importance
                        csv = st.session_state.feature_importance.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Feature Importance",
                            data=csv,
                            file_name="feature_importance.csv",
                            mime="text/csv",
                        )
                
                # Display forecast results for time series
                if st.session_state.forecast_results is not None:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("🔮 Forecast Results")
                    
                    # Display forecast summary
                    forecast_df = st.session_state.forecast_results
                    st.markdown(f"**Forecast Periods:** {len(forecast_df)} time points")
                    
                    # Display forecast table with styling
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Add download button for forecast results
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Results",
                        data=csv,
                        file_name="forecast_results.csv",
                        mime="text/csv",
                    )
            else:
                # Display a message if no model has been trained yet
                st.info("⚠️ No model results available. Please train a model in the 'Model Setup' tab.")
                
                # Add some example images or placeholders
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 0.5rem;">
                    <h3>What to expect after training a model:</h3>
                    <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem;">📊</div>
                            <p>Performance Metrics</p>
                        </div>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem;">📈</div>
                            <p>Visualizations</p>
                        </div>
                        <div style="text-align: center; padding: 1rem;">
                            <div style="font-size: 2rem;">🔍</div>
                            <p>Feature Importance</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    app = DataAnalyticsApp()
    app.run()

if __name__ == "__main__":
    main()