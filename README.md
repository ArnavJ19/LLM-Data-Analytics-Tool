# AI-Powered Data Analytics Tool

A comprehensive data analytics web application that combines traditional statistical analysis with Large Language Model (LLM) powered insights and predictive analytics capabilities. Upload your dataset and get instant exploratory data analysis, visualizations, AI-generated insights, and predictive models!

##  Features

###  Core Capabilities
- **Multi-format Data Upload**: Supports CSV, Excel (.xlsx, .xls), and JSON files
- **Automatic EDA**: Comprehensive exploratory data analysis with statistical summaries
- **Smart Visualizations**: Auto-generated interactive charts using Plotly
- **AI-Powered Insights**: Natural language insights powered by OLLAMA and Qwen model
- **Predictive Analytics**: Build, evaluate, and visualize machine learning models
- **Time Series Forecasting**: Predict future values based on historical data
- **Data Quality Assessment**: Automated detection of missing values, outliers, and data issues
- **Interactive Chat**: Ask questions about your data in natural language
- **Export Functionality**: Generate comprehensive reports in Markdown format

### Analysis Features
- Statistical summaries (mean, median, mode, std, skewness, kurtosis)
- Distribution analysis and normality testing
- Correlation analysis with interactive heatmaps
- Outlier detection using IQR and Z-score methods
- Missing value pattern analysis
- Categorical variable analysis
- Time series visualization (when applicable)
- Regression and classification modeling
- Feature importance analysis
- Model performance evaluation
- Time series forecasting

###  Visualization Types
- **Distributions**: Histograms with KDE overlays
- **Relationships**: Scatter plots, correlation heatmaps, scatter matrices
- **Categorical**: Bar charts, box plots by category
- **Outliers**: Box plots with outlier highlighting
- **Missing Values**: Missing value patterns and heatmaps
- **Time Series**: Temporal trend analysis
- **Predictive Models**: Actual vs predicted plots, residual plots, confusion matrices
- **Feature Importance**: Bar charts of feature importance
- **Forecasts**: Time series forecast visualizations

##  Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **OLLAMA** installed and running (for AI insights)
3. **Qwen model** loaded in OLLAMA

### Installation

1. **Clone or download the application files:**
   ```bash
   # If using git
   git clone <repository-url>
   cd data-analytics-tool
   
   # Or download and extract the files to a folder
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup OLLAMA (for AI insights):**
   
   **On macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   **On Windows:**
   Download and install from [ollama.ai](https://ollama.ai)

4. **Pull the Qwen model:**
   ```bash
   ollama pull qwen2.5:8b
   ```

5. **Start OLLAMA server:**
   ```bash
   ollama serve
   ```

### Create Requirements File

Create a `requirements.txt` file with the following content:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scipy>=1.10.0
scikit-learn>=1.3.0
requests>=2.31.0
openpyxl>=3.1.0
xlrd>=2.0.0
seaborn>=0.12.0
```

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your dataset** and start exploring!

## Project Structure

```
data-analytics-tool/
│
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data loading and EDA engine
├── visualizer.py          # Auto-visualization generator
├── llm_insights.py        # LLM integration and insights
├── predictive_analytics.py # Predictive modeling and forecasting
├── utils.py               # Utilities and export functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

##  How to Use

### 1. Upload Your Data
- Drag and drop or browse for your file (CSV, Excel, or JSON)
- The app automatically detects format and loads your data
- Preview your dataset and column information

### 2. Explore Auto-EDA
- View comprehensive statistical summaries
- Analyze data quality and missing value patterns
- Examine correlation matrices and outlier detection results

### 3. Interactive Visualizations
- Automatically generated charts based on your data types
- Interactive Plotly visualizations you can zoom, pan, and explore
- Distribution plots, scatter matrices, and correlation heatmaps

### 4. AI-Powered Insights
- Get natural language explanations of your data patterns
- Understand statistical findings in plain English
- Receive recommendations for data cleaning and analysis

### 5. Predictive Analytics
- Build regression, classification, or time series forecasting models
- Select target variables and features for prediction
- Choose from multiple algorithms or let the system select the best model
- Evaluate model performance with metrics and visualizations
- Analyze feature importance to understand key drivers
- Generate forecasts for time series data

### 6. Chat with Your Data
- Ask questions in natural language
- Get specific insights about relationships and patterns
- Explore your data conversationally

### 7. Export Reports
- Generate comprehensive analysis reports
- Download in Markdown format for documentation
- Include all insights, visualizations, and recommendations

## Configuration

### OLLAMA Configuration
The app connects to OLLAMA running on `http://localhost:11434` by default. To change this:

1. Open `llm_insights.py`
2. Modify the `base_url` parameter in the `LLMInsightGenerator` class
3. Restart the application

### Model Selection
To use a different OLLAMA model:

1. Pull your desired model: `ollama pull <model-name>`
2. Update the `model_name` parameter in `llm_insights.py`
3. Restart the application

## Troubleshooting

### Common Issues

**1. "OLLAMA is not available" error:**
- Ensure OLLAMA is installed and running: `ollama serve`
- Check if the Qwen model is pulled: `ollama list`
- Verify the connection URL is correct

**2. File upload issues:**
- Ensure your file is in supported format (CSV, Excel, JSON)
- Check file encoding (UTF-8 recommended for CSV)
- Try with a smaller file first to test

**3. Visualization not showing:**
- Ensure Plotly is properly installed
- Check browser compatibility (modern browsers recommended)
- Clear browser cache and reload

**4. Memory issues with large datasets:**
- Try with smaller datasets first
- Close other applications to free memory
- Consider data sampling for very large files

### Performance Tips

- **Large datasets**: The app works best with datasets under 100MB
- **Many columns**: Performance may slow with >50 columns
- **Missing values**: High missing value percentages may affect analysis
- **Text data**: Very long text fields may need preprocessing

## Customization

### Adding New Visualizations
1. Open `visualizer.py`
2. Add your visualization function to the `DataVisualizer` class
3. Update the `generate_auto_visualizations` method to include your new viz

### Custom LLM Prompts
1. Open `llm_insights.py`
2. Modify the prompt templates in the insight generation methods
3. Adjust `max_tokens` and temperature settings as needed

### UI Modifications
1. Edit `app.py` to modify the Streamlit interface
2. Add custom CSS in the `st.markdown` sections
3. Reorganize tabs and sections as desired

## Example Use Cases

### Business Analytics
- Customer behavior analysis
- Sales performance tracking and forecasting
- Market research data exploration
- Financial data analysis and prediction
- Customer churn prediction
- Demand forecasting

### Research & Academia
- Survey data analysis
- Experimental result exploration and prediction
- Literature review data compilation
- Statistical analysis and modeling for papers
- Research outcome prediction

### Data Science Projects
- Dataset exploration and profiling
- Feature engineering insights
- Model preparation, training, and evaluation
- Data quality assessment
- Predictive modeling and forecasting

### Personal Projects
- Personal finance tracking
- Health and fitness data analysis
- Social media analytics
- IoT sensor data exploration

## Contributing

This is a modular application designed for easy extension:

1. **Data Processors**: Add new file format support in `data_processor.py`
2. **Visualizations**: Extend chart types in `visualizer.py`
3. **LLM Models**: Add support for other models in `llm_insights.py`
4. **Predictive Models**: Add new algorithms in `predictive_analytics.py`
5. **Export Formats**: Add new export options in `utils.py`

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Streamlit** for the amazing web app framework
- **Plotly** for interactive visualizations
- **OLLAMA** for local LLM deployment
- **Qwen** for the powerful language model
- **Pandas** and **NumPy** for data processing

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify OLLAMA is running and model is loaded
4. Test with sample datasets first

---


**Happy Data Exploring! 🚀📊**
