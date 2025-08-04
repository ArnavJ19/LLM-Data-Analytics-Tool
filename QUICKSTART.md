# 🚀 Quick Start Guide

Get the AI-Powered Data Analytics Tool running in under 5 minutes!

## 📋 Prerequisites

- **Python 3.8+** installed
- **10GB free disk space** (for OLLAMA models)
- **4GB RAM minimum** (8GB recommended)

## ⚡ Option 1: Automated Setup (Recommended)

### 1. Download and Setup
```bash
# Download the application files
# Extract to a folder called 'data-analytics-tool'
cd data-analytics-tool

# Run the automated setup
python setup.py
```

The setup script will:
- ✅ Install Python dependencies
- ✅ Set up OLLAMA and AI models
- ✅ Generate sample datasets
- ✅ Run tests to verify everything works
- ✅ Create startup scripts

### 2. Launch the Application
```bash
# Option A: Use the startup script
./start_app.sh        # Linux/Mac
start_app.bat          # Windows

# Option B: Manual launch
streamlit run app.py
```

### 3. Open Your Browser
- Go to `http://localhost:8501`
- Upload a dataset or use the provided sample data
- Start exploring!

---

## 🛠️ Option 2: Manual Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install OLLAMA (for AI insights)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### 3. Start OLLAMA and Install Model
```bash
# Start OLLAMA server
ollama serve

# In another terminal, install the AI model
ollama pull qwen2.5:8b
```

### 4. Generate Sample Data (Optional)
```bash
python sample_data_generator.py
```

### 5. Test the Application
```bash
python test_app.py
```

### 6. Launch
```bash
streamlit run app.py
```

---

## 🐳 Option 3: Docker Setup

### Quick Docker Launch
```bash
# Build and run with Docker Compose
docker-compose up -d

# Wait for setup (first time takes ~10 minutes to download AI model)
docker-compose logs -f ollama-init

# Access at http://localhost:8501
```

### Stop Docker Services
```bash
docker-compose down
```

---

## 📊 What to Expect

### 1. Upload Your Data
- **Supported formats**: CSV, Excel (.xlsx, .xls), JSON
- **File size limit**: 100MB (configurable)
- **Auto-detection**: File format and data types

### 2. Instant Analysis
- **Statistical summaries** for all columns
- **Data quality assessment** (missing values, outliers, duplicates)
- **Automatic visualizations** based on data types
- **Correlation analysis** between variables

### 3. AI-Powered Insights
- **Natural language explanations** of your data patterns
- **Business recommendations** based on statistical findings
- **Interactive chat** to ask questions about your data
- **Comprehensive reports** you can export

### 4. Interactive Features
- **Dynamic visualizations** you can zoom and explore
- **Real-time filtering** and data exploration
- **Export capabilities** (Markdown reports)
- **Multiple analysis perspectives**

---

## 🎯 Sample Datasets Included

The application comes with realistic sample datasets:

1. **Sales Data** (1,000 records)
   - Product sales across regions and time
   - Revenue, quantities, customer demographics
   - Perfect for business analytics practice

2. **Customer Data** (500 records)
   - Customer demographics and behavior
   - Segmentation, lifetime value, satisfaction scores
   - Great for customer analysis

3. **Marketing Campaigns** (2,000 records)
   - Multi-channel campaign performance
   - ROI, conversion rates, engagement metrics
   - Ideal for marketing analytics

4. **IoT Sensor Data** (5,000 records)
   - Time-series sensor readings
   - Temperature, humidity, motion detection
   - Perfect for operational analytics

5. **Financial Data** (1,000 records)
   - Stock market data with OHLC prices
   - Volume, volatility, sector information
   - Excellent for financial analysis

---

## 🔧 Troubleshooting

### Common Issues

**"OLLAMA not available" error:**
```bash
# Make sure OLLAMA is running
ollama serve

# Check if model is installed
ollama list

# Install model if missing
ollama pull qwen2.5:8b
```

**Port already in use:**
```bash
# Change port in the command
streamlit run app.py --server.port 8502
```

**Memory issues:**
- Try smaller datasets first
- Close other applications
- Use the Docker version for better resource management

**File upload fails:**
- Check file format (CSV, Excel, JSON only)
- Ensure file size is under 100MB
- Try with sample datasets first

### Getting Help

1. **Check the logs** in the terminal for error messages
2. **Run the test script**: `python test_app.py`
3. **Verify setup**: `python setup.py --quick`
4. **Use sample data** to test functionality first

---

## 🎊 You're Ready!

Once you see the Streamlit interface:

1. 📁 **Upload a dataset** or select sample data
2. 📊 **Explore the auto-generated analysis** in different tabs
3. 🤖 **Check out AI insights** for natural language explanations
4. 💬 **Ask questions** in the chat interface
5. 📋 **Export reports** when you're done

**Happy analyzing! 🚀📈**

---

*Need more detailed instructions? Check the full [README.md](README.md) for comprehensive documentation.*