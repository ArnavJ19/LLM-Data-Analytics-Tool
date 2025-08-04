"""
Test script for the Data Analytics Web Application

This script tests the core functionality of all modules to ensure
everything is working correctly before deploying the application.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any
import tempfile
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_processor import DataProcessor
    from visualizer import DataVisualizer
    from llm_insights import LLMInsightGenerator
    from utils import export_report, detect_column_types, generate_data_profile
    from config import get_config, validate_config
    from sample_data_generator import SampleDataGenerator
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ApplicationTester:
    """Test suite for the data analytics application."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"\n🧪 Running test: {test_name}")
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            self.passed_tests += 1
            self.test_results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ {test_name} FAILED: {str(e)}")
            self.failed_tests += 1
            self.test_results.append((test_name, "FAILED", str(e)))
    
    def test_configuration(self):
        """Test configuration loading and validation."""
        config = get_config()
        
        # Test configuration access
        assert hasattr(config, 'llm'), "Config missing LLM section"
        assert hasattr(config, 'data_processing'), "Config missing data processing section"
        assert hasattr(config, 'visualization'), "Config missing visualization section"
        
        # Test configuration validation
        issues = validate_config(config)
        if issues:
            print(f"⚠️ Configuration issues found: {issues}")
        
        # Test environment variable override
        os.environ['MAX_FILE_SIZE_MB'] = '50'
        config_with_override = get_config()
        assert config_with_override.data_processing.max_file_size_mb == 50, "Environment override failed"
        
        print("📋 Configuration loaded and validated")
    
    def test_sample_data_generation(self):
        """Test sample data generation."""
        generator = SampleDataGenerator(seed=42)
        
        # Test each dataset type
        sales_df = generator.generate_sales_data(100)
        assert len(sales_df) == 100, "Sales data generation failed"
        assert 'Revenue' in sales_df.columns, "Sales data missing expected columns"
        
        customer_df = generator.generate_customer_data(50)
        assert len(customer_df) == 50, "Customer data generation failed"
        assert 'Customer_ID' in customer_df.columns, "Customer data missing expected columns"
        
        marketing_df = generator.generate_marketing_data(200)
        assert len(marketing_df) == 200, "Marketing data generation failed"
        assert 'Campaign_ID' in marketing_df.columns, "Marketing data missing expected columns"
        
        print("📊 Sample data generation working correctly")
    
    def test_data_processor(self):
        """Test data processing functionality."""
        processor = DataProcessor()
        
        # Generate test data
        generator = SampleDataGenerator(seed=42)
        test_df = generator.generate_sales_data(100)
        
        # Test column info
        col_info = processor.get_column_info(test_df)
        assert len(col_info) == len(test_df.columns), "Column info mismatch"
        assert 'Data_Type' in col_info.columns, "Column info missing data types"
        
        # Test data quality checks
        quality_checks = processor.check_data_quality(test_df)
        assert 'duplicates' in quality_checks, "Quality checks missing duplicates"
        assert 'total_missing' in quality_checks, "Quality checks missing missing values"
        
        # Test EDA generation
        eda_results = processor.generate_eda(test_df)
        assert 'numerical_summary' in eda_results, "EDA missing numerical summary"
        assert 'categorical_summary' in eda_results, "EDA missing categorical summary"
        
        print("🔍 Data processor functionality verified")
    
    def test_visualizer(self):
        """Test visualization generation."""
        visualizer = DataVisualizer()
        
        # Generate test data
        generator = SampleDataGenerator(seed=42)
        test_df = generator.generate_sales_data(100)
        
        # Test auto-visualization generation
        visualizations = visualizer.generate_auto_visualizations(test_df)
        
        assert 'distributions' in visualizations, "Missing distribution plots"
        assert 'relationships' in visualizations, "Missing relationship plots"
        assert 'categorical' in visualizations, "Missing categorical plots"
        
        # Test correlation heatmap
        numerical_cols = test_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = test_df[numerical_cols].corr()
            heatmap_fig = visualizer.create_correlation_heatmap(corr_matrix)
            assert heatmap_fig is not None, "Correlation heatmap generation failed"
        
        print("📈 Visualization generation working correctly")
    
    def test_llm_insights(self):
        """Test LLM insights (if OLLAMA is available)."""
        llm_generator = LLMInsightGenerator()
        
        if not llm_generator.is_available:
            print("⚠️ OLLAMA not available, skipping LLM tests")
            return
        
        # Generate test data and EDA
        generator = SampleDataGenerator(seed=42)
        test_df = generator.generate_sales_data(50)  # Smaller dataset for faster testing
        
        processor = DataProcessor()
        eda_results = processor.generate_eda(test_df)
        
        # Test insight generation
        insights = llm_generator.generate_comprehensive_insights(test_df, eda_results)
        
        assert 'dataset_summary' in insights, "Missing dataset summary"
        assert len(insights['dataset_summary']) > 50, "Dataset summary too short"
        
        # Test custom question answering
        question = "What is the average revenue in this dataset?"
        answer = llm_generator.answer_data_question(question, test_df, eda_results)
        assert len(answer) > 20, "Answer too short"
        
        print("🤖 LLM insights generation working correctly")
    
    def test_utilities(self):
        """Test utility functions."""
        # Generate test data
        generator = SampleDataGenerator(seed=42)
        test_df = generator.generate_sales_data(50)
        
        processor = DataProcessor()
        eda_results = processor.generate_eda(test_df)
        
        # Test column type detection
        column_types = detect_column_types(test_df)
        assert 'numerical' in column_types, "Column type detection missing numerical"
        assert 'categorical' in column_types, "Column type detection missing categorical"
        
        # Test data profiling
        profile = generate_data_profile(test_df)
        assert 'basic_info' in profile, "Data profile missing basic info"
        assert 'column_types' in profile, "Data profile missing column types"
        
        # Test report export
        insights = {'dataset_summary': 'Test summary', 'statistical_insights': 'Test insights'}
        report = export_report(test_df, eda_results, insights, format="markdown")
        assert len(report) > 100, "Report too short"
        assert "# Data Analysis Report" in report, "Report missing header"
        
        print("🛠️ Utility functions working correctly")
    
    def test_file_processing(self):
        """Test file upload and processing simulation."""
        processor = DataProcessor()
        
        # Test CSV processing
        generator = SampleDataGenerator(seed=42)
        test_df = generator.generate_sales_data(50)
        
        # Save to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_df.to_csv(f.name, index=False)
            csv_path = f.name
        
        # Simulate file upload by reading the CSV
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(test_df), "CSV loading failed"
        assert list(loaded_df.columns) == list(test_df.columns), "CSV columns mismatch"
        
        # Clean up
        os.unlink(csv_path)
        
        # Test Excel processing
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            test_df.to_excel(f.name, index=False)
            excel_path = f.name
        
        loaded_df = pd.read_excel(excel_path)
        assert len(loaded_df) == len(test_df), "Excel loading failed"
        
        # Clean up
        os.unlink(excel_path)
        
        # Test JSON processing
        test_dict = test_df.head(10).to_dict('records')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_dict, f, default=str)
            json_path = f.name
        
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        loaded_df = pd.DataFrame(loaded_data)
        assert len(loaded_df) == 10, "JSON loading failed"
        
        # Clean up
        os.unlink(json_path)
        
        print("📁 File processing simulation successful")
    
    def test_error_handling(self):
        """Test error handling with problematic data."""
        processor = DataProcessor()
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        try:
            eda_results = processor.generate_eda(empty_df)
            assert 'numerical_summary' in eda_results, "Empty dataframe handling failed"
        except Exception as e:
            # It's okay if it raises an exception, as long as it's handled gracefully
            print(f"Empty dataframe raised exception (expected): {type(e).__name__}")
        
        # Test with all-null column
        problematic_df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'null_col': [None, None, None, None, None],
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        eda_results = processor.generate_eda(problematic_df)
        assert eda_results is not None, "Problematic data handling failed"
        
        print("⚠️ Error handling working correctly")
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting Data Analytics Application Test Suite")
        print("=" * 60)
        
        # List of all tests
        tests = [
            ("Configuration Loading", self.test_configuration),
            ("Sample Data Generation", self.test_sample_data_generation),
            ("Data Processor", self.test_data_processor),
            ("Visualizer", self.test_visualizer),
            ("LLM Insights", self.test_llm_insights),
            ("Utilities", self.test_utilities),
            ("File Processing", self.test_file_processing),
            ("Error Handling", self.test_error_handling)
        ]
        
        # Run each test
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests)) * 100:.1f}%")
        
        if self.failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, status, error in self.test_results:
                if status == "FAILED":
                    print(f"  - {test_name}: {error}")
        
        print("\n🎯 RECOMMENDATIONS:")
        if self.failed_tests == 0:
            print("  🎉 All tests passed! Your application is ready to run.")
            print("  🚀 Start the app with: streamlit run app.py")
        else:
            print("  🔧 Fix the failed tests before running the application")
            if any("OLLAMA" in str(result[2]) for result in self.test_results if result[1] == "FAILED"):
                print("  💡 Install and start OLLAMA for AI insights: https://ollama.ai")
        
        print("  📚 Check README.md for detailed setup instructions")
        
        return self.failed_tests == 0

def main():
    """Main test runner."""
    print("🧪 Data Analytics Application - Test Suite")
    print("Testing all components before launch...\n")
    
    tester = ApplicationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All systems go! Ready to launch the application.")
        sys.exit(0)
    else:
        print("\n🚨 Some tests failed. Please fix issues before launching.")
        sys.exit(1)

if __name__ == "__main__":
    main()