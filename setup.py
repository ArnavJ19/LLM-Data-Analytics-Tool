"""
Setup and Deployment Script for Data Analytics Web Application

This script helps users set up the environment, install dependencies,
and get the application running quickly.
"""

import subprocess
import sys
import os
import platform
import urllib.request
import json
from pathlib import Path
from typing import List, Dict, Any

class SetupManager:
    """Manages the setup and deployment of the application."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.setup_log = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log setup messages."""
        print(f"[{level}] {message}")
        self.setup_log.append(f"[{level}] {message}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        self.log("Checking Python version...")
        
        if self.python_version >= (3, 8):
            self.log(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible")
            return True
        else:
            self.log(f"❌ Python {self.python_version.major}.{self.python_version.minor} is too old. Requires Python 3.8+", "ERROR")
            return False
    
    def check_pip(self) -> bool:
        """Check if pip is available."""
        self.log("Checking pip availability...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            self.log("✅ pip is available")
            return True
        except subprocess.CalledProcessError:
            self.log("❌ pip is not available", "ERROR")
            return False
    
    def install_requirements(self) -> bool:
        """Install Python requirements."""
        self.log("Installing Python requirements...")
        
        requirements_file = "requirements.txt"
        if not os.path.exists(requirements_file):
            self.log("❌ requirements.txt not found", "ERROR")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], check=True)
            self.log("✅ Python requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to install requirements: {e}", "ERROR")
            return False
    
    def check_ollama_availability(self) -> Dict[str, Any]:
        """Check if OLLAMA is installed and running."""
        self.log("Checking OLLAMA availability...")
        
        ollama_status = {
            'installed': False,
            'running': False,
            'models': [],
            'recommended_model': 'qwen3:8b'
        }
        
        # Check if OLLAMA is installed
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                ollama_status['installed'] = True
                self.log("✅ OLLAMA is installed")
            else:
                self.log("❌ OLLAMA is not installed", "WARNING")
        except FileNotFoundError:
            self.log("❌ OLLAMA is not installed", "WARNING")
        
        # Check if OLLAMA is running and get models
        if ollama_status['installed']:
            try:
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    ollama_status['running'] = True
                    self.log("✅ OLLAMA is running")
                    
                    # Parse model list
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    models = [line.split()[0] for line in lines if line.strip()]
                    ollama_status['models'] = models
                    
                    if models:
                        self.log(f"📋 Available models: {', '.join(models)}")
                    else:
                        self.log("⚠️ No models found", "WARNING")
                else:
                    self.log("❌ OLLAMA is installed but not running", "WARNING")
            except Exception as e:
                self.log(f"❌ Error checking OLLAMA status: {e}", "WARNING")
        
        return ollama_status
    
    def install_ollama(self) -> bool:
        """Provide instructions for installing OLLAMA."""
        self.log("OLLAMA Installation Instructions:")
        
        if self.system == "darwin":  # macOS
            self.log("📥 Download OLLAMA for macOS from: https://ollama.ai")
            self.log("🔧 Or install via Homebrew: brew install ollama")
        elif self.system == "linux":
            self.log("🔧 Install OLLAMA on Linux:")
            self.log("   curl -fsSL https://ollama.ai/install.sh | sh")
        elif self.system == "windows":
            self.log("📥 Download OLLAMA for Windows from: https://ollama.ai")
        else:
            self.log("📥 Visit https://ollama.ai for installation instructions")
        
        self.log("⚠️ After installation, run: ollama serve")
        return False  # Manual installation required
    
    def setup_ollama_model(self, model_name: str = "qwen2.5:8b") -> bool:
        """Setup recommended OLLAMA model."""
        self.log(f"Setting up OLLAMA model: {model_name}")
        
        try:
            self.log("📥 Pulling model (this may take several minutes)...")
            subprocess.run(['ollama', 'pull', model_name], check=True)
            self.log(f"✅ Model {model_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed to install model {model_name}: {e}", "ERROR")
            return False
        except FileNotFoundError:
            self.log("❌ OLLAMA not found. Please install OLLAMA first.", "ERROR")
            return False
    
    def generate_sample_data(self) -> bool:
        """Generate sample datasets for testing."""
        self.log("Generating sample datasets...")
        
        try:
            from sample_data_generator import SampleDataGenerator
            
            generator = SampleDataGenerator()
            generator.save_sample_datasets()
            
            self.log("✅ Sample datasets generated in 'sample_datasets' folder")
            return True
        except Exception as e:
            self.log(f"❌ Failed to generate sample data: {e}", "ERROR")
            return False
    
    def test_application(self) -> bool:
        """Run application tests."""
        self.log("Running application tests...")
        
        try:
            result = subprocess.run([sys.executable, "test_app.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ All application tests passed")
                return True
            else:
                self.log("❌ Some application tests failed", "WARNING")
                self.log(f"Test output: {result.stdout}", "INFO")
                return False
        except Exception as e:
            self.log(f"❌ Failed to run tests: {e}", "ERROR")
            return False
    
    def create_startup_script(self) -> bool:
        """Create a startup script for easy launching."""
        self.log("Creating startup script...")
        
        if self.system == "windows":
            script_content = """@echo off
echo Starting Data Analytics Web Application...
python -m streamlit run app.py
pause
"""
            script_name = "start_app.bat"
        else:
            script_content = """#!/bin/bash
echo "Starting Data Analytics Web Application..."
python -m streamlit run app.py
"""
            script_name = "start_app.sh"
        
        try:
            with open(script_name, 'w') as f:
                f.write(script_content)
            
            if self.system != "windows":
                os.chmod(script_name, 0o755)  # Make executable
            
            self.log(f"✅ Startup script created: {script_name}")
            return True
        except Exception as e:
            self.log(f"❌ Failed to create startup script: {e}", "ERROR")
            return False
    
    def print_setup_summary(self, setup_results: Dict[str, bool]):
        """Print a summary of the setup process."""
        self.log("\n" + "="*60)
        self.log("📊 SETUP SUMMARY")
        self.log("="*60)
        
        for component, success in setup_results.items():
            status = "✅ READY" if success else "❌ NEEDS ATTENTION"
            self.log(f"{component}: {status}")
        
        all_ready = all(setup_results.values())
        
        if all_ready:
            self.log("\n🎉 Setup completed successfully!")
            self.log("🚀 You can now start the application:")
            self.log("   python -m streamlit run app.py")
            self.log("   or run the startup script created for you")
        else:
            self.log("\n⚠️ Setup completed with some issues")
            self.log("📋 Please address the items marked as 'NEEDS ATTENTION'")
        
        self.log("\n💡 NEXT STEPS:")
        self.log("1. Start OLLAMA if not running: ollama serve")
        self.log("2. Launch the application: streamlit run app.py")
        self.log("3. Open your browser to the URL shown by Streamlit")
        self.log("4. Upload a dataset or use the sample data provided")
        
        return all_ready
    
    def run_full_setup(self, skip_ollama: bool = False, skip_tests: bool = False):
        """Run the complete setup process."""
        self.log("🚀 Starting Data Analytics Application Setup")
        self.log("="*60)
        
        setup_results = {}
        
        # Core requirements
        setup_results['Python Version'] = self.check_python_version()
        setup_results['pip Available'] = self.check_pip()
        setup_results['Python Requirements'] = self.install_requirements()
        
        # OLLAMA setup (optional but recommended)
        if not skip_ollama:
            ollama_status = self.check_ollama_availability()
            
            if not ollama_status['installed']:
                self.install_ollama()
                setup_results['OLLAMA Installed'] = False
            else:
                setup_results['OLLAMA Installed'] = True
                
                if not ollama_status['running']:
                    self.log("⚠️ OLLAMA is installed but not running", "WARNING")
                    self.log("💡 Start OLLAMA with: ollama serve")
                    setup_results['OLLAMA Running'] = False
                else:
                    setup_results['OLLAMA Running'] = True
                    
                    # Check for recommended model
                    if ollama_status['recommended_model'] not in ollama_status['models']:
                        self.log(f"⚠️ Recommended model {ollama_status['recommended_model']} not found")
                        
                        if input("📥 Would you like to install the recommended model? (y/n): ").lower() == 'y':
                            setup_results['OLLAMA Model'] = self.setup_ollama_model()
                        else:
                            setup_results['OLLAMA Model'] = False
                    else:
                        setup_results['OLLAMA Model'] = True
        
        # Generate sample data
        setup_results['Sample Data'] = self.generate_sample_data()
        
        # Run tests
        if not skip_tests:
            setup_results['Application Tests'] = self.test_application()
        
        # Create startup script
        setup_results['Startup Script'] = self.create_startup_script()
        
        # Print summary
        return self.print_setup_summary(setup_results)

def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Data Analytics Web Application")
    parser.add_argument("--skip-ollama", action="store_true", 
                       help="Skip OLLAMA installation and setup")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip application tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick setup (skip OLLAMA and tests)")
    
    args = parser.parse_args()
    
    setup_manager = SetupManager()
    
    if args.quick:
        success = setup_manager.run_full_setup(skip_ollama=True, skip_tests=True)
    else:
        success = setup_manager.run_full_setup(
            skip_ollama=args.skip_ollama, 
            skip_tests=args.skip_tests
        )
    
    if success:
        print("\n🎊 Setup completed successfully! Ready to launch.")
        sys.exit(0)
    else:
        print("\n⚠️ Setup completed with some issues. Check the summary above.")
        sys.exit(1)

if __name__ == "__main__":
    main()