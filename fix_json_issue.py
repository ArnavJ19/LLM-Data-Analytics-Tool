#!/usr/bin/env python3
"""
OLLAMA Timeout Fix Script for Data Analytics Application

This script diagnoses and fixes OLLAMA timeout issues.
"""

import subprocess
import requests
import time
import json
import sys
import os

class OllamaTimeoutFixer:
    """Diagnose and fix OLLAMA timeout issues."""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "qwen3:8b"
        self.alternative_models = ["qwen3:8b", "llama3.1:8b", "mistral:7b", "phi3:mini"]
    
    def check_ollama_status(self):
        """Check OLLAMA server status."""
        print("🔍 Checking OLLAMA status...")
        
        try:
            # Check if OLLAMA is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✅ OLLAMA is running with {len(models)} models loaded")
                
                # List available models
                if models:
                    print("📋 Available models:")
                    for model in models:
                        size_mb = model.get('size', 0) / (1024*1024)
                        print(f"  - {model['name']} ({size_mb:.1f} MB)")
                else:
                    print("⚠️ No models found")
                
                return True, models
            else:
                print(f"❌ OLLAMA responded with status {response.status_code}")
                return False, []
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to OLLAMA. Server is not running.")
            return False, []
        except requests.exceptions.Timeout:
            print("⏰ OLLAMA server is running but very slow to respond")
            return True, []  # Running but slow
        except Exception as e:
            print(f"❌ Error checking OLLAMA: {e}")
            return False, []
    
    def test_model_performance(self, model_name):
        """Test model response time with a simple prompt."""
        print(f"🧪 Testing model performance: {model_name}")
        
        test_prompt = "Hello! Please respond with just 'OK' to test performance."
        
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.1
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                result = response.json()
                response_text = result.get('response', '')
                
                print(f"✅ Model responded in {response_time:.2f} seconds")
                print(f"📝 Response: {response_text.strip()}")
                
                return True, response_time
            else:
                print(f"❌ Model request failed: {response.status_code}")
                return False, 0
                
        except requests.exceptions.Timeout:
            print("⏰ Model test timed out (>30s)")
            return False, 30
        except Exception as e:
            print(f"❌ Model test error: {e}")
            return False, 0
    
    def restart_ollama_suggestion(self):
        """Provide instructions for restarting OLLAMA."""
        print("\n🔄 OLLAMA Restart Instructions:")
        print("=" * 40)
        
        # Check OS
        import platform
        system = platform.system().lower()
        
        if system == "windows":
            print("Windows:")
            print("1. Press Ctrl+C in the OLLAMA terminal")
            print("2. Wait for it to stop completely") 
            print("3. Run: ollama serve")
        elif system == "darwin":  # macOS
            print("macOS:")
            print("1. If running in terminal: Press Ctrl+C")
            print("2. If running as service: brew services restart ollama")
            print("3. Or manually: ollama serve")
        else:  # Linux
            print("Linux:")
            print("1. If running in terminal: Press Ctrl+C")
            print("2. If running as systemd service:")
            print("   sudo systemctl restart ollama")
            print("3. Or manually: ollama serve")
        
        print("\n💡 After restarting, wait 30 seconds before testing again")
    
    def suggest_model_alternatives(self, available_models):
        """Suggest faster model alternatives."""
        print("\n🚀 Model Performance Recommendations:")
        print("=" * 45)
        
        available_model_names = [model['name'] for model in available_models]
        
        # Recommend models by size/speed
        recommendations = [
            ("phi3:mini", "Fastest, smallest model (3.8B parameters)"),
            ("qwen2.5:7b", "Good balance of speed and quality"),
            ("mistral:7b", "Fast and efficient"),
            ("llama3.1:8b", "Good quality, moderate speed"),
            ("qwen3:8b", "Current model (slower but higher quality)")
        ]
        
        print("Models ranked by speed (fastest first):")
        for model, description in recommendations:
            if any(model in available for available in available_model_names):
                status = "✅ Available"
            else:
                status = "⬇️ Need to pull"
            
            print(f"  {model}: {description} - {status}")
        
        print("\n📥 To install a faster model:")
        print("   ollama pull phi3:mini")
        print("   ollama pull mistral:7b")
    
    def update_app_timeout_settings(self):
        """Update application timeout settings."""
        print("\n⚙️ Updating application timeout settings...")
        
        # Check if config.py exists and update timeout
        if os.path.exists('config.py'):
            try:
                with open('config.py', 'r') as f:
                    content = f.read()
                
                # Update timeout settings
                if 'timeout: int = 30' in content:
                    content = content.replace('timeout: int = 30', 'timeout: int = 120')
                    
                with open('config.py', 'w') as f:
                    f.write(content)
                
                print("✅ Updated default timeout to 120 seconds")
                
            except Exception as e:
                print(f"⚠️ Could not update config.py: {e}")
        
        # Create or update .env file
        env_content = """# OLLAMA Timeout Settings
OLLAMA_TIMEOUT=120
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
"""
        
        with open('.env.timeout', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env.timeout with optimized settings")
        print("💡 Copy these settings to your .env file if needed")
    
    def run_comprehensive_fix(self):
        """Run comprehensive OLLAMA timeout diagnostics and fixes."""
        print("🔧 OLLAMA Timeout Diagnostic and Fix Tool")
        print("=" * 50)
        
        # Step 1: Check OLLAMA status
        is_running, models = self.check_ollama_status()
        
        if not is_running:
            print("\n❌ OLLAMA is not running!")
            self.restart_ollama_suggestion()
            return False
        
        if not models:
            print("\n⚠️ No models found. Installing recommended fast model...")
            try:
                subprocess.run(['ollama', 'pull', 'phi3:mini'], check=True, timeout=300)
                print("✅ Installed phi3:mini model")
            except Exception as e:
                print(f"❌ Failed to install model: {e}")
                return False
        
        # Step 2: Test current model performance
        current_model_works = False
        if any(self.model_name in model['name'] for model in models):
            success, response_time = self.test_model_performance(self.model_name)
            if success and response_time < 30:
                print(f"✅ Current model {self.model_name} is working well!")
                current_model_works = True
            elif success:
                print(f"⚠️ Current model {self.model_name} is slow ({response_time:.1f}s)")
            else:
                print(f"❌ Current model {self.model_name} is not responding")
        
        # Step 3: Test alternative models if current model is slow
        if not current_model_works:
            print("\n🔍 Testing alternative models...")
            best_model = None
            best_time = float('inf')
            
            available_model_names = [model['name'] for model in models]
            
            for alt_model in self.alternative_models:
                if any(alt_model in available for available in available_model_names):
                    success, response_time = self.test_model_performance(alt_model)
                    if success and response_time < best_time:
                        best_model = alt_model
                        best_time = response_time
            
            if best_model:
                print(f"\n✅ Recommended model: {best_model} ({best_time:.1f}s response time)")
                
                # Update app to use best model
                self.model_name = best_model
            else:
                print("\n⚠️ All models are slow or unresponsive")
                self.suggest_model_alternatives(models)
        
        # Step 4: Update application settings
        self.update_app_timeout_settings()
        
        # Step 5: Final recommendations
        print(f"\n🎯 Final Recommendations:")
        print("=" * 30)
        
        if current_model_works:
            print("✅ Your current setup is working fine!")
            print("💡 The timeout errors may be intermittent")
        else:
            print(f"🔄 Switch to faster model: {self.model_name}")
            print("⏰ Increased timeout settings to 120 seconds")
        
        print("\n🚀 Next Steps:")
        print("1. Restart your Streamlit app: streamlit run app.py")
        print("2. Try the AI Insights feature again")
        print("3. If still slow, consider using phi3:mini model")
        
        return True

def main():
    """Main function."""
    fixer = OllamaTimeoutFixer()
    success = fixer.run_comprehensive_fix()
    
    if success:
        print("\n🎉 OLLAMA timeout fixes applied!")
        print("Your application should now work better with AI insights.")
    else:
        print("\n⚠️ Some issues remain. Please follow the manual steps above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)