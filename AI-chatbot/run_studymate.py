#!/usr/bin/env python3
"""
StudyMate Launcher - Easy startup script for StudyMate AI Assistant
Automatically detects and uses the best available AI model without requiring cloud services
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is running locally"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'streamlit', 'PyMuPDF', 'numpy', 'faiss-cpu', 
        'sentence-transformers', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError:
        return False

def print_ai_status():
    """Print the status of available AI models"""
    print("\n" + "="*60)
    print("🤖 AI MODEL STATUS")
    print("="*60)
    
    # Check Ollama
    if check_ollama():
        print("✅ Ollama: Running locally (FREE, PRIVATE, OFFLINE)")
        print("   → Best option! No API keys needed, completely private")
    else:
        print("❌ Ollama: Not running")
        print("   → Install from https://ollama.ai for best experience")
    
    # Check OpenAI
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI: API key found (PAID, HIGH QUALITY)")
    else:
        print("❌ OpenAI: No API key found")
        print("   → Set OPENAI_API_KEY environment variable to use")
    
    # Check Watsonx
    if os.getenv('WATSONX_API_KEY') and os.getenv('WATSONX_PROJECT_ID'):
        print("✅ IBM Watsonx: Credentials found (PAID, ENTERPRISE)")
    else:
        print("❌ IBM Watsonx: No credentials found")
        print("   → Set WATSONX_API_KEY and WATSONX_PROJECT_ID to use")
    
    # Hugging Face (always available)
    print("✅ Hugging Face: Available (FREE, ONLINE)")
    print("   → Free tier with rate limits")
    
    # Fallback (always available)
    print("✅ Fallback System: Available (FREE, BASIC)")
    print("   → Simple context-based responses, no AI model needed")
    
    print("="*60)

def main():
    """Main launcher function"""
    print("🚀 Starting StudyMate AI Assistant...")
    print("📚 AI-Powered Academic Assistant for PDF Documents")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n⚠️  Missing required packages: {', '.join(missing)}")
        print("Installing packages automatically...")
        
        if not install_packages(missing):
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing)}")
            return 1
        
        print("✅ Packages installed successfully!")
    
    # Print AI model status
    print_ai_status()
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not check_ollama():
        print("   • Install Ollama for the best free, private experience")
        print("     1. Visit https://ollama.ai")
        print("     2. Download and install Ollama")
        print("     3. Run: ollama pull llama2")
        print("     4. Restart StudyMate")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("   • For high-quality responses, get an OpenAI API key")
        print("     1. Visit https://openai.com/api")
        print("     2. Set environment variable: OPENAI_API_KEY=your_key")
    
    print("\n🎯 StudyMate will automatically use the best available AI model!")
    print("   No configuration needed - just upload your PDFs and start asking questions!")
    
    # Start Streamlit
    print("\n🌐 Starting web interface...")
    try:
        # Use app.py as the main application
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"])
    except KeyboardInterrupt:
        print("\n👋 StudyMate stopped. Thanks for using StudyMate!")
    except Exception as e:
        print(f"\n❌ Error starting StudyMate: {e}")
        print("Please make sure all dependencies are installed and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
