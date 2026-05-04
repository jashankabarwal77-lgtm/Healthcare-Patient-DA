#!/usr/bin/env python3
"""
Healthcare Patient Data Analysis Dashboard Launcher

This script launches the interactive Streamlit dashboard with all features.
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'seaborn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_data_file():
    """Check if patient data exists"""
    if not os.path.exists('patient_data.csv'):
        print("📊 Generating patient data...")
        try:
            subprocess.check_call([sys.executable, "data_generator.py"])
            print("✅ Patient data generated successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate patient data.")
            return False
    else:
        print("✅ Patient data found!")
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching Healthcare Dashboard...")
    print("="*50)
    print("Dashboard Features:")
    print("📊 Overview - Key metrics and summary")
    print("🏥 Disease Analysis - Disease patterns by age")
    print("⏱️ Recovery Analysis - Treatment effectiveness")
    print("🤖 ML Prediction - Readmission risk prediction")
    print("📈 Trends - Temporal analysis")
    print("📋 Data Explorer - Raw data and filters")
    print("="*50)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:8501')
    
    # Start browser thread
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main function"""
    print("🏥 Healthcare Patient Data Analysis Dashboard")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data file
    if not check_data_file():
        sys.exit(1)
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
