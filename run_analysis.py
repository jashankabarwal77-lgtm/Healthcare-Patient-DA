#!/usr/bin/env python3
"""
Healthcare Patient Data Analysis - Complete Runner Script

This script runs the entire analysis pipeline:
1. Generate patient data
2. Perform all three analytical tasks
3. Display summary results
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'scipy'
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
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to run the complete analysis"""
    print("🏥 Healthcare Patient Data Analysis")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Step 1: Generate data
    if not run_command("python data_generator.py", "Data Generation"):
        print("Failed to generate data. Exiting.")
        sys.exit(1)
    
    # Step 2: Run analysis
    if not run_command("python data_analysis.py", "Data Analysis"):
        print("Failed to run analysis. Exiting.")
        sys.exit(1)
    
    # Check for output files
    expected_files = [
        'patient_data.csv',
        'disease_analysis_by_age.png',
        'recovery_time_analysis.png',
        'readmission_prediction_analysis.png'
    ]
    
    print(f"\n{'='*60}")
    print("CHECKING OUTPUT FILES")
    print('='*60)
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - Missing")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE! 🎉")
    print('='*60)
    print("\nGenerated files:")
    print("- patient_data.csv: Patient dataset")
    print("- disease_analysis_by_age.png: Disease analysis visualizations")
    print("- recovery_time_analysis.png: Recovery time analysis charts")
    print("- readmission_prediction_analysis.png: ML model results")
    
    print("\nNext steps:")
    print("1. Review the generated PNG files for insights")
    print("2. Check the console output for detailed statistics")
    print("3. Modify the scripts to customize the analysis")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
