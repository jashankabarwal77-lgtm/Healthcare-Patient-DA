#!/usr/bin/env python3
"""
Healthcare Patient Data Analysis - Demo Script

This script provides a quick demonstration of the analysis results
"""

import pandas as pd
import numpy as np

def show_demo_results():
    """Display key results from the analysis"""
    
    print("🏥 Healthcare Patient Data Analysis - Demo Results")
    print("="*60)
    
    # Load the data
    try:
        data = pd.read_csv('patient_data.csv')
        print(f"✅ Successfully loaded {len(data)} patient records")
    except FileNotFoundError:
        print("❌ Data file not found. Please run 'python data_generator.py' first.")
        return
    
    # Basic statistics
    print(f"\n📊 Dataset Overview:")
    print(f"   Total patients: {len(data):,}")
    print(f"   Age range: {data['age'].min()} - {data['age'].max()} years")
    print(f"   Gender distribution: {dict(data['gender'].value_counts())}")
    print(f"   Readmission rate: {data['readmitted'].mean():.1%}")
    
    # Disease analysis
    print(f"\n🏥 Disease Analysis by Age Groups:")
    data['age_group'] = pd.cut(data['age'], 
                               bins=[0, 30, 50, 65, 100], 
                               labels=['18-30', '31-50', '51-65', '65+'])
    
    for age_group in ['18-30', '31-50', '51-65', '65+']:
        group_data = data[data['age_group'] == age_group]
        if len(group_data) > 0:
            most_common = group_data['diagnosis'].mode().iloc[0]
            percentage = (group_data['diagnosis'] == most_common).mean() * 100
            print(f"   {age_group}: {most_common} ({percentage:.1f}%)")
    
    # Recovery time analysis
    print(f"\n⏱️  Recovery Time Analysis:")
    surgery_avg = data[data['treatment'] == 'Surgery']['recovery_time_days'].mean()
    medication_avg = data[data['treatment'] == 'Medication']['recovery_time_days'].mean()
    print(f"   Surgery average: {surgery_avg:.1f} days")
    print(f"   Medication average: {medication_avg:.1f} days")
    print(f"   Difference: {surgery_avg - medication_avg:.1f} days")
    
    # Readmission risk factors
    print(f"\n⚠️  Readmission Risk Factors:")
    high_risk = data[data['readmitted'] == 1]
    low_risk = data[data['readmitted'] == 0]
    
    print(f"   High-risk patients (readmitted):")
    print(f"     Average age: {high_risk['age'].mean():.1f} years")
    print(f"     Average recovery time: {high_risk['recovery_time_days'].mean():.1f} days")
    print(f"     Average length of stay: {high_risk['length_of_stay_days'].mean():.1f} days")
    
    print(f"   Low-risk patients (not readmitted):")
    print(f"     Average age: {low_risk['age'].mean():.1f} years")
    print(f"     Average recovery time: {low_risk['recovery_time_days'].mean():.1f} days")
    print(f"     Average length of stay: {low_risk['length_of_stay_days'].mean():.1f} days")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    print(f"   1. Disease patterns vary significantly by age group")
    print(f"   2. Surgery requires {surgery_avg - medication_avg:.1f} more days recovery than medication")
    print(f"   3. Readmitted patients are {high_risk['age'].mean() - low_risk['age'].mean():.1f} years older on average")
    print(f"   4. The dataset shows realistic healthcare patterns")
    
    print(f"\n📁 Generated Files:")
    print(f"   - patient_data.csv: Patient dataset")
    print(f"   - disease_analysis_by_age.png: Disease analysis visualizations")
    print(f"   - recovery_time_analysis.png: Recovery time analysis charts")
    print(f"   - readmission_prediction_analysis.png: ML model results")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run 'python data_analysis.py' for complete analysis")
    print(f"   2. Review the generated PNG files for detailed visualizations")
    print(f"   3. Modify the scripts to customize the analysis")
    print(f"   4. Check README.md for detailed documentation")

if __name__ == "__main__":
    show_demo_results()
