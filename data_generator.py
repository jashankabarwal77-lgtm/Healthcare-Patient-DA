import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_patient_data(n_patients=1000):
    """
    Generate realistic patient data for healthcare analysis
    """
    np.random.seed(42)
    random.seed(42)
    
    # Age distribution (more patients in middle age)
    ages = np.random.normal(55, 20, n_patients)
    ages = np.clip(ages, 18, 95).astype(int)
    
    # Gender
    genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    
    # Common diseases by age group
    diseases = []
    for age in ages:
        if age < 30:
            diseases.append(np.random.choice(['Appendicitis', 'Pneumonia', 'Diabetes Type 1', 'Asthma'], p=[0.3, 0.25, 0.25, 0.2]))
        elif age < 50:
            diseases.append(np.random.choice(['Hypertension', 'Diabetes Type 2', 'Heart Disease', 'Cancer', 'Pneumonia'], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
        else:
            diseases.append(np.random.choice(['Heart Disease', 'Cancer', 'Diabetes Type 2', 'Hypertension', 'Stroke', 'Pneumonia'], p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]))
    
    # Treatment types
    treatments = []
    for disease in diseases:
        if disease in ['Appendicitis', 'Cancer']:
            treatments.append('Surgery')
        elif disease in ['Pneumonia', 'Asthma']:
            treatments.append('Medication')
        elif disease in ['Heart Disease', 'Stroke']:
            treatments.append(np.random.choice(['Surgery', 'Medication'], p=[0.6, 0.4]))
        else:
            treatments.append('Medication')
    
    # Recovery times (in days) based on treatment and age
    recovery_times = []
    for i, (age, treatment, disease) in enumerate(zip(ages, treatments, diseases)):
        base_time = 0
        if treatment == 'Surgery':
            base_time = np.random.normal(14, 5)
        else:
            base_time = np.random.normal(7, 3)
        
        # Age factor
        if age > 65:
            base_time *= 1.5
        elif age > 50:
            base_time *= 1.2
        
        # Disease factor
        if disease == 'Cancer':
            base_time *= 2
        elif disease == 'Stroke':
            base_time *= 1.8
        
        recovery_times.append(max(1, int(base_time)))
    
    # Readmission risk factors
    readmission_risk = []
    for i, (age, treatment, disease, recovery_time) in enumerate(zip(ages, treatments, diseases, recovery_times)):
        risk = 0.1  # Base risk
        
        # Age factor
        if age > 70:
            risk += 0.2
        elif age > 60:
            risk += 0.1
        
        # Treatment factor
        if treatment == 'Surgery':
            risk += 0.15
        
        # Disease factor
        if disease in ['Cancer', 'Heart Disease', 'Stroke']:
            risk += 0.2
        
        # Recovery time factor
        if recovery_time > 20:
            risk += 0.1
        
        readmission_risk.append(min(0.9, risk))
    
    # Generate readmission outcomes
    readmitted = np.random.binomial(1, readmission_risk, n_patients)
    
    # Admission dates (last 2 years)
    start_date = datetime(2022, 1, 1)
    admission_dates = []
    for _ in range(n_patients):
        days_offset = np.random.randint(0, 730)  # 2 years
        admission_dates.append(start_date + timedelta(days=days_offset))
    
    # Length of stay (correlated with recovery time)
    length_of_stay = []
    for recovery_time in recovery_times:
        if recovery_time > 15:
            los = np.random.normal(recovery_time * 0.3, 2)
        else:
            los = np.random.normal(recovery_time * 0.4, 1.5)
        length_of_stay.append(max(1, int(los)))
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': ages,
        'gender': genders,
        'diagnosis': diseases,
        'treatment': treatments,
        'recovery_time_days': recovery_times,
        'length_of_stay_days': length_of_stay,
        'admission_date': admission_dates,
        'readmission_risk': readmission_risk,
        'readmitted': readmitted
    })
    
    return data

if __name__ == "__main__":
    # Generate data
    patient_data = generate_patient_data(1000)
    
    # Save to CSV
    patient_data.to_csv('patient_data.csv', index=False)
    print(f"Generated {len(patient_data)} patient records")
    print("Data saved to 'patient_data.csv'")
    
    # Display sample
    print("\nSample data:")
    print(patient_data.head())
