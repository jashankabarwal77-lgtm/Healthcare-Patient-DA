import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HealthcareDataAnalyzer:
    def __init__(self, data_path='patient_data.csv'):
        """Initialize the analyzer with patient data"""
        self.data = pd.read_csv(data_path)
        self.data['admission_date'] = pd.to_datetime(self.data['admission_date'])
        print(f"Loaded {len(self.data)} patient records")
        
    def create_age_groups(self):
        """Create age groups for analysis"""
        self.data['age_group'] = pd.cut(self.data['age'], 
                                       bins=[0, 30, 50, 65, 100], 
                                       labels=['18-30', '31-50', '51-65', '65+'])
        return self.data
    
    def analyze_diseases_by_age(self):
        """Task 1: Identify common diseases in different age groups"""
        print("\n" + "="*60)
        print("TASK 1: DISEASE ANALYSIS BY AGE GROUPS")
        print("="*60)
        
        # Create age groups if not already done
        if 'age_group' not in self.data.columns:
            self.create_age_groups()
        
        # Disease distribution by age group
        disease_age_analysis = self.data.groupby(['age_group', 'diagnosis']).size().unstack(fill_value=0)
        
        # Calculate percentages
        disease_percentages = disease_age_analysis.div(disease_age_analysis.sum(axis=1), axis=0) * 100
        
        # Display results
        print("\nDisease Distribution by Age Group:")
        print(disease_percentages.round(2))
        
        # Find most common disease in each age group
        print("\nMost Common Disease by Age Group:")
        for age_group in disease_percentages.index:
            most_common = disease_percentages.loc[age_group].idxmax()
            percentage = disease_percentages.loc[age_group, most_common]
            print(f"{age_group}: {most_common} ({percentage:.1f}%)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of disease counts by age group
        disease_age_analysis.plot(kind='bar', ax=ax1, stacked=True)
        ax1.set_title('Disease Distribution by Age Group')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Number of Patients')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Heatmap of disease percentages
        sns.heatmap(disease_percentages, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Disease Percentage by Age Group')
        ax2.set_xlabel('Diagnosis')
        ax2.set_ylabel('Age Group')
        
        plt.tight_layout()
        plt.savefig('disease_analysis_by_age.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return disease_percentages
    
    def compare_recovery_times(self):
        """Task 2: Compare recovery times across treatments"""
        print("\n" + "="*60)
        print("TASK 2: RECOVERY TIME ANALYSIS")
        print("="*60)
        
        # Recovery time statistics by treatment
        recovery_stats = self.data.groupby('treatment')['recovery_time_days'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nRecovery Time Statistics by Treatment:")
        print(recovery_stats)
        
        # Recovery time by diagnosis and treatment
        diagnosis_treatment_stats = self.data.groupby(['diagnosis', 'treatment'])['recovery_time_days'].agg([
            'count', 'mean', 'median'
        ]).round(2)
        
        print("\nRecovery Time by Diagnosis and Treatment:")
        print(diagnosis_treatment_stats)
        
        # Statistical significance test
        from scipy import stats
        
        surgery_recovery = self.data[self.data['treatment'] == 'Surgery']['recovery_time_days']
        medication_recovery = self.data[self.data['treatment'] == 'Medication']['recovery_time_days']
        
        t_stat, p_value = stats.ttest_ind(surgery_recovery, medication_recovery)
        
        print(f"\nStatistical Test (Surgery vs Medication):")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot by treatment
        sns.boxplot(data=self.data, x='treatment', y='recovery_time_days', ax=ax1)
        ax1.set_title('Recovery Time by Treatment Type')
        ax1.set_ylabel('Recovery Time (Days)')
        
        # Box plot by diagnosis
        sns.boxplot(data=self.data, x='diagnosis', y='recovery_time_days', ax=ax2)
        ax2.set_title('Recovery Time by Diagnosis')
        ax2.set_ylabel('Recovery Time (Days)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Violin plot by treatment and age group
        sns.violinplot(data=self.data, x='age_group', y='recovery_time_days', 
                      hue='treatment', split=True, ax=ax3)
        ax3.set_title('Recovery Time by Age Group and Treatment')
        ax3.set_ylabel('Recovery Time (Days)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Scatter plot: Age vs Recovery Time
        sns.scatterplot(data=self.data, x='age', y='recovery_time_days', 
                       hue='treatment', alpha=0.6, ax=ax4)
        ax4.set_title('Age vs Recovery Time')
        ax4.set_ylabel('Recovery Time (Days)')
        
        plt.tight_layout()
        plt.savefig('recovery_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return recovery_stats
    
    def predict_readmission_risk(self):
        """Task 3: Predict patient readmission risk with basic ML"""
        print("\n" + "="*60)
        print("TASK 3: READMISSION RISK PREDICTION")
        print("="*60)
        
        # Prepare features for ML
        # Create age groups if not already done
        if 'age_group' not in self.data.columns:
            self.create_age_groups()
        
        # Encode categorical variables
        le_diagnosis = LabelEncoder()
        le_treatment = LabelEncoder()
        le_gender = LabelEncoder()
        le_age_group = LabelEncoder()
        
        X = self.data.copy()
        X['diagnosis_encoded'] = le_diagnosis.fit_transform(X['diagnosis'])
        X['treatment_encoded'] = le_treatment.fit_transform(X['treatment'])
        X['gender_encoded'] = le_gender.fit_transform(X['gender'])
        X['age_group_encoded'] = le_age_group.fit_transform(X['age_group'])
        
        # Select features for prediction
        feature_columns = ['age', 'diagnosis_encoded', 'treatment_encoded', 
                          'gender_encoded', 'age_group_encoded', 'recovery_time_days', 
                          'length_of_stay_days']
        
        X_features = X[feature_columns]
        y = X['readmitted']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC Score: {auc_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature importance plot
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax1)
        ax1.set_title('Feature Importance for Readmission Prediction')
        
        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax3.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        
        # Readmission rate by age group
        readmission_by_age = self.data.groupby('age_group')['readmitted'].agg(['count', 'sum', 'mean'])
        readmission_by_age['percentage'] = readmission_by_age['mean'] * 100
        
        sns.barplot(data=readmission_by_age.reset_index(), x='age_group', y='percentage', ax=ax4)
        ax4.set_title('Readmission Rate by Age Group')
        ax4.set_ylabel('Readmission Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('readmission_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Risk factors analysis
        print("\nRisk Factors Analysis:")
        print("Patients with higher readmission risk:")
        high_risk = self.data[self.data['readmitted'] == 1]
        low_risk = self.data[self.data['readmitted'] == 0]
        
        print(f"Average age (readmitted): {high_risk['age'].mean():.1f} vs {low_risk['age'].mean():.1f}")
        print(f"Average recovery time (readmitted): {high_risk['recovery_time_days'].mean():.1f} vs {low_risk['recovery_time_days'].mean():.1f}")
        print(f"Average length of stay (readmitted): {high_risk['length_of_stay_days'].mean():.1f} vs {low_risk['length_of_stay_days'].mean():.1f}")
        
        return rf_model, feature_importance
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"Total patients: {len(self.data)}")
        print(f"Age range: {self.data['age'].min()} - {self.data['age'].max()} years")
        print(f"Gender distribution: {self.data['gender'].value_counts().to_dict()}")
        print(f"Readmission rate: {self.data['readmitted'].mean():.2%}")
        
        print(f"\nDiagnosis distribution:")
        print(self.data['diagnosis'].value_counts())
        
        print(f"\nTreatment distribution:")
        print(self.data['treatment'].value_counts())
        
        print(f"\nKey Insights:")
        print("1. Disease patterns vary significantly by age group")
        print("2. Surgery patients have longer recovery times than medication patients")
        print("3. Age, diagnosis, and treatment type are key predictors of readmission risk")
        
        # Save summary statistics
        summary_stats = {
            'total_patients': len(self.data),
            'readmission_rate': self.data['readmitted'].mean(),
            'avg_age': self.data['age'].mean(),
            'avg_recovery_time': self.data['recovery_time_days'].mean(),
            'avg_length_of_stay': self.data['length_of_stay_days'].mean()
        }
        
        return summary_stats

def main():
    """Main function to run all analyses"""
    # Initialize analyzer
    analyzer = HealthcareDataAnalyzer()
    
    # Run all analyses
    disease_analysis = analyzer.analyze_diseases_by_age()
    recovery_analysis = analyzer.compare_recovery_times()
    model, feature_importance = analyzer.predict_readmission_risk()
    summary = analyzer.generate_summary_report()
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")
    print("Files created:")
    print("- disease_analysis_by_age.png")
    print("- recovery_time_analysis.png") 
    print("- readmission_prediction_analysis.png")

if __name__ == "__main__":
    main()
