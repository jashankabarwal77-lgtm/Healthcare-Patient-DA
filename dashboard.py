import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Healthcare Patient Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the patient data"""
    try:
        data = pd.read_csv('patient_data.csv')
        data['admission_date'] = pd.to_datetime(data['admission_date'])
        data['age_group'] = pd.cut(data['age'], 
                                   bins=[0, 30, 50, 65, 100], 
                                   labels=['18-30', '31-50', '51-65', '65+'])
        return data
    except FileNotFoundError:
        st.error("❌ Data file not found. Please run 'python data_generator.py' first.")
        return None

@st.cache_data
def train_ml_model(data):
    """Train and cache the ML model"""
    # Prepare features
    le_diagnosis = LabelEncoder()
    le_treatment = LabelEncoder()
    le_gender = LabelEncoder()
    le_age_group = LabelEncoder()
    
    X = data.copy()
    X['diagnosis_encoded'] = le_diagnosis.fit_transform(X['diagnosis'])
    X['treatment_encoded'] = le_treatment.fit_transform(X['treatment'])
    X['gender_encoded'] = le_gender.fit_transform(X['gender'])
    X['age_group_encoded'] = le_age_group.fit_transform(X['age_group'])
    
    feature_columns = ['age', 'diagnosis_encoded', 'treatment_encoded', 
                      'gender_encoded', 'age_group_encoded', 'recovery_time_days', 
                      'length_of_stay_days']
    
    X_features = X[feature_columns]
    y = X['readmitted']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_test, y_test, feature_columns, le_diagnosis, le_treatment, le_gender, le_age_group

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare Patient Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(data['age'].min()),
        max_value=int(data['age'].max()),
        value=(int(data['age'].min()), int(data['age'].max()))
    )
    
    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=data['gender'].unique(),
        default=data['gender'].unique()
    )
    
    # Diagnosis filter
    diagnosis_filter = st.sidebar.multiselect(
        "Diagnosis",
        options=data['diagnosis'].unique(),
        default=data['diagnosis'].unique()
    )
    
    # Treatment filter
    treatment_filter = st.sidebar.multiselect(
        "Treatment",
        options=data['treatment'].unique(),
        default=data['treatment'].unique()
    )
    
    # Age group filter
    age_group_filter = st.sidebar.multiselect(
        "Age Group",
        options=data['age_group'].unique(),
        default=data['age_group'].unique()
    )
    
    # Readmission filter
    readmission_filter = st.sidebar.multiselect(
        "Readmission Status",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Readmitted" if x == 1 else "Not Readmitted"
    )
    
    # Apply filters
    filtered_data = data[
        (data['age'].between(age_range[0], age_range[1])) &
        (data['gender'].isin(gender_filter)) &
        (data['diagnosis'].isin(diagnosis_filter)) &
        (data['treatment'].isin(treatment_filter)) &
        (data['age_group'].isin(age_group_filter)) &
        (data['readmitted'].isin(readmission_filter))
    ]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_data):,} / {len(data):,}")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🏥 Disease Analysis", "⏱️ Recovery Analysis", 
        "🤖 ML Prediction", "📈 Trends", "📋 Data Explorer"
    ])
    
    with tab1:
        show_overview_tab(filtered_data, data)
    
    with tab2:
        show_disease_analysis_tab(filtered_data)
    
    with tab3:
        show_recovery_analysis_tab(filtered_data)
    
    with tab4:
        show_ml_prediction_tab(filtered_data, data)
    
    with tab5:
        show_trends_tab(filtered_data)
    
    with tab6:
        show_data_explorer_tab(filtered_data)

def show_overview_tab(filtered_data, original_data):
    """Overview tab with key metrics and summary"""
    st.markdown("## 📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=f"{len(filtered_data):,}",
            delta=f"{len(filtered_data) - len(original_data):,}" if len(filtered_data) != len(original_data) else None
        )
    
    with col2:
        readmission_rate = filtered_data['readmitted'].mean() * 100
        st.metric(
            label="Readmission Rate",
            value=f"{readmission_rate:.1f}%"
        )
    
    with col3:
        avg_age = filtered_data['age'].mean()
        st.metric(
            label="Average Age",
            value=f"{avg_age:.1f} years"
        )
    
    with col4:
        avg_recovery = filtered_data['recovery_time_days'].mean()
        st.metric(
            label="Avg Recovery Time",
            value=f"{avg_recovery:.1f} days"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            filtered_data, x='age', nbins=20,
            title="Age Distribution",
            labels={'age': 'Age', 'count': 'Number of Patients'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = filtered_data['gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values, names=gender_counts.index,
            title="Gender Distribution"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Diagnosis and treatment summary
    col1, col2 = st.columns(2)
    
    with col1:
        diagnosis_counts = filtered_data['diagnosis'].value_counts()
        fig_diagnosis = px.bar(
            x=diagnosis_counts.index, y=diagnosis_counts.values,
            title="Diagnosis Distribution",
            labels={'x': 'Diagnosis', 'y': 'Number of Patients'}
        )
        fig_diagnosis.update_xaxes(tickangle=45)
        st.plotly_chart(fig_diagnosis, use_container_width=True)
    
    with col2:
        treatment_counts = filtered_data['treatment'].value_counts()
        fig_treatment = px.bar(
            x=treatment_counts.index, y=treatment_counts.values,
            title="Treatment Distribution",
            labels={'x': 'Treatment', 'y': 'Number of Patients'}
        )
        st.plotly_chart(fig_treatment, use_container_width=True)

def show_disease_analysis_tab(filtered_data):
    """Disease analysis tab"""
    st.markdown("## 🏥 Disease Analysis by Age Groups")
    
    # Disease by age group heatmap
    disease_age_pivot = pd.crosstab(filtered_data['age_group'], filtered_data['diagnosis'])
    disease_age_percentages = disease_age_pivot.div(disease_age_pivot.sum(axis=1), axis=0) * 100
    
    fig_heatmap = px.imshow(
        disease_age_percentages,
        title="Disease Percentage by Age Group",
        labels=dict(x="Diagnosis", y="Age Group", color="Percentage"),
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Most common diseases by age group
    st.markdown("### Most Common Diseases by Age Group")
    
    age_groups = filtered_data['age_group'].unique()
    for age_group in age_groups:
        group_data = filtered_data[filtered_data['age_group'] == age_group]
        if len(group_data) > 0:
            most_common = group_data['diagnosis'].mode().iloc[0]
            percentage = (group_data['diagnosis'] == most_common).mean() * 100
            st.markdown(f"**{age_group}**: {most_common} ({percentage:.1f}%)")
    
    # Disease trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease vs recovery time
        fig_disease_recovery = px.box(
            filtered_data, x='diagnosis', y='recovery_time_days',
            title="Recovery Time by Diagnosis",
            labels={'diagnosis': 'Diagnosis', 'recovery_time_days': 'Recovery Time (Days)'}
        )
        fig_disease_recovery.update_xaxes(tickangle=45)
        st.plotly_chart(fig_disease_recovery, use_container_width=True)
    
    with col2:
        # Disease vs readmission rate
        disease_readmission = filtered_data.groupby('diagnosis')['readmitted'].agg(['count', 'mean']).reset_index()
        disease_readmission['readmission_rate'] = disease_readmission['mean'] * 100
        
        fig_disease_readmission = px.bar(
            disease_readmission, x='diagnosis', y='readmission_rate',
            title="Readmission Rate by Diagnosis",
            labels={'diagnosis': 'Diagnosis', 'readmission_rate': 'Readmission Rate (%)'}
        )
        fig_disease_readmission.update_xaxes(tickangle=45)
        st.plotly_chart(fig_disease_readmission, use_container_width=True)

def show_recovery_analysis_tab(filtered_data):
    """Recovery time analysis tab"""
    st.markdown("## ⏱️ Recovery Time Analysis")
    
    # Recovery time statistics
    col1, col2 = st.columns(2)
    
    with col1:
        recovery_stats = filtered_data.groupby('treatment')['recovery_time_days'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        st.markdown("### Recovery Time Statistics by Treatment")
        st.dataframe(recovery_stats)
    
    with col2:
        # Treatment comparison
        fig_treatment_recovery = px.box(
            filtered_data, x='treatment', y='recovery_time_days',
            title="Recovery Time by Treatment Type",
            labels={'treatment': 'Treatment', 'recovery_time_days': 'Recovery Time (Days)'}
        )
        st.plotly_chart(fig_treatment_recovery, use_container_width=True)
    
    # Age vs Recovery time
    fig_age_recovery = px.scatter(
        filtered_data, x='age', y='recovery_time_days', color='treatment',
        title="Age vs Recovery Time by Treatment",
        labels={'age': 'Age', 'recovery_time_days': 'Recovery Time (Days)'},
        opacity=0.6
    )
    st.plotly_chart(fig_age_recovery, use_container_width=True)
    
    # Recovery time by age group and treatment
    fig_violin = px.violin(
        filtered_data, x='age_group', y='recovery_time_days', color='treatment',
        title="Recovery Time Distribution by Age Group and Treatment",
        labels={'age_group': 'Age Group', 'recovery_time_days': 'Recovery Time (Days)'}
    )
    st.plotly_chart(fig_violin, use_container_width=True)

def show_ml_prediction_tab(filtered_data, original_data):
    """Machine learning prediction tab"""
    st.markdown("## 🤖 Readmission Risk Prediction")
    
    # Train model on filtered data
    if len(filtered_data) > 50:  # Need sufficient data
        model, X_test, y_test, feature_columns, le_diagnosis, le_treatment, le_gender, le_age_group = train_ml_model(filtered_data)
        
        # Model performance
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{model.score(X_test, y_test):.2%}")
        
        with col2:
            st.metric("AUC Score", f"{auc_score:.3f}")
        
        with col3:
            st.metric("Data Points", f"{len(filtered_data):,}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(
            feature_importance, x='importance', y='feature',
            title="Feature Importance for Readmission Prediction",
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    else:
        st.warning("⚠️ Insufficient data for ML model training. Please adjust filters to include more data points.")
    
    # Risk factors analysis
    st.markdown("### Risk Factors Analysis")
    
    high_risk = filtered_data[filtered_data['readmitted'] == 1]
    low_risk = filtered_data[filtered_data['readmitted'] == 0]
    
    if len(high_risk) > 0 and len(low_risk) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High-risk patients (readmitted):**")
            st.write(f"Average age: {high_risk['age'].mean():.1f} years")
            st.write(f"Average recovery time: {high_risk['recovery_time_days'].mean():.1f} days")
            st.write(f"Average length of stay: {high_risk['length_of_stay_days'].mean():.1f} days")
        
        with col2:
            st.markdown("**Low-risk patients (not readmitted):**")
            st.write(f"Average age: {low_risk['age'].mean():.1f} years")
            st.write(f"Average recovery time: {low_risk['recovery_time_days'].mean():.1f} days")
            st.write(f"Average length of stay: {low_risk['length_of_stay_days'].mean():.1f} days")

def show_trends_tab(filtered_data):
    """Trends analysis tab"""
    st.markdown("## 📈 Temporal Trends Analysis")
    
    # Monthly trends
    filtered_data['month'] = filtered_data['admission_date'].dt.to_period('M')
    monthly_stats = filtered_data.groupby('month').agg({
        'patient_id': 'count',
        'readmitted': 'mean',
        'recovery_time_days': 'mean',
        'age': 'mean'
    }).reset_index()
    monthly_stats['month'] = monthly_stats['month'].astype(str)
    
    # Monthly patient count
    fig_monthly_patients = px.line(
        monthly_stats, x='month', y='patient_id',
        title="Monthly Patient Admissions",
        labels={'month': 'Month', 'patient_id': 'Number of Patients'}
    )
    st.plotly_chart(fig_monthly_patients, use_container_width=True)
    
    # Monthly readmission rate
    fig_monthly_readmission = px.line(
        monthly_stats, x='month', y='readmitted',
        title="Monthly Readmission Rate",
        labels={'month': 'Month', 'readmitted': 'Readmission Rate'}
    )
    st.plotly_chart(fig_monthly_readmission, use_container_width=True)
    
    # Seasonal analysis
    filtered_data['season'] = filtered_data['admission_date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_stats = filtered_data.groupby('season').agg({
        'patient_id': 'count',
        'readmitted': 'mean',
        'recovery_time_days': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_seasonal_patients = px.bar(
            seasonal_stats, x='season', y='patient_id',
            title="Patient Admissions by Season",
            labels={'season': 'Season', 'patient_id': 'Number of Patients'}
        )
        st.plotly_chart(fig_seasonal_patients, use_container_width=True)
    
    with col2:
        fig_seasonal_readmission = px.bar(
            seasonal_stats, x='season', y='readmitted',
            title="Readmission Rate by Season",
            labels={'season': 'Season', 'readmitted': 'Readmission Rate'}
        )
        st.plotly_chart(fig_seasonal_readmission, use_container_width=True)

def show_data_explorer_tab(filtered_data):
    """Data explorer tab"""
    st.markdown("## 📋 Data Explorer")
    
    # Data summary
    st.markdown("### Data Summary")
    st.write(f"**Total records:** {len(filtered_data):,}")
    st.write(f"**Columns:** {len(filtered_data.columns)}")
    st.write(f"**Date range:** {filtered_data['admission_date'].min().strftime('%Y-%m-%d')} to {filtered_data['admission_date'].max().strftime('%Y-%m-%d')}")
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Types")
        st.dataframe(filtered_data.dtypes.to_frame('Data Type'))
    
    with col2:
        st.markdown("### Missing Values")
        missing_data = filtered_data.isnull().sum().to_frame('Missing Count')
        missing_data['Percentage'] = (missing_data['Missing Count'] / len(filtered_data)) * 100
        st.dataframe(missing_data)
    
    # Raw data viewer
    st.markdown("### Raw Data")
    
    # Pagination
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
    total_pages = len(filtered_data) // page_size + (1 if len(filtered_data) % page_size > 0 else 0)
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered_data.iloc[start_idx:end_idx]
    else:
        page_data = filtered_data
    
    st.dataframe(page_data, use_container_width=True)
    
    # Download data
    st.markdown("### Download Data")
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_patient_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
