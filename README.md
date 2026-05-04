# Healthcare Patient Data Analysis

A comprehensive data analysis project that examines patient records to identify disease patterns, compare treatment outcomes, and predict readmission risks using machine learning.

## Project Overview

This project performs three main analytical tasks:

1. **Disease Analysis by Age Groups**: Identifies common diseases across different age demographics
2. **Recovery Time Comparison**: Compares recovery times across different treatments and diagnoses
3. **Readmission Risk Prediction**: Uses machine learning to predict patient readmission risk

## Features

- **Realistic Data Generation**: Creates synthetic but realistic patient data for analysis
- **Comprehensive Visualizations**: Multiple charts and graphs for insights
- **Statistical Analysis**: Includes significance testing and correlation analysis
- **Machine Learning Model**: Random Forest classifier for readmission prediction
- **Detailed Reporting**: Summary statistics and key insights

## Project Structure

```
ecommerce/
├── requirements.txt              # Python dependencies
├── data_generator.py            # Script to generate patient data
├── data_analysis.py             # Main analysis script
├── dashboard.py                 # Interactive Streamlit dashboard
├── run_dashboard.py             # Dashboard launcher script
├── demo.py                      # Quick demo script
├── run_analysis.py              # Complete analysis runner
├── README.md                    # This file
├── patient_data.csv             # Generated patient data (after running)
├── disease_analysis_by_age.png  # Disease analysis visualizations
├── recovery_time_analysis.png   # Recovery time analysis plots
└── readmission_prediction_analysis.png  # ML model results
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset**:
   ```bash
   python data_generator.py
   ```

4. **Choose your analysis method**:
   - **Interactive Dashboard** (Recommended): `python run_dashboard.py`
   - **Command Line Analysis**: `python data_analysis.py`
   - **Quick Demo**: `python demo.py`

## Usage

### Option 1: Interactive Dashboard (Recommended)
```bash
python run_dashboard.py
```
This launches a comprehensive web dashboard with:
- **Real-time filtering** by age, gender, diagnosis, treatment, and more
- **Interactive visualizations** with Plotly charts
- **Multiple analysis tabs** for different insights
- **Machine learning predictions** with live model training
- **Data export capabilities**
- **Temporal trend analysis**

### Option 2: Command Line Analysis
```bash
python data_analysis.py
```
This performs all three analytical tasks and generates:
- Console output with detailed statistics
- Three visualization files (PNG format)
- Model performance metrics

### Option 3: Quick Demo
```bash
python demo.py
```
This provides a quick overview of key insights without generating plots.

### Data Generation
```bash
python data_generator.py
```
This creates a `patient_data.csv` file with 1000 realistic patient records including:
- Patient demographics (age, gender)
- Medical information (diagnosis, treatment)
- Outcomes (recovery time, readmission status)

## Analysis Tasks

### Task 1: Disease Analysis by Age Groups
- Identifies most common diseases in each age group (18-30, 31-50, 51-65, 65+)
- Shows disease distribution percentages
- Creates interactive heatmaps and bar charts
- **Dashboard feature**: Real-time filtering and drill-down analysis

### Task 2: Recovery Time Comparison
- Compares recovery times between surgery and medication treatments
- Analyzes recovery patterns by diagnosis and age
- Performs statistical significance testing
- Creates interactive box plots, violin plots, and scatter plots
- **Dashboard feature**: Dynamic filtering and trend analysis

### Task 3: Readmission Risk Prediction
- Trains a Random Forest model to predict readmission risk
- Uses features: age, diagnosis, treatment, gender, recovery time, length of stay
- Provides model performance metrics (accuracy, precision, recall, AUC)
- Shows feature importance and risk factor analysis
- **Dashboard feature**: Live model training and interactive ROC curves

### Additional Dashboard Features
- **Temporal Trends**: Monthly and seasonal analysis of patient patterns
- **Data Explorer**: Raw data viewer with pagination and export capabilities
- **Real-time Filtering**: Apply multiple filters simultaneously
- **Interactive Visualizations**: Zoom, pan, and hover details on all charts

## Key Insights

The analysis typically reveals:

1. **Age-related disease patterns**: Different age groups show distinct disease prevalence
2. **Treatment effectiveness**: Surgery generally requires longer recovery than medication
3. **Risk factors**: Age, diagnosis type, and treatment method significantly impact readmission risk
4. **Predictive power**: The ML model achieves good performance in identifying high-risk patients

## Output Files

After running the analysis, you'll get:

- **`disease_analysis_by_age.png`**: Disease distribution visualizations
- **`recovery_time_analysis.png`**: Recovery time comparison charts
- **`readmission_prediction_analysis.png`**: ML model performance and feature importance

**Dashboard Outputs:**
- Interactive web interface accessible at `http://localhost:8501`
- Real-time filtered visualizations
- Exportable filtered datasets
- Live machine learning model performance

## Technical Details

### Data Features
- **Patient ID**: Unique identifier
- **Age**: Patient age (18-95)
- **Gender**: Male/Female
- **Diagnosis**: Medical condition (8 different types)
- **Treatment**: Surgery or Medication
- **Recovery Time**: Days to recovery
- **Length of Stay**: Hospital stay duration
- **Readmission Risk**: Calculated risk score
- **Readmitted**: Binary outcome (0/1)

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: 7 engineered features
- **Evaluation**: Classification report, ROC-AUC, confusion matrix
- **Cross-validation**: 80/20 train-test split with stratification

## Customization

You can modify the analysis by:

1. **Changing data size**: Edit `n_patients` parameter in `data_generator.py`
2. **Adding new diseases**: Modify the disease lists in the generator
3. **Adjusting age groups**: Change the bins in `create_age_groups()`
4. **Trying different ML models**: Replace RandomForest with other algorithms

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- streamlit (for dashboard)
- plotly (for interactive charts)

## License

This project is for educational and demonstration purposes.

## Contributing

Feel free to extend the analysis with additional features like:
- More sophisticated ML models
- Additional visualization types
- Real-time data processing
- Web dashboard interface
