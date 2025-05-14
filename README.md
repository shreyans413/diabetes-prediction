# Diabetes Prediction using Machine Learning (MATLAB)

> 🎓 A course project for **MC106 - MATLAB Programming** at [Your College Name], submitted by **Shreyans Jain (24/B06/040)**.

This project uses the **Pima Indians Diabetes Dataset** from Kaggle to build and evaluate machine learning models that predict whether a person is likely to have diabetes based on diagnostic measurements.

## 📊 Dataset
- Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Format: CSV
- Features include Glucose, BMI, Insulin, SkinThickness, BloodPressure, Age, etc.
- Target variable: `Outcome` (0 = No Diabetes, 1 = Diabetes)

## 🧪 Workflow

### 🔹 Data Preprocessing
- Missing values handled (0s replaced with NaNs)
- Mean/Median imputation
- Data normalization
- Visualization of distributions and correlations

### 🔹 Model Training
- **Random Forest (TreeBagger)**
- **XGBoost-like Boosting (LogitBoost)**
- **Support Vector Machine (SVM)**
- **Model Stacking** using LogitBoost as a meta-model

### 🔹 Evaluation
- Accuracy comparison between base models and stacked model
- Custom-built `classificationReport` function for:
  - Precision
  - Recall
  - F1 Score (Weighted)
- ROC Curve with AUC for the stacked model

## 📈 Results
- All models were compared on test accuracy.
- The stacked model achieved the highest performance.
- ROC-AUC score provided for the final ensemble.

## 📁 Project Structure
```plaintext
├── diabetes.csv                  # Input dataset
├── diabetes_prediction.m         # Main MATLAB code
└── README.md                     # Project documentation
