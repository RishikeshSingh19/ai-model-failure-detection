# 📊 Explainable AI System for Telecom Customer Churn Prediction

An end-to-end Machine Learning and Explainable AI (XAI) project that predicts telecom customer churn and explains **why** customers are likely to leave using SHAP interpretability.

This project demonstrates the complete Data Science lifecycle — from data preprocessing and model building to deployment of an interactive Streamlit dashboard.

---

## 🎯 Project Overview

Telecommunication companies face significant revenue loss due to customer churn.  
Identifying customers likely to leave enables proactive retention strategies.

This project builds an **Explainable Machine Learning System** that:

✅ Predicts churn probability  
✅ Segments customers into risk categories  
✅ Explains model decisions using SHAP  
✅ Provides a business-friendly dashboard  
✅ Deploys as an interactive web application

## 📁 Project Structure

```
ai-model-failure-detection
│
├── app/
│   ├── .gitkeep
│   └── app.py                # Streamlit application
│
├── data/
│   ├── .gitkeep
│   └── dataset.csv           # Telecom churn dataset
│
├── models/
│   ├── .gitkeep
│   └── baseline_pipeline.pkl # Trained ML pipeline
│
├── notebooks/
│   ├── .gitkeep
│   ├── day1_baseline.ipynb
│   ├── day2_failure_detection.ipynb
│   ├── day3_xai_failure_analysis.ipynb
│   └── day4_risk_segmentation_business_insights.ipynb
│
├── src/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

This structure separates experimentation (notebooks), production assets (models), and deployment code (Streamlit app), following industry-standard ML project organization.

---

## 📊 Dataset Description

The project uses a telecom customer dataset containing customer demographics, subscription details, and billing information.

### Key Features

| Feature | Description |
|--------|-------------|
| tenure | Number of months customer stayed |
| MonthlyCharges | Monthly subscription cost |
| Contract | Contract duration type |
| InternetService | Type of internet service |
| TechSupport | Whether tech support is included |
| PaymentMethod | Customer payment method |

### Target Variable

**Churn**
- `1` → Customer likely to leave
- `0` → Customer likely to stay

The dataset represents real-world churn behavior commonly analyzed in telecom analytics.

---

## 🤖 Model Approach

The project builds an end-to-end machine learning pipeline to predict telecom customer churn.

### Machine Learning Workflow

Data Preprocessing  
↓  
Feature Encoding  
↓  
Pipeline Transformation  
↓  
Logistic Regression Model  
↓  
Churn Probability Prediction

### Model Details

- **Algorithm:** Logistic Regression
- **Class Handling:** Balanced class weights
- **Pipeline:** Scikit-learn Pipeline
- **Output:** Probability of customer churn

The pipeline ensures consistent preprocessing during both training and inference, reducing prediction errors in deployment.

---

## 🧠 Explainable AI (SHAP)

To make predictions transparent and business-interpretable, the system integrates **SHAP (SHapley Additive Explanations)**.

### Why Explainability Matters

Telecom companies must understand *why* customers churn — not just predictions.

SHAP explains:

- Which features increase churn risk
- Which features reduce churn risk
- Individual customer prediction reasoning

### Explanation Features

✅ SHAP Waterfall Plot  
Shows how each feature pushes prediction toward churn or retention.

✅ Feature Contribution Analysis  
Displays top drivers influencing churn predictions.

Example churn drivers:

- Month-to-month contract → increases risk
- High monthly charges → increases risk
- Long tenure → reduces risk

This transforms the model from a **black box** into an interpretable decision-support system.
