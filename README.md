# Financial-Risk-Assessment
Machine learning project for credit default prediction using the Kaggle “Give Me Some Credit” dataset. Includes EDA, preprocessing pipelines, SMOTE balancing, model benchmarking, hyperparameter tuning, and ensemble learning with XGBoost, Random Forest, SVM, and MLP models. Achieved ROC-AUC of 0.837 using an XGB+RF ensemble.
# Financial Risk Assessment — Credit Default Prediction

A complete end-to-end machine learning project for predicting borrower default risk using the Kaggle **Give Me Some Credit** dataset.

This repository demonstrates:

* Exploratory Data Analysis (EDA)
* Data preprocessing pipelines
* Class imbalance handling with SMOTE
* Model training and tuning
* Ensemble learning
* Performance evaluation and visualization
* Comparative benchmarking across multiple ML models

---

# Project Overview

Financial institutions rely on credit risk models to identify borrowers who may default on loans. In this project, multiple machine learning algorithms were trained and evaluated to predict serious delinquency/default using borrower financial history.

The project includes:

* A full preprocessing pipeline
* Cross-validation evaluation
* Hyperparameter tuning
* Ensemble modeling
* Heatmaps and metric comparison visualizations
* Performance benchmarking

---

# Dataset

**Dataset:** Give Me Some Credit (Kaggle)

* ~150,000 borrower records
* 10 financial/risk-related features
* Binary classification target:

  * `1` → borrower defaulted
  * `0` → borrower did not default

### Example Features

* Revolving utilization ratio
* Age
* Debt ratio
* Monthly income
* Number of open credit lines
* Past due payment history
* Real estate loans
* Dependents

---

# Problem Statement

The dataset is highly imbalanced, making default prediction challenging.

The main goals were:

1. Build robust preprocessing pipelines
2. Handle missing values effectively
3. Address class imbalance using SMOTE
4. Compare multiple ML algorithms
5. Optimize predictive performance using tuning and ensembling
6. Evaluate tradeoffs between ROC-AUC, Precision, Recall, and F1-score

---

# Tech Stack

## Languages & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

## Environment

* Jupyter Notebook

---

# Repository Structure

```bash
financial-risk-assessment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── SMOTE.ipynb
│
├── images/
│   ├── heatmap.png
│   └── model_comparison_table.png
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Exploratory Data Analysis (EDA)

The EDA phase included:

* Class imbalance analysis
* Missing value visualization
* Correlation heatmaps
* Feature distributions
* Boxplots by target class
* Delinquency trend analysis
* Risk driver analysis

### Key Findings

* Severe class imbalance existed in the target variable
* Monthly income contained substantial missing values
* Delinquency-related variables were highly predictive
* Several outliers required robust preprocessing

---

# Preprocessing Pipelines

Three preprocessing pipelines were compared using stratified 5-fold cross-validation.

## Pipeline A

* Median/Mode Imputation
* StandardScaler

## Pipeline B

* KNN Imputation
* MinMaxScaler

## Pipeline C (Best Pipeline)

* Iterative Imputer
* RobustScaler
* SMOTE oversampling

Pipeline C achieved the strongest cross-validation performance and was used for final model training.

---

# Models Evaluated

The following machine learning models were trained and compared:

| Model               | Description                  |
| ------------------- | ---------------------------- |
| Logistic Regression | Baseline linear model        |
| Decision Tree       | Depth-limited classifier     |
| Random Forest       | Ensemble bagging model       |
| XGBoost             | Gradient boosting model      |
| Tuned XGBoost       | Hyperparameter-optimized XGB |
| SVM (RBF)           | Kernel-based classifier      |
| MLP Neural Network  | Feed-forward neural network  |
| XGB + RF Ensemble   | Soft voting ensemble         |

---

# Final Model Performance

| Model               | ROC-AUC | Precision | Recall | F1     |
| ------------------- | ------- | --------- | ------ | ------ |
| Logistic Regression | 0.7866  | 0.2461    | 0.6110 | 0.3509 |
| Decision Tree       | 0.8256  | 0.3333    | 0.3956 | 0.3618 |
| Random Forest       | 0.8348  | 0.3898    | 0.3916 | 0.3907 |
| XGBoost (default)   | 0.8175  | 0.3691    | 0.3019 | 0.3321 |
| SVM (RBF)           | 0.8216  | 0.2309    | 0.6576 | 0.3417 |
| XGBoost (tuned)     | 0.8293  | 0.3223    | 0.4289 | 0.3680 |
| MLP                 | 0.8340  | 0.3350    | 0.4500 | 0.3820 |
| XGB + RF Ensemble   | 0.8370  | 0.3206    | 0.4641 | 0.3792 |

---

# Performance Insights

## Best ROC-AUC

The **XGB + RF Ensemble** achieved the highest ROC-AUC:

* ROC-AUC: **0.8370**

## Best Precision + F1

The **Random Forest** achieved:

* Precision: **0.3898**
* F1-score: **0.3907**

## Best Recall

The **SVM (RBF)** achieved:

* Recall: **0.6576**

This demonstrates the tradeoff between minimizing false negatives and maximizing balanced classification performance.

---

# Visualizations

The project includes:

* Model comparison heatmaps
* ROC-AUC comparison charts
* Correlation heatmaps
* Missing value heatmaps
* Feature importance rankings
* Distribution plots

---

# Ensemble Learning

A soft-voting ensemble combining:

* Tuned XGBoost
* Random Forest

was created to improve generalization.

The ensemble produced the strongest ROC-AUC across all tested models.

---

# Feature Importance

Feature importance analysis showed that the strongest predictors of default included:

* Revolving utilization ratio
* Number of past-due payments
* Debt ratio
* Monthly income
* History of serious delinquency

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/financial-risk-assessment.git
cd financial-risk-assessment
```

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Launch Notebook

```bash
jupyter notebook
```

Open:

```bash
notebooks/SMOTE.ipynb
```

---

# Example Requirements

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
jupyter
```

---

# Future Improvements

Potential future enhancements:

* Threshold optimization
* SHAP explainability
* Probability calibration
* Deep learning architectures
* AutoML experimentation
* Real-time deployment API
* Streamlit dashboard

---

# Learning Outcomes

This project demonstrates practical experience in:

* Machine learning workflows
* Imbalanced classification
* Data preprocessing
* Feature engineering
* Hyperparameter tuning
* Ensemble methods
* Model evaluation
* Financial risk analytics

---

# Author

Add your name, LinkedIn, and GitHub profile here.

Example:

```txt
Name: Your Name
LinkedIn: https://linkedin.com/in/yourprofile
GitHub: https://github.com/yourusername
```

---

# License

This project is open-source and available under the MIT License.
