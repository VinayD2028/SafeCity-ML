# SafeCity-ML 🏙️

> **Predicting urban crime patterns and recommending safer neighborhoods using state-of-the-art Machine Learning** — trained on 2M+ real-world Chicago crime records with **94.28% peak accuracy**.

---

## 📌 Overview

SafeCity-ML is an end-to-end machine learning project that applies data-driven intelligence to one of society's most pressing challenges: urban crime. By training four distinct ML models — Artificial Neural Networks (ANN), Decision Trees, Random Forests, and XGBoost — on the [Chicago Data Portal Crime Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2), this project predicts the type of crime likely to occur based on spatiotemporal and contextual features, enabling smarter allocation of public safety resources.

The project covers the **full data science lifecycle**: data collection, exploratory data analysis (EDA), preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation — with every decision documented and reproducible.

---

## 🚀 Key Highlights

- 📊 **Dataset**: Chicago Crime Dataset (City of Chicago Data Portal) — 2M+ records spanning multiple years
- 🎯 **Best Model**: Hyperparameter-tuned Decision Tree — **94.28% Test Accuracy**, F1-Score: 0.940
- 🧠 **Models Implemented**: Baseline ANN, Tuned ANN (5-layer, 10 epochs), Decision Tree (baseline + GridSearchCV), Random Forest (baseline + RandomizedSearchCV), XGBoost (baseline + GridSearchCV)
- 🔬 **Feature Engineering**: Lasso Regression (L1 regularization) + domain expertise for feature selection
- 📈 **Evaluation**: ROC-AUC curves, confusion matrices, precision/recall/F1 for all models
- 💾 **Model Persistence**: Best model saved via `joblib` for production inference

---

## 📊 Model Performance Summary

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline ANN (1 hidden layer, 5 epochs) | 81.30% | 0.79 | 0.81 | 0.79 |
| **Tuned ANN (5 hidden layers, 10 epochs)** | **92.01%** | **0.91** | **0.92** | **0.91** |
| Decision Tree (Baseline) | 92.77% | 0.928 | 0.928 | 0.928 |
| **Decision Tree (GridSearchCV Tuned)** | **94.28%** | **0.939** | **0.943** | **0.940** |
| Random Forest (Baseline) | 74.54% | — | — | — |
| **Random Forest (RandomizedSearchCV Tuned)** | **92.98%** | — | — | **0.928** |
| XGBoost (Baseline) | 90.90% | — | 0.91 | 0.90 |
| XGBoost (GridSearchCV Tuned) | 90.18% | — | — | 0.89 |

> The **tuned Decision Tree** (criterion=entropy, max_depth=15, GridSearchCV with 5-fold CV) achieved the best balance of accuracy and interpretability — making it ideal for deployment in real-world public safety applications.

---

## 🏗️ Project Architecture

```
SafeCity-ML/
│
├── Data Collection_Preprocessing_Transformation_Feature_Engineering.py
│   └── EDA, missing value imputation, label encoding, Lasso feature selection
│
├── Baseline ANN.py                   # 1-hidden-layer ANN baseline
├── Hyper Tuned ANN 10 epochs .py    # 5-hidden-layer ANN, hyperparameter tuned
│
├── Decision_Tree.py                  # Baseline + GridSearchCV tuned Decision Tree
├── RandomForest_Model.py             # Baseline + RandomizedSearchCV tuned RF
├── XGBOOST.py                        # Baseline + GridSearchCV tuned XGBoost
│
├── preprocessed_crimes_data.csv      # Feature-engineered dataset (see Drive link)
├── dt_hyperparameter_tuning_model.joblib   # Best saved model (Decision Tree)
│
├── Report.pdf                        # Full technical report
└── Presentation.pdf                  # Project presentation slides
```

---

## 🔧 Tech Stack

**Machine Learning & Data Science**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

**Additional Skills**

![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=openjdk&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Angular](https://img.shields.io/badge/Angular-DD0031?style=flat&logo=angular&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat&logo=amazonaws&logoColor=white)

---

## 📋 Methodology

### 1. Data Collection
The dataset is sourced from the **Chicago Data Portal** — a rich, publicly available dataset containing millions of crime incidents reported in Chicago from 2001 to the present, including crime type, location, date/time, arrest status, and more.

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of all 36 crime types
- Geospatial analysis of crime frequency by **community area**
- Correlation heatmaps across all features
- Crime frequency breakdown by **location description** (street, residence, apartment, etc.)

### 3. Data Preprocessing
- **Missing value imputation**: mean for numerical columns, mode for categorical columns
- **Duplicate removal**: eliminated all redundant records
- **Label encoding**: converted categorical variables to numeric representations

### 4. Feature Engineering
- Applied **Lasso Regression (L1 regularization)** with cross-validated alpha selection to identify the most predictive features
- Combined with **domain knowledge** to finalize 6 key features: `Date`, `Year`, `Longitude`, `Latitude`, `Location Description`, `Description`
- Saved as `preprocessed_crimes_data.csv` for downstream modelling

### 5. Model Training & Tuning
- **Train/Validation/Test split**: 70% / 10% / 20%
- All models were trained on the training set, tuned on the validation set, and evaluated on the held-out test set
- **GridSearchCV** (Decision Tree, XGBoost) and **RandomizedSearchCV** (Random Forest) used for hyperparameter optimization with **5-fold cross-validation**
- ANN architecture scaled from 1 hidden layer (baseline) to 5 hidden layers with `ReLU` activations and `Softmax` output; trained with the **Adam optimizer** and **categorical cross-entropy loss**

### 6. Evaluation
Each model is evaluated with:
- Accuracy, Precision, Recall, F1-Score (weighted)
- Multi-class ROC-AUC curves (top 10 crime classes)
- Confusion matrices (heatmap visualizations)
- Classification reports per crime category

---

## 📂 Dataset

The preprocessed dataset (`preprocessed_crimes_data.csv`) is available on Google Drive due to its size:

🔗 [Download preprocessed_crimes_data.csv](https://drive.google.com/drive/folders/1nIlykOEeJ5UtiuVZTfG7rf-eKG0MY4DD?usp=drive_link)

**Original source**: [Chicago Data Portal — Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

---

## ⚡ Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn joblib
```

### Run the Pipeline

```bash
# Step 1: Data preprocessing and feature engineering
python "Data Collection_Preprocessing_Transformation_Feature_Engineering.py"

# Step 2: Train and evaluate models (choose one)
python Decision_Tree.py
python RandomForest_Model.py
python XGBOOST.py
python "Baseline ANN.py"
python "Hyper Tuned ANN 10 epochs .py"
```

### Load Saved Model for Inference

```python
import joblib
import pandas as pd

# Load the best-performing model
model = joblib.load('dt_hyperparameter_tuning_model.joblib')

# Predict crime type from feature vector
sample = pd.DataFrame([{
    'Date': 1620000000, 'Year': 2023, 'Longitude': -87.65,
    'Latitude': 41.85, 'Location Description': 5, 'Description': 12
}])
prediction = model.predict(sample)
print(prediction)
```

---

## 📁 File Reference

| File | Description |
|---|---|
| `Data Collection_Preprocessing_Transformation_Feature_Engineering.py` | Full pipeline: EDA, Preprocessing, Feature Engineering |
| `Baseline ANN.py` | 1-hidden-layer ANN with StandardScaler, Adam optimizer, 5 epochs |
| `Hyper Tuned ANN 10 epochs .py` | 5-hidden-layer deep ANN, 10 epochs, tuned architecture |
| `Decision_Tree.py` | Decision Tree: baseline + GridSearchCV tuned (best model: 94.28%) |
| `RandomForest_Model.py` | Random Forest: baseline + RandomizedSearchCV tuned |
| `XGBOOST.py` | XGBoost: baseline + GridSearchCV tuned, multi-class classification |
| `dt_hyperparameter_tuning_model.joblib` | Serialized best model ready for inference |
| `Report.pdf` | Technical report with full methodology and findings |
| `Presentation.pdf` | Project presentation slides |

---

## 👤 Author

**Vinay Devabhaktuni**

A passionate software engineer and data scientist with expertise spanning machine learning, full-stack development, and cloud technologies. This project demonstrates hands-on proficiency in the complete ML lifecycle — from raw data ingestion and EDA through to model deployment — applied to a high-impact, real-world problem domain.

**Skills**: Python · TensorFlow · Keras · Scikit-learn · XGBoost · Pandas · NumPy · SQL · Java · React · Angular · Apache Spark · AWS

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with ❤️ to make data work for safer communities.</i>
</p>
