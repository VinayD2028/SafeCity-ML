"""
Data Collection, Preprocessing, Transformation & Feature Engineering — SafeCity-ML
=====================================================================================
Author : Vinay Devabhaktuni
Project: SafeCity-ML — Urban Crime Prediction using Machine Learning
Dataset: Chicago Crime Dataset (Chicago Data Portal, 2001–Present)
         Source: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

Description:
    This module implements the full data preparation pipeline for the SafeCity-ML
    project. It transforms raw Chicago crime records into a clean, feature-engineered
    dataset ready for machine learning model training.

    Steps covered:
        1. Data Collection   — Load raw CSV from the Chicago Data Portal
        2. EDA (Raw)         — Visualize crime type distributions and community area patterns
        3. Preprocessing     — Missing value imputation, duplicate removal
        4. EDA (Clean)       — Post-processing visualizations
        5. Transformation    — Label encoding of categorical features
        6. Feature Selection — Lasso Regression (L1) + domain knowledge
        7. Output            — Save preprocessed_crimes_data.csv for modelling

    Output file: preprocessed_crimes_data.csv
    Selected features: Date, Year, Longitude, Latitude,
                       Location Description, Primary Type (target), Description
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# 1. Data Collection
# ---------------------------------------------------------------------------
# Load the raw Chicago crime dataset.
# The dataset includes millions of crime incidents reported in Chicago from
# 2001 to the present, with columns such as crime type, location, date,
# arrest status, community area, and geographic coordinates.
data = pd.read_csv('crime_dataset.csv')

# Basic information about the raw dataset
print(data.info())
print("\nFirst 5 rows of raw data:")
print(data.head(5))
print("\nRandom sample (5 rows):")
print(data.sample(5))

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA) — Raw Data
# ---------------------------------------------------------------------------

# ---- 2a. Crime Type Distribution ----
# Visualize the frequency of each crime type to understand class distribution
# before any preprocessing. Identifies dominant classes (e.g., Theft, Battery)
# and rare ones (e.g., Ritualism, Human Trafficking).
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(
    x='Primary Type', data=data,
    order=data['Primary Type'].value_counts().index
)
plt.title('Distribution of Crime Types (Raw Data)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Top 10 most frequent crime types
top_primary_types = data['Primary Type'].value_counts().nlargest(10).index
print("\nTop 10 Crime Types (Raw):")
print(top_primary_types)

# ---------------------------------------------------------------------------
# 3. Data Preprocessing
# ---------------------------------------------------------------------------

# ---- 3a. Missing Value Handling ----
# Check for missing values across all columns
print("\nMissing values before imputation:")
print(data.isnull().sum())

# Impute numerical columns with column mean (preserves distribution)
num_cols = data.select_dtypes(include=np.number).columns
imputer  = SimpleImputer(strategy='mean')
data[num_cols] = imputer.fit_transform(data[num_cols])

# Impute categorical columns with column mode (most frequent value)
cat_cols = data.select_dtypes(include='object').columns
imputer  = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer.fit_transform(data[cat_cols])

# Confirm no remaining missing values
print("\nMissing values after imputation:")
print(data.isnull().sum())

# ---- 3b. Duplicate Removal ----
# Remove exact duplicate rows to avoid data leakage during model training
print(f"\nDuplicate rows before removal: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Duplicate rows after removal : {data.duplicated().sum()}")

# ---- 3c. Basic Statistics & Data Profiling ----
print("\nBasic Statistics (Numerical Columns):")
print(data.describe())

# Unique values per categorical column — helps identify encoding requirements
print("\nUnique values in categorical columns:")
for col in data.select_dtypes(include=['object']).columns:
    print(f"  {col}: {data[col].unique()[:5]} ...")  # Show first 5 unique values

print("\nRandom sample after preprocessing:")
print(data.sample(5))

# ---------------------------------------------------------------------------
# 4. EDA — Post-Preprocessing Visualizations
# ---------------------------------------------------------------------------

# ---- 4a. Crime Type Distribution (Clean Data) ----
plt.figure(figsize=(12, 6))
sns.countplot(
    x='Primary Type', data=data,
    order=data['Primary Type'].value_counts().index,
)
plt.title('Distribution of Crime Types (After Preprocessing)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

top_primary_types = data['Primary Type'].value_counts().nlargest(10).index
print("\nTop 10 Crime Types (Clean):")
print(top_primary_types)

# ---- 4b. Top 10 Community Areas by Crime Volume ----
# Stacked bar chart showing crime composition within the 10 most crime-prone
# community areas — useful for identifying neighbourhood-level risk patterns.
top_community_areas = data['Community Area'].value_counts().nlargest(10).index
data_top_community  = data[data['Community Area'].isin(top_community_areas)]

crime_counts_by_community = data_top_community.groupby(
    ['Community Area', 'Primary Type']
).size().unstack(fill_value=0)

plt.figure(figsize=(15, 8))
crime_counts_by_community.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Community Area')
plt.ylabel('Crime Frequency')
plt.title('Crime Type Frequency by Top 10 Community Areas')
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 4c. Top 10 Crime Types × Top 10 Location Descriptions ----
# Reveals where specific crimes are most likely to occur
# (e.g., theft on streets, domestic violence in residences).
top_location_descriptions = data['Location Description'].value_counts().nlargest(10).index
data_filtered = data[
    data['Primary Type'].isin(top_primary_types) &
    data['Location Description'].isin(top_location_descriptions)
]

crime_counts_by_location = data_filtered.groupby(
    ['Location Description', 'Primary Type']
).size().unstack(fill_value=0)

plt.figure(figsize=(15, 15))
crime_counts_by_location.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Location Description')
plt.ylabel('Crime Frequency')
plt.title('Top 10 Crime Types by Top 10 Location Descriptions')
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Data Transformation — Label Encoding
# ---------------------------------------------------------------------------
# Convert all categorical string features to integer representations.
# LabelEncoder maps each unique string to a unique integer consistently.
# Required because ML models only accept numerical inputs.
label_encoder = LabelEncoder()
for col in cat_cols:
    data[col] = label_encoder.fit_transform(data[col])

print("\nSample after label encoding:")
print(data.head(5))

# ---- 5a. Correlation Heatmap (Post-Encoding) ----
# Visualize pairwise feature correlations to identify redundant or
# highly correlated features that may need to be removed.
corr_matrix = data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix (After Label Encoding)')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6. Feature Selection — Lasso Regression (L1 Regularization)
# ---------------------------------------------------------------------------
# LassoCV applies L1 regularization, which drives coefficients of less
# informative features toward zero. Features with non-zero coefficients
# are selected as the most predictive.
# Cross-validated alpha selection ensures robust feature importance estimation.

X = data.drop(['Primary Type'], axis=1)   # All features (excluding target)
y = data['Primary Type']                   # Target: crime type

# Standardize features before Lasso (regularization is scale-sensitive)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Features with non-zero Lasso coefficients are considered informative
selected_features_lasso = X.columns[lasso.coef_ != 0]
print("\nFeatures selected by Lasso (L1):", list(selected_features_lasso))

# ---- 6a. Final Feature Set (Lasso + Domain Knowledge) ----
# Combining Lasso results with domain knowledge about crime prediction,
# the following 7 columns are selected as the final feature set:
#   - Date, Year      : temporal context (time of crime)
#   - Longitude, Latitude : geographic location of the crime
#   - Location Description: type of location (street, residence, etc.)
#   - Description     : specific sub-type of the crime
#   - Primary Type    : TARGET variable (crime category)
selected_features = [
    'Date', 'Year', 'Longitude', 'Latitude',
    'Location Description', 'Primary Type', 'Description',
]
data_selected = data[selected_features]

# ---------------------------------------------------------------------------
# 7. Save Preprocessed Dataset
# ---------------------------------------------------------------------------
# Export the clean, feature-selected dataset as a CSV file.
# This file is the input for all downstream model training scripts.
data_selected.to_csv('preprocessed_crimes_data.csv', index=False)
print("\nPreprocessed dataset saved to: preprocessed_crimes_data.csv")

# ---------------------------------------------------------------------------
# 8. Reload & Validate Output Dataset
# ---------------------------------------------------------------------------
# Reload to confirm the saved file is correct and complete.
data_selected = pd.read_csv('preprocessed_crimes_data.csv')
print("\nReloaded dataset shape:", data_selected.shape)
print(data_selected.head(10))

# Rename target column for downstream consistency
data_selected.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)
print(data_selected.head(10))

# Confirm no missing values in the final output
print("\nMissing values in final dataset:")
print(data_selected.isnull().sum())

# ---------------------------------------------------------------------------
# 9. Data Preparation Summary
# ---------------------------------------------------------------------------
# The preprocessed_crimes_data.csv contains:
#   - Rows   : ~2M+ crime records
#   - Columns: 7 (6 features + 1 target)
#   - Target : Crime_Type — 36 unique crime categories (integer-encoded)
#
# This dataset is split 70/20/10 (train/test/val) in each model training script.
print("\nFinal dataset summary:")
print(data_selected.describe())
