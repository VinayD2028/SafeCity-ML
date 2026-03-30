"""
XGBoost Classifier — SafeCity-ML
===================================
Author : Vinay Devabhaktuni
Project: SafeCity-ML — Urban Crime Prediction using Machine Learning
Dataset: Chicago Crime Dataset (Chicago Data Portal, 2001–Present)

Description:
    This module trains and evaluates an XGBoost (Extreme Gradient Boosting)
    classifier to predict the type of crime (Crime_Type) from spatiotemporal
    and contextual features extracted from the Chicago crime dataset.

    Two models are built:
        1. Baseline XGBoost  — manually configured hyperparameters
                               (n_estimators=75, max_depth=6, lr=0.1)
                               Test Accuracy: 90.90%, weighted F1: 0.90
        2. Tuned XGBoost     — optimized via 3-fold StratifiedKFold GridSearchCV
                               Best params: learning_rate=0.01, n_estimators=100
                               Test Accuracy: 90.18%

Pipeline:
    Load Data → Split (70/20/10) → Train Baseline → Evaluate →
    GridSearchCV → Train Tuned Model → Evaluate → Visualize
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
# Load the feature-engineered dataset produced by the preprocessing pipeline.
# Features: Date, Year, Longitude, Latitude, Location Description, Description
# Target  : Primary Type (crime category, integer-encoded)
data = pd.read_csv('preprocessed_crimes_data.csv')

# Check for missing values — should be zero after preprocessing
print("Missing values per column:")
print(data.isnull().sum())

# ---------------------------------------------------------------------------
# 2. Feature / Target Split & Train-Validation-Test Split
# ---------------------------------------------------------------------------
X = data.drop('Primary Type', axis=1)   # Feature matrix
y = data['Primary Type']                 # Target: crime type label

# Split into 70% training, 20% testing, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Confirm all crime classes appear in both training and validation splits
unique_classes_train = set(y_train)
unique_classes_val   = set(y_val)
print(f'\nUnique classes in training set  : {len(unique_classes_train)}')
print(f'Unique classes in validation set: {len(unique_classes_val)}')

# ---------------------------------------------------------------------------
# 3. Baseline XGBoost Model
# ---------------------------------------------------------------------------
# XGBoost with manually selected hyperparameters — provides a strong baseline
# due to the gradient boosting framework's built-in regularization.
# multi:softmax is used for multi-class prediction.

model = XGBClassifier(
    objective='multi:softmax',    # Multi-class classification
    eval_metric='mlogloss',       # Log-loss evaluation during training
    n_estimators=75,              # Number of boosting rounds
    max_depth=6,                  # Maximum tree depth (controls complexity)
    learning_rate=0.1,            # Step size shrinkage to prevent overfitting
    n_jobs=-1,                    # Use all available CPU cores
)

# Train with early evaluation on both training and validation sets
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
)

# Evaluate baseline model on validation set
y_val_pred   = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'\n[Baseline XGB] Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate baseline model on held-out test set
y_test_pred  = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'[Baseline XGB] Test Accuracy       : {test_accuracy * 100:.2f}%')

# ---------------------------------------------------------------------------
# 4. ROC-AUC Curves — Baseline Model (Top 10 Crime Classes)
# ---------------------------------------------------------------------------
y_prob    = model.predict_proba(X_test)
n_classes = 10  # Display ROC curves for the top 10 most frequent crime classes

plt.figure(figsize=(8, 8))
for i in range(n_classes):
    fpr, tpr, _  = roc_curve(y_test == i, y_prob[:, i])
    roc_auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'[Baseline XGB] ROC Curve — Top {n_classes} Crime Classes')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Confusion Matrices — Baseline Model
# ---------------------------------------------------------------------------
# Test set confusion matrix — shows per-class prediction quality
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.set_style("dark")
plt.figure(figsize=(14, 14))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(unique_classes_train),
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=100)
plt.yticks(rotation=0)
plt.title("[Baseline XGB] Confusion Matrix — Test Set")
plt.tight_layout()
plt.show()

# Validation set confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(14, 14))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(unique_classes_train),
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=100)
plt.title("[Baseline XGB] Confusion Matrix — Validation Set")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6. Classification Reports — Baseline Model
# ---------------------------------------------------------------------------
print("\n[Baseline XGB] Classification Report — Validation Set:")
print(classification_report(y_val, model.predict(X_val)))

print("[Baseline XGB] Classification Report — Test Set:")
print(classification_report(y_test, y_test_pred))

# ---------------------------------------------------------------------------
# 7. Hyperparameter Tuning — GridSearchCV with StratifiedKFold
# ---------------------------------------------------------------------------
# GridSearchCV explores learning_rate and n_estimators combinations,
# using 3-fold stratified cross-validation to account for class imbalance.

param_grid = {
    'n_estimators'  : [100, 125],    # Number of boosting rounds
    'learning_rate' : [0.01],        # Fixed small learning rate for stability
}

model_gs = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    n_jobs=-1,
)

stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model_gs,
    param_grid=param_grid,
    scoring='accuracy',
    cv=stratified_cv,
)
grid_search.fit(X_train, y_train)

print(f'\nBest Parameters (GridSearchCV): {grid_search.best_params_}')

# Retrieve the best model from the search
best_model = grid_search.best_estimator_

# Evaluate tuned model on validation set
y_val_pred   = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'[Tuned XGB] Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate tuned model on held-out test set
y_test_pred   = best_model.predict(X_test)
test_accuracy  = accuracy_score(y_test, y_test_pred)
print(f'[Tuned XGB] Test Accuracy      : {test_accuracy * 100:.2f}%')

# Full classification report for the tuned model
print("\n[Tuned XGB] Classification Report — Test Set:")
print(classification_report(y_test, y_test_pred))

# ---------------------------------------------------------------------------
# 8. ROC-AUC Curves — Tuned Model (Top 10 Crime Classes)
# ---------------------------------------------------------------------------
y_prob = best_model.predict_proba(X_test)
plt.figure(figsize=(8, 8))
for i in range(n_classes):
    fpr, tpr, _   = roc_curve(y_test == i, y_prob[:, i])
    roc_auc_score  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'[Tuned XGB] ROC Curve — Top {n_classes} Crime Classes (After Tuning)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 9. Confusion Matrix — Tuned Model
# ---------------------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_,
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Tuned XGB] Confusion Matrix — Test Set')
plt.tight_layout()
plt.show()
