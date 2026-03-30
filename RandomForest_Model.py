"""
Random Forest Classifier — SafeCity-ML
========================================
Author : Vinay Devabhaktuni
Project: SafeCity-ML — Urban Crime Prediction using Machine Learning
Dataset: Chicago Crime Dataset (Chicago Data Portal, 2001–Present)

Description:
    This module trains and evaluates a Random Forest ensemble classifier
    to predict the type of crime (Crime_Type) from spatiotemporal and
    contextual features extracted from the Chicago crime dataset.

    Two models are built:
        1. Baseline Random Forest  — manually selected hyperparameters
                                     (n_estimators=150, max_depth=10)
                                     Test Accuracy: 74.54%
        2. Tuned Random Forest     — optimized via RandomizedSearchCV
                                     (5-fold stratified CV)
                                     Best params: n_estimators=50,
                                     min_samples_split=2, min_samples_leaf=1,
                                     max_depth=None
                                     Test Accuracy: 92.98%, F1: 0.928

Pipeline:
    Load Data → Split (70/20/10) → Train Baseline → Evaluate →
    RandomizedSearch → Train Tuned Model → Evaluate → Visualize
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
# Load the feature-engineered dataset (produced by the preprocessing pipeline).
# Features: Date, Year, Longitude, Latitude, Location Description, Description
# Target  : Primary Type (crime category)
data_selected = pd.read_csv('preprocessed_crimes_data.csv')
data_selected.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)
print(data_selected.head(10))

# ---------------------------------------------------------------------------
# 2. Feature / Target Split & Train-Validation-Test Split
# ---------------------------------------------------------------------------
X = data_selected.drop('Crime_Type', axis=1)   # Feature matrix
y = data_selected['Crime_Type']                  # Target: crime type label

# Split into 70% training, 20% testing, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples      : {len(X_test)}")

# ---------------------------------------------------------------------------
# 3. Baseline Random Forest Model
# ---------------------------------------------------------------------------
# An initial Random Forest with manually chosen hyperparameters is trained to
# establish a baseline. This helps quantify the improvement from tuning.

rf_model = RandomForestClassifier(
    n_estimators=150,     # Number of trees in the forest
    max_depth=10,         # Maximum depth of each tree (limits overfitting)
    min_samples_split=5,  # Minimum samples required to split an internal node
    min_samples_leaf=2,   # Minimum samples required at a leaf node
    random_state=42,
)
rf_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred  = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'\n[Baseline RF] Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate on held-out test set
y_test_pred  = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'[Baseline RF] Test Accuracy       : {test_accuracy * 100:.2f}%')

# ---------------------------------------------------------------------------
# 4. Classification Reports — Baseline Model
# ---------------------------------------------------------------------------
# Validation set report
y_pred = rf_model.predict(X_val)
print("\n[Baseline RF] Classification Report — Validation Set:")
print(classification_report(y_val, y_pred))

# Test set report
y_pred = rf_model.predict(X_test)
print("[Baseline RF] Classification Report — Test Set:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 5. Confusion Matrices — Baseline Model
# ---------------------------------------------------------------------------
# Validation confusion matrix
y_pred = rf_model.predict(X_val)
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Baseline RF] Confusion Matrix — Validation Set')
plt.tight_layout()
plt.show()

# Test confusion matrix
y_pred = rf_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Baseline RF] Confusion Matrix — Test Set')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6. ROC-AUC Curves — Baseline Model (Top 10 Crime Classes)
# ---------------------------------------------------------------------------
y_prob = rf_model.predict_proba(X_test)
n_classes = 10    # Display ROC curves for the top 10 crime classes

plt.figure(figsize=(8, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'[Baseline RF] ROC Curve — Top {n_classes} Crime Classes')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 7. Hyperparameter Tuning — RandomizedSearchCV
# ---------------------------------------------------------------------------
# Use RandomizedSearchCV with StratifiedKFold to efficiently search the
# hyperparameter space for the best Random Forest configuration.

param_dist = {
    'n_estimators'    : [50, 100, 150],
    'max_depth'       : [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
}

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=5,
    cv=StratifiedKFold(n_splits=5),
    scoring='accuracy',
    random_state=42,
)
random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
print("\nBest Hyperparameters (RandomizedSearchCV):", best_params_random)

# ---------------------------------------------------------------------------
# 8. Tuned Random Forest Model — Best Configuration
# ---------------------------------------------------------------------------
# Instantiate the tuned model with the best parameters identified above.
# Best config: n_estimators=50, min_samples_split=2,
#              min_samples_leaf=1, max_depth=None
# Result: Test Accuracy 92.98%, F1-Score 0.928

rf_model_tuned = RandomForestClassifier(
    n_estimators=50,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
)
rf_model_tuned.fit(X_train, y_train)

# Evaluate tuned model on validation set
y_val_pred  = rf_model_tuned.predict(X_val)
accuracy    = accuracy_score(y_val, y_val_pred)
f1          = f1_score(y_val, y_val_pred, average='weighted')
print(f'\n[Tuned RF] Validation Accuracy: {accuracy * 100:.2f}%')
print(f'[Tuned RF] Validation F1 Score : {f1 * 100:.2f}%')

# Evaluate tuned model on held-out test set
y_test_pred  = rf_model_tuned.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test       = f1_score(y_test, y_test_pred, average='weighted')
print(f'[Tuned RF] Test Accuracy       : {accuracy_test * 100:.2f}%')
print(f'[Tuned RF] Test F1 Score       : {f1_test * 100:.2f}%')

# ---------------------------------------------------------------------------
# 9. Classification Reports & Visualizations — Tuned Model
# ---------------------------------------------------------------------------
# Validation set classification report
y_pred = rf_model_tuned.predict(X_val)
print("\n[Tuned RF] Classification Report — Validation Set:")
print(classification_report(y_val, y_pred))

# Test set classification report
y_pred = rf_model_tuned.predict(X_test)
print("[Tuned RF] Classification Report — Test Set:")
print(classification_report(y_test, y_pred))

# ROC-AUC curves for tuned model (top 10 classes)
y_prob = rf_model_tuned.predict_proba(X_test)
plt.figure(figsize=(8, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'[Tuned RF] ROC Curve — Top {n_classes} Crime Classes (After Tuning)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Confusion matrices for tuned model
y_pred = rf_model_tuned.predict(X_val)
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Tuned RF] Confusion Matrix — Validation Set')
plt.tight_layout()
plt.show()

y_pred = rf_model_tuned.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Tuned RF] Confusion Matrix — Test Set')
plt.tight_layout()
plt.show()
