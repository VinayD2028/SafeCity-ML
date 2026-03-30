"""
Decision Tree Classifier — SafeCity-ML
========================================
Author : Vinay Devabhaktuni
Project: SafeCity-ML — Urban Crime Prediction using Machine Learning
Dataset: Chicago Crime Dataset (Chicago Data Portal, 2001–Present)

Description:
    This module trains and evaluates a Decision Tree classifier to predict
    the type of crime (Crime_Type) based on spatiotemporal and contextual
    features extracted from the Chicago crime dataset.

    Two models are built:
        1. Baseline Decision Tree  — default hyperparameters
        2. Tuned Decision Tree     — optimized via 5-fold GridSearchCV
                                     Best params: criterion='entropy',
                                     max_depth=15, min_samples_leaf=4,
                                     min_samples_split=5
                                     Test Accuracy: 94.28%, F1: 0.940

Pipeline:
    Load Data → Split (70/20/10) → Train → Evaluate → GridSearch → Save Model
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
# Load the feature-engineered dataset produced by the preprocessing pipeline.
# Columns: Date, Year, Longitude, Latitude, Location Description, Description,
#          Primary Type (target)
file_path = 'preprocessed_crimes_data.csv'
df = pd.read_csv(file_path)

# Rename target column for clarity
df.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)

# Sanity check — confirm there are no residual missing values
print("Missing values:\n", df.isnull().sum())
print(df.head(5))

# ---------------------------------------------------------------------------
# 2. Helper Functions — Evaluation Metrics
# ---------------------------------------------------------------------------

def print_metrics(y_test, y_test_pred):
    """
    Print weighted Precision, Recall, and F1-Score for a multi-class classifier.

    Parameters
    ----------
    y_test      : array-like — true class labels
    y_test_pred : array-like — predicted class labels
    """
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall    = recall_score(y_test, y_test_pred, average='weighted')
    f1        = f1_score(y_test, y_test_pred, average='weighted')
    print(f'Precision : {precision:.3f}')
    print(f'Recall    : {recall:.3f}')
    print(f'F1 Score  : {f1:.3f}')


def print_class_report_conf_matrix(y_test, y_test_pred):
    """
    Print the full classification report and confusion matrix,
    then report overall accuracy and weighted metrics.

    Parameters
    ----------
    y_test      : array-like — true class labels
    y_test_pred : array-like — predicted class labels
    """
    conf_matrix  = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred)
    accuracy     = accuracy_score(y_test, y_test_pred)

    print('Confusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(class_report)
    print(f'Accuracy Score: {accuracy * 100:.2f}%')
    print_metrics(y_test, y_test_pred)


def show_roc_auc_curve(X_test, y_test, model):
    """
    Plot per-class ROC curves and their AUC scores for a multi-class model.

    Parameters
    ----------
    X_test : array-like — test feature matrix
    y_test : array-like — true class labels
    model  : fitted sklearn classifier with predict_proba support
    """
    # Binarise labels for one-vs-rest ROC computation
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    n_classes  = y_test_bin.shape[1]

    fpr, tpr, roc_auc_scores = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
        roc_auc_scores[i]  = auc(fpr[i], tpr[i])

    # Plot ROC curves for each crime class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_scores[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Decision Tree (Multi-Class)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
               fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 3. Feature / Target Split & Train-Validation-Test Split
# ---------------------------------------------------------------------------
df.info()

target_col = 'Crime_Type'
X = df.drop(target_col, axis=1)   # Feature matrix
y = df[target_col]                 # Target: crime type label (integer encoded)

# Split into 70% training, 20% testing, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val     = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples      : {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. Baseline Decision Tree Model
# ---------------------------------------------------------------------------
# Train a default Decision Tree to establish a performance baseline before
# any hyperparameter optimisation is applied.

start_time = time.time()
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'\n[Baseline DT] Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate on held-out test set
y_test_pred  = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'[Baseline DT] Test Accuracy       : {test_accuracy * 100:.2f}%')

end_time = time.time()
print(f'[Baseline DT] Training Time       : {end_time - start_time:.2f}s')

# Detailed metrics for the baseline model
print_metrics(y_test, y_test_pred)
print_class_report_conf_matrix(y_test, y_test_pred)
show_roc_auc_curve(X_test, y_test, classifier)

# Persist the baseline model to disk
joblib.dump(classifier, 'dt_model_baseline.joblib')
print('Baseline model saved to dt_model_baseline.joblib')

# ---------------------------------------------------------------------------
# 5. Hyperparameter Tuning — GridSearchCV
# ---------------------------------------------------------------------------
# Perform exhaustive grid search over key Decision Tree hyperparameters
# using 5-fold stratified cross-validation to select the optimal configuration.

dtree    = DecisionTreeClassifier()
param_grid = {
    'criterion'       : ['gini', 'entropy'],
    'max_depth'       : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'    : [None, 'sqrt', 'log2'],
}

grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid_search.best_params_)
best_model  = grid_search.best_estimator_
grid_accuracy = best_model.score(X_test, y_test)
print(f"Grid Search — Test Accuracy: {grid_accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------
# 6. Tuned Decision Tree Model — Best Configuration
# ---------------------------------------------------------------------------
# Instantiate and train the Decision Tree with the optimal hyperparameters
# identified by GridSearchCV:
#   criterion='entropy', max_depth=15, max_features=None,
#   min_samples_leaf=4,  min_samples_split=5
# Result: Test Accuracy 94.28%, F1-Score 0.940

start_time   = time.time()
best_classifier = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=15,
    max_features=None,
    min_samples_leaf=4,
    min_samples_split=5,
)
best_classifier.fit(X_train, y_train)

# Evaluate tuned model on validation set
y_val_pred  = best_classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'\n[Tuned DT] Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluate tuned model on held-out test set
y_test_pred  = best_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'[Tuned DT] Test Accuracy      : {test_accuracy * 100:.2f}%')

end_time = time.time()
print(f'[Tuned DT] Training Time      : {end_time - start_time:.2f}s')

# Full evaluation of the tuned model
print_metrics(y_test, y_test_pred)
print_class_report_conf_matrix(y_test, y_test_pred)
show_roc_auc_curve(X_test, y_test, best_classifier)

# ---------------------------------------------------------------------------
# 7. Persist Best Model
# ---------------------------------------------------------------------------
# Save the best-performing tuned model for downstream inference.
joblib.dump(best_classifier, 'dt_hyperparameter_tuning_model.joblib')
print('Best tuned model saved to dt_hyperparameter_tuning_model.joblib')
