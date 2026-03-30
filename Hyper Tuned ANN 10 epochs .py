"""
Hyperparameter-Tuned Artificial Neural Network (ANN) — SafeCity-ML
====================================================================
Author : Vinay Devabhaktuni
Project: SafeCity-ML — Urban Crime Prediction using Machine Learning
Dataset: Chicago Crime Dataset (Chicago Data Portal, 2001–Present)

Description:
    This module implements a deep, hyperparameter-tuned feedforward ANN
    using TensorFlow/Keras to predict the type of crime (Crime_Type)
    from spatiotemporal and contextual features.

    Architecture (Tuned):
        Input → Dense(64, ReLU) × 5 hidden layers → Dense(num_classes, Softmax)
        Optimizer : Adam
        Loss      : Categorical Cross-Entropy
        Epochs    : 10  |  Batch Size: 64

    Results (Tuned ANN):
        Test Accuracy : 92.01%
        Precision     : 0.91  (weighted)
        Recall        : 0.92  (weighted)
        F1-Score      : 0.91  (weighted)

    Compared to the Baseline ANN (single hidden layer, 5 epochs, 81.30% accuracy),
    scaling to 5 hidden layers and 10 epochs delivers a +10.71% accuracy uplift.

Pipeline:
    Load Data → Scale Features → Split (70/20/10) →
    One-Hot Encode Labels → Build Deep ANN → Train →
    Plot Learning Curves → Evaluate → Visualize
"""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
# Load the feature-engineered dataset produced by the preprocessing pipeline.
# Features: Date, Year, Longitude, Latitude, Location Description, Description
# Target  : Crime_Type (integer-encoded crime category)
data_selected = pd.read_csv('preprocessed_crimes_data.csv')
print(data_selected.head(5))

# Rename target column for clarity
data_selected.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)
print(data_selected.head(5))

# ---------------------------------------------------------------------------
# 2. Feature / Target Split
# ---------------------------------------------------------------------------
X = data_selected.drop('Crime_Type', axis=1)   # Feature matrix
y = data_selected['Crime_Type']                  # Target labels

# ---------------------------------------------------------------------------
# 3. Feature Scaling
# ---------------------------------------------------------------------------
# StandardScaler normalizes features to zero mean and unit variance.
# Consistent scaling from the baseline model ensures fair comparison.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 4. Train-Validation-Test Split
# ---------------------------------------------------------------------------
# Split into 70% training, 20% testing, ~10% validation (same seed as baseline)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42
)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples      : {len(X_test)}")

# ---------------------------------------------------------------------------
# 5. One-Hot Encode Target Labels
# ---------------------------------------------------------------------------
# Keras' categorical_crossentropy requires one-hot encoded targets.
num_classes    = len(y_train.unique())
y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot   = to_categorical(y_val,   num_classes)
y_test_onehot  = to_categorical(y_test,  num_classes)

print(f"Number of crime classes: {num_classes}")

# ---------------------------------------------------------------------------
# 6. Build Tuned Deep ANN
# ---------------------------------------------------------------------------
# Key architectural change from the baseline:
#   - 5 hidden Dense(64, ReLU) layers instead of 1
#   - 10 training epochs instead of 5
#   - Batch size increased to 64 for faster gradient updates
#
# Adding depth allows the network to learn more complex non-linear patterns
# in the crime data, resulting in significantly higher accuracy.
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),  # Hidden layer 1
    Dense(64, activation='relu'),                               # Hidden layer 2
    Dense(64, activation='relu'),                               # Hidden layer 3
    Dense(64, activation='relu'),                               # Hidden layer 4
    Dense(64, activation='relu'),                               # Hidden layer 5
    Dense(num_classes, activation='softmax'),                   # Output layer
])

# Compile with Adam optimizer and categorical cross-entropy loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ---------------------------------------------------------------------------
# 7. Train the Tuned ANN
# ---------------------------------------------------------------------------
history = model.fit(
    X_train, y_train_onehot,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val_onehot),
)

# ---------------------------------------------------------------------------
# 8. Evaluate on Test Set
# ---------------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f'\n[Tuned ANN] Test Loss    : {loss:.4f}')
print(f'[Tuned ANN] Test Accuracy: {accuracy * 100:.2f}%')

# ---------------------------------------------------------------------------
# 9. Learning Curve — Training vs. Validation Accuracy
# ---------------------------------------------------------------------------
# Visualize how training and validation accuracy evolve across epochs.
# Convergence without significant divergence indicates good generalisation.
training_accuracy   = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

plt.plot(range(1, 11), training_accuracy,   label='Training Accuracy',   marker='o')
plt.plot(range(1, 11), validation_accuracy, label='Validation Accuracy', marker='*')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('[Tuned ANN] Training and Validation Accuracy Over 10 Epochs')
plt.legend(loc='lower right')
plt.xticks(range(1, 11))
plt.yticks(np.arange(0.8, 1.0, 0.05))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 10. Predictions & Classification Metrics
# ---------------------------------------------------------------------------
y_pred_probabilities = model.predict(X_test)
y_pred               = np.argmax(y_pred_probabilities, axis=-1)

# Weighted metrics account for class imbalance across all 36 crime types
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test,    y_pred, average='weighted')
f1        = f1_score(y_test,        y_pred, average='weighted')

print(f'[Tuned ANN] Precision: {precision:.2f}')
print(f'[Tuned ANN] Recall   : {recall:.2f}')
print(f'[Tuned ANN] F1-Score : {f1:.2f}')

# Full per-class classification report
report = classification_report(y_test, y_pred)
print(f'\n[Tuned ANN] Classification Report:\n{report}')

# ---------------------------------------------------------------------------
# 11. Crime Type Label Mapping (for chart annotations)
# ---------------------------------------------------------------------------
crime_type_mapping = {
    0 : 'ARSON',                           1 : 'ASSAULT',
    2 : 'BATTERY',                         3 : 'BURGLARY',
    4 : 'CONCEALED CARRY LICENSE VIOLATION',5 : 'CRIM SEXUAL ASSAULT',
    6 : 'CRIMINAL DAMAGE',                 7 : 'CRIMINAL SEXUAL ASSAULT',
    8 : 'CRIMINAL TRESPASS',               9 : 'DECEPTIVE PRACTICE',
    10: 'DOMESTIC VIOLENCE',               11: 'GAMBLING',
    12: 'HOMICIDE',                        13: 'HUMAN TRAFFICKING',
    14: 'INTERFERENCE WITH PUBLIC OFFICER',15: 'INTIMIDATION',
    16: 'KIDNAPPING',                      17: 'LIQUOR LAW VIOLATION',
    18: 'MOTOR VEHICLE THEFT',             19: 'NARCOTICS',
    20: 'NON - CRIMINAL',                  21: 'NON-CRIMINAL',
    22: 'NON-CRIMINAL (SUBJECT SPECIFIED)',23: 'OBSCENITY',
    24: 'OFFENSE INVOLVING CHILDREN',      25: 'OTHER NARCOTIC VIOLATION',
    26: 'OTHER OFFENSE',                   27: 'PROSTITUTION',
    28: 'PUBLIC INDECENCY',                29: 'PUBLIC PEACE VIOLATION',
    30: 'RITUALISM',                       31: 'ROBBERY',
    32: 'SEX OFFENSE',                     33: 'STALKING',
    34: 'THEFT',                           35: 'WEAPONS VIOLATION',
}

# ---------------------------------------------------------------------------
# 12. ROC-AUC Curves — Top 10 Crime Classes
# ---------------------------------------------------------------------------
y_test_pred    = model.predict(X_test)
n_classes      = y_test_onehot.shape[1]
class_counts   = np.sum(y_test_onehot, axis=0)
top_10_classes = np.argsort(class_counts)[-10:][::-1]   # Most frequent 10

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_test_pred[:, i])
    roc_auc[i]         = auc(fpr[i], tpr[i])

# Micro-average ROC aggregates across all classes
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_test_pred.ravel())
roc_auc["micro"]               = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))
for i in top_10_classes:
    plt.plot(fpr[i], tpr[i],
             label=f'{crime_type_mapping[i]} (AUC = {roc_auc[i]:.5f})')
plt.plot(
    fpr["micro"], tpr["micro"],
    label=f'Micro-average (AUC = {roc_auc["micro"]:.5f})',
    color='deeppink', linestyle=':', linewidth=4,
)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('[Tuned ANN] Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='right', bbox_to_anchor=(1.6, 0.5))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 13. Confusion Matrix
# ---------------------------------------------------------------------------
cm   = confusion_matrix(y_test, y_pred)
mask = cm == 0    # Mask zero cells to reduce visual clutter

plt.figure(figsize=(16, 12))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Reds',
    linewidths=0.5, square=True, annot_kws={"size": 9},
    mask=mask, cbar_kws={"shrink": 0.8},
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('[Tuned ANN] Confusion Matrix — Test Set')
plt.tight_layout()
plt.show()
plt.savefig('tuned_ann_confusion_matrix.png')
