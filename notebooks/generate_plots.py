
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

data_path_prefix = '../data/processed/'
if not os.path.exists(data_path_prefix):
    data_path_prefix = 'data/processed/'

print(f"Loading data from: {data_path_prefix}")

try:
    X_class = pd.read_csv(os.path.join(data_path_prefix, 'X_classification.csv'))
    y_class = pd.read_csv(os.path.join(data_path_prefix, 'y_classification.csv'))
except FileNotFoundError:
    print("Error loading data")
    exit(1)

print("\n" + "="*40)
print(" PARTIE 1: CLASSIFICATION RESULTATS")
print("="*40)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)
y_train_class = y_train_class.values.ravel()
y_test_class = y_test_class.values.ravel()

scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_class_scaled, y_train_class)
y_pred_log = log_reg.predict(X_test_class_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_class, y_pred_log):.4f}")

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_class_scaled, y_train_class)
y_pred_rf = rf_clf.predict(X_test_class_scaled)
acc_rf = accuracy_score(y_test_class, y_pred_rf)
print(f"Random Forest Accuracy:     {acc_rf:.4f}")

# MLP Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_clf.fit(X_train_class_scaled, y_train_class)
y_pred_mlp = mlp_clf.predict(X_test_class_scaled)
acc_mlp = accuracy_score(y_test_class, y_pred_mlp)
print(f"MLP Classifier Accuracy:      {acc_mlp:.4f}")

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train_class_scaled, y_train_class)
y_pred_gb = gb_clf.predict(X_test_class_scaled)
acc_gb = accuracy_score(y_test_class, y_pred_gb)
print(f"Gradient Boosting Accuracy:   {acc_gb:.4f}")

# Dummy Classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_class_scaled, y_train_class)
y_pred_dummy = dummy_clf.predict(X_test_class_scaled)
acc_dummy = accuracy_score(y_test_class, y_pred_dummy)
print(f"Dummy Classifier Accuracy:    {acc_dummy:.4f}")

# Comparison Bar Plot
models = ['LogReg', 'RandomForest', 'MLP', 'GradBoost', 'Dummy']
accuracies = [accuracy_score(y_test_class, y_pred_log), acc_rf, acc_mlp, acc_gb, acc_dummy]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#95a5a6']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)
plt.ylim(0.4, 0.7) # Zoom in a bit
plt.title('Comparaison Finale (avec Boosting)')
plt.ylabel('Accuracy')
plt.axhline(y=acc_dummy, color='r', linestyle='--', label='Baseline')
plt.legend()

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('classification_comparison_v2.png')
print("Graphique sauvegardé: classification_comparison_v2.png")
