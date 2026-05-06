import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # 0.2 sadala 80% apmācība, 20% testi
)

# Experimentu hiperparametri
configs = [
    {"var_smoothing": 1e-12},
    {"var_smoothing": 1e-9},
    {"var_smoothing": 1e-6},
]

results = []

print("=== TRAINING EXPERIMENTS (Naive Bayes) ===\n")

for i, cfg in enumerate(configs, 1):
    model = GaussianNB(**cfg)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_train)
    
    acc = accuracy_score(y_train, pred)
    precision = precision_score(y_train, pred, pos_label="Besni")
    recall = recall_score(y_train, pred, pos_label="Besni")
    f1 = f1_score(y_train, pred, pos_label="Besni")
    
    results.append({
        "Experiment": i,
        "Params": cfg,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    
    print(f"Experiment {i}")
    print(f"Params: {cfg}")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 50)

# --- Results table ---
results_df = pd.DataFrame(results)
print("\n=== RESULTS TABLE ===")
print(results_df)

# Labākā modeļa izvēle priekš testiem
best_exp = max(results, key=lambda x: x["F1"])

print("\n=== BEST MODEL ===")
print(best_exp)

best_model = GaussianNB(**best_exp["Params"])
best_model.fit(X_train, y_train)

print("\n=== TEST RESULTS ===")

y_test_pred = best_model.predict(X_test)

acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {acc_test:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Kļūdu matrica
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=["Besni", "Kecimen"], columns=["Besni", "Kecimen"])

print("\nConfusion Matrix:")
print(cm_df)