import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # 0.2 sadala 80% apmācība, 20% testi
)

# Eksperimentu hiperparametri
configs = [
    {"n_estimators": 50, "max_depth": 5, "max_features": 2, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 10, "max_features": 3, "min_samples_split": 5, "min_samples_leaf": 3},
    {"n_estimators": 200, "max_depth": None, "max_features": None, "min_samples_split": 2, "min_samples_leaf": 1},
]

results = []

print("=== TRAINING EXPERIMENTS ===\n")

for i, cfg in enumerate(configs, 1):
    model = RandomForestClassifier(**cfg, random_state=42)
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

results_df = pd.DataFrame(results)
print("\n=== RESULTS TABLE ===")
print(results_df)

# Labākā modeļa izvēle
valid_results = [r for r in results if r["Accuracy"] < 1.0]  # izslēdz tos ar 100%
best_exp = max(valid_results, key=lambda x: x["F1"])
print("\n=== BEST MODEL ===")
print(best_exp)

best_model = RandomForestClassifier(**best_exp["Params"], random_state=42)
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