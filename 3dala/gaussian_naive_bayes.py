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

# Samazinātā pazīmju kopa (MajorAxisLength, MinorAxisLength, Extent)
selected_features = ["MajorAxisLength", "MinorAxisLength", "Extent"]
# Pilnā pazīmju kopa
full_features = [col for col in df.columns if col != "Class"]

y = df["Class"]

X_full = df[full_features]
X_selected = df[selected_features]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

X_train_selected = X_selected.loc[X_train_full.index]
X_test_selected = X_selected.loc[X_test_full.index]

# Eksperimentu hiperparametri (1-3 pilnā kopa, 4-6 samazinātā kopa)
configs = [
    {   # 1. eksperiments (pilnā kopa)
        "name": "1. eksperiments (pilnā kopa)",
        "features": "full",
        "var_smoothing": 1e-12,
    },
    {   # 2. eksperiments (pilnā kopa)
        "name": "2. eksperiments (pilnā kopa)",
        "features": "full",
        "var_smoothing": 1e-9,
    },
    {   # 3. eksperiments (pilnā kopa)
        "name": "3. eksperiments (pilnā kopa)",
        "features": "full",
        "var_smoothing": 1e-6,
    },
    {   # 4. eksperiments (samazinātā kopa) - tie paši parametri kā 1.
        "name": "4. eksperiments (samazinātā kopa)",
        "features": "selected",
        "var_smoothing": 1e-12,
    },
    {   # 5. eksperiments (samazinātā kopa) - tie paši parametri kā 2.
        "name": "5. eksperiments (samazinātā kopa)",
        "features": "selected",
        "var_smoothing": 1e-9,
    },
    {   # 6. eksperiments (samazinātā kopa) - tie paši parametri kā 3.
        "name": "6. eksperiments (samazinātā kopa)",
        "features": "selected",
        "var_smoothing": 1e-6,
    },
]

results = []

print("=== TRAINING EXPERIMENTS (Naive Bayes) ===\n")
print("Pilnā pazīmju kopa (7 pazīmes): Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter")
print("Samazinātā pazīmju kopa (3 pazīmes): MajorAxisLength, MinorAxisLength, Extent")
print("-" * 80)

for i, cfg in enumerate(configs, 1):
    print(f"\n{cfg['name']}")
    print(f"Params: var_smoothing={cfg['var_smoothing']}")
    
    # Izvēlas pareizo datu kopu
    if cfg['features'] == 'full':
        X_train = X_train_full
    else:
        X_train = X_train_selected
    
    model = GaussianNB(var_smoothing=cfg['var_smoothing'])
    model.fit(X_train, y_train)
    
    pred = model.predict(X_train)
    
    acc = accuracy_score(y_train, pred)
    precision = precision_score(y_train, pred, pos_label="Besni")
    recall = recall_score(y_train, pred, pos_label="Besni")
    f1 = f1_score(y_train, pred, pos_label="Besni")
    
    results.append({
        "Experiment": i,
        "Name": cfg['name'],
        "Features": "Pilnā (7)" if cfg['features'] == 'full' else "Samazinātā (3)",
        "Params": cfg,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 60)

results_df = pd.DataFrame(results)
print("\n=== RESULTS TABLE ===")
print(results_df[["Experiment", "Name", "Features", "Accuracy", "Precision", "Recall", "F1"]])

# Labākā modeļa izvēle priekš testiem
best_exp = max(results, key=lambda x: x["F1"])

print("\n=== BEST MODEL ===")
print(f"Eksperiments: {best_exp['Experiment']} - {best_exp['Name']}")
print(f"Pazīmju kopa: {best_exp['Features']}")
print(f"Precizitāte: {best_exp['Accuracy']:.4f}, F1: {best_exp['F1']:.4f}")


best_cfg = best_exp["Params"]

# Izvēlas pareizo datu kopu
if best_cfg['features'] == 'full':
    X_train_best = X_train_full
    X_test_best = X_test_full
else:
    X_train_best = X_train_selected
    X_test_best = X_test_selected

best_model = GaussianNB(var_smoothing=best_cfg['var_smoothing'])
best_model.fit(X_train_best, y_train)

print("\n=== TEST RESULTS ===")

y_test_pred = best_model.predict(X_test_best)

acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {acc_test:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Kecimen", "Besni"]))

# Kļūdu matrica
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=["Kecimen", "Besni"], columns=["Kecimen", "Besni"])

print("\nConfusion Matrix:")
print(cm_df)