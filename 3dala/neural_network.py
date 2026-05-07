import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")

# Samazinātā pazīmju kopa (MajorAxisLength, MinorAxisLength, Extent)
selected_features = ["MajorAxisLength", "MinorAxisLength", "Extent"]
# Pilnā pazīmju kopa (visas, izņemot Class)
full_features = [col for col in df.columns if col != "Class"]

y = df["Class"].map({"Kecimen": 0, "Besni": 1})

X_full = df[full_features]
X_selected = df[selected_features]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

X_train_selected = X_selected.loc[X_train_full.index]
X_test_selected = X_selected.loc[X_test_full.index]

# Normalizācija pilnajai kopai
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

# Normalizācija samazinātajai kopai
scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_test_selected_scaled = scaler_selected.transform(X_test_selected)

# Pārveido uz PyTorch tensoriem (pilnā kopa)
X_train_full_tensor = torch.tensor(X_train_full_scaled, dtype=torch.float32)
X_test_full_tensor = torch.tensor(X_test_full_scaled, dtype=torch.float32)

# Pārveido uz PyTorch tensoriem (samazinātā kopa)
X_train_selected_tensor = torch.tensor(X_train_selected_scaled, dtype=torch.float32)
X_test_selected_tensor = torch.tensor(X_test_selected_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# MLP klases definēšana
class MLP(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


# Eksperimentu hiperparametri (1-3 pilnā kopa, 4-6 samazinātā kopa)
configs = [
    {   # 1. eksperiments (pilnā kopa)
        "name": "1. eksperiments (pilnā kopa)",
        "features": "full",
        "input_size": 7,
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.005,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {   # 2. eksperiments (pilnā kopa)
        "name": "2. eksperiments (pilnā kopa)",
        "features": "full",
        "input_size": 7,
        "hidden1": 64,
        "hidden2": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.01,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {   # 3. eksperiments (pilnā kopa)
        "name": "3. eksperiments (pilnā kopa)",
        "features": "full",
        "input_size": 7,
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0,
        "learning_rate": 0.001,
        "epochs": 5000,
        "optimizer": "Adam",
    },
    {   # 4. eksperiments (samazinātā kopa) - tie paši parametri kā 1.
        "name": "4. eksperiments (samazinātā kopa)",
        "features": "selected",
        "input_size": 3,
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.005,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {   # 5. eksperiments (samazinātā kopa) - tie paši parametri kā 2.
        "name": "5. eksperiments (samazinātā kopa)",
        "features": "selected",
        "input_size": 3,
        "hidden1": 64,
        "hidden2": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.01,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {   # 6. eksperiments (samazinātā kopa) - tie paši parametri kā 3.
        "name": "6. eksperiments (samazinātā kopa)",
        "features": "selected",
        "input_size": 3,
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0,
        "learning_rate": 0.001,
        "epochs": 5000,
        "optimizer": "Adam",
    },
]

results = []

print("=== TRAINING EXPERIMENTS (Neural Network) ===\n")
print("Pilnā pazīmju kopa (7 pazīmes): Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter")
print("Samazinātā pazīmju kopa (3 pazīmes): MajorAxisLength, MinorAxisLength, Extent")
print("-" * 80)

for i, cfg in enumerate(configs, 1):
    print(f"\n{cfg['name']}")
    print(f"Params: input_size={cfg['input_size']}, hidden1={cfg['hidden1']}, hidden2={cfg['hidden2']}, "
          f"dropout={cfg['dropout_rate']}, lr={cfg['learning_rate']}, "
          f"epochs={cfg['epochs']}, optimizer={cfg['optimizer']}")
    
    # Izvēlas pareizos datu tensorus
    if cfg['features'] == 'full':
        X_train_tensor = X_train_full_tensor
    else:
        X_train_tensor = X_train_selected_tensor
    
    model = MLP(
        input_size=cfg['input_size'],
        hidden1=cfg['hidden1'],
        hidden2=cfg['hidden2'],
        dropout_rate=cfg['dropout_rate']
    )
    
    criterion = nn.BCELoss()
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=0.9)
    
    # Apmācība
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Novērtēšana uz apmācības datiem
    with torch.no_grad():
        train_output = model(X_train_tensor)
        train_pred = (train_output > 0.5).float().numpy().flatten()
        y_train_np = y_train_tensor.numpy().flatten()
        
        acc = accuracy_score(y_train_np, train_pred)
        precision = precision_score(y_train_np, train_pred)
        recall = recall_score(y_train_np, train_pred)
        f1 = f1_score(y_train_np, train_pred)
    
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

# Labākā modeļa izvēle (izslēdz modeļus ar >99% precizitāti, lai izvairītos no pārmācītiem modeļiem)
valid_results = [r for r in results if r["Accuracy"] < 0.99]
best_exp = max(valid_results, key=lambda x: x["F1"])

print("\n=== BEST MODEL (based on training F1, excluding 100% accuracy) ===")
print(f"Eksperiments: {best_exp['Experiment']} - {best_exp['Name']}")
print(f"Pazīmju kopa: {best_exp['Features']}")
print(f"Precizitāte: {best_exp['Accuracy']:.4f}, F1: {best_exp['F1']:.4f}")

# Apmāca labāko modeli testēšanai
best_cfg = best_exp["Params"]
best_model = MLP(
    input_size=best_cfg['input_size'],
    hidden1=best_cfg['hidden1'],
    hidden2=best_cfg['hidden2'],
    dropout_rate=best_cfg['dropout_rate']
)

criterion = nn.BCELoss()
if best_cfg['optimizer'] == 'Adam':
    optimizer = optim.Adam(best_model.parameters(), lr=best_cfg['learning_rate'])
else:
    optimizer = optim.SGD(best_model.parameters(), lr=best_cfg['learning_rate'], momentum=0.9)

# Izvēlas pareizos apmācības datus labākajam modelim
if best_cfg['features'] == 'full':
    X_train_best = X_train_full_tensor
    X_test_best = X_test_full_tensor
else:
    X_train_best = X_train_selected_tensor
    X_test_best = X_test_selected_tensor

for epoch in range(best_cfg['epochs']):
    optimizer.zero_grad()
    output = best_model(X_train_best)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

print("\n=== TEST RESULTS ===")

with torch.no_grad():
    test_output = best_model(X_test_best)
    test_pred = (test_output > 0.5).float().numpy().flatten()
    y_test_np = y_test_tensor.numpy().flatten()

acc_test = accuracy_score(y_test_np, test_pred)
print(f"Test Accuracy: {acc_test:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_np, test_pred, target_names=["Kecimen", "Besni"]))

# Kļūdu matrica
cm = confusion_matrix(y_test_np, test_pred)
cm_df = pd.DataFrame(cm, index=["Kecimen", "Besni"], columns=["Kecimen", "Besni"])

print("\nConfusion Matrix:")
print(cm_df)