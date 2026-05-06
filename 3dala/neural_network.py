import numpy as np
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

# --- Load data ---
df = pd.read_csv("data.csv")

X = df.drop("Class", axis=1)
y = df["Class"].map({"Kecimen": 0, "Besni": 1})  # 0 = Kecimen, 1 = Besni

# --- Train/Test split (stratify to maintain class balance) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Normalization (important for neural networks) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Convert to PyTorch tensors ---
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)


# --- Neural Network Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_size=7, hidden1=64, hidden2=32, dropout_rate=0.2):
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


# --- Hyperparameter configurations (based on DOCX file) ---
configs = [
    {  # 1. eksperiments - noklusējuma konfigurācija
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.005,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {  # 2. eksperiments - lielāks mācīšanās ātrums
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.01,
        "epochs": 3000,
        "optimizer": "Adam",
    },
    {  # 3. eksperiments - mazāks mācīšanās ātrums un vairāk epohu
        "hidden1": 64,
        "hidden2": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 5000,
        "optimizer": "Adam",
    },
]

results = []

print("=== TRAINING EXPERIMENTS (Neural Network) ===\n")

for i, cfg in enumerate(configs, 1):
    print(f"Experiment {i}")
    print(f"Params: hidden1={cfg['hidden1']}, hidden2={cfg['hidden2']}, "
          f"dropout={cfg['dropout_rate']}, lr={cfg['learning_rate']}, "
          f"epochs={cfg['epochs']}, optimizer={cfg['optimizer']}")
    
    # --- Create model ---
    model = MLP(
        input_size=7,
        hidden1=cfg['hidden1'],
        hidden2=cfg['hidden2'],
        dropout_rate=cfg['dropout_rate']
    )
    
    # --- Loss function and optimizer ---
    criterion = nn.BCELoss()
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=0.9)
    
    # --- Training ---
    losses = []
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # --- Evaluation on training data ---
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
        "Params": cfg,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 60)

# --- Results table ---
results_df = pd.DataFrame(results)
print("\n=== RESULTS TABLE ===")
print(results_df)

# --- Choose best model (based on F1) ---
best_exp = max(results, key=lambda x: x["F1"])

print("\n=== BEST MODEL ===")
print(best_exp)

# --- Train best model on full training data ---
best_cfg = best_exp["Params"]
best_model = MLP(
    input_size=7,
    hidden1=best_cfg['hidden1'],
    hidden2=best_cfg['hidden2'],
    dropout_rate=best_cfg['dropout_rate']
)

criterion = nn.BCELoss()
if best_cfg['optimizer'] == 'Adam':
    optimizer = optim.Adam(best_model.parameters(), lr=best_cfg['learning_rate'])
else:
    optimizer = optim.SGD(best_model.parameters(), lr=best_cfg['learning_rate'], momentum=0.9)

# Train the best model
for epoch in range(best_cfg['epochs']):
    optimizer.zero_grad()
    output = best_model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# --- TESTING ---
print("\n=== TEST RESULTS ===")

with torch.no_grad():
    test_output = best_model(X_test_tensor)
    test_pred = (test_output > 0.5).float().numpy().flatten()
    y_test_np = y_test_tensor.numpy().flatten()

acc_test = accuracy_score(y_test_np, test_pred)
print(f"Test Accuracy: {acc_test:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_np, test_pred, target_names=["Kecimen", "Besni"]))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_np, test_pred)
cm_df = pd.DataFrame(cm, index=["Kecimen", "Besni"], columns=["Kecimen", "Besni"])

print("\nConfusion Matrix:")
print(cm_df)