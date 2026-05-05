import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ReLU = sliekšņveida funkcija + lineārā funkcija.
# Neironu tīkla arhitektūra: Daudzslāņu perceptrons (MLP) ar 2 slēptajiem slāņiem;
# 1. slēptajā slānī ir 64 neironi; 2. slēptajā slānī ir 32 neironi.
# Slēpto slāņu aktivācijas funkcija ir ReLU; Izejas slāņa AF ir loģistiskā funkcija (sigmoid).

Raisin = np.dtype(
    [
        ("Area", np.uint16),
        ("MajorAxisLength", np.float32),
        ("MinorAxisLength", np.float32),
        ("Eccentricity", np.float32),
        ("ConvexArea", np.float32),
        ("Extent", np.float32),
        ("Perimeter", np.float32),
        ("Class", str),
    ]
)

df = pd.read_csv("./data.csv", dtype=Raisin)

X = df.drop("Class", axis=1)
Y = df["Class"].map({"Kecimen": 0, "Besni": 1})

# set the same initial weights each time the model is run
# torch.manual_seed(42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32).reshape(-1, 1)


class mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # experiment with different layer sizes and activation functions here
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


net = mlp()

criterion = torch.nn.BCELoss()
# experiment here
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# try more epochs and higher lr with Adam
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

losses = []
for epoch in range(3000):  # experiment with epoch count
    optimizer.zero_grad()
    output = net(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

correct = 0
total = 0
with torch.no_grad():
    output = net(X_test)
    predictions = (output > 0.5).float()
    correct = (predictions == Y_test).sum().item()
    total = Y_test.shape[0]

    # convert predictions and test labels to numpy arrays for classification report
    Y_pred_np = predictions.numpy().flatten()
    Y_test_np = Y_test.numpy().flatten()

print(f"Accuracy: {100 * correct // total} %")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

classification_rep = classification_report(Y_test_np, Y_pred_np)
print(classification_rep)

cm = pd.crosstab(pd.Series(Y_test_np), pd.Series(Y_pred_np)).values
labels = ["Besni", "Kecimen"]
plt.figure(layout="tight")
plt.axis(False)
plt.title("MLP Neural Network")
plt.table(cellText=cm, rowLabels=labels, colLabels=labels, loc="center").scale(1, 2.5)
plt.show()
