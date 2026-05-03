import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

os.makedirs("supervised_ml", exist_ok=True)

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
Y = df["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

gnbc = GaussianNB()
gnbc.fit(X_train, Y_train)

Y_pred = gnbc.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)

cm = pd.crosstab(Y_test, Y_pred).values
labels = ["Besni", "Kecimen"]
plt.figure(layout="tight")
plt.axis(False)
plt.title("Gaussian Naive Bayes")
plt.table(cellText=cm, rowLabels=labels, colLabels=labels, loc="center").scale(1, 2.5)
plt.savefig("supervised_ml/nb_confusion_matrix.png")

# ---experiment with---
# smoothing

# for vs in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
#     gnbc = GaussianNB(var_smoothing=vs)
#     gnbc.fit(X_train, Y_train)
#     Y_pred = gnbc.predict(X_test)
#     print(f"var_smoothing={vs}: {accuracy_score(Y_test, Y_pred):.4f}")
