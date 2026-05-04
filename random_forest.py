import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

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
# Note that number 42 is an arbitrary seed for Python's Mersenne Twister random number generator.
# It is fixed to ensure reproducibility.

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, Y_train)

Y_pred = rfc.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)

cm = pd.crosstab(Y_test, Y_pred).values
labels = ["Besni", "Kecimen"]
importances = rfc.feature_importances_

plt.figure(layout="tight")
plt.axis(False)
plt.title("Random Forest")
plt.table(cellText=cm, rowLabels=labels, colLabels=labels, loc="center").scale(1, 2.5)

plt.figure(layout="tight")
plt.barh(X.columns, importances)

plt.show()

# ---experiment with---
# n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
