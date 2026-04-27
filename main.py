#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Raisin = np.dtype(
    [
        ("Area", np.uint32),
        ("MajorAxisLength", np.float32),
        ("MinorAxisLength", np.float32),
        ("Eccentricity", np.float32),
        ("ConvexArea", np.float32),
        ("Extent", np.float32),
        ("Perimeter", np.float32),
        ("Class", str),
    ]
)

os.makedirs("graphs/scatterplots", exist_ok=True)

df = pd.read_csv("./data.csv", dtype=Raisin)

color_map = {"Kecimen": "blue", "Besni": "orange"}

plt.figure(figsize=(10, 6))
for label, color in color_map.items():
    subset = df[df["Class"] == label]

    plt.scatter(
        subset["MajorAxisLength"],
        subset["MinorAxisLength"],
        c=color,
        label=label,
        alpha=0.6,
        edgecolors="w",
        s=60,
    )

plt.title("Raisin Classification: Major vs Minor Axis Length", fontsize=14)
plt.xlabel("MajorAxisLength", fontsize=12)
plt.ylabel("MinorAxisLength", fontsize=12)
plt.legend(title="Raisin Variety")
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("graphs/scatterplots/raisin_scatterplot_1.png", dpi=300)

plt.figure(figsize=(10, 6))
for label, color in color_map.items():
    subset = df[df["Class"] == label]

    plt.scatter(
        subset["Area"],
        subset["Extent"],
        c=color,
        label=label,
        alpha=0.6,
        edgecolors="w",
        s=60,
    )

    plt.title("Raisin Classification: Area vs Extent", fontsize=14)
    plt.xlabel("Area", fontsize=12)
    plt.ylabel("Extent", fontsize=12)
    plt.legend(title="Raisin Variety")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("graphs/scatterplots/raisin_scatterplot_2.png", dpi=300)
