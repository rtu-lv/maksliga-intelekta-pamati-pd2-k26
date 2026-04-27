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

df = pd.read_csv("./data.csv")

color_map = {"Kecimen": "blue", "Besni": "orange"}


def scatter_plot(df, x_col, y_col, filename, color_map):
    plt.figure(figsize=(10, 6))
    for label, color in color_map.items():
        subset = df[df["Class"] == label]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            c=color,
            label=label,
            alpha=0.6,
            edgecolors="w",
            s=60,
        )
    plt.title(f"Raisin Classification: {x_col} vs {y_col}", fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend(title="Raisin Variety")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)


scatter_plot(
    df,
    "MajorAxisLength",
    "MinorAxisLength",
    "graphs/scatterplots/raisin_scatterplot_1.png",
    color_map,
)
scatter_plot(
    df, "Area", "Extent", "graphs/scatterplots/raisin_scatterplot_2.png", color_map
)
