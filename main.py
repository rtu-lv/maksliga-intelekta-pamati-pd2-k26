import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

df = pd.read_csv("./data.csv", dtype=Raisin)  # dataframe input

# ----SCATTER PLOT----
color_map = {"Kecimen": "blue", "Besni": "orange"}


def scatter_plot(df, x_col, y_col, filename, color_map):
    plt.figure(figsize=(6, 4))
    for (
        label,
        color,
    ) in (
        color_map.items()
    ):  # items() returns a view of the dictionary's key-value pairs
        subset = df[
            df["Class"] == label
        ]  # filter the dataframe by class label all at once (broadcasting)
        plt.scatter(
            subset[x_col],  # take the filtered x column
            subset[y_col],  # take the filtered y column
            c=color,
            label=label,
            alpha=0.6,
            edgecolors="w",
            s=30,
        )
    plt.title(f"{x_col} vs {y_col}", fontsize=11)
    plt.xlabel(x_col, fontsize=8)
    plt.ylabel(y_col, fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(title="Raisin Type", fontsize=8, title_fontsize=8)
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

os.makedirs("graphs/histograms", exist_ok=True)


# ----HISTOGRAM----
def histogram(df, col, filename, color_map):
    plt.figure(figsize=(6, 4))

    for label, color in color_map.items():
        subset = df[df["Class"] == label]
        plt.hist(
            subset[col],  # take the filtered column
            bins=30,
            color=color,
            alpha=0.6,
            label=label,
        )
    plt.title(f"Distribution of {col}", fontsize=11)
    plt.xlabel(col, fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(title="Raisin Type", fontsize=8, title_fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)


histogram(df, "MajorAxisLength", "graphs/histograms/raisin_histogram_1.png", color_map)
histogram(df, "MinorAxisLength", "graphs/histograms/raisin_histogram_2.png", color_map)

# ----DISTRIBUTION PLOTS----


def box_plot(df, col, filename):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Class", y=col, data=df)
    plt.title(f"Raisin Distribution: {col}", fontsize=14)
    plt.savefig(filename, dpi=300)


def violin_plot(df, col, filename):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Class", y=col, data=df)
    plt.title(f"Raisin Distribution: {col}", fontsize=14)
    plt.savefig(filename, dpi=300)


os.makedirs("graphs/distributions", exist_ok=True)

box_plot(df, "ConvexArea", "graphs/distributions/raisin_boxplot.png")
violin_plot(df, "Eccentricity", "graphs/distributions/raisin_violinplot.png")

# ----STATISTIC CALCULATIONS----

os.makedirs("statistics", exist_ok=True)

stats = df.groupby("Class").agg(["mean", "var"])
stats.to_excel("statistics/statistics.xlsx")
