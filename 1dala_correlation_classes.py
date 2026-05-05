import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv("data.csv")
for cls in csv["Class"].unique():
    subset = csv[csv["Class"] == cls]

    corr = subset.drop(columns=["Class"], errors="ignore").corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    va='center', ha='center')

    plt.title(f"Class: {cls}")
plt.show()