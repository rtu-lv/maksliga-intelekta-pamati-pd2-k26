import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hier

classes = {"Besni": np.uint32(1), "Kecimen": np.uint32(2)}
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

targets = df["Class"].map(classes)
data = df.loc[:, "Area":"Perimeter"]
norm = (data - data.mean()) / data.std()

linkage = hier.ward(norm)


hier.dendrogram(linkage, truncate_mode="level", p=5)
plt.savefig("unsupervised_ml/dendrogram.png")

for count in [2, 3, 5]:
    prediction = hier.fcluster(linkage, criterion="maxclust", t=count)
    confusion = pd.crosstab(prediction, targets).values.T

    plt.figure(layout="tight")
    plt.suptitle(f"Hierarhiskā klasterēšana, t={count}")
    plt.axis(False)

    if count == len(classes):
        rows = cols = list(classes.keys())

        precision = np.diag(confusion).sum() / confusion.sum()
        plt.figtext(0.025, 0.025, f"Precizitāte: {precision}")
    else:
        rows = [f"{c} ({v})" for c, v in classes.items()]
        cols = range(1, count + 1)

    plt.table(cellText=confusion, rowLabels=rows, colLabels=cols, loc="center").scale(
        1, 2.5
    )
plt.show()
