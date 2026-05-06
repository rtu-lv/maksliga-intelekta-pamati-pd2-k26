from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform


def silhouette(points, labels):
	distances = squareform(pdist(points, metric='euclidean'))
	clusters = np.unique(labels)
	scores = []

	for index, label in enumerate(labels):
		same = labels == label
		same[index] = False

		if same.sum() == 0:
			scores.append(0.0)
			continue

		a = distances[index, same].mean()
		b = min(
			distances[index, labels == other].mean()
			for other in clusters
			if other != label
		)

		scores.append((b - a) / max(a, b))

	return float(np.mean(scores))


classes = {'Besni': np.uint32(1), 'Kecimen': np.uint32(2)}
Raisin = np.dtype([
	('Area', np.float32),
	('MajorAxisLength', np.float32),
	('MinorAxisLength', np.float32),
	('Eccentricity', np.float32),
	('ConvexArea', np.float32),
	('Extent', np.float32),
	('Perimeter', np.float32),
	('Class', str),
])

data_path = Path(__file__).resolve().parents[1] / 'data.csv'

df = pd.read_csv(data_path, dtype=Raisin)

targets = df['Class'].map(classes)
data = df.loc[:, 'Area':'Perimeter']
norm = (data - data.mean()) / data.std()

results = []

for count in [2, 3, 4, 5, 6]:
	_, prediction = kmeans2(norm.to_numpy(), count, iter=50, minit='points', seed=42)
	score = silhouette(norm.to_numpy(), prediction)
	results.append((count, score))

	confusion = pd.crosstab(prediction + 1, targets).values.T

	print(f'K-vidējie, k={count}')
	print(f'Silueta koeficients: {score:.4f}')

	plt.figure(layout='tight')
	plt.suptitle(f'K-vidējie, k={count}')
	plt.axis(False)

	rows = [f'{c} ({v})' for c, v in classes.items()]
	cols = range(1, count + 1)

	plt.table(cellText=confusion,
	          rowLabels=rows, colLabels=cols,
	          loc='center').scale(1, 2.5)

summary = pd.DataFrame(results, columns=['k', 'silhouette'])
best = summary.loc[summary['silhouette'].idxmax()]

print('\nSilueta koeficientu kopsavilkums')
print(summary.to_string(index=False))
print(f"\nLabākais k pēc silueta koeficienta: {int(best['k'])}")

plt.figure(layout='tight')
plt.plot(summary['k'], summary['silhouette'], marker='o')
plt.title('K-vidējo silueta koeficienti')
plt.xlabel('k')
plt.ylabel('Silueta koeficients')
plt.xticks(summary['k'])
plt.grid(alpha=0.3)

plt.show()
