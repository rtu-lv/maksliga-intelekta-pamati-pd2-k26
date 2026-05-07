import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hier

classes = {'Kecimen': np.uint32(1), 'Besni': np.uint32(2)}
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

df = pd.read_csv('./data.csv', dtype=Raisin)

targets = df['Class'].map(classes)
data = df.iloc[:, :-1]
norm = (data - data.mean()) / data.std()

linkage = hier.ward(norm)


plt.title('Dendrogramma')
hier.dendrogram(linkage, truncate_mode='level', p=5)

for count in [2, 3, 5]:
	prediction = hier.fcluster(linkage, criterion='maxclust', t=count)
	confusion = pd.crosstab(prediction, targets).values.T

	plt.figure(layout='tight')
	plt.suptitle(f'Hierarhiskā klasterēšana, t={count}')
	plt.axis(False)

	rows = [f'Patiesi {v} ({c})' for c, v in classes.items()]
	cols = [f'Prognozēti {i+1}' for i in range(count)]

	if count == len(classes):
		precision = np.diag(confusion).sum() / confusion.sum()
		plt.figtext(0.025, 0.025, f'Precizitāte: {precision:.2f}')

	plt.table(cellText=confusion,
	          rowLabels=rows, colLabels=cols,
	          loc='center').scale(1, 2.5)

plt.show()

