import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hir

classes = {b'Besni': np.uint32(1), b'Kecimen': np.uint32(2)}
Raisin = np.dtype([
	('Area', np.uint16),
	('MajorAxisLength', np.float32),
	('MinorAxisLength', np.float32),
	('Eccentricity', np.float32),
	('ConvexArea', np.float32),
	('Extent', np.float32),
	('Perimeter', np.float32),
	('Class', 'S7'),
])

df = pd.read_csv('./data.csv', dtype=Raisin)

targets = df['Class'].map(classes)
data = df.loc[:, 'Area':'Perimeter']
norm = (data - data.mean()) / data.std()


linkage = hir.ward(norm)
hir.dendrogram(linkage, truncate_mode='level', p=5)

for count in [2, 3, 5]:
	prediction = hir.fcluster(linkage, criterion='maxclust', t=count)

	positive = np.sum(prediction == targets)
	negative = len(data) - positive
	precision = positive / len(data)

	plt.figure()
	plt.bar(['Pareizi pozitīvi', 'Kļūdaini negatīvi'], [positive, negative])
	plt.title(f't={count}, precizitāte: {precision:.2f}')

plt.show()

