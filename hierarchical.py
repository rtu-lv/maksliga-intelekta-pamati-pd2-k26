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

data = df.loc[:, 'Area':'Perimeter']
norm = (data - data.mean()) / data.std()

targets = df['Class'].map(classes)


linkage = hir.ward(norm)

hir.dendrogram(linkage, truncate_mode='level', p=5)
plt.show()


predict = hir.fcluster(linkage, criterion='maxclust', t=2)

plt.bar(['Pareizi pozitīvi', 'Kļūdaini negatīvi'],
        [np.sum(predict == targets), np.sum(predict != targets)])
plt.show()

