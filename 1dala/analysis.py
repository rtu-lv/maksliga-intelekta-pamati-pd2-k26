import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

stats = df.groupby('Class').agg(['mean', 'var'])
print(stats.T)


color_map = {
    "Kecimen": "red",
    "Besni": "blue"
}

for x, y in [('MajorAxisLength', 'MinorAxisLength'), ('Area', 'Perimeter'), ('Area', 'Extent')]:
	ax = df.plot.scatter(x, y, c=df["Class"].map(color_map), alpha=0.6)

for measure in ['MajorAxisLength', 'MinorAxisLength', 'Area']:
	ax = df.pivot(columns='Class', values=measure) \
	  .plot.hist(bins=30, xlabel=measure, alpha=0.6)
	ax.set_xlabel(measure)

for measure in ['ConvexArea', 'Eccentricity']:
	ax = df.plot.box(by='Class', column=measure)
plt.show()

