#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Raisin = np.dtype([
	('Area', np.uint16),
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


for x, y in [('MajorAxisLength', 'MinorAxisLength'), ('Area', 'Perimeter'), ('Area', 'Extent')]:
	df.plot.scatter(x, y, c='Class', alpha=0.6)

for measure in ['MajorAxisLength', 'MinorAxisLength', 'Area']:
	df.pivot(columns='Class', values=measure) \
	  .plot.hist(bins=30, xlabel=measure, alpha=0.6)

for measure in ['ConvexArea', 'Eccentricity']:
	df.plot.box(by='Class', column=measure)

plt.show()

