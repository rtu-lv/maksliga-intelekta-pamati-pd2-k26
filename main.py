#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Raisin = np.dtype([
	('Area', np.uint32),
	('MajorAxisLength', np.float32),
	('MinorAxisLength', np.float32),
	('Eccentricity', np.float32),
	('ConvexArea', np.float32),
	('Extent', np.float32),
	('Perimeter', np.float32),
	('Class', str),
])

df = pd.read_csv('./data.csv', dtype=Raisin)

df.plot.scatter(x='MinorAxisLength', y='Extent')
plt.show()

