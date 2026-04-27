#!/usr/bin/env python3

import numpy as np
import pandas as pd

Building = np.dtype([
	('compactness', np.float32),
	('surface_area', np.float32),
	('wall_area', np.float32),
	('roof_area', np.float32),
	('height', np.float32),
	('orientation', np.uint8),
	('glazing_area', np.float32),
	('glazing_distribution', np.float32),
	('heat_load', np.float32),
	('cool_load', np.float32),
])

df = pd.read_csv('./data.csv', dtype=Building)


