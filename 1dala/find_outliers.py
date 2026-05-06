# specifiski atrod izlecošus datu punktus (izmantojot z-score jeb Standard score)
import pandas as pd
import numpy as np

csv = pd.read_csv("./data.csv")
numeric_cols = csv.select_dtypes(include=[np.number])
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
threshold = 4

# atrod all lines where they are beyond the 3rd standard deviation
outlier_mask = (np.abs(z_scores) > threshold).any(axis=1)
outliers = csv[outlier_mask].copy()
# z_scores_outliers = z_scores[outlier_mask]
# z_scores_outliers = z_scores_outliers.add_suffix("_zscore")
# outliers = pd.concat([outliers, z_scores_outliers], axis=1)

outliers.to_csv("./1dala/out/outliers_data.csv", index=False)
data_no_outliers = csv[~outlier_mask]
data_no_outliers.to_csv("./1dala/out/data_with_no_outliers.csv", index=False)