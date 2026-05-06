# atrod izceļošas vērtības
import pandas as pd
import numpy as np

csv = pd.read_csv("./data.csv")
numeric_cols = csv.select_dtypes(include=[np.number])

z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
threshold = 4

outliers_list = []

for col in numeric_cols.columns:
    col_outliers = csv[np.abs(z_scores[col]) > threshold].copy()
    col_outliers["Outlier_Column"] = col
    outliers_list.append(col_outliers)

outliers_columnwise = pd.concat(outliers_list)
outliers_columnwise.to_csv("./1dala/out/outliers_attribute_columnwise.csv", index=False)
mask = (np.abs(z_scores) > threshold).any(axis=1)
data_no_outliers = csv[~mask]

data_no_outliers.to_csv("./1dala/out/attribute_with_no_outliers.csv", index=False)