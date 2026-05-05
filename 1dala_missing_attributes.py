# ---- Finds unfilled attrobutes in the dataset ----
# SHOULD BE EMPTY FOR OUR DATA SET

import pandas as pd
csv = pd.read_csv("data.csv")
missing_data = csv[csv.isnull().any(axis=1)]
missing_data.to_csv("missing_data.csv", index=False)