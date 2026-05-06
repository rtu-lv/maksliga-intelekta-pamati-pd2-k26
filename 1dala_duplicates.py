# ---- Finds duplicate data in the dataset ----
# SHOULD BE EMPTY FOR OUR DATA SET

import pandas as pd
csv = pd.read_csv("data.csv")
duplicates = csv[csv.duplicated()]
duplicates.to_csv("data_duplicate.csv", index=False)