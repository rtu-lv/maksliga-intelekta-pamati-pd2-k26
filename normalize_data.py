import pandas as pd

csv = pd.read_csv("data.csv")

features = csv.columns[:-2]

csv[features] = (csv[features] - csv[features].mean()) / csv[features].std()

csv.to_csv("data_normalized.csv", index=False)
