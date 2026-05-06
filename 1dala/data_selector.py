import pandas as pd

df = pd.read_csv("./data.csv")

# Keep only selected columns (case-sensitive)
df_filtered = df[["MinorAxisLength", "MajorAxisLength", "Extent","Class"]]
df_filtered.to_csv("./data_selected.csv", index=False)
