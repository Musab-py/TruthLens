
import pandas as pd

data = pd.read_csv("data/raw/bahria_news.csv")

print("\nActual CSV column names:")
print(data.columns.tolist())

print("\nFirst five records:")
print(data.head().to_string())

print("\nDataset shape:")
print(data.shape)
