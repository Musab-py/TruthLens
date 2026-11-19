
import pandas as pd

data = pd.read_csv("data/raw/bahria_news.csv")

print("\nDataset Shape:")
print(data.shape)

print("\nLabel Distribution:")
print(data["label"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Articles:")
print(data["text"].duplicated().sum())

print("\nSample Articles:")
print(data[["text", "label"]].head(10))
