
import pandas as pd

data = pd.read_csv("data/processed/bahria_news_cleaned.csv")

# Only inspect REAL records because they have source URLs
real_data = data[data["label"] == 0]

print("\nTotal REAL records:", len(real_data))
print("Unique REAL source URLs:", real_data["source_url"].nunique())

print("\nRecords grouped by source URL:")

source_counts = (
    real_data.groupby("source_url")
    .size()
    .sort_values(ascending=False)
)

print(source_counts.to_string())

print("\nSources used by multiple records:")
print(source_counts[source_counts > 1].to_string())
