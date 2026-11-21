
import pandas as pd

# Load dataset
file_path = "data/raw/bahria_news.csv"
data = pd.read_csv(file_path)

print("\n========== TRUTHLENS DATASET REPORT ==========")

# 1. Dataset size
print("\nDataset Shape:")
print(data.shape)

# 2. Check required columns
required_columns = [
    "text",
    "label",
    "category",
    "source",
    "source_url",
    "verification_note"
]

print("\nMissing Columns:")
for column in required_columns:
    if column not in data.columns:
        print("Missing:", column)

# 3. Label distribution
print("\nLabel Distribution:")
print(data["label"].value_counts(dropna=False))

print("\nClass Names:")
print("REAL:", (data["label"] == 0).sum())
print("FAKE:", (data["label"] == 1).sum())

# 4. Missing values
print("\nMissing Values:")
print(data[required_columns].isnull().sum())

# 5. Duplicate articles
print("\nDuplicate Articles:")
print(data["text"].duplicated().sum())

# 6. Duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# 7. Category distribution
print("\nCategory Distribution:")
print(data["category"].value_counts())

# 8. Source distribution
print("\nSource Distribution:")
print(data["source"].value_counts())

# 9. Invalid labels
print("\nInvalid Labels:")
print(data[~data["label"].isin([0, 1])])

# 10. Empty article text
print("\nEmpty Article Text:")
print(data[data["text"].isna() | (data["text"].astype(str).str.strip() == "")])

print("\n========== REPORT COMPLETED ==========")
