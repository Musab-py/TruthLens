
import pandas as pd
from pathlib import Path

# Load original dataset
input_path = "data/raw/bahria_news.csv"

data = pd.read_csv(input_path)

print("Original dataset loaded:", data.shape)

# Standardize category names
data["category"] = (
    data["category"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.title()
)

# Standardize source names
data["source"] = data["source"].astype(str).str.strip()

source_mapping = {
    "bahria university (official website)": "Bahria University Official Website",
    "bahria university official website": "Bahria University Official Website",
    "student-generated example": "Student-generated example",
    "synthetic example": "Synthetic example"
}

data["source"] = data["source"].replace(source_mapping)

# Remove unnecessary spaces from text fields
data["text"] = data["text"].astype(str).str.strip()
data["verification_note"] = (
    data["verification_note"].astype(str).str.strip()
)

# Save cleaned dataset
output_path = Path("data/processed/bahria_news_cleaned.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output_path, index=False)

print("\nDataset cleaning completed!")
print("Cleaned dataset shape:", data.shape)

print("\nUpdated category distribution:")
print(data["category"].value_counts())

print("\nUpdated source distribution:")
print(data["source"].value_counts())

print("\nFinal label distribution:")
print(data["label"].value_counts())

print("\nSaved to:", output_path)
