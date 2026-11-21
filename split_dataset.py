
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from urllib.parse import urlsplit, urlunsplit

# Load cleaned dataset
data = pd.read_csv("data/processed/bahria_news_cleaned.csv")

# Separate REAL and FAKE records
real_data = data[data["label"] == 0].copy()
fake_data = data[data["label"] == 1].copy()

# Normalize URLs so www and non-www versions count as one source
def normalize_url(url):
    parts = urlsplit(url.strip())

    hostname = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"

    return urlunsplit((
        parts.scheme.lower(),
        hostname,
        path,
        parts.query,
        ""
    ))

real_data["source_group"] = real_data["source_url"].apply(normalize_url)

# Split REAL records by source group
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(real_data, groups=real_data["source_group"])
)

real_train = real_data.iloc[train_indices].copy()
real_test = real_data.iloc[test_indices].copy()

# Split synthetic FAKE records separately
fake_train, fake_test = train_test_split(
    fake_data,
    test_size=0.20,
    random_state=42
)

# Combine REAL and FAKE records
train_data = pd.concat([real_train, fake_train])
test_data = pd.concat([real_test, fake_test])

# Shuffle both datasets
train_data = train_data.sample(frac=1, random_state=42)
test_data = test_data.sample(frac=1, random_state=42)

# Remove temporary grouping column
train_data = train_data.drop(columns=["source_group"])
test_data = test_data.drop(columns=["source_group"])

# Save datasets
train_data.to_csv("data/processed/train.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)

# Display report
print("\n========== DATASET SPLIT REPORT ==========")

print("\nTraining dataset:", train_data.shape)
print(train_data["label"].value_counts().sort_index())

print("\nTesting dataset:", test_data.shape)
print(test_data["label"].value_counts().sort_index())

print("\nREAL source groups in training:",
      real_train["source_group"].nunique())

print("REAL source groups in testing:",
      real_test["source_group"].nunique())

# Verify no REAL source appears in both sets
train_sources = set(real_train["source_group"])
test_sources = set(real_test["source_group"])

overlap = train_sources.intersection(test_sources)

print("\nOverlapping REAL sources:", len(overlap))

if len(overlap) == 0:
    print("SUCCESS: No source leakage detected!")
else:
    print("WARNING: Source leakage detected!")

print("\nDataset splitting completed!")
