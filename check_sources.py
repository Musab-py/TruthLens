
import pandas as pd

data = pd.read_csv("data/raw/bahria_news.csv")

print("\nMissing URLs by label:")
print(data.groupby("label")["source_url"].apply(lambda x: x.isna().sum()))

print("\nRecords with missing URLs:")
missing = data[data["source_url"].isna()]

print(missing[["text", "label", "source", "verification_note"]].to_string(index=False))

print("\nREAL records with missing URLs:")
print(
    data[
        (data["label"] == 0) &
        (data["source_url"].isna())
    ][["text", "source", "verification_note"]].to_string(index=False)
)
