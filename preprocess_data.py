
from src.truthlens.data_loader import CSVDataLoader
from src.truthlens.preprocessor import TextPreprocessor


# Create objects
loader = CSVDataLoader("data/raw/bahria_news.csv")
preprocessor = TextPreprocessor()


# Load dataset
data = loader.load()


if data is not None:

    # Validate dataset
    if loader.validate(data):

        # Clean every news article
        data["clean_text"] = data["text"].apply(
            preprocessor.preprocess
        )

        # Save cleaned dataset
        data.to_csv(
            "data/processed/bahria_news_cleaned.csv",
            index=False
        )

        print("\nPreprocessing completed!")
        print(data[["text", "clean_text"]].head())

    else:
        print("Dataset validation failed!")
