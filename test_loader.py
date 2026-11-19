from src.truthlens.data_loader import CSVDataLoader

loader = CSVDataLoader("data/raw/bahria_news.csv")

dataset = loader.load()

if dataset is not None:

    print(dataset.head())

    loader.validate(dataset)