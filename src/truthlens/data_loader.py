
import pandas as pd


class CSVDataLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully!")
            return data

        except FileNotFoundError:
            print("Dataset file not found!")
            return None

    def validate(self, data):
        required_columns = [
            "text",
            "label",
            "category",
            "source",
            "source_url",
            "verification_note"
        ]

        for column in required_columns:
            if column not in data.columns:
                print(f"Missing column: {column}")
                return False

        print("Dataset validation successful!")
        return True
