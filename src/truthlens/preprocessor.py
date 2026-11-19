
import re

class TextPreprocessor:

    def to_lowercase(self, text):
        return text.lower()

    def clean_text(self, text):
        # Remove punctuation and special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def preprocess(self, text):
        # Convert to lowercase
        text = self.to_lowercase(text)

        # Clean the text
        text = self.clean_text(text)

        return text


# Testing the class
if __name__ == "__main__":

    preprocessor = TextPreprocessor()

    news = "BREAKING!!! Bahria UNIVERSITY announces new admissions for 2026..."

    result = preprocessor.preprocess(news)

    print("Original Text:", news)
    print("Processed Text:", result)
