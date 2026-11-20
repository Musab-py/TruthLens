
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


class TfidfClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        self.model = LogisticRegression(
            max_iter=1000
        )

    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        self.model.fit(X_train_tfidf, y_train)

        print("Model trained successfully!")

    def predict(self, text):
        text_tfidf = self.vectorizer.transform(text)

        predictions = self.model.predict(text_tfidf)

        return predictions

    def save(self):
        joblib.dump(self.model, "models/tfidf_model.pkl")
        joblib.dump(self.vectorizer, "models/tfidf_vectorizer.pkl")

        print("Model and vectorizer saved successfully!")

    def load(self):
        self.model = joblib.load("models/tfidf_model.pkl")
        self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

        print("Model and vectorizer loaded successfully!")

    
    def predict_probability(self, text):
        text_tfidf = self.vectorizer.transform(text)
        probabilities = self.model.predict_proba(text_tfidf)
        return probabilities


