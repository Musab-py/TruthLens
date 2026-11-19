
from src.truthlens.classifiers.tfidf_classifier import TfidfClassifier
from src.truthlens.preprocessor import TextPreprocessor


class TruthLensPipeline:

    def __init__(self):

        self.preprocessor = TextPreprocessor()

        self.classifier = TfidfClassifier()

        # Load previously trained model
        self.classifier.load()


    def predict_news(self, news_text):

        cleaned_text = self.preprocessor.preprocess(news_text)

        prediction = self.classifier.predict([cleaned_text])

        probabilities = self.classifier.predict_probability([cleaned_text])

        if prediction[0] == 0:
            result = "REAL"
            confidence = probabilities[0][0] * 100
        else:
            result = "FAKE"
            confidence = probabilities[0][1] * 100

        return result, confidence
