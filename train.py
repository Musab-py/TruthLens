from src.truthlens.evaluation.evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.truthlens.data_loader import CSVDataLoader
from src.truthlens.preprocessor import TextPreprocessor
from src.truthlens.classifiers.tfidf_classifier import TfidfClassifier
from src.truthlens.evaluation.visualizer import ModelVisualizer

# Load dataset
loader = CSVDataLoader("data/raw/bahria_news.csv")
data = loader.load()

if data is not None and loader.validate(data):

    # Preprocess news articles
    preprocessor = TextPreprocessor()

    data["clean_text"] = data["text"].apply(
        preprocessor.preprocess
    )

    # Separate features and labels
    X = data["clean_text"]
    y = data["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create and train classifier
    classifier = TfidfClassifier()
    classifier.train(X_train, y_train)
    classifier.save()

    # Make predictions
    predictions = classifier.predict(X_test)

# Evaluate model
evaluator = ModelEvaluator()

results = evaluator.evaluate(y_test, predictions)

# Visualize evaluation results
visualizer = ModelVisualizer()

visualizer.plot_confusion_matrix(
    results["confusion_matrix"]
)