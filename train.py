
import pandas as pd

from src.truthlens.preprocessor import TextPreprocessor
from src.truthlens.classifiers.tfidf_classifier import TfidfClassifier
from src.truthlens.evaluation.evaluator import ModelEvaluator
from src.truthlens.evaluation.visualizer import ModelVisualizer


# Load the prepared training and testing datasets
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))


# Preprocess news articles
preprocessor = TextPreprocessor()

X_train = train_data["text"].apply(preprocessor.preprocess)
y_train = train_data["label"]

X_test = test_data["text"].apply(preprocessor.preprocess)
y_test = test_data["label"]


# Create and train classifier
classifier = TfidfClassifier()

classifier.train(X_train, y_train)

# Save trained model and vectorizer
classifier.save()


# Make predictions
predictions = classifier.predict(X_test)
# Identify incorrect predictions
print("\n========== MISCLASSIFIED ARTICLES ==========")

for index, actual, predicted in zip(
    test_data.index,
    y_test,
    predictions
):
    if actual != predicted:
        print("\nArticle:", test_data.loc[index, "text"])
        print("Actual:", "REAL" if actual == 0 else "FAKE")
        print("Predicted:", "REAL" if predicted == 0 else "FAKE")



print("\n========== ERROR SOURCE DETAILS ==========")

for index, actual, predicted in zip(
    test_data.index,
    y_test,
    predictions
):
    if actual != predicted:

        article = test_data.loc[index]

        print("\nArticle:", article["text"])
        print("Category:", article["category"])
        print("Source:", article["source"])
        print("Source URL:", article["source_url"])
        print("Verification:", article["verification_note"])



# Evaluate model
evaluator = ModelEvaluator()

results = evaluator.evaluate(y_test, predictions)

print("\n========== TRUTHLENS MODEL RESULTS ==========")
print(results)


# Visualize evaluation results
visualizer = ModelVisualizer()

visualizer.plot_confusion_matrix(
    results["confusion_matrix"]
)

print("\nTraining and evaluation completed!")
