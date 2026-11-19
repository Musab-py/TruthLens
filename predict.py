from src.truthlens.pipeline import TruthLensPipeline


# Create TruthLens object
truthlens = TruthLensPipeline()


print("\n===== TRUTHLENS FAKE NEWS DETECTOR =====")

news = input("\nEnter a news article: ")

result, confidence = truthlens.predict_news(news)

print("\nPrediction:", result)
print(f"Model Probability: {confidence:.2f}%")

print("\nPrediction:", result)
