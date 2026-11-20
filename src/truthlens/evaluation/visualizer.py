
import matplotlib.pyplot as plt
import seaborn as sns


class ModelVisualizer:

    def plot_confusion_matrix(self, matrix):

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"]
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.title("TruthLens Confusion Matrix")

        plt.tight_layout()

        plt.savefig("confusion_matrix.png")

        plt.show()

        print("Confusion matrix saved successfully!")
