
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:

    def evaluate(self, y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )

        recall = recall_score(
            y_true, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        matrix = confusion_matrix(y_true, y_pred)

        print("\n===== MODEL EVALUATION =====")

        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 Score:  {f1 * 100:.2f}%")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                zero_division=0
            )
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix
        }