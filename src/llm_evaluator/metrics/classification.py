from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from llm_evaluator.results import EvaluationResult


@dataclass
class ClassificationEvaluator:
    """Evaluate classification predictions."""

    labels: list[str]

    def __post_init__(self):
        if not self.labels:
            raise ValueError("labels cannot be empty.")

        if len(self.labels) != len(set(self.labels)):
            raise ValueError("labels must be unique.")

    def evaluate(
        self,
        y_true: list[str],
        y_pred: list[str],
    ) -> EvaluationResult:
        """Evaluate classification predictions."""

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        if not y_true:
            raise ValueError("y_true and y_pred cannot be empty.")

        unknown_labels = (set(y_true) | set(y_pred)) - set(self.labels)

        if unknown_labels:
            raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=self.labels,
            zero_division=0,
        )

        per_class = {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(self.labels)
        }

        averages = {}

        for average in ["micro", "macro", "weighted"]:
            p, r, f, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=self.labels,
                average=average,
                zero_division=0,
            )

            averages[average] = {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
            }

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=self.labels,
        )

        return EvaluationResult(
            evaluator="classification",
            metrics={
                "accuracy": float(accuracy),
                "averages": averages,
            },
            details={
                "per_class": per_class,
                "confusion_matrix": matrix.tolist(),
                "labels": self.labels,
            },
        )
