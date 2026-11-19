from llm_evaluator.results import EvaluationResult


class ModelComparison:
    """Compare evaluation results from different models."""

    def __init__(self):
        self.results: dict[str, EvaluationResult] = {}

    def add_result(self, model_name: str, result: EvaluationResult) -> None:
        """Add an evaluation result for a model."""
        if model_name in self.results:
            raise ValueError(f"Model '{model_name}' already exists.")

        self.results[model_name] = result

    def compare(self) -> dict:
        """Return a comparison of model metrics."""
        comparison = {}

        for model_name, result in self.results.items():
            comparison[model_name] = {
                "accuracy": result.metrics.get("accuracy"),
                "macro_precision": result.metrics.get("averages", {})
                .get("macro", {})
                .get("precision"),
                "macro_recall": result.metrics.get("averages", {})
                .get("macro", {})
                .get("recall"),
                "macro_f1": result.metrics.get("averages", {})
                .get("macro", {})
                .get("f1_score"),
            }

        return comparison
