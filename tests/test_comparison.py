import pytest

from llm_evaluator.comparison import ModelComparison
from llm_evaluator.metrics.classification import ClassificationEvaluator


@pytest.fixture
def evaluator():
    return ClassificationEvaluator(
        labels=["ambiguidade", "incompletude", "inconsistencia"]
    )


def test_compare_two_models(evaluator):
    y_true = [
        "ambiguidade",
        "incompletude",
        "inconsistencia",
        "ambiguidade",
        "incompletude",
        "inconsistencia",
    ]

    model_a = [
        "ambiguidade",
        "incompletude",
        "ambiguidade",
        "ambiguidade",
        "inconsistencia",
        "inconsistencia",
    ]

    model_b = [
        "ambiguidade",
        "incompletude",
        "inconsistencia",
        "ambiguidade",
        "incompletude",
        "ambiguidade",
    ]

    result_a = evaluator.evaluate(y_true, model_a)
    result_b = evaluator.evaluate(y_true, model_b)

    comparison = ModelComparison()
    comparison.add_result("Model A", result_a)
    comparison.add_result("Model B", result_b)

    results = comparison.compare()

    assert results["Model A"]["accuracy"] == pytest.approx(4 / 6)
    assert results["Model B"]["accuracy"] == pytest.approx(5 / 6)


def test_prevent_duplicate_model(evaluator):
    result = evaluator.evaluate(
        ["ambiguidade"],
        ["ambiguidade"],
    )

    comparison = ModelComparison()
    comparison.add_result("Model A", result)

    with pytest.raises(ValueError):
        comparison.add_result("Model A", result)
