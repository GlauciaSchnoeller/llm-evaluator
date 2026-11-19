import json

import pytest

from llm_evaluator.results import EvaluationResult


@pytest.fixture
def evaluation_result():
    return EvaluationResult(
        evaluator="classification",
        metrics={
            "accuracy": 0.6,
            "averages": {
                "micro": {
                    "precision": 0.6,
                    "recall": 0.6,
                    "f1_score": 0.6,
                }
            },
        },
        details={
            "per_class": {
                "ambiguidade": {
                    "precision": 0.8,
                    "recall": 1.0,
                    "f1_score": 0.8,
                    "support": 2,
                }
            },
            "confusion_matrix": [[2, 0], [0, 1]],
        },
    )


def test_evaluation_result_creation(evaluation_result):
    assert evaluation_result.evaluator == "classification"
    assert evaluation_result.metrics["accuracy"] == pytest.approx(0.6)


def test_evaluation_result_to_dict(evaluation_result):
    result = evaluation_result.to_dict()

    assert isinstance(result, dict)
    assert result["evaluator"] == "classification"
    assert result["metrics"]["accuracy"] == pytest.approx(0.6)
    assert "per_class" in result["details"]


def test_evaluation_result_empty_defaults():
    result = EvaluationResult(evaluator="classification")

    assert result.metrics == {}
    assert result.details == {}


def test_evaluation_result_independent_instances():
    result_1 = EvaluationResult(evaluator="classification")
    result_2 = EvaluationResult(evaluator="rag")

    result_1.metrics["accuracy"] = 0.9

    assert result_2.metrics == {}


def test_evaluation_result_to_json(evaluation_result):
    result = evaluation_result.to_json()

    assert isinstance(result, str)
    assert '"evaluator": "classification"' in result


def test_evaluation_result_save_json(evaluation_result, tmp_path):
    filepath = tmp_path / "results" / "classification.json"

    evaluation_result.save_json(filepath)

    assert filepath.exists()

    data = json.loads(filepath.read_text(encoding="utf-8"))

    assert data["evaluator"] == "classification"
    assert data["metrics"]["accuracy"] == pytest.approx(0.6)
