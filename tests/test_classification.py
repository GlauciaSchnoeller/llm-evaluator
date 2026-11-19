import pytest

from llm_evaluator.metrics.classification import (
    ClassificationEvaluator,
)


@pytest.fixture
def evaluator():
    return ClassificationEvaluator(
        labels=[
            "ambiguidade",
            "incompletude",
            "inconsistencia",
        ]
    )


@pytest.fixture
def classification_data():
    y_true = [
        "ambiguidade",
        "incompletude",
        "inconsistencia",
        "ambiguidade",
        "incompletude",
    ]

    y_pred = [
        "ambiguidade",
        "incompletude",
        "ambiguidade",
        "ambiguidade",
        "inconsistencia",
    ]

    return y_true, y_pred


def test_accuracy(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.metrics["accuracy"] == pytest.approx(0.6)


def test_precision_per_class(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["per_class"]["ambiguidade"]["precision"] == pytest.approx(
        2 / 3
    )
    assert results.details["per_class"]["incompletude"]["precision"] == pytest.approx(
        1.0
    )
    assert results.details["per_class"]["inconsistencia"]["precision"] == pytest.approx(
        0.0
    )


def test_recall_per_class(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["per_class"]["ambiguidade"]["recall"] == pytest.approx(1.0)
    assert results.details["per_class"]["incompletude"]["recall"] == pytest.approx(0.5)
    assert results.details["per_class"]["inconsistencia"]["recall"] == pytest.approx(
        0.0
    )


def test_f1_score_per_class(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["per_class"]["ambiguidade"]["f1_score"] == pytest.approx(0.8)
    assert results.details["per_class"]["incompletude"]["f1_score"] == pytest.approx(
        2 / 3
    )
    assert results.details["per_class"]["inconsistencia"]["f1_score"] == pytest.approx(
        0.0
    )


def test_confusion_matrix(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["confusion_matrix"] == [
        [2, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
    ]


def test_empty_data_raises_error(evaluator):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate([], [])


def test_different_lengths_raise_error(evaluator):
    y_true = ["ambiguidade", "incompletude"]
    y_pred = ["ambiguidade"]

    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate(y_true, y_pred)


def test_class_without_predictions(evaluator):
    y_true = ["ambiguidade", "incompletude"]
    y_pred = ["ambiguidade", "ambiguidade"]

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["per_class"]["inconsistencia"]["precision"] == 0.0
    assert results.details["per_class"]["inconsistencia"]["recall"] == 0.0
    assert results.details["per_class"]["inconsistencia"]["f1_score"] == 0.0


def test_aggregate_metrics(evaluator, classification_data):
    y_true, y_pred = classification_data

    results = evaluator.evaluate(y_true, y_pred)

    assert results.metrics["averages"]["micro"]["f1_score"] == pytest.approx(0.6)

    assert results.metrics["averages"]["macro"]["precision"] == pytest.approx(5 / 9)

    assert results.metrics["averages"]["macro"]["recall"] == pytest.approx(0.5)

    assert results.metrics["averages"]["weighted"]["f1_score"] == pytest.approx(44 / 75)


def test_unknown_label_raises_error(evaluator):
    y_true = ["ambiguidade"]
    y_pred = ["outro"]

    with pytest.raises(ValueError, match="Unknown labels"):
        evaluator.evaluate(y_true, y_pred)


def test_duplicate_labels_raise_error():
    with pytest.raises(ValueError, match="must be unique"):
        ClassificationEvaluator(labels=["ambiguidade", "ambiguidade"])


def test_empty_labels_raise_error():
    with pytest.raises(ValueError, match="cannot be empty"):
        ClassificationEvaluator(labels=[])


def test_confusion_matrix_respects_label_order(evaluator):
    y_true = [
        "inconsistencia",
        "ambiguidade",
        "incompletude",
    ]

    y_pred = [
        "inconsistencia",
        "ambiguidade",
        "incompletude",
    ]

    results = evaluator.evaluate(y_true, y_pred)

    assert results.details["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
