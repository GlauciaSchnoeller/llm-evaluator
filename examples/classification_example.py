from llm_evaluator.metrics.classification import (
    ClassificationEvaluator,
)

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

evaluator = ClassificationEvaluator(
    labels=[
        "ambiguidade",
        "incompletude",
        "inconsistencia",
    ]
)

results = evaluator.evaluate(y_true, y_pred)

print(results)
