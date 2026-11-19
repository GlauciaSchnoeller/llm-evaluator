from llm_evaluator.metrics.classification import (
    ClassificationEvaluator,
)

# Expected classifications (ground truth)
y_true = [
    "ambiguidade",
    "incompletude",
    "inconsistencia",
    "ambiguidade",
    "incompletude",
    "inconsistencia",
]

# Simulated predictions from Model A
model_a_predictions = [
    "ambiguidade",
    "incompletude",
    "ambiguidade",
    "ambiguidade",
    "inconsistencia",
    "inconsistencia",
]

# Simulated predictions from Model B
model_b_predictions = [
    "ambiguidade",
    "incompletude",
    "inconsistencia",
    "ambiguidade",
    "incompletude",
    "ambiguidade",
]

evaluator = ClassificationEvaluator(
    labels=[
        "ambiguidade",
        "incompletude",
        "inconsistencia",
    ]
)

# Evaluate models
model_a_results = evaluator.evaluate(
    y_true,
    model_a_predictions,
)

model_b_results = evaluator.evaluate(
    y_true,
    model_b_predictions,
)

# Save results
model_a_results.save_json("results/model_a.json")
model_b_results.save_json("results/model_b.json")

# Compare accuracy
print("Model A Accuracy:", model_a_results.metrics["accuracy"])
print("Model B Accuracy:", model_b_results.metrics["accuracy"])

print("\nModel comparison:")

if model_a_results.metrics["accuracy"] > model_b_results.metrics["accuracy"]:
    print("Model A performed better.")

elif model_b_results.metrics["accuracy"] > model_a_results.metrics["accuracy"]:
    print("Model B performed better.")

else:
    print("Both models achieved the same accuracy.")
