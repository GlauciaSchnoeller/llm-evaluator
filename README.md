# LLM Evaluator

A Python library for evaluating LLM responses and comparing model performance.

## Features

- Classification metrics (accuracy, precision, recall, F1-score).
- Confusion matrix and classification reports.
- Faithfulness evaluation using LLM-as-a-Judge.
- Standardized evaluation results.
- Model comparison.
- JSON export of evaluation results.


## Installation

Clone the repository:

```bash
git clone https://github.com/GlauciaSchnoeller/llm-evaluator.git
cd llm-evaluator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Classification Evaluation

```python
from llm_evaluator.metrics.classification import ClassificationEvaluator

evaluator = ClassificationEvaluator(
    labels=["ambiguity", "incompleteness", "inconsistency"]
)

result = evaluator.evaluate(
    y_true=["ambiguity", "incompleteness", "inconsistency"],
    y_pred=["ambiguity", "incompleteness", "ambiguity"],
)

print(result.metrics)
```

### Faithfulness Evaluation

The `FaithfulnessEvaluator` uses an LLM Judge to evaluate whether an answer is supported by the provided context.

```python
from llm_evaluator.metrics.faithfulness import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(judge=my_judge)

result = evaluator.evaluate(
    question="How can the user reset their password?",
    answer="Through a link sent by email.",
    context="The system allows password reset through an email link.",
)

print(result.metrics)
```

The judge must implement the `BaseJudge` interface and return a `JudgeResult`.

### Export Results

```python
result.save_json("results/evaluation.json")
```

## Running Tests

```bash
pytest -v
```

## Development

This project is under development. New evaluation metrics and LLM integrations will be added.
