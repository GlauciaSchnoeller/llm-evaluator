# LLM Evaluator

A Python library for evaluating LLM responses and RAG applications.

The project provides evaluation metrics, standardized results, model comparison, and LLM-as-a-Judge capabilities. It is designed to support experiments involving Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems.

## Features

- **Classification metrics:** accuracy, precision, recall, F1-score, and confusion matrix.
- **Standardized evaluation results:** structured results with metrics and evaluation details.
- **Model comparison:** compare evaluation results from different models.
- **Faithfulness evaluation:** assess whether an LLM-generated answer is supported by the provided context.
- **LLM-as-a-Judge:** use language models to evaluate generated responses.
- **Ollama integration:** run evaluations using locally hosted language models.


## Installation

Clone the repository:

```bash
git clone https://github.com/GlauciaSchnoeller/llm-evaluator.git
cd llm-evaluator
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

## Ollama Integration

The library provides an `OllamaJudge` implementation that uses locally hosted language models through the Ollama API.

The integration uses Python's standard library (`urllib`), so no additional Python SDK is required.

### Prerequisites

1. Install [Ollama](https://ollama.com/download).
2. Download a model. For example:

```bash
ollama pull llama3.2
```

3. Make sure the Ollama service is running.

### Configuration

Configure the judge with the desired model:

```python
from llm_evaluator.judges import OllamaJudge

judge = OllamaJudge(
    model="llama3.2",
    base_url="http://localhost:11434",
    timeout=300,
)
```

The `OllamaJudge` implements the `BaseJudge` interface and returns standardized `JudgeResult` objects. It can be integrated with the `FaithfulnessEvaluator` to evaluate LLM-generated responses against a given context.

## Examples

The `examples/` directory contains usage examples:

- **`compare_classifications.py`** — compares classification evaluation results from different models.
- **`evaluate_faithfulness.py`** — evaluates response faithfulness using an LLM Judge powered by Ollama.

Run an example from the project root:

```bash
python examples/compare_classifications.py
```

Or run the faithfulness evaluation:

```bash
python examples/evaluate_faithfulness.py
```

## Running Tests

Run the test suite with:

```bash
pytest -v
```

## Code Quality

The project uses Ruff for linting and formatting, and ty for static type checking.

```bash
ruff check .
ruff format .
ty check
```
