import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from llm_evaluator.judges import JudgeResult, OllamaJudge


@patch("llm_evaluator.judges.ollama.urlopen")
def test_ollama_judge_returns_judge_result(mock_urlopen):
    api_response = {
        "response": json.dumps(
            {
                "score": 0.9,
                "reasoning": "The answer is supported by the context.",
            }
        )
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(api_response).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    judge = OllamaJudge()

    result = judge.evaluate("Evaluate this answer.")

    assert isinstance(result, JudgeResult)
    assert result.score == 0.9
    assert result.reasoning == "The answer is supported by the context."


@patch("llm_evaluator.judges.ollama.urlopen")
def test_ollama_judge_rejects_invalid_score(mock_urlopen):
    api_response = {
        "response": json.dumps(
            {
                "score": 1.5,
                "reasoning": "Invalid score.",
            }
        )
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(api_response).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    judge = OllamaJudge()

    with pytest.raises(ValueError, match="between 0 and 1"):
        judge.evaluate("Evaluate this answer.")


@patch("llm_evaluator.judges.ollama.urlopen")
def test_ollama_judge_rejects_empty_reasoning(mock_urlopen):
    api_response = {
        "response": json.dumps(
            {
                "score": 0.8,
                "reasoning": "",
            }
        )
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(api_response).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    judge = OllamaJudge()

    with pytest.raises(ValueError, match="non-empty string"):
        judge.evaluate("Evaluate this answer.")


@patch("llm_evaluator.judges.ollama.urlopen")
def test_ollama_judge_handles_connection_error(mock_urlopen):
    mock_urlopen.side_effect = URLError("Connection refused")

    judge = OllamaJudge()

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        judge.evaluate("Evaluate this answer.")
