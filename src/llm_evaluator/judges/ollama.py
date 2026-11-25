import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_evaluator.judges.base import BaseJudge
from llm_evaluator.judges.judge_result import JudgeResult


class OllamaJudge(BaseJudge):
    """Evaluate prompts using a local Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate(self, prompt: str) -> JudgeResult:
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }

        request = Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(request_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                api_result = json.loads(response.read().decode("utf-8"))

        except HTTPError as error:
            raise RuntimeError(f"Ollama returned HTTP error {error.code}.") from error

        except URLError as error:
            raise RuntimeError("Could not connect to Ollama.") from error

        try:
            judge_data = json.loads(api_result["response"])
            score = judge_data["score"]
            reasoning = judge_data["reasoning"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Ollama returned an invalid evaluation response."
            ) from error

        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
        ):
            raise ValueError("Judge score must be between 0 and 1.")

        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Judge reasoning must be a non-empty string.")

        return JudgeResult(
            score=float(score),
            reasoning=reasoning,
        )
