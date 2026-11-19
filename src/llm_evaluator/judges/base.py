from abc import ABC, abstractmethod

from llm_evaluator.judges.judge_result import JudgeResult


class BaseJudge(ABC):
    @abstractmethod
    def evaluate(self, prompt: str) -> JudgeResult:
        pass
