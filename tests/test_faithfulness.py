import pytest

from llm_evaluator.judges import BaseJudge
from llm_evaluator.judges.judge_result import JudgeResult
from llm_evaluator.metrics.faithfulness import FaithfulnessEvaluator


class MockJudge(BaseJudge):
    def evaluate(self, prompt: str) -> JudgeResult:
        return JudgeResult(
            score=0.5,
            reasoning="One claim is not supported by the context.",
        )


def test_faithfulness_evaluation():
    evaluator = FaithfulnessEvaluator(judge=MockJudge())

    result = evaluator.evaluate(
        question="Como o usuário altera sua senha?",
        answer="Por meio de um link enviado por e-mail, válido por 24 horas.",
        context="O sistema permite alterar a senha por um link enviado por e-mail.",
    )

    assert result.evaluator == "faithfulness"
    assert result.metrics["faithfulness"] == 0.5
    assert "reasoning" in result.details


def test_faithfulness_rejects_invalid_score():
    class InvalidJudge(BaseJudge):
        def evaluate(self, prompt: str) -> JudgeResult:
            return JudgeResult(
                score=1.5,
                reasoning="Invalid score.",
            )

    evaluator = FaithfulnessEvaluator(judge=InvalidJudge())

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluator.evaluate("Question", "Answer", "Context")


def test_faithfulness_rejects_missing_reasoning():
    class InvalidJudge(BaseJudge):
        def evaluate(self, prompt: str) -> JudgeResult:
            return JudgeResult(
                score=0.8,
                reasoning="",
            )

    evaluator = FaithfulnessEvaluator(judge=InvalidJudge())

    with pytest.raises(ValueError, match="non-empty string"):
        evaluator.evaluate("Question", "Answer", "Context")
