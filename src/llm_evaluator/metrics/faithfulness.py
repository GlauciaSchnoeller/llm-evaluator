from llm_evaluator.judges import BaseJudge
from llm_evaluator.results import EvaluationResult


class FaithfulnessEvaluator:
    """Evaluate how faithful an answer is to the provided context."""

    def __init__(self, judge: BaseJudge):
        self.judge = judge

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:

        prompt = f"""
        Evaluate whether the answer is supported by the context.

        Question:
        {question}

        Context:
        {context}

        Answer:
        {answer}

        Return:
        - score: a value between 0 and 1
        - reasoning: explanation of the score
        """

        judge_result = self.judge.evaluate(prompt)

        score = judge_result.score
        reasoning = judge_result.reasoning

        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("Judge score must be a numeric value.")

        if not 0 <= score <= 1:
            raise ValueError("Judge score must be between 0 and 1.")

        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Judge reasoning must be a non-empty string.")

        return EvaluationResult(
            evaluator="faithfulness",
            metrics={
                "faithfulness": float(score),
            },
            details={
                "reasoning": reasoning,
                "question": question,
                "answer": answer,
                "context": context,
            },
        )
