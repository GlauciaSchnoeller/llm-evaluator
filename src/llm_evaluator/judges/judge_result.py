from dataclasses import dataclass


@dataclass
class JudgeResult:
    """Represent the result returned by an LLM judge."""

    score: float
    reasoning: str
