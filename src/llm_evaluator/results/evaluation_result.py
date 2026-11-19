import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    """Represent the result of an evaluation."""

    evaluator: str
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the evaluation result into a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert the evaluation result into a JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=4,
            ensure_ascii=False,
        )

    def save_json(self, filepath: str | Path) -> None:
        """Save the evaluation result into a JSON file."""
        path = Path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            self.to_json(),
            encoding="utf-8",
        )
