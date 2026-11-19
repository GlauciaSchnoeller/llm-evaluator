import inspect

from llm_evaluator.judges import BaseJudge


def test_base_judge_is_abstract():
    assert inspect.isabstract(BaseJudge)