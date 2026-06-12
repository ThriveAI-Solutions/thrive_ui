import asyncio

from pydantic_ai.models.test import TestModel

from evals.judge import JudgeVerdict, build_judge, judge_turn, render_judge_prompt


def test_prompt_includes_question_answer_and_evidence():
    p = render_judge_prompt(
        question="Does this patient have diabetes?",
        answer="Yes, A1C 7.2 on 2026-03-15.",
        tool_summaries=["labs: 2 rows; data_availability=data_present"],
    )
    assert "Does this patient have diabetes?" in p
    assert "A1C 7.2" in p
    assert "data_present" in p


def test_judge_turn_returns_verdict_dict():
    judge = build_judge(model=TestModel())
    out = asyncio.run(judge_turn(judge, question="Q?", answer="A.", tool_summaries=[]))
    assert out is not None
    assert out["suggestion"] in {"looks_correct", "looks_wrong", "unsure"}
    assert isinstance(out["reason"], str)
    JudgeVerdict(**out)  # dict round-trips back into the model


def test_judge_turn_swallows_failures():
    class _Boom:
        async def run(self, *a, **k):
            raise RuntimeError("model down")

    out = asyncio.run(judge_turn(_Boom(), question="Q?", answer="A.", tool_summaries=[]))
    assert out is None
