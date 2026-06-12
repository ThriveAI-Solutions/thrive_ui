"""Local-LLM judge — triage only, never the verdict.

Uses the same provider/model as the agent (agent.models.build_model), so
on prod this runs against the local Ollama server and no PHI leaves the
network. A judge failure degrades to None (card shows "judge unavailable");
it never fails the harness run.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

JUDGE_INSTRUCTIONS = """\
You are a careful clinical-data QA reviewer. You are given a question that
was asked about a single patient, the answering assistant's final reply, and
summaries of the database tool results the assistant saw.

Assess ONLY whether the reply is consistent with the tool evidence:
- looks_correct: the reply's claims match the tool result summaries.
- looks_wrong: the reply contradicts the evidence, or asserts specifics
  (dates, values, names) the evidence does not support, or fabricates data
  when the tools returned nothing.
- unsure: the evidence is insufficient to tell either way.

You cannot verify ground truth — only internal consistency. When in doubt,
say unsure. Keep the reason to one short sentence."""


class JudgeVerdict(BaseModel):
    suggestion: Literal["looks_correct", "looks_wrong", "unsure"]
    reason: str


def build_judge(model=None) -> Agent:
    if model is None:
        from agent.models import build_model

        model = build_model()
    return Agent(model, output_type=JudgeVerdict, instructions=JUDGE_INSTRUCTIONS, retries=2)


def render_judge_prompt(question: str, answer: str, tool_summaries: list[str]) -> str:
    evidence = "\n".join(f"- {s}" for s in tool_summaries) or "- (no tool calls were made)"
    return f"QUESTION ASKED:\n{question}\n\nASSISTANT'S REPLY:\n{answer}\n\nTOOL RESULT SUMMARIES:\n{evidence}"


async def judge_turn(judge, question: str, answer: str, tool_summaries: list[str]) -> Optional[dict]:
    try:
        result = await judge.run(render_judge_prompt(question, answer, tool_summaries))
        return result.output.model_dump()
    except Exception as exc:
        logger.warning("judge failed: %s", exc)
        return None
