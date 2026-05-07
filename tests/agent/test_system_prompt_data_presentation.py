"""The system prompt must explicitly tell the model to present findings
when a tool returns data — otherwise the model hedges with "I can offer
to summarize..." instead of actually answering. Doctors using the
platform need the data, not a meta-description of it.
"""

from __future__ import annotations

from agent.system_prompt import SYSTEM_PROMPT


def test_prompt_requires_presenting_data_when_present():
    """Explicit phrasing tying data_availability=data_present to actual presentation."""
    p = SYSTEM_PROMPT.lower()
    assert "data_present" in p
    # Must instruct the model to present, list, summarize, or show the data.
    presents_data = any(verb in p for verb in ("present", "list ", "show ", "summarize the", "report the"))
    assert presents_data, "system prompt does not instruct the model to present data"


def test_prompt_forbids_hedging_when_data_present():
    """When data is in hand, the model should not pivot to 'would you like
    me to...' — that turns useful clinical answers into useless meta-talk."""
    p = SYSTEM_PROMPT.lower()
    # The forbidden behavior must be explicitly called out somewhere in the
    # prompt, not just absent. Search for any of these phrasings.
    explicit_anti_hedge = any(
        phrase in p
        for phrase in (
            "do not offer",
            "do not ask whether",
            "do not hedge",
            "do not describe what you found instead",
            "must present the findings",
            "must not describe",
            "answer directly",
        )
    )
    assert explicit_anti_hedge, "system prompt does not explicitly forbid hedging when data is present"


def test_prompt_gives_long_list_guidance():
    """For 50+ rows the model needs to know to group/summarize rather
    than dump or refuse."""
    p = SYSTEM_PROMPT.lower()
    # Some shape of "for many results, group/summarize/most-recent/by class".
    has_long_list_guidance = any(
        phrase in p
        for phrase in (
            "group by",
            "by class",
            "by category",
            "most recent",
            "summarize counts",
            "if more than",
            "if the result set is long",
        )
    )
    assert has_long_list_guidance, "system prompt does not guide handling of long result lists"
