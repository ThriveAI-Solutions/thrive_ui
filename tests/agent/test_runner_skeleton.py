"""Phase 0: just the skeleton. Phase 1 fills in the real agent."""

import pytest
from agent.runner import AgenticRunner


def test_runner_can_be_constructed():
    runner = AgenticRunner()
    assert runner is not None


def test_runner_run_raises_not_implemented():
    runner = AgenticRunner()
    with pytest.raises(NotImplementedError):
        runner.run("anything", deps=None)
