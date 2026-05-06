"""AgenticRunner — the entrypoint for the new agent path.

Phase 0: skeleton with NotImplementedError.
Phase 1: implements run() with Pydantic AI Agent + streaming.
"""

from __future__ import annotations
from typing import Any


class AgenticRunner:
    """Owns the Pydantic AI Agent and its tool registrations.

    A single runner is constructed at process start and reused across
    requests via Streamlit's @st.cache_resource. Per-request state lives
    in AgentDeps, not on the runner.
    """

    def __init__(self) -> None:
        self._agent = None

    def run(self, question: str, deps: Any) -> Any:
        raise NotImplementedError("Phase 1 — see plan Task 21.")
