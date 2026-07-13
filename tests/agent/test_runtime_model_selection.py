"""Issue #236: the agent runner must honor the user's in-app model selection.

`agent.runtime._selected_model()` turns the persisted session selection into a
(provider, model) override for `build_model`, but only when the provider is a
real, configured registry provider — otherwise it falls back to the secrets
default. `_runner(provider, model)` threads that pair into the AgenticRunner and
is cache-keyed on it so switching models actually rebuilds the runner.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.runtime as runtime_mod


class _St:
    def __init__(self, session_state: dict, secrets: dict | None = None):
        self.session_state = session_state
        self.secrets = secrets if secrets is not None else {}


def _patch_registry(monkeypatch, *, configured_providers: set[str]):
    """Fake registry: get_provider returns a provider whose is_configured is
    True only for the named providers; None for anything unknown."""

    def _get_provider(pid):
        if pid in configured_providers:
            return SimpleNamespace(is_configured=lambda secrets: True)
        if pid == "known-but-unconfigured":
            return SimpleNamespace(is_configured=lambda secrets: False)
        return None

    fake_registry = SimpleNamespace(get_provider=_get_provider)
    monkeypatch.setattr(
        "utils.llm_registry.registry.get_registry", lambda: fake_registry, raising=True
    )


def test_selected_model_returns_pair_for_configured_provider(monkeypatch):
    _patch_registry(monkeypatch, configured_providers={"ollama"})
    monkeypatch.setattr(
        runtime_mod,
        "st",
        _St({"selected_llm_provider": "ollama", "selected_llm_model": "gpt-oss:20b"}),
        raising=True,
    )
    assert runtime_mod._selected_model() == ("ollama", "gpt-oss:20b")


def test_selected_model_none_when_unset(monkeypatch):
    _patch_registry(monkeypatch, configured_providers={"ollama"})
    monkeypatch.setattr(runtime_mod, "st", _St({}), raising=True)
    assert runtime_mod._selected_model() == (None, None)


def test_selected_model_none_when_only_provider_set(monkeypatch):
    _patch_registry(monkeypatch, configured_providers={"ollama"})
    monkeypatch.setattr(
        runtime_mod, "st", _St({"selected_llm_provider": "ollama"}), raising=True
    )
    assert runtime_mod._selected_model() == (None, None)


def test_selected_model_none_for_unknown_provider(monkeypatch):
    _patch_registry(monkeypatch, configured_providers={"ollama"})
    monkeypatch.setattr(
        runtime_mod,
        "st",
        _St({"selected_llm_provider": "mystery", "selected_llm_model": "x"}),
        raising=True,
    )
    assert runtime_mod._selected_model() == (None, None)


def test_selected_model_none_for_unconfigured_provider(monkeypatch):
    _patch_registry(monkeypatch, configured_providers=set())
    monkeypatch.setattr(
        runtime_mod,
        "st",
        _St({"selected_llm_provider": "known-but-unconfigured", "selected_llm_model": "x"}),
        raising=True,
    )
    assert runtime_mod._selected_model() == (None, None)


def test_selected_model_falls_back_on_registry_error(monkeypatch):
    monkeypatch.setattr(
        "utils.llm_registry.registry.get_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True,
    )
    monkeypatch.setattr(
        runtime_mod,
        "st",
        _St({"selected_llm_provider": "ollama", "selected_llm_model": "x"}),
        raising=True,
    )
    assert runtime_mod._selected_model() == (None, None)


def test_runner_threads_overrides_into_agentic_runner(monkeypatch):
    spy = MagicMock(return_value="runner-instance")
    monkeypatch.setattr(runtime_mod, "AgenticRunner", spy, raising=True)
    monkeypatch.setattr(runtime_mod, "configure_observability", lambda: None, raising=True)

    # Unique args so st.cache_resource doesn't return a value cached by another
    # test; assert the override kwargs reach AgenticRunner.
    result = runtime_mod._runner("ollama", "unit-test-model-236")
    assert result == "runner-instance"
    spy.assert_called_once_with(provider_override="ollama", model_override="unit-test-model-236")
