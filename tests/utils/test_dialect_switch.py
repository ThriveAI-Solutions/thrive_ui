"""Tests for the Postgres/Redshift dialect switch.

Dialect is configured via [postgres].dialect in secrets.toml. It must
propagate to every Vanna backend so the LLM system prompt names the
right warehouse, and the few hardcoded introspection queries we own
(STRING_AGG vs LISTAGG, etc.) must branch on it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.thriveai_base import ThriveAI_Base
from utils.vanna_calls import (
    MyVannaAnthropicChromaDB,
    MyVannaGeminiChromaDB,
    MyVannaOllamaChromaDB,
    VannaService,
    extract_vanna_config_from_secrets,
)


# ---------- shared fixtures ----------


def _secrets_dict(test_chromadb_path: str, dialect: str | None) -> dict:
    postgres: dict = {
        "host": "localhost",
        "port": 5432,
        "database": "thrive",
        "user": "postgres",
        "password": "postgres",
    }
    if dialect is not None:
        postgres["dialect"] = dialect
    return {
        "ai_keys": {
            "ollama_model": "llama3",
            "vanna_api": "mock_vanna_api",
            "vanna_model": "mock_vanna_model",
            "anthropic_api": "mock_anthropic_api",
            "anthropic_model": "claude-3-sonnet-20240229",
            "gemini_model": "gemini-1.5-flash",
            "gemini_api": "mock_gemini_api",
        },
        "rag_model": {"chroma_path": test_chromadb_path},
        "postgres": postgres,
        "security": {"allow_llm_to_see_data": True},
    }


@pytest.fixture
def secrets_with_redshift_dialect(test_chromadb_path):
    with patch("streamlit.secrets", new=_secrets_dict(test_chromadb_path, "redshift")):
        yield


@pytest.fixture
def secrets_without_dialect(test_chromadb_path):
    with patch("streamlit.secrets", new=_secrets_dict(test_chromadb_path, None)):
        yield


# ---------- config extraction ----------


def test_extract_vanna_config_passes_dialect_through(secrets_with_redshift_dialect):
    config = extract_vanna_config_from_secrets()
    assert config["postgres"]["dialect"] == "redshift"


def test_extract_vanna_config_omits_dialect_when_unset(secrets_without_dialect):
    config = extract_vanna_config_from_secrets()
    # Caller treats absence as "postgresql".
    assert "dialect" not in config["postgres"]


# ---------- backend dialect propagation ----------


def test_ollama_chromadb_sets_self_dialect_from_secrets(secrets_with_redshift_dialect, test_chromadb_path):
    with (
        patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__", return_value=None),
        patch("utils.vanna_calls.Ollama.__init__", return_value=None),
        patch("utils.vanna_calls.ThriveAI_Ollama.__init__", return_value=None),
    ):
        backend = MyVannaOllamaChromaDB(user_role=1, config={"path": test_chromadb_path})
        assert backend.dialect == "redshift"


def test_anthropic_chromadb_sets_self_dialect_from_secrets(secrets_with_redshift_dialect, test_chromadb_path):
    with (
        patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__", return_value=None),
        patch("utils.vanna_calls.Anthropic_Chat.__init__", return_value=None),
    ):
        backend = MyVannaAnthropicChromaDB(user_role=1, config={"path": test_chromadb_path})
        assert backend.dialect == "redshift"


def test_gemini_chromadb_sets_self_dialect_from_secrets(secrets_with_redshift_dialect, test_chromadb_path):
    with (
        patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__", return_value=None),
        patch("utils.vanna_calls.GoogleGeminiChat.__init__", return_value=None),
        patch("google.generativeai.configure"),
        patch("google.generativeai.GenerativeModel"),
    ):
        backend = MyVannaGeminiChromaDB(user_role=1, config={"path": test_chromadb_path})
        assert backend.dialect == "redshift"


def test_anthropic_chromadb_defaults_dialect_to_postgresql(secrets_without_dialect, test_chromadb_path):
    with (
        patch("utils.vanna_calls.ThriveAI_ChromaDB.__init__", return_value=None),
        patch("utils.vanna_calls.Anthropic_Chat.__init__", return_value=None),
    ):
        backend = MyVannaAnthropicChromaDB(user_role=1, config={"path": test_chromadb_path})
        assert backend.dialect == "postgresql"


# ---------- system prompt content ----------


def _make_prompt_stub(dialect: str) -> ThriveAI_Base:
    """Return a ThriveAI_Base subclass instance suitable for testing get_sql_prompt."""

    class _Stub(ThriveAI_Base):
        def add_ddl_to_prompt(self, prompt, ddl_list, **kwargs):
            return prompt

        def add_documentation_to_prompt(self, prompt, doc_list, **kwargs):
            return prompt

        def system_message(self, m):
            return {"role": "system", "content": m}

        def user_message(self, m):
            return {"role": "user", "content": m}

        def assistant_message(self, m):
            return {"role": "assistant", "content": m}

        def generate_embedding(self, data, **kwargs):
            return [0.1]

        def submit_prompt(self, *a, **kw):
            return ""

    # VannaBase has many abstract storage methods we don't care about for
    # prompt-content tests — neutralize the ABC machinery so we can build an
    # instance without supplying them.
    _Stub.__abstractmethods__ = frozenset()
    stub = _Stub.__new__(_Stub)
    stub.config = {"dialect": dialect, "schema": "public"}
    stub.dialect = dialect
    stub.schema = "public"
    stub.max_tokens = 32_000
    stub.static_documentation = ""
    return stub


def test_system_prompt_names_dialect():
    redshift = _make_prompt_stub("redshift")
    msgs = redshift.get_sql_prompt(
        initial_prompt=None,
        question="how many patients?",
        question_sql_list=[],
        ddl_list=[],
        doc_list=[],
    )
    system_text = msgs[0]["content"]
    assert "redshift expert" in system_text.lower()


def test_system_prompt_includes_redshift_cheatsheet_only_for_redshift():
    redshift = _make_prompt_stub("redshift")
    postgres = _make_prompt_stub("postgresql")

    redshift_text = redshift.get_sql_prompt(
        initial_prompt=None, question="q", question_sql_list=[], ddl_list=[], doc_list=[]
    )[0]["content"]
    postgres_text = postgres.get_sql_prompt(
        initial_prompt=None, question="q", question_sql_list=[], ddl_list=[], doc_list=[]
    )[0]["content"]

    # The cheat-sheet calls out LISTAGG as the Redshift equivalent of STRING_AGG.
    assert "LISTAGG" in redshift_text
    assert "LISTAGG" not in postgres_text
    # And explicitly tells the LLM to avoid CORR on Redshift.
    assert "CORR" in redshift_text
    assert "CORR" not in postgres_text


# ---------- _build_index_query dialect branch ----------


def test_build_index_query_uses_string_agg_for_postgres():
    svc = VannaService.__new__(VannaService)
    svc.dialect = "postgresql"
    sql = svc._build_index_query("public", "")
    assert "STRING_AGG" in sql
    assert "LISTAGG" not in sql


def test_build_index_query_uses_listagg_for_redshift():
    svc = VannaService.__new__(VannaService)
    svc.dialect = "redshift"
    sql = svc._build_index_query("public", "")
    assert "LISTAGG" in sql
    assert "STRING_AGG" not in sql


def test_build_index_query_defaults_to_postgres_when_dialect_missing():
    svc = VannaService.__new__(VannaService)
    # No svc.dialect attribute set — current callers may not have one yet.
    sql = svc._build_index_query("public", "")
    assert "STRING_AGG" in sql
