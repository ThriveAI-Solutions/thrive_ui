"""Render the schema-reference + few-shot SQL block injected into run_sql's
pydantic-ai tool description.

Lives next to the SQL-template modules in agent/db/queries/* because it
shares their schema_prefix discipline. Rendered text goes ONLY to the
LLM via the run_sql tool description — never executed.
"""

from __future__ import annotations
from typing import Optional

from agent.rag.seed import IDENTITY_DOCS, RUN_SQL_EXAMPLES, SCHEMA_DOCS


def _header(schema_prefix: str) -> str:
    if schema_prefix:
        prefix_no_dot = schema_prefix.rstrip(".")
        return (
            f"SCHEMA REFERENCE — run_sql may ONLY read from the {prefix_no_dot}.* views below.\n"
            f"ALWAYS write fully-qualified names ({prefix_no_dot}.<view>); the warehouse search_path\n"
            f"does NOT include {prefix_no_dot}, so unqualified names will fail. Do not invent tables.\n"
            "If the answer needs a domain not listed, say so plainly instead of guessing."
        )
    return (
        "SCHEMA REFERENCE — run_sql may ONLY read from the views below.\n"
        "ALWAYS use the view names exactly as listed; do not invent tables.\n"
        "If the answer needs a domain not listed, say so plainly instead of guessing."
    )


def _render_catalog(schema_prefix: str) -> str:
    lines: list[str] = []
    for doc in SCHEMA_DOCS:
        view = doc.get("view") or ""
        text = doc["text"].strip()
        if view:
            qualified = f"{schema_prefix}{view}"
            # Replace bare "<view>:" prefix in the seed text with qualified
            # form, leaving the rest of the description untouched.
            if text.startswith(f"{view}:"):
                text = f"{qualified}:" + text[len(view) + 1 :]
        lines.append(text)
    catalog = "\n\n".join(lines)
    # IDENTITY_DOCS has a critical warning that names federated_*_v.patient_id;
    # qualify the wildcard reference for consistency.
    identity = "\n\n".join(d["text"].strip() for d in IDENTITY_DOCS)
    if schema_prefix:
        identity = identity.replace(
            "federated_*_v.patient_id",
            f"{schema_prefix}federated_*_v.patient_id",
        )
    return f"{catalog}\n\n{identity}"


def _render_examples(schema_prefix: str) -> str:
    rendered = [doc["text"].replace("{p}", schema_prefix) for doc in RUN_SQL_EXAMPLES]
    body = "\n\n".join(rendered)
    return "EXAMPLE QUERIES — copy these idioms; do not invent new join shapes.\n\n" + body


def schema_context_for_sql(schema_prefix: str, question: Optional[str] = None) -> str:
    """Render the schema reference + few-shot block for run_sql's description.

    `schema_prefix` is "dw." in production / "" in SQLite tests. It is baked
    into every rendered view name AND every example SQL body.

    `question` is currently unused; reserved for the A2 swap to RAG retrieval
    (`rag.search(query=question, kind="schema"|"examples")`).
    """
    return "\n\n".join(
        [
            _header(schema_prefix),
            _render_catalog(schema_prefix),
            _render_examples(schema_prefix),
        ]
    )
