# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Detailed Documentation

- [Database Architecture](llmdocs/database.md) — SQLite (app state), PostgreSQL (analytics), ChromaDB / Milvus (vector stores)
- [User Guide](docs/USER_GUIDE.md) — end-user feature walkthrough
- Agentic replatform spec: `docs/superpowers/specs/2026-05-06-agentic-replatform-design.md` (canonical design for the `agent/` package; tool caps, run-logging, event streaming)

## Sample Database

For local agent development without PHI, load the synthetic ~200-patient sample:

```bash
docker compose up -d
uv run python scripts/load_sample_db.py
```

See `data/sample/README.md` and `docs/superpowers/specs/2026-05-13-sample-database-design.md`
for details. Tests requiring the loaded DB are marked `@pytest.mark.sample_db`
and skipped by default — run with `uv run pytest -m sample_db`.

## Project Overview

Thrive AI is an intelligent data analysis platform built with Streamlit. It converts natural language to SQL using Vanna AI, runs queries against PostgreSQL, and provides interactive visualizations and AI-generated summaries.

## Common Commands

```bash
# Install dependencies
uv sync

# Start PostgreSQL (if using Docker)
docker compose up -d

# Run application
uv run streamlit run app.py

# Run tests
uv run pytest                       # All tests
uv run pytest -m unit               # Unit tests only
uv run pytest -m "not milvus"       # Skip Milvus tests
uv run pytest -m milvus             # Milvus tests only
uv run pytest -vs tests/path/to/test_file.py::test_name  # Single test

# Code quality
uv run ruff check                   # Lint
uv run ruff format                  # Format
```

## Architecture

### Application Flow

```
app.py (entry point, cookie/auth setup)
    → views/chat_bot.py (main chat UI)
        → utils/chat_bot_helper.py (renderers, message pipeline)
            → utils/vanna_calls.py (VannaService - SQL gen, run, summary)
                → utils/chromadb_vector.py or utils/milvus_vector.py (RAG)
```

### Key Modules

- **`utils/vanna_calls.py`**: Central `VannaService` class orchestrating AI backends. Backend selection based on `st.secrets['ai_keys']` and `rag_model` config. Backends: `MyVannaAnthropicChromaDB`, `MyVannaOllamaChromaDB`, `MyVannaGeminiChromaDB`, `MyVannaOllamaMilvus`, `MyVannaGeminiMilvus`, `VannaDefault`.

- **`utils/chat_bot_helper.py`**: Message rendering pipeline, follow-ups, chart generation, summary streaming/caching. Uses `MESSAGE_RENDERERS` dict for type dispatch. Critical session state keys: `my_question`, `messages`, `last_run_sql_error`, `pending_sql_error`, `streamed_summary`.

- **`utils/magic_functions.py`**: 20+ slash commands (`/describe`, `/distribution`, `/clusters`, `/pca`, etc.) for statistical analysis and ML.

- **`orm/models.py`**: SQLAlchemy models (`User`, `UserRole`, `Message`) using SQLite for app state. `RoleTypeEnum`: ADMIN=0, DOCTOR=1, NURSE=2, PATIENT=3.

- **`orm/functions.py`**: Auth helpers, user preference load/save, recent message retrieval.

- **`agent/`** (pydantic-ai): Multi-step agentic alternative to the Vanna single-shot SQL pipeline. `agent/runner.py` owns the `Agent` and tool registrations; nodes stream via `agent.iter()`. Hard caps from `[agent]` in secrets (`max_tool_calls`, `max_wall_clock_s`). Tools live in `agent/tools/` (`find_patient`, `search_patients_by_criteria`, `get_patient_clinical_data`, `list_patient_documents`, `search_codes`, `search_knowledge_base`, `run_sql`, `make_chart`, `summarize_results`). `agent/db/` is the read-only analytics adapter (separate from the Streamlit Postgres connection — driven by `[analytics_db]`). `agent/rag/` wraps Chroma for the agent's knowledge base; `agent/codes/` loads code-system reference data. `agent/run_logger.py` persists per-run events (subject to `[agent_logging].mode`).

### Data Flow

1. User question → guardrails check
2. `VannaService.generate_sql()` → SQL from RAG + LLM
3. `VannaService.run_sql()` → Execute against Postgres
4. DataFrame → chart/summary/follow-ups via `chat_bot_helper.py`
5. Each step persists as `orm.models.Message` to SQLite

### Vector Stores

- **ChromaDB**: Default. 8-D deterministic embeddings in dev/test with automatic dimension coercion.
- **Milvus Lite**: Dense + sparse (BM25) hybrid retrieval with RRF. File-backed. Changing `text_dim` requires collection recreation.
- **Ollama embeddings**: Optional via `ai_keys.ollama_embed_model` - dimension must match Milvus `text_dim`.

## Configuration

Configuration via `.streamlit/secrets.toml`. Key sections:

```toml
[ai_keys]
# One of: anthropic_api/model, ollama_model, gemini_api/model, vanna_api/model
anthropic_api = "<key>"
anthropic_model = "claude-3-5-sonnet-latest"

[rag_model]
chroma_path = "./chromadb"          # ChromaDB
# OR milvus_uri = "./milvus_demo.db"  # Milvus Lite

[postgres]
host = "localhost"
port = 5469              # docker-compose maps host 5469 → container 5432
database = "postgres"
user = "postgres"
password = "postgres"
schema_name = "public"
object_type = "tables"   # or "views"
# dialect = "postgresql" # "postgresql" (default) or "redshift" — controls the
                         # SQL dialect the LLM is prompted to emit. When set to
                         # "redshift" the system prompt includes a cheat-sheet
                         # (LISTAGG vs STRING_AGG, no CORR, etc.) and the
                         # hardcoded index-introspection query branches.

[analytics_db]
# Read-only warehouse the agent's clinical-data tools query against.
# Required when agentic mode is enabled. SQLAlchemy URL; for Redshift use
# redshift+psycopg2:// (plain postgresql+psycopg2:// runs a statement Redshift rejects).
# url = "redshift+psycopg2://user:pass@host:5439/db"
# dialect = "redshift"   # adapter taxonomy independent of URL prefix

[agent]
# max_tool_calls = 7         # hard cap on tool invocations per run
# max_wall_clock_s = 120.0   # raise for slow local Ollama models
# expose_query_details_to = ["admin"]   # roles allowed to see SQL/raw rows in tool cards
# ollama_think = true        # global thinking toggle; per-model override under [agent.ollama_think_per_model]

[sqlite]
database = "./pgDatabase/db.sqlite3"

[cookie]
password = "<random-strong-secret>"

[security]
allow_llm_to_see_data = false
restrict_rag_by_role = true

[agent_logging]
mode = "full"                    # full | scrubbed | disabled
max_logged_result_bytes = 5000000
max_logged_event_bytes = 5000000
retention_days = 0               # 0 = keep indefinitely
```

## Development Guidelines

### Schema Changes (Alembic)
The SQLite app DB is migrated by Alembic. `app.py` → `init_db()` runs `alembic upgrade head` on every startup and the result is cached for the process. To add a column:

1. Edit `orm/models.py`
2. `uv run alembic revision --autogenerate -m "..."`
3. Review `alembic/versions/<latest>.py` — SQLite uses `op.batch_alter_table(...)` for most ALTERs; verify it
4. `uv run alembic upgrade head` (or just restart the app)
5. Commit the new revision file

Useful: `uv run alembic current`, `uv run alembic history`, `uv run alembic downgrade -1`. Pre-Alembic raw-SQL scripts in `scripts/legacy_migrations/` are forensic only — do not run them.

### Adding New Settings
1. Add column to `orm/models.User` (then make an Alembic revision per above)
2. Update load/save in `orm/functions.*`
3. Expose in `views/chat_bot.py` sidebar

### Adding New Message Types
1. Extend `utils/enums.MessageType`
2. Implement renderer function
3. Register in `MESSAGE_RENDERERS` dict in `chat_bot_helper.py`
4. Ensure serialization in `orm.models.Message`

### Session State Keys to Preserve
`my_question`, `messages`, `last_run_sql_error`, `last_failed_sql`, `pending_sql_error`, `streamed_summary`, `_vn_instance`

## Gotchas

- Gemini backends require manual `self.chat_model` fix (already implemented); use non-streaming summary fallback
- `Message` persists DataFrame as JSON; pass `dataframe` consistently when creating summaries/charts
- SQL errors persist in session state for retry panel UX
- Missing `cookie.password` prevents login - app stops until cookies ready
- Multiple deployments on the same hostname (e.g. prod at `/` and a dev at `/dev`) MUST set distinct `cookie.prefix` values, or they'll read each other's session cookies and point at user_ids from the wrong DB
- Ollama defaults to `http://localhost:11434` - ensure model is pulled and running
- Role-restricted RAG retrieval on by default; disable with `security.restrict_rag_by_role = false`
- Agentic run logging defaults to `full` fidelity (verbatim PHI in the app SQLite DB); set `[agent_logging].mode = "scrubbed"` to hash SQL literals and drop full result rows, or `disabled` to turn it off. Protect DB backups accordingly.
