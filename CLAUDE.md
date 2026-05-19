# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Detailed Documentation

For deeper context on specific systems, see:
- [Authentication & Users](llmdocs/auth.md) - Cookie sessions, roles, permissions, user preferences
- [Database Architecture](llmdocs/database.md) - SQLite, PostgreSQL, ChromaDB, Milvus
- [Streamlit UI Patterns](llmdocs/streamlit-ui.md) - Views, session state, message rendering
- [Vanna AI Integration](llmdocs/vanna-integration.md) - LLM backends, SQL generation, streaming
- [Magic Functions (Slash Commands)](llmdocs/magic-functions.md) - Complete command reference
- [Configuration Reference](llmdocs/configuration.md) - All secrets.toml options
- [Testing Patterns](llmdocs/testing.md) - Fixtures, mocking, test organization
- [LLM Guide](llmdocs/LLM_GUIDE.md) - Quick reference for LLM collaboration
- [Project Outline](llmdocs/PROJECT_OUTLINE.md) - High-level architecture overview
- [LLM Server & Local Models](llmdocs/llm-server.md) - aillm01 specs, A10G capacity, gemma4 / gpt-oss / qwen3.6 capability notes

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
port = 5432
database = "analytics"
user = "user"
password = "pass"
schema_name = "public"
object_type = "tables"  # or "views"

[sqlite]
database = "./pgDatabase/db.sqlite3"

[cookie]
password = "<random-strong-secret>"

[security]
allow_llm_to_see_data = false
restrict_rag_by_role = true
```

## Development Guidelines

### Adding New Settings
1. Add column to `orm/models.User`
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
- Ollama defaults to `http://localhost:11434` - ensure model is pulled and running
- Role-restricted RAG retrieval on by default; disable with `security.restrict_rag_by_role = false`
