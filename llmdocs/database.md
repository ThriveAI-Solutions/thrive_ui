# Database Architecture

Thrive AI uses a multi-layered database architecture with four distinct database systems serving different purposes.

## Overview

| Database    | Purpose                      | Library    | Storage         |
| ----------- | ---------------------------- | ---------- | --------------- |
| SQLite      | App state (users, messages)  | SQLAlchemy | File-backed     |
| PostgreSQL  | Analytics data source        | psycopg2   | External server |
| ChromaDB    | Default vector store for RAG | chromadb   | File-backed     |
| Milvus Lite | Hybrid vector store for RAG  | pymilvus   | File-backed     |

## Key Files

| Component       | File                       |
| --------------- | -------------------------- |
| ORM Models      | `orm/models.py`            |
| ORM Functions   | `orm/functions.py`         |
| ChromaDB Vector | `utils/chromadb_vector.py` |
| Milvus Vector   | `utils/milvus_vector.py`   |
| VannaService    | `utils/vanna_calls.py`     |

---

## SQLite - Application State

### Configuration

```toml
[sqlite]
database = "./pgDatabase/db.sqlite3"
```

### Connection

```python
# orm/models.py
DATABASE_URL = f"sqlite:///{db_settings['database']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### Schema

**thrive_user_role**

```sql
CREATE TABLE thrive_user_role (
    id INTEGER PRIMARY KEY,
    role_name VARCHAR UNIQUE NOT NULL,
    description VARCHAR,
    role INTEGER  -- RoleTypeEnum value
);
```

**thrive_user**

```sql
CREATE TABLE thrive_user (
    id INTEGER PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    password VARCHAR NOT NULL,
    first_name VARCHAR,
    last_name VARCHAR,
    user_role_id INTEGER REFERENCES thrive_user_role(id),
    -- 15+ preference boolean columns
    show_sql BOOLEAN DEFAULT true,
    show_table BOOLEAN DEFAULT true,
    show_chart BOOLEAN DEFAULT true,
    show_summary BOOLEAN DEFAULT true,
    voice_input BOOLEAN DEFAULT false,
    speak_summary BOOLEAN DEFAULT false,
    show_question_history BOOLEAN DEFAULT false,
    show_suggested BOOLEAN DEFAULT true,
    show_followup BOOLEAN DEFAULT true,
    show_elapsed_time BOOLEAN DEFAULT true,
    show_plotly_code BOOLEAN DEFAULT false,
    llm_fallback BOOLEAN DEFAULT true,
    min_message_id INTEGER DEFAULT 0,
    theme VARCHAR DEFAULT 'WellTellAI',
    selected_llm_provider VARCHAR,
    selected_llm_model VARCHAR,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);
```

**thrive_message**

```sql
CREATE TABLE thrive_message (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES thrive_user(id) ON DELETE CASCADE,
    group_id VARCHAR,  -- UUID for message grouping
    role VARCHAR,      -- RoleType: 'assistant' | 'user'
    type VARCHAR,      -- MessageType enum value
    content TEXT,
    query TEXT,        -- SQL query if applicable
    question TEXT,     -- Original user question
    dataframe TEXT,    -- JSON-serialized DataFrame
    elapsed_time FLOAT,
    feedback INTEGER,  -- 1=thumbs up, -1=thumbs down, NULL=none
    feedback_comment TEXT,
    training_status VARCHAR,  -- pending/approved/rejected
    reviewed_by INTEGER,
    reviewed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX ix_message_user_id ON thrive_message(user_id);
CREATE INDEX ix_message_created_at ON thrive_message(created_at);
CREATE INDEX ix_message_type ON thrive_message(type);
CREATE INDEX ix_message_feedback ON thrive_message(feedback);
CREATE INDEX ix_message_training_status ON thrive_message(training_status);
```

### Session Management Pattern

```python
from orm.models import SessionLocal

def some_db_operation():
    session = SessionLocal()
    try:
        # ... operations
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

### Eager Loading

```python
# Avoid N+1 queries for user roles
session.query(User).options(joinedload(User.role)).filter(...)
```

### Migration Pattern

Schema changes are managed by **Alembic**. The app calls `init_db()` at
startup, which runs `alembic upgrade head` and then seeds default users.
Adding a column is `edit orm/models.py` → `alembic revision --autogenerate`
→ commit. See README §Database Migrations for the full workflow.

Pre-Alembic raw-SQLite scripts live in `scripts/legacy_migrations/`. Their
effects are baked into the baseline revision and they are not run on new
databases.

---

## PostgreSQL - Analytics Data Source

### Configuration

```toml
[postgres]
host = "localhost"
port = 5432
database = "analytics"
user = "user"
password = "pass"
schema_name = "public"       # Configurable schema
object_type = "tables"       # "tables" or "views"
dialect = "postgres"         # or "redshift"
```

### Connection

Established once per VannaService instance via `connect_to_postgres()`:

```python
# utils/vanna_calls.py:653-659
self.vn.connect_to_postgres(
    host=postgres_config["host"],
    dbname=postgres_config["database"],
    user=postgres_config["user"],
    password=postgres_config["password"],
    port=postgres_config["port"]
)
```

### Query Execution

```python
# VannaService.run_sql()
df = self.vn.run_sql(sql=sql)
```

**Safety Features**:

- LIMIT clause enforcement via `config_helper.ensure_query_has_limit()`
- Forbidden table/column checking via `check_references()`
- Error persistence for retry UI

### Schema Discovery

Vanna introspects PostgreSQL for DDL training:

- Tables or views (based on `object_type` config)
- Specific schema (based on `schema_name` config)

---

## ChromaDB - Default Vector Store

### Configuration

```toml
[rag_model]
chroma_path = "./chromadb"
```

### Implementation (`utils/chromadb_vector.py`)

**Collections** (three separate collections):

| Collection                 | Purpose            | Content                                   |
| -------------------------- | ------------------ | ----------------------------------------- |
| `sql_collection`           | Question-SQL pairs | JSON: `{"question": "...", "sql": "..."}` |
| `ddl_collection`           | Schema definitions | DDL statements                            |
| `documentation_collection` | Table docs         | Custom documentation strings              |

### Embeddings

**Development/Test**: 8-dimensional deterministic embeddings

- Auto-coercion via `_CoercingCollection` wrapper handles dimension mismatches

**Production with Ollama**:

```toml
[ai_keys]
ollama_embed_model = "nomic-embed-text"
```

### Role-Based Filtering

```python
# ThriveAI_ChromaDB._prepare_retrieval_metadata()
def _prepare_retrieval_metadata(self, metadata=None):
    if metadata is None:
        metadata = {}
    metadata["user_role"] = {"$gte": self._get_effective_role()}
    return metadata
```

### Retrieval Configuration

```python
n_results_sql = 10      # Number of similar SQL examples to retrieve
n_results_ddl = 10      # Number of related DDL statements
n_results_documentation = 10  # Number of related docs
```

---

## Milvus Lite - Hybrid Vector Store

### Configuration

```toml
[rag_model.milvus]
uri = "./milvus_demo.db"        # File path for Milvus Lite
text_dim = 768                   # Embedding dimension (MUST match model)
collection_prefix = "thrive"     # Collection name prefix
```

**Warning**: Changing `text_dim` requires collection recreation.

### Implementation (`utils/milvus_vector.py`)

**Collections**:

- `{prefix}_sql` - Question-SQL pairs
- `{prefix}_ddl` - DDL statements
- `{prefix}_docs` - Documentation

### Schema Per Collection

```python
schema = CollectionSchema([
    FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True),
    FieldSchema("text_dense", DataType.FLOAT_VECTOR, dim=text_dim),
    FieldSchema("text_sparse", DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema("user_role", DataType.INT64)
])
```

### Hybrid Retrieval Strategy

Milvus uses **dense + sparse** retrieval with **Reciprocal Rank Fusion (RRF)**:

```python
def _hybrid_docs(self, collection, query_text, n_results, role_filter):
    # 1. Dense search (semantic similarity)
    dense_results = collection.search(
        data=[dense_embedding],
        anns_field="text_dense",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        filter=f"user_role >= {role_filter}",
        limit=n_results * 2
    )

    # 2. Sparse search (BM25 keyword matching)
    sparse_results = collection.search(
        data=[sparse_embedding],
        anns_field="text_sparse",
        param={"metric_type": "IP"},
        filter=f"user_role >= {role_filter}",
        limit=n_results * 2
    )

    # 3. Combine with RRF (k=60)
    return reciprocal_rank_fusion(dense_results, sparse_results, k=60)[:n_results]
```

### Embeddings

**Dense**: Deterministic HashingVectorizer (768-D) for development

```python
# Falls back to hashing if Ollama unavailable
vectorizer = HashingVectorizer(n_features=text_dim, norm='l2')
```

**Sparse**: BM25 vectors auto-generated from text field by Milvus

**Ollama Support**: Uses Ollama embeddings when `ollama_embed_model` is configured

### Advantages Over ChromaDB

- Hybrid dense + sparse gives both keyword AND semantic search
- RRF ranking is robust to single search type failure
- File-backed (no server required)
- Supports role-based metadata filtering at retrieval time

---

## Vector Store Selection Logic

In `VannaService._setup_vanna()` (`utils/vanna_calls.py:602-643`):

```python
# Priority order:
if milvus_config:  # Milvus preferred if configured
    if ollama_configured:
        self.vn = MyVannaOllamaMilvus(...)
    elif gemini_configured:
        self.vn = MyVannaGeminiMilvus(...)
    elif openai_configured:
        self.vn = MyVannaOpenAIMilvus(...)
elif chroma_config:  # ChromaDB fallback
    if ollama_configured:
        self.vn = MyVannaOllamaChromaDB(...)
    elif anthropic_configured:
        self.vn = MyVannaAnthropicChromaDB(...)
    elif gemini_configured:
        self.vn = MyVannaGeminiChromaDB(...)
else:  # Remote VannaDB (cloud)
    self.vn = VannaDefault(...)
```

---

## Data Flow Through Databases

```
1. User submits question
    ↓
2. RAG Retrieval (ChromaDB or Milvus)
    ├─ Query sql_collection for similar questions
    ├─ Query ddl_collection for relevant schema
    └─ Query documentation_collection for context
    ↓
3. LLM generates SQL using RAG context
    ↓
4. PostgreSQL executes generated SQL
    ↓
5. Results stored in SQLite
    ├─ Message with SQL (MessageType.SQL)
    ├─ Message with DataFrame (MessageType.DATAFRAME)
    ├─ Message with Summary (MessageType.SUMMARY)
    └─ Message with Chart (MessageType.PLOTLY_CHART)
```

---

## Gotchas

1. **Milvus `text_dim`**: Must match embedding model dimension. Changing requires collection recreation.

2. **ChromaDB Dimension Coercion**: Uses `_CoercingCollection` wrapper to handle dimension mismatches automatically.

3. **DataFrame Serialization**: DataFrames are stored as JSON strings in SQLite `thrive_message.dataframe` column.

4. **Cascade Delete**: Deleting a User cascades to all their Messages.

5. **Schema Selection**: PostgreSQL queries use `schema_name` from config, not hardcoded "public".

6. **Role Filtering**: Both vector stores filter by `user_role >= X` at query time when `restrict_rag_by_role` is enabled.

7. **Ollama Embeddings**: When configured, embedding dimension MUST match Milvus `text_dim` setting.

## Sample Database (Synthetic)

The repo ships a committed synthetic dataset that mirrors the production
Redshift `dw` schema (the 17 V1-whitelist views) with ~200 patients (Synthea
seed pin generates ~237), no PHI.

- Source artifact: `data/sample/thrive_sample.sql.zst` (5.4 MB compressed)
- Loader: `scripts/load_sample_db.py` (Postgres default, SQLite optional)
- ETL: `scripts/sample_db/etl.py` (transforms Synthea CSVs)
- Regeneration: requires Java 11+, run `./scripts/synthea/generate.sh` then
  the ETL. Rarely needed — only when schema or noise model changes.

The sample DB deliberately mirrors the warehouse's code-system polyglot
(mixed `code_type` spellings, ICD-10/SNOMED dual coding for problems,
~22% empty `code_type` in labs, ~46% empty in orders). See the design doc
at `docs/superpowers/specs/2026-05-13-sample-database-design.md` for the
full noise model.
