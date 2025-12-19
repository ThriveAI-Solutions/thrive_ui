# Issue #9: Automatic RAG Enhancement for SQL Generation Accuracy

**GitHub Issue:** https://github.com/ThriveAI-Solutions/thrive_ui/issues/9
**BD Issue:** thrive_ui-mgd

## Problem Summary

The current RAG-based SQL generation system relies on:
- Manually curated training data in `utils/config/training_data.json` (~174 Q-SQL pairs)
- Basic schema introspection via `training_plan()` and `train_ddl()` functions
- Static documentation entries for business rules

Limitations:
1. **Manual effort** - Adding new training examples requires developer intervention
2. **Coverage gaps** - New tables/columns may not have adequate context
3. **Stale data** - Training data does not automatically reflect schema changes
4. **Missing patterns** - Common query patterns users attempt are not captured automatically
5. **Limited semantic context** - Column purposes and relationships are not richly described

## Phase 1 Implementation Plan: Schema Enrichment (Full)

### 1.1 Enhanced Metadata Extraction

**Objective:** Automatically extract richer metadata from INFORMATION_SCHEMA and pg_catalog

**Implementation:**

#### 1.1.1 Column Statistics Collection
- **Location:** New function `collect_column_statistics()` in `utils/vanna_calls.py`
- **Features:**
  - Min/max/avg values for numeric columns
  - Distinct count and null ratios
  - Top N most common values for categorical columns (respecting forbidden columns)
  - Data distribution patterns (normal, skewed, etc.)

#### 1.1.2 Index and Usage Pattern Extraction
- **Location:** New function `extract_index_information()` in `utils/vanna_calls.py`
- **Features:**
  - Index definitions from `pg_indexes`
  - Index usage statistics from `pg_stat_user_indexes` (if available)
  - Primary key and unique constraint documentation

#### 1.1.3 View Definition Extraction
- **Location:** New function `extract_view_definitions()` in `utils/vanna_calls.py`
- **Features:**
  - Extract view SQL definitions from `information_schema.views`
  - Document view dependencies
  - Train RAG with view semantics

### 1.2 Relationship Discovery

**Objective:** Auto-detect relationships (explicit FK and implicit naming patterns)

**Implementation:**

#### 1.2.1 Explicit Foreign Key Discovery
- **Enhancement:** Improve `training_plan()` FK extraction
- **Features:**
  - Generate comprehensive relationship documentation
  - Create JOIN pattern suggestions for related tables
  - Document cardinality (one-to-many, many-to-many)

#### 1.2.2 Implicit Relationship Detection
- **Location:** New function `discover_implicit_relationships()` in `utils/vanna_calls.py`
- **Features:**
  - Detect `*_id` columns that reference other tables by naming convention
  - Match column names across tables (e.g., `user_id` in multiple tables)
  - Generate suggested relationship documentation

#### 1.2.3 Table Dependency Graph
- **Location:** New class `SchemaGraph` in `utils/schema_enrichment.py`
- **Features:**
  - Build directed graph of table relationships
  - Calculate table centrality (most connected tables)
  - Generate optimal JOIN path suggestions

### 1.3 Column Semantic Inference

**Objective:** Infer column semantics beyond simple pattern matching

**Implementation:**

#### 1.3.1 Enhanced Semantic Classification
- **Enhancement:** Extend semantic classification in `training_plan()` and `train_ddl()`
- **Additional patterns:**
  - Boolean detection: Y/N, 0/1, true/false patterns
  - Gender detection: M/F, male/female patterns
  - Country/state code detection
  - URL/path detection
  - JSON/array detection

#### 1.3.2 Sample Value Analysis
- **Location:** New function `analyze_sample_values()` in `utils/vanna_calls.py`
- **Features:**
  - Sample top N values per column
  - LLM-based semantic description generation
  - Pattern recognition (regex-based)

#### 1.3.3 Automatic Documentation Generation
- **Location:** New function `generate_column_documentation()` in `utils/vanna_calls.py`
- **Features:**
  - Generate natural language descriptions for columns
  - Create usage examples for common column types
  - Document value constraints and business rules

## Key Files to Modify/Create

### New Files
1. `utils/schema_enrichment.py` - Schema analysis and graph utilities
2. `tests/utils/test_schema_enrichment.py` - Unit tests

### Modified Files
1. `utils/vanna_calls.py`:
   - Add `collect_column_statistics()`
   - Add `extract_index_information()`
   - Add `extract_view_definitions()`
   - Add `discover_implicit_relationships()`
   - Add `generate_column_documentation()`
   - Enhance `training_plan()` with new capabilities
   - Enhance `train_ddl()` with additional metadata

2. `views/training.py` (if exists, or create):
   - Add UI for triggering enhanced training
   - Show training progress and statistics

## Implementation Steps

### Step 1: Create Schema Enrichment Module
- [ ] Create `utils/schema_enrichment.py` with core classes
- [ ] Implement `SchemaGraph` for relationship tracking
- [ ] Implement column statistics collection utilities
- [ ] Add comprehensive tests

### Step 2: Implement Column Statistics Collection
- [ ] Add `collect_column_statistics()` function
- [ ] Integrate with forbidden references filtering
- [ ] Add security measures for sensitive data
- [ ] Generate training documentation

### Step 3: Implement Enhanced Relationship Discovery
- [ ] Enhance FK discovery in existing functions
- [ ] Add implicit relationship detection
- [ ] Generate JOIN pattern documentation
- [ ] Build table dependency tracking

### Step 4: Implement Semantic Inference
- [ ] Extend semantic pattern matching
- [ ] Add sample value analysis
- [ ] Generate LLM-based column descriptions
- [ ] Create automatic documentation

### Step 5: Integration & UI
- [ ] Integrate all components into enhanced training workflow
- [ ] Add training page UI components (optional)
- [ ] Add progress tracking and logging
- [ ] Write integration tests

### Step 6: Testing & Validation
- [ ] Run full test suite
- [ ] Manual testing with database
- [ ] Validate SQL generation improvements
- [ ] Performance benchmarking

## Acceptance Criteria

1. **Column Statistics:**
   - [ ] Min/max/distinct counts collected for all allowed tables
   - [ ] Top N values captured for categorical columns
   - [ ] Null ratios documented

2. **Relationship Discovery:**
   - [ ] All explicit FK relationships documented
   - [ ] Implicit relationships (by naming) detected
   - [ ] JOIN patterns generated for related tables

3. **Semantic Inference:**
   - [ ] Boolean columns (Y/N) correctly identified
   - [ ] Temporal columns identified
   - [ ] LLM descriptions generated for key columns

4. **Integration:**
   - [ ] Enhanced training can be triggered
   - [ ] Respects forbidden tables/columns
   - [ ] No performance regression (< 30s for < 100 tables)

## Technical Notes

### Security Considerations
- All sample values must respect forbidden_references.json
- No sensitive data (passwords, PII) in training
- Column-level filtering applied

### Performance Targets
- Schema scan < 30 seconds for < 100 tables
- Memory-efficient processing (streaming where possible)
- Incremental updates supported

### Dependencies
- SQLGlot (optional) for SQL parsing
- psycopg2 for PostgreSQL access
- pandas for statistics

## References

- Current training infrastructure: `utils/vanna_calls.py:1494-2429`
- Vector stores: `utils/chromadb_vector.py`, `utils/milvus_vector.py`
- Forbidden references: `utils/config/forbidden_references.json`
