"""
Schema Enrichment Module for Automatic RAG Enhancement

This module provides utilities for extracting rich metadata from PostgreSQL databases
to enhance RAG-based SQL generation accuracy.

Key components:
- SchemaGraph: Relationship graph for table dependencies
- ColumnStatistics: Statistical analysis of column data
- SemanticClassifier: Enhanced semantic type classification
- SchemaEnricher: Main orchestrator for enrichment
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnStatistics:
    """Statistics for a single database column."""

    table_name: str
    column_name: str
    data_type: str
    nullable: bool = True
    distinct_count: int = 0
    null_count: int = 0
    total_count: int = 0
    min_value: Any = None
    max_value: Any = None
    avg_value: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    semantic_type: str = "general"
    sample_values: list[Any] = field(default_factory=list)

    @property
    def null_ratio(self) -> float:
        """Calculate null ratio as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.null_count / self.total_count) * 100

    @property
    def distinct_ratio(self) -> float:
        """Calculate distinct value ratio as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.distinct_count / self.total_count) * 100

    def is_likely_primary_key(self) -> bool:
        """Heuristic to detect likely primary key columns."""
        return self.distinct_ratio > 95 and self.null_count == 0

    def is_likely_categorical(self) -> bool:
        """Heuristic to detect likely categorical columns."""
        return self.distinct_ratio < 10 and self.distinct_count < 50

    def to_documentation(self) -> str:
        """Generate training documentation for this column."""
        parts = [f"Column {self.table_name}.{self.column_name} ({self.data_type})"]

        if self.semantic_type != "general":
            parts.append(f"is a {self.semantic_type} field")

        if self.is_likely_primary_key():
            parts.append("appears to be a primary key or unique identifier")
        elif self.is_likely_categorical():
            parts.append("is a categorical/enum-like field")

        if self.null_ratio > 0:
            parts.append(f"has {self.null_ratio:.1f}% null values")

        if self.top_values:
            top_vals_str = ", ".join([f"'{v}'" for v, _ in self.top_values[:5]])
            parts.append(f"common values include: {top_vals_str}")

        if self.min_value is not None and self.max_value is not None:
            parts.append(f"ranges from {self.min_value} to {self.max_value}")

        return ". ".join(parts) + "."


@dataclass
class TableRelationship:
    """Represents a relationship between two tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "foreign_key"  # foreign_key, implicit, inferred
    confidence: float = 1.0  # 1.0 for explicit FK, < 1.0 for inferred

    def to_documentation(self) -> str:
        """Generate training documentation for this relationship."""
        if self.relationship_type == "foreign_key":
            return (
                f"Table {self.source_table} has a foreign key relationship: "
                f"{self.source_table}.{self.source_column} references {self.target_table}.{self.target_column}. "
                f"Use JOIN {self.target_table} ON {self.source_table}.{self.source_column} = "
                f"{self.target_table}.{self.target_column} to combine data from these tables."
            )
        else:
            return (
                f"Table {self.source_table} likely relates to {self.target_table} via "
                f"{self.source_table}.{self.source_column} -> {self.target_table}.{self.target_column} "
                f"(inferred relationship, confidence: {self.confidence:.0%})."
            )


class SchemaGraph:
    """
    Graph representation of database schema relationships.

    Tracks tables as nodes and relationships (FK, implicit) as edges.
    Provides utilities for finding optimal JOIN paths.
    """

    def __init__(self):
        self.tables: set[str] = set()
        self.relationships: list[TableRelationship] = []
        self._adjacency: dict[str, list[tuple[str, TableRelationship]]] = {}

    def add_table(self, table_name: str):
        """Add a table to the graph."""
        self.tables.add(table_name)
        if table_name not in self._adjacency:
            self._adjacency[table_name] = []

    def add_relationship(self, relationship: TableRelationship):
        """Add a relationship to the graph."""
        self.add_table(relationship.source_table)
        self.add_table(relationship.target_table)
        self.relationships.append(relationship)

        # Add bidirectional edges
        self._adjacency[relationship.source_table].append((relationship.target_table, relationship))
        if relationship.target_table not in self._adjacency:
            self._adjacency[relationship.target_table] = []
        self._adjacency[relationship.target_table].append((relationship.source_table, relationship))

    def get_related_tables(self, table_name: str) -> list[str]:
        """Get tables directly related to the given table."""
        if table_name not in self._adjacency:
            return []
        return [target for target, _ in self._adjacency[table_name]]

    def get_relationships_for_table(self, table_name: str) -> list[TableRelationship]:
        """Get all relationships involving the given table."""
        return [r for r in self.relationships if r.source_table == table_name or r.target_table == table_name]

    def find_join_path(self, source: str, target: str, max_depth: int = 3) -> list[TableRelationship] | None:
        """
        Find a path of relationships to JOIN from source to target table.

        Uses BFS to find shortest path.
        """
        if source not in self.tables or target not in self.tables:
            return None

        if source == target:
            return []

        visited = {source}
        queue: list[tuple[str, list[TableRelationship]]] = [(source, [])]

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            for neighbor, relationship in self._adjacency.get(current, []):
                if neighbor == target:
                    return path + [relationship]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [relationship]))

        return None

    def get_table_centrality(self) -> dict[str, int]:
        """
        Calculate centrality (number of connections) for each table.

        Higher centrality indicates more interconnected tables.
        """
        centrality = {}
        for table in self.tables:
            centrality[table] = len(self._adjacency.get(table, []))
        return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True))

    def generate_join_documentation(self) -> list[str]:
        """Generate JOIN pattern documentation for all relationships."""
        docs = []
        for rel in self.relationships:
            docs.append(rel.to_documentation())
        return docs


class SemanticClassifier:
    """
    Enhanced semantic type classifier for database columns.

    Extends basic pattern matching with additional detection patterns
    for boolean, enum, and other common data patterns.
    """

    # Pattern definitions for semantic classification
    # Note: Order matters - more specific patterns should come first
    # Using a list of tuples to preserve order (dicts maintain insertion order in Python 3.7+)
    PATTERNS = {
        "boolean": [
            r"^is_",
            r"^has_",
            r"^can_",
            r"^should_",
            r"^was_",
            r"^will_",
            r"_flag$",
            r"_yn$",
            r"_ind$",
        ],
        "identifier": [
            r"^id$",
            r"_id$",
            r"^.*_pk$",
            r"^pk_",
            r"^key$",
            r"_key$",
        ],
        "name": [
            r"name",
            r"title",
            r"label",
            r"description",
        ],
        "email": [r"email", r"e_mail", r"mail_address"],
        "phone": [r"phone", r"telephone", r"mobile", r"cell", r"fax"],
        "temporal": [
            r"date",
            r"time",
            r"_at$",
            r"created",
            r"updated",
            r"modified",
            r"timestamp",
            r"^dt_",
            r"_dt$",
        ],
        "monetary": [
            r"price",
            r"cost",
            r"amount",
            r"fee",
            r"charge",
            r"payment",
            r"balance",
            r"salary",
            r"wage",
            r"revenue",
        ],
        "quantity": [
            r"count",
            r"quantity",
            r"number",
            r"total",
            r"num_",
            r"_cnt$",
            r"qty",
        ],
        "status": [
            r"status",
            r"state",
            r"^active$",
            r"^enabled$",
            r"^deleted$",
        ],
        "location": [
            r"address",
            r"city",
            r"state",
            r"zip",
            r"postal",
            r"country",
            r"region",
            r"latitude",
            r"longitude",
            r"lat$",
            r"lng$",
            r"lon$",
        ],
        "url": [r"url", r"link", r"href", r"website", r"uri"],
        "code": [
            r"code",
            r"_cd$",
            r"^cd_",
        ],
        "percentage": [r"percent", r"pct", r"rate", r"ratio"],
    }

    # Value patterns for data-driven classification
    VALUE_PATTERNS = {
        "boolean_yn": re.compile(r"^[YNyn]$"),
        "boolean_tf": re.compile(r"^(true|false|TRUE|FALSE|True|False|t|f|T|F)$"),
        "boolean_01": re.compile(r"^[01]$"),
        "gender": re.compile(r"^(M|F|male|female|MALE|FEMALE|Male|Female)$"),
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "phone_us": re.compile(r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"),
        "zip_us": re.compile(r"^\d{5}(-\d{4})?$"),
        "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}"),
        "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
        "state_us": re.compile(r"^(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$"),
    }

    @classmethod
    def classify_by_name(cls, column_name: str) -> str:
        """Classify semantic type based on column name patterns."""
        column_lower = column_name.lower()

        for semantic_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, column_lower, re.IGNORECASE):
                    return semantic_type

        return "general"

    @classmethod
    def classify_by_values(cls, sample_values: list[Any]) -> str | None:
        """
        Classify semantic type based on sample values.

        Returns None if no clear pattern detected.
        """
        if not sample_values:
            return None

        # Filter out None/NaN values
        valid_values = [str(v) for v in sample_values if v is not None and pd.notna(v)]
        if not valid_values:
            return None

        # Count pattern matches
        pattern_counts = {}
        for pattern_name, pattern in cls.VALUE_PATTERNS.items():
            matches = sum(1 for v in valid_values if pattern.match(v))
            if matches > 0:
                pattern_counts[pattern_name] = matches / len(valid_values)

        # Return pattern with highest match ratio (if > 50%)
        if pattern_counts:
            best_pattern = max(pattern_counts.items(), key=lambda x: x[1])
            if best_pattern[1] > 0.5:
                # Map pattern names to semantic types
                pattern_to_type = {
                    "boolean_yn": "boolean",
                    "boolean_tf": "boolean",
                    "boolean_01": "boolean",
                    "gender": "categorical",
                    "email": "email",
                    "phone_us": "phone",
                    "zip_us": "location",
                    "date_iso": "temporal",
                    "uuid": "identifier",
                    "state_us": "location",
                }
                return pattern_to_type.get(best_pattern[0])

        return None

    @classmethod
    def classify(cls, column_name: str, sample_values: list[Any] | None = None) -> str:
        """
        Classify semantic type using both name and value patterns.

        Value-based classification takes precedence when available.
        """
        # Try value-based classification first (more accurate)
        if sample_values:
            value_type = cls.classify_by_values(sample_values)
            if value_type:
                return value_type

        # Fall back to name-based classification
        return cls.classify_by_name(column_name)


class SchemaEnricher:
    """
    Main orchestrator for schema enrichment.

    Coordinates extraction of:
    - Column statistics
    - Table relationships
    - Semantic classifications
    - Index information
    - View definitions
    """

    def __init__(self, connection, forbidden_tables: list[str] | None = None, forbidden_columns: list[str] | None = None):
        """
        Initialize the schema enricher.

        Args:
            connection: psycopg2 connection to PostgreSQL database
            forbidden_tables: List of table names to exclude from enrichment
            forbidden_columns: List of column names to exclude from sampling
        """
        self.conn = connection
        self.forbidden_tables = set(forbidden_tables or [])
        self.forbidden_columns = set(forbidden_columns or [])
        self.schema_graph = SchemaGraph()
        self.column_stats: dict[str, list[ColumnStatistics]] = {}

    def get_allowed_tables(self, schema_name: str = "public") -> list[str]:
        """Get list of tables excluding forbidden ones."""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (schema_name,))
            tables = [row[0] for row in cur.fetchall()]

        return [t for t in tables if t not in self.forbidden_tables]

    def get_allowed_views(self, schema_name: str = "public") -> list[str]:
        """Get list of views excluding forbidden ones."""
        query = """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = %s
            ORDER BY table_name
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (schema_name,))
            views = [row[0] for row in cur.fetchall()]

        return [v for v in views if v not in self.forbidden_tables]

    def collect_column_statistics(
        self, table_name: str, schema_name: str = "public", sample_limit: int = 1000
    ) -> list[ColumnStatistics]:
        """
        Collect comprehensive statistics for all columns in a table.

        Args:
            table_name: Name of the table to analyze
            schema_name: Schema name (default: public)
            sample_limit: Maximum rows to sample for statistics

        Returns:
            List of ColumnStatistics for each column
        """
        if table_name in self.forbidden_tables:
            logger.warning(f"Skipping forbidden table: {table_name}")
            return []

        stats_list = []

        try:
            # Get column metadata
            col_query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """
            with self.conn.cursor() as cur:
                cur.execute(col_query, (schema_name, table_name))
                columns = cur.fetchall()

            # Get total row count
            from psycopg2 import sql as psycopg2_sql

            count_query = psycopg2_sql.SQL("SELECT COUNT(*) FROM {}").format(psycopg2_sql.Identifier(table_name))
            with self.conn.cursor() as cur:
                cur.execute(count_query)
                total_count = cur.fetchone()[0]

            for col_name, data_type, is_nullable in columns:
                # Skip forbidden columns
                if col_name in self.forbidden_columns:
                    continue

                stats = ColumnStatistics(
                    table_name=table_name,
                    column_name=col_name,
                    data_type=data_type,
                    nullable=is_nullable == "YES",
                    total_count=total_count,
                )

                # Collect detailed statistics
                try:
                    stats = self._collect_column_stats(table_name, col_name, data_type, stats, sample_limit)
                except Exception as e:
                    logger.warning(f"Error collecting stats for {table_name}.{col_name}: {e}")

                # Classify semantic type
                stats.semantic_type = SemanticClassifier.classify(col_name, stats.sample_values)

                stats_list.append(stats)

            self.column_stats[table_name] = stats_list

        except Exception as e:
            logger.error(f"Error collecting statistics for table {table_name}: {e}")

        return stats_list

    def _collect_column_stats(
        self, table_name: str, column_name: str, data_type: str, stats: ColumnStatistics, sample_limit: int
    ) -> ColumnStatistics:
        """Collect detailed statistics for a single column."""
        from psycopg2 import sql as psycopg2_sql

        col_ident = psycopg2_sql.Identifier(column_name)
        tbl_ident = psycopg2_sql.Identifier(table_name)

        # Get null count
        null_query = psycopg2_sql.SQL("SELECT COUNT(*) FROM {} WHERE {} IS NULL").format(tbl_ident, col_ident)
        with self.conn.cursor() as cur:
            cur.execute(null_query)
            stats.null_count = cur.fetchone()[0]

        # Get distinct count
        distinct_query = psycopg2_sql.SQL("SELECT COUNT(DISTINCT {}) FROM {}").format(col_ident, tbl_ident)
        with self.conn.cursor() as cur:
            cur.execute(distinct_query)
            stats.distinct_count = cur.fetchone()[0]

        # Get min/max/avg for numeric types
        numeric_types = [
            "integer",
            "bigint",
            "smallint",
            "decimal",
            "numeric",
            "real",
            "double precision",
            "float",
        ]
        if data_type.lower() in numeric_types:
            agg_query = psycopg2_sql.SQL("SELECT MIN({}), MAX({}), AVG({}) FROM {}").format(
                col_ident, col_ident, col_ident, tbl_ident
            )
            with self.conn.cursor() as cur:
                cur.execute(agg_query)
                row = cur.fetchone()
                stats.min_value = row[0]
                stats.max_value = row[1]
                stats.avg_value = float(row[2]) if row[2] is not None else None

        # Get top N values (for categorical columns)
        if stats.distinct_count < 100:  # Only for low-cardinality columns
            top_query = psycopg2_sql.SQL(
                "SELECT {}, COUNT(*) as cnt FROM {} WHERE {} IS NOT NULL "
                "GROUP BY {} ORDER BY cnt DESC LIMIT 10"
            ).format(col_ident, tbl_ident, col_ident, col_ident)
            with self.conn.cursor() as cur:
                cur.execute(top_query)
                stats.top_values = [(row[0], row[1]) for row in cur.fetchall()]

        # Get sample values for semantic classification
        sample_query = psycopg2_sql.SQL("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL LIMIT %s").format(
            col_ident, tbl_ident, col_ident
        )
        with self.conn.cursor() as cur:
            cur.execute(sample_query, (min(sample_limit, 100),))
            stats.sample_values = [row[0] for row in cur.fetchall()]

        return stats

    def discover_explicit_relationships(self, schema_name: str = "public") -> list[TableRelationship]:
        """
        Discover explicit foreign key relationships from database metadata.

        Returns:
            List of TableRelationship objects for all FK constraints
        """
        fk_query = """
            SELECT
                tc.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
        """

        relationships = []
        with self.conn.cursor() as cur:
            cur.execute(fk_query, (schema_name,))
            for row in cur.fetchall():
                source_table, source_column, target_table, target_column = row

                # Skip if involves forbidden tables
                if source_table in self.forbidden_tables or target_table in self.forbidden_tables:
                    continue

                rel = TableRelationship(
                    source_table=source_table,
                    source_column=source_column,
                    target_table=target_table,
                    target_column=target_column,
                    relationship_type="foreign_key",
                    confidence=1.0,
                )
                relationships.append(rel)
                self.schema_graph.add_relationship(rel)

        logger.info(f"Discovered {len(relationships)} explicit FK relationships")
        return relationships

    def discover_implicit_relationships(self, schema_name: str = "public") -> list[TableRelationship]:
        """
        Discover implicit relationships based on naming conventions.

        Looks for patterns like:
        - table_name.other_table_id -> other_table.id
        - table_name.other_table_name_id -> other_table_name.id

        Returns:
            List of TableRelationship objects for inferred relationships
        """
        relationships = []
        allowed_tables = set(self.get_allowed_tables(schema_name))

        # Get all columns ending with _id
        id_col_query = """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s
            AND column_name LIKE '%%_id'
            AND column_name != 'id'
        """

        with self.conn.cursor() as cur:
            cur.execute(id_col_query, (schema_name,))
            id_columns = cur.fetchall()

        for source_table, column_name in id_columns:
            if source_table in self.forbidden_tables:
                continue

            # Try to find referenced table
            # Pattern: {table_name}_id -> table_name
            potential_target = column_name.rsplit("_id", 1)[0]

            # Check variations
            targets_to_check = [
                potential_target,
                potential_target + "s",  # Plural
                potential_target.rstrip("s"),  # Singular
            ]

            for target_table in targets_to_check:
                if target_table in allowed_tables and target_table != source_table:
                    # Verify target table has an 'id' column
                    check_query = """
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s AND column_name = 'id'
                    """
                    with self.conn.cursor() as cur:
                        cur.execute(check_query, (schema_name, target_table))
                        if cur.fetchone():
                            # Check if this relationship already exists (from explicit FK)
                            exists = any(
                                r.source_table == source_table
                                and r.source_column == column_name
                                and r.target_table == target_table
                                for r in self.schema_graph.relationships
                            )

                            if not exists:
                                rel = TableRelationship(
                                    source_table=source_table,
                                    source_column=column_name,
                                    target_table=target_table,
                                    target_column="id",
                                    relationship_type="implicit",
                                    confidence=0.8,
                                )
                                relationships.append(rel)
                                self.schema_graph.add_relationship(rel)
                            break

        logger.info(f"Discovered {len(relationships)} implicit relationships")
        return relationships

    def extract_index_information(self, schema_name: str = "public") -> list[dict]:
        """
        Extract index information from the database.

        Returns:
            List of dictionaries with index metadata
        """
        index_query = """
            SELECT
                t.relname AS table_name,
                i.relname AS index_name,
                a.attname AS column_name,
                ix.indisunique AS is_unique,
                ix.indisprimary AS is_primary
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = %s
            AND t.relkind = 'r'
            ORDER BY t.relname, i.relname
        """

        indexes = []
        with self.conn.cursor() as cur:
            cur.execute(index_query, (schema_name,))
            for row in cur.fetchall():
                table_name, index_name, column_name, is_unique, is_primary = row

                if table_name in self.forbidden_tables:
                    continue

                indexes.append(
                    {
                        "table_name": table_name,
                        "index_name": index_name,
                        "column_name": column_name,
                        "is_unique": is_unique,
                        "is_primary": is_primary,
                    }
                )

        logger.info(f"Extracted {len(indexes)} index entries")
        return indexes

    def extract_view_definitions(self, schema_name: str = "public") -> list[dict]:
        """
        Extract view definitions from the database.

        Returns:
            List of dictionaries with view name and definition
        """
        view_query = """
            SELECT table_name, view_definition
            FROM information_schema.views
            WHERE table_schema = %s
        """

        views = []
        with self.conn.cursor() as cur:
            cur.execute(view_query, (schema_name,))
            for row in cur.fetchall():
                view_name, definition = row

                if view_name in self.forbidden_tables:
                    continue

                views.append({"view_name": view_name, "definition": definition})

        logger.info(f"Extracted {len(views)} view definitions")
        return views

    def generate_training_documentation(self) -> list[str]:
        """
        Generate comprehensive training documentation from enrichment results.

        Returns:
            List of documentation strings ready for RAG training
        """
        docs = []

        # Column statistics documentation
        for table_name, stats_list in self.column_stats.items():
            for stats in stats_list:
                docs.append(stats.to_documentation())

        # Relationship documentation
        docs.extend(self.schema_graph.generate_join_documentation())

        # Table centrality documentation
        centrality = self.schema_graph.get_table_centrality()
        if centrality:
            central_tables = [t for t, c in centrality.items() if c > 2]
            if central_tables:
                docs.append(
                    f"Central tables with many relationships: {', '.join(central_tables)}. "
                    "These tables are commonly used in JOIN operations."
                )

        return docs

    def enrich_schema(self, schema_name: str = "public", include_views: bool = True) -> dict:
        """
        Run full schema enrichment process.

        Args:
            schema_name: Schema to enrich
            include_views: Whether to include view definitions

        Returns:
            Dictionary with enrichment results and statistics
        """
        results = {
            "tables_processed": 0,
            "columns_analyzed": 0,
            "explicit_relationships": 0,
            "implicit_relationships": 0,
            "documentation_generated": 0,
        }

        # Get allowed tables
        tables = self.get_allowed_tables(schema_name)
        logger.info(f"Starting enrichment for {len(tables)} tables in schema '{schema_name}'")

        # Collect column statistics
        for table in tables:
            stats = self.collect_column_statistics(table, schema_name)
            results["columns_analyzed"] += len(stats)
            results["tables_processed"] += 1

        # Discover relationships
        explicit_rels = self.discover_explicit_relationships(schema_name)
        results["explicit_relationships"] = len(explicit_rels)

        implicit_rels = self.discover_implicit_relationships(schema_name)
        results["implicit_relationships"] = len(implicit_rels)

        # Extract indexes and views
        self.extract_index_information(schema_name)
        if include_views:
            self.extract_view_definitions(schema_name)

        # Generate documentation
        docs = self.generate_training_documentation()
        results["documentation_generated"] = len(docs)

        logger.info(f"Schema enrichment complete: {results}")
        return results
