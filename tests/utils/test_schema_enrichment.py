"""
Tests for Schema Enrichment Module

Tests cover:
- ColumnStatistics dataclass and methods
- TableRelationship dataclass and methods
- SchemaGraph relationship tracking and path finding
- SemanticClassifier pattern matching
- SchemaEnricher integration (with mocked database)
"""

from unittest.mock import MagicMock, patch

import pytest

from utils.schema_enrichment import (
    ColumnStatistics,
    SchemaEnricher,
    SchemaGraph,
    SemanticClassifier,
    TableRelationship,
)


class TestColumnStatistics:
    """Tests for ColumnStatistics dataclass."""

    def test_basic_creation(self):
        """Test creating a basic ColumnStatistics instance."""
        stats = ColumnStatistics(
            table_name="users",
            column_name="id",
            data_type="integer",
        )
        assert stats.table_name == "users"
        assert stats.column_name == "id"
        assert stats.data_type == "integer"
        assert stats.nullable is True
        assert stats.semantic_type == "general"

    def test_null_ratio_calculation(self):
        """Test null ratio is calculated correctly."""
        stats = ColumnStatistics(
            table_name="test",
            column_name="col",
            data_type="text",
            total_count=100,
            null_count=25,
        )
        assert stats.null_ratio == 25.0

    def test_null_ratio_zero_total(self):
        """Test null ratio with zero total count."""
        stats = ColumnStatistics(
            table_name="test",
            column_name="col",
            data_type="text",
            total_count=0,
            null_count=0,
        )
        assert stats.null_ratio == 0.0

    def test_distinct_ratio_calculation(self):
        """Test distinct ratio is calculated correctly."""
        stats = ColumnStatistics(
            table_name="test",
            column_name="col",
            data_type="text",
            total_count=100,
            distinct_count=80,
        )
        assert stats.distinct_ratio == 80.0

    def test_is_likely_primary_key(self):
        """Test primary key detection heuristic."""
        # High distinctness, no nulls -> likely PK
        stats = ColumnStatistics(
            table_name="test",
            column_name="id",
            data_type="integer",
            total_count=100,
            distinct_count=100,
            null_count=0,
        )
        assert stats.is_likely_primary_key() is True

        # Has nulls -> not PK
        stats.null_count = 1
        assert stats.is_likely_primary_key() is False

    def test_is_likely_categorical(self):
        """Test categorical detection heuristic."""
        # Low distinct ratio -> categorical
        stats = ColumnStatistics(
            table_name="test",
            column_name="status",
            data_type="text",
            total_count=1000,
            distinct_count=5,
        )
        assert stats.is_likely_categorical() is True

        # High distinct ratio -> not categorical
        stats.distinct_count = 500
        assert stats.is_likely_categorical() is False

    def test_to_documentation_basic(self):
        """Test basic documentation generation."""
        stats = ColumnStatistics(
            table_name="users",
            column_name="email",
            data_type="varchar",
            semantic_type="email",
        )
        doc = stats.to_documentation()
        assert "users.email" in doc
        assert "varchar" in doc
        assert "email field" in doc

    def test_to_documentation_with_stats(self):
        """Test documentation includes statistics when available."""
        stats = ColumnStatistics(
            table_name="products",
            column_name="price",
            data_type="decimal",
            semantic_type="monetary",
            total_count=100,
            null_count=10,
            min_value=9.99,
            max_value=999.99,
        )
        doc = stats.to_documentation()
        assert "10.0% null" in doc
        assert "9.99 to 999.99" in doc

    def test_to_documentation_with_top_values(self):
        """Test documentation includes top values."""
        stats = ColumnStatistics(
            table_name="orders",
            column_name="status",
            data_type="text",
            top_values=[("pending", 50), ("shipped", 30), ("delivered", 20)],
        )
        doc = stats.to_documentation()
        assert "pending" in doc
        assert "shipped" in doc


class TestTableRelationship:
    """Tests for TableRelationship dataclass."""

    def test_basic_creation(self):
        """Test creating a TableRelationship instance."""
        rel = TableRelationship(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
        )
        assert rel.source_table == "orders"
        assert rel.target_table == "users"
        assert rel.relationship_type == "foreign_key"
        assert rel.confidence == 1.0

    def test_to_documentation_foreign_key(self):
        """Test FK relationship documentation."""
        rel = TableRelationship(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
            relationship_type="foreign_key",
        )
        doc = rel.to_documentation()
        assert "foreign key" in doc
        assert "orders.user_id references users.id" in doc
        assert "JOIN" in doc

    def test_to_documentation_implicit(self):
        """Test implicit relationship documentation."""
        rel = TableRelationship(
            source_table="orders",
            source_column="customer_id",
            target_table="customer",
            target_column="id",
            relationship_type="implicit",
            confidence=0.8,
        )
        doc = rel.to_documentation()
        assert "likely relates" in doc
        assert "80%" in doc


class TestSchemaGraph:
    """Tests for SchemaGraph relationship tracking."""

    def test_add_table(self):
        """Test adding tables to graph."""
        graph = SchemaGraph()
        graph.add_table("users")
        graph.add_table("orders")
        assert "users" in graph.tables
        assert "orders" in graph.tables

    def test_add_relationship(self):
        """Test adding relationships to graph."""
        graph = SchemaGraph()
        rel = TableRelationship(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
        )
        graph.add_relationship(rel)

        assert "orders" in graph.tables
        assert "users" in graph.tables
        assert len(graph.relationships) == 1

    def test_get_related_tables(self):
        """Test finding related tables."""
        graph = SchemaGraph()
        graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )
        graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="product_id",
                target_table="products",
                target_column="id",
            )
        )

        related = graph.get_related_tables("orders")
        assert "users" in related
        assert "products" in related

    def test_find_join_path_direct(self):
        """Test finding direct join path."""
        graph = SchemaGraph()
        rel = TableRelationship(
            source_table="orders",
            source_column="user_id",
            target_table="users",
            target_column="id",
        )
        graph.add_relationship(rel)

        path = graph.find_join_path("orders", "users")
        assert path is not None
        assert len(path) == 1
        assert path[0].source_table == "orders"

    def test_find_join_path_indirect(self):
        """Test finding indirect join path through intermediate table."""
        graph = SchemaGraph()
        graph.add_relationship(
            TableRelationship(
                source_table="order_items",
                source_column="order_id",
                target_table="orders",
                target_column="id",
            )
        )
        graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )

        path = graph.find_join_path("order_items", "users")
        assert path is not None
        assert len(path) == 2

    def test_find_join_path_no_path(self):
        """Test when no join path exists."""
        graph = SchemaGraph()
        graph.add_table("users")
        graph.add_table("products")  # No relationship

        path = graph.find_join_path("users", "products")
        assert path is None

    def test_get_table_centrality(self):
        """Test centrality calculation."""
        graph = SchemaGraph()
        # Users is connected to orders and profiles
        graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )
        graph.add_relationship(
            TableRelationship(
                source_table="profiles",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )

        centrality = graph.get_table_centrality()
        # users has 2 connections, orders and profiles have 1 each
        assert centrality["users"] == 2
        assert centrality["orders"] == 1
        assert centrality["profiles"] == 1

    def test_generate_join_documentation(self):
        """Test generating join documentation."""
        graph = SchemaGraph()
        graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )

        docs = graph.generate_join_documentation()
        assert len(docs) == 1
        assert "JOIN" in docs[0]


class TestSemanticClassifier:
    """Tests for SemanticClassifier."""

    def test_classify_by_name_identifier(self):
        """Test identifier pattern detection."""
        assert SemanticClassifier.classify_by_name("id") == "identifier"
        assert SemanticClassifier.classify_by_name("user_id") == "identifier"
        assert SemanticClassifier.classify_by_name("order_pk") == "identifier"

    def test_classify_by_name_temporal(self):
        """Test temporal pattern detection."""
        assert SemanticClassifier.classify_by_name("created_at") == "temporal"
        assert SemanticClassifier.classify_by_name("order_date") == "temporal"
        assert SemanticClassifier.classify_by_name("updated_timestamp") == "temporal"

    def test_classify_by_name_monetary(self):
        """Test monetary pattern detection."""
        assert SemanticClassifier.classify_by_name("price") == "monetary"
        assert SemanticClassifier.classify_by_name("total_cost") == "monetary"
        assert SemanticClassifier.classify_by_name("payment_amount") == "monetary"

    def test_classify_by_name_boolean(self):
        """Test boolean pattern detection."""
        assert SemanticClassifier.classify_by_name("is_active") == "boolean"
        assert SemanticClassifier.classify_by_name("has_paid") == "boolean"
        assert SemanticClassifier.classify_by_name("active_flag") == "boolean"

    def test_classify_by_name_general(self):
        """Test fallback to general type."""
        assert SemanticClassifier.classify_by_name("xyz_column") == "general"
        assert SemanticClassifier.classify_by_name("foo_bar") == "general"

    def test_classify_by_values_boolean_yn(self):
        """Test Y/N value pattern detection."""
        values = ["Y", "N", "Y", "Y", "N"]
        assert SemanticClassifier.classify_by_values(values) == "boolean"

    def test_classify_by_values_boolean_tf(self):
        """Test true/false value pattern detection."""
        values = ["true", "false", "true", "true", "false"]
        assert SemanticClassifier.classify_by_values(values) == "boolean"

    def test_classify_by_values_email(self):
        """Test email value pattern detection."""
        values = ["user@example.com", "test@test.org", "admin@company.net"]
        assert SemanticClassifier.classify_by_values(values) == "email"

    def test_classify_by_values_empty(self):
        """Test empty values return None."""
        assert SemanticClassifier.classify_by_values([]) is None
        assert SemanticClassifier.classify_by_values([None, None]) is None

    def test_classify_combined(self):
        """Test combined name and value classification."""
        # Value-based takes precedence
        assert SemanticClassifier.classify("some_column", ["Y", "N", "Y"]) == "boolean"
        # Falls back to name when no value pattern
        assert SemanticClassifier.classify("user_email", ["abc", "def"]) == "email"
        # Uses name when no values
        assert SemanticClassifier.classify("created_at") == "temporal"


class TestSchemaEnricher:
    """Tests for SchemaEnricher with mocked database."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=None)
        return conn, cursor

    def test_initialization(self, mock_connection):
        """Test SchemaEnricher initialization."""
        conn, _ = mock_connection
        enricher = SchemaEnricher(
            conn,
            forbidden_tables=["secret_table"],
            forbidden_columns=["password"],
        )
        assert "secret_table" in enricher.forbidden_tables
        assert "password" in enricher.forbidden_columns

    def test_get_allowed_tables(self, mock_connection):
        """Test getting allowed tables."""
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            ("users",),
            ("orders",),
            ("thrive_user",),
        ]

        enricher = SchemaEnricher(conn, forbidden_tables=["thrive_user"])
        tables = enricher.get_allowed_tables()

        assert "users" in tables
        assert "orders" in tables
        assert "thrive_user" not in tables

    def test_forbidden_table_excluded_from_stats(self, mock_connection):
        """Test that forbidden tables are excluded from statistics collection."""
        conn, _ = mock_connection
        enricher = SchemaEnricher(conn, forbidden_tables=["secret_data"])

        stats = enricher.collect_column_statistics("secret_data")
        assert stats == []

    def test_discover_explicit_relationships(self, mock_connection):
        """Test explicit FK discovery."""
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            ("orders", "user_id", "users", "id"),
            ("order_items", "order_id", "orders", "id"),
        ]

        enricher = SchemaEnricher(conn)
        relationships = enricher.discover_explicit_relationships()

        assert len(relationships) == 2
        assert relationships[0].source_table == "orders"
        assert relationships[0].target_table == "users"

    def test_discover_explicit_relationships_excludes_forbidden(self, mock_connection):
        """Test that forbidden tables are excluded from relationship discovery."""
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            ("orders", "user_id", "users", "id"),
            ("secret_orders", "user_id", "secret_users", "id"),
        ]

        enricher = SchemaEnricher(conn, forbidden_tables=["secret_orders", "secret_users"])
        relationships = enricher.discover_explicit_relationships()

        assert len(relationships) == 1
        assert relationships[0].source_table == "orders"

    @patch.object(SchemaEnricher, "get_allowed_tables")
    def test_discover_implicit_relationships(self, mock_get_tables, mock_connection):
        """Test implicit relationship discovery."""
        conn, cursor = mock_connection
        mock_get_tables.return_value = ["users", "orders", "products"]

        # First call: get columns with _id
        # Second call: check if target has 'id' column
        cursor.fetchall.side_effect = [
            [("orders", "user_id"), ("orders", "product_id")],
        ]
        cursor.fetchone.side_effect = [(1,), (1,)]  # Both targets have 'id' column

        enricher = SchemaEnricher(conn)
        enricher.discover_explicit_relationships = MagicMock(return_value=[])

        relationships = enricher.discover_implicit_relationships()

        # Should find user_id -> users.id and product_id -> products.id
        assert len(relationships) >= 1

    def test_generate_training_documentation(self, mock_connection):
        """Test training documentation generation."""
        conn, _ = mock_connection
        enricher = SchemaEnricher(conn)

        # Add some column stats
        enricher.column_stats["users"] = [
            ColumnStatistics(
                table_name="users",
                column_name="email",
                data_type="varchar",
                semantic_type="email",
            )
        ]

        # Add a relationship
        enricher.schema_graph.add_relationship(
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        )

        docs = enricher.generate_training_documentation()

        assert len(docs) >= 2  # At least column doc + relationship doc
        assert any("email" in doc for doc in docs)
        assert any("JOIN" in doc for doc in docs)

    @patch.object(SchemaEnricher, "collect_column_statistics")
    @patch.object(SchemaEnricher, "discover_explicit_relationships")
    @patch.object(SchemaEnricher, "discover_implicit_relationships")
    @patch.object(SchemaEnricher, "extract_index_information")
    @patch.object(SchemaEnricher, "extract_view_definitions")
    @patch.object(SchemaEnricher, "get_allowed_tables")
    def test_enrich_schema(
        self,
        mock_get_tables,
        mock_extract_views,
        mock_extract_indexes,
        mock_discover_implicit,
        mock_discover_explicit,
        mock_collect_stats,
        mock_connection,
    ):
        """Test full schema enrichment."""
        conn, _ = mock_connection

        mock_get_tables.return_value = ["users", "orders"]
        mock_collect_stats.return_value = [
            ColumnStatistics(table_name="test", column_name="col", data_type="text")
        ]
        mock_discover_explicit.return_value = [
            TableRelationship(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="id",
            )
        ]
        mock_discover_implicit.return_value = []
        mock_extract_indexes.return_value = []
        mock_extract_views.return_value = []

        enricher = SchemaEnricher(conn)
        results = enricher.enrich_schema()

        assert results["tables_processed"] == 2
        assert results["explicit_relationships"] == 1
        mock_collect_stats.assert_called()
        mock_discover_explicit.assert_called_once()
        mock_discover_implicit.assert_called_once()


class TestSemanticClassifierEdgeCases:
    """Edge case tests for SemanticClassifier."""

    def test_classify_mixed_case_column_names(self):
        """Test classification works with mixed case."""
        assert SemanticClassifier.classify_by_name("EMAIL") == "email"
        assert SemanticClassifier.classify_by_name("Created_At") == "temporal"
        assert SemanticClassifier.classify_by_name("IS_ACTIVE") == "boolean"

    def test_classify_values_with_nulls(self):
        """Test classification handles null values."""
        values = ["Y", None, "N", None, "Y"]
        assert SemanticClassifier.classify_by_values(values) == "boolean"

    def test_classify_values_insufficient_pattern_match(self):
        """Test returns None when pattern match is insufficient."""
        # Less than 50% match
        values = ["Y", "N", "maybe", "unknown", "other"]
        result = SemanticClassifier.classify_by_values(values)
        # Should not return boolean since less than 50% match
        assert result is None or result != "boolean"

    def test_classify_uuid_values(self):
        """Test UUID pattern detection."""
        values = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ]
        assert SemanticClassifier.classify_by_values(values) == "identifier"

    def test_classify_us_state_codes(self):
        """Test US state code detection."""
        values = ["NY", "CA", "TX", "FL", "WA"]
        assert SemanticClassifier.classify_by_values(values) == "location"
