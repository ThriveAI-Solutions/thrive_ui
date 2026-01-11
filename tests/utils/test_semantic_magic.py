"""Tests for semantic magic command recognition."""

import pytest
from unittest.mock import MagicMock, patch

from utils.semantic_magic_service import SemanticMagicService, ClassificationResult


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_default_values(self):
        """Test that ClassificationResult has sensible defaults."""
        result = ClassificationResult(
            command_pattern=None,
            command_name=None,
        )
        assert result.extracted_params == {}
        assert result.confidence == 0.0
        assert result.explanation == ""
        assert result.alternatives == []

    def test_full_initialization(self):
        """Test ClassificationResult with all values."""
        result = ClassificationResult(
            command_pattern=r"^/distribution\s+(?P<table>\w+)\.(?P<column>\w+)$",
            command_name="/distribution",
            extracted_params={"table": "patients", "column": "age"},
            confidence=0.92,
            explanation="User wants distribution analysis",
            alternatives=[("/describe", 0.5)],
        )
        assert result.command_name == "/distribution"
        assert result.confidence == 0.92
        assert result.extracted_params["table"] == "patients"


class TestSemanticMagicService:
    """Unit tests for SemanticMagicService."""

    @pytest.fixture
    def mock_command_dict(self):
        """Minimal command dict for testing."""
        return {
            r"^/distribution\s+(?P<table>\w+)\.(?P<column>\w+)$": {
                "func": MagicMock(__name__="_distribution_analysis"),
                "description": "Analyze distribution of a specific column with statistical tests",
                "sample_values": {"table": "patients", "column": "age"},
                "category": "Statistical Analysis",
            },
            r"^/describe\s+(?P<table>\w+)$": {
                "func": MagicMock(__name__="_describe_table"),
                "description": "Generate comprehensive descriptive statistics for a table",
                "sample_values": {"table": "patients"},
                "category": "Data Exploration & Basic Info",
            },
            r"^/heatmap\s+(?P<table>\w+)$": {
                "func": MagicMock(__name__="_generate_heatmap"),
                "description": "Generate a correlation heatmap visualization for a table",
                "sample_values": {"table": "patients"},
                "category": "Visualizations",
            },
            r"^/help$": {
                "func": MagicMock(__name__="_help"),
                "description": "Show available magic commands",
                "sample_values": {},
                "category": "Help & System Commands",
            },
        }

    def test_extract_command_name_from_pattern(self):
        """Test command name extraction from regex patterns."""
        assert SemanticMagicService._extract_command_name(r"^/distribution\s+(?P<table>\w+)$") == "/distribution"
        assert SemanticMagicService._extract_command_name(r"^/help$") == "/help"
        assert (
            SemanticMagicService._extract_command_name(
                r"^/correlation\s+(?P<table>\w+)\.(?P<column1>\w+)\.(?P<column2>\w+)$"
            )
            == "/correlation"
        )

    def test_build_command_catalog_excludes_help(self, mock_command_dict):
        """Test that _help command is excluded from catalog."""
        catalog = SemanticMagicService._build_command_catalog(mock_command_dict)
        command_names = [cmd["name"] for cmd in catalog]
        assert "/help" not in command_names
        assert "/distribution" in command_names
        assert "/describe" in command_names

    def test_build_command_catalog_extracts_params(self, mock_command_dict):
        """Test that parameters are correctly extracted from patterns."""
        catalog = SemanticMagicService._build_command_catalog(mock_command_dict)
        dist_cmd = next(cmd for cmd in catalog if cmd["name"] == "/distribution")
        assert "table" in dist_cmd["parameters"]
        assert "column" in dist_cmd["parameters"]

    def test_get_cache_key_is_case_insensitive(self):
        """Test that cache keys are case-insensitive."""
        key1 = SemanticMagicService._get_cache_key("Show distribution of AGE")
        key2 = SemanticMagicService._get_cache_key("show distribution of age")
        assert key1 == key2

    def test_get_cache_key_strips_whitespace(self):
        """Test that cache keys ignore leading/trailing whitespace."""
        key1 = SemanticMagicService._get_cache_key("  show distribution  ")
        key2 = SemanticMagicService._get_cache_key("show distribution")
        assert key1 == key2

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_classify_high_confidence_distribution(self, mock_objects, mock_llm, mock_command_dict):
        """Test classification of clear distribution request."""
        mock_objects.return_value = {"tables": ["patients"], "columns_by_table": {"patients": ["age", "name"]}}
        mock_llm.return_value = """
        {
            "command": "/distribution",
            "table": "patients",
            "column": "age",
            "confidence": 0.92,
            "explanation": "User wants distribution analysis of age",
            "alternatives": []
        }
        """

        # Clear cache before test
        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "show me the distribution of age in patients",
            mock_command_dict,
        )

        assert result.command_name == "/distribution"
        assert result.confidence >= 0.85
        assert result.extracted_params.get("table") == "patients"
        assert result.extracted_params.get("column") == "age"

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_classify_low_confidence_returns_alternatives(self, mock_objects, mock_llm, mock_command_dict):
        """Test that ambiguous queries return alternatives."""
        mock_objects.return_value = {"tables": ["patients"], "columns_by_table": {}}
        mock_llm.return_value = """
        {
            "command": "/describe",
            "table": "patients",
            "confidence": 0.65,
            "explanation": "Might want describe or profile",
            "alternatives": [{"command": "/profile", "confidence": 0.55}]
        }
        """

        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "analyze the patients table",
            mock_command_dict,
        )

        assert result.confidence < 0.85
        assert result.confidence >= 0.50
        assert len(result.alternatives) > 0

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_classify_no_match_returns_low_confidence(self, mock_objects, mock_llm, mock_command_dict):
        """Test that data queries don't match magic commands."""
        mock_objects.return_value = {"tables": ["patients"], "columns_by_table": {}}
        mock_llm.return_value = """
        {
            "command": null,
            "confidence": 0.1,
            "explanation": "This is a data query, not a command",
            "alternatives": []
        }
        """

        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "how many patients are over 65?",
            mock_command_dict,
        )

        assert result.command_name is None
        assert result.confidence < 0.50

    def test_cache_returns_same_result(self, mock_command_dict):
        """Test that repeated classifications use cache."""
        # Pre-populate cache
        question = "show distribution of age"
        cache_key = SemanticMagicService._get_cache_key(question)
        cached_result = ClassificationResult(
            command_pattern=r"^/distribution\s+(?P<table>\w+)\.(?P<column>\w+)$",
            command_name="/distribution",
            extracted_params={"table": "test", "column": "age"},
            confidence=0.9,
            explanation="cached",
            alternatives=[],
        )
        SemanticMagicService._cache[cache_key] = cached_result

        # Should return cached without calling LLM
        with patch.object(SemanticMagicService, "_call_llm") as mock_llm:
            result = SemanticMagicService.classify(question, mock_command_dict)
            mock_llm.assert_not_called()
            assert result.explanation == "cached"

        # Cleanup
        SemanticMagicService.clear_cache()

    def test_clear_cache(self, mock_command_dict):
        """Test that clear_cache removes all cached results."""
        # Add something to cache
        question = "test question"
        cache_key = SemanticMagicService._get_cache_key(question)
        SemanticMagicService._cache[cache_key] = ClassificationResult(
            command_pattern=None,
            command_name=None,
        )

        assert len(SemanticMagicService._cache) > 0

        SemanticMagicService.clear_cache()

        assert len(SemanticMagicService._cache) == 0

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_parse_response_handles_markdown_code_blocks(self, mock_objects, mock_llm, mock_command_dict):
        """Test that JSON inside markdown code blocks is parsed correctly."""
        mock_objects.return_value = {"tables": ["patients"], "columns_by_table": {}}
        mock_llm.return_value = """
        Here is the classification:
        ```json
        {
            "command": "/heatmap",
            "table": "patients",
            "confidence": 0.88,
            "explanation": "User wants a heatmap",
            "alternatives": []
        }
        ```
        """

        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "create a heatmap for patients",
            mock_command_dict,
        )

        assert result.command_name == "/heatmap"
        assert result.confidence == 0.88

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_parse_response_handles_malformed_json(self, mock_objects, mock_llm, mock_command_dict):
        """Test graceful handling of malformed LLM response."""
        mock_objects.return_value = {"tables": [], "columns_by_table": {}}
        mock_llm.return_value = "This is not JSON at all"

        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "some question",
            mock_command_dict,
        )

        # Should return a valid result with low confidence
        assert result.confidence == 0.0
        assert result.command_name is None

    @patch.object(SemanticMagicService, "_call_llm")
    @patch.object(SemanticMagicService, "_get_available_objects")
    def test_llm_call_failure_returns_empty_result(self, mock_objects, mock_llm, mock_command_dict):
        """Test that LLM call failure is handled gracefully."""
        mock_objects.return_value = {"tables": [], "columns_by_table": {}}
        mock_llm.side_effect = Exception("LLM unavailable")

        SemanticMagicService.clear_cache()

        result = SemanticMagicService.classify(
            "describe patients",
            mock_command_dict,
        )

        # Should return empty result, not crash
        assert result.confidence == 0.0


class TestBuildCommandPreview:
    """Tests for _build_command_preview function."""

    def test_table_and_column(self):
        """Test preview with table and column."""
        from utils.magic_functions import _build_command_preview

        result = ClassificationResult(
            command_pattern=r"^/distribution\s+(?P<table>\w+)\.(?P<column>\w+)$",
            command_name="/distribution",
            extracted_params={"table": "patients", "column": "age"},
            confidence=0.9,
        )

        preview = _build_command_preview(result)
        assert preview == "/distribution patients.age"

    def test_table_only(self):
        """Test preview with table only."""
        from utils.magic_functions import _build_command_preview

        result = ClassificationResult(
            command_pattern=r"^/describe\s+(?P<table>\w+)$",
            command_name="/describe",
            extracted_params={"table": "patients"},
            confidence=0.9,
        )

        preview = _build_command_preview(result)
        assert preview == "/describe patients"

    def test_correlation_with_two_columns(self):
        """Test preview with table and two columns."""
        from utils.magic_functions import _build_command_preview

        result = ClassificationResult(
            command_pattern=r"^/correlation\s+(?P<table>\w+)\.(?P<column1>\w+)\.(?P<column2>\w+)$",
            command_name="/correlation",
            extracted_params={"table": "patients", "column1": "age", "column2": "weight"},
            confidence=0.9,
        )

        preview = _build_command_preview(result)
        assert preview == "/correlation patients.age.weight"

    def test_no_params(self):
        """Test preview with no parameters."""
        from utils.magic_functions import _build_command_preview

        result = ClassificationResult(
            command_pattern=r"^/help$",
            command_name="/help",
            extracted_params={},
            confidence=0.9,
        )

        preview = _build_command_preview(result)
        assert preview == "/help"

    def test_none_command(self):
        """Test preview with None command."""
        from utils.magic_functions import _build_command_preview

        result = ClassificationResult(
            command_pattern=None,
            command_name=None,
            confidence=0.0,
        )

        preview = _build_command_preview(result)
        assert preview == ""


class TestValidateAndFuzzyMatchParams:
    """Tests for _validate_and_fuzzy_match_params function."""

    @patch("utils.magic_functions.find_closest_object_name")
    def test_table_fuzzy_matching(self, mock_find_object):
        """Test that table names are fuzzy matched."""
        from utils.magic_functions import _validate_and_fuzzy_match_params

        mock_find_object.return_value = "public.patients"

        params = {"table": "patient"}  # Slightly misspelled
        validated = _validate_and_fuzzy_match_params(params)

        assert validated["table"] == "patients"
        mock_find_object.assert_called_once_with("patient")

    @patch("utils.magic_functions.find_closest_column_name")
    @patch("utils.magic_functions.find_closest_object_name")
    def test_column_fuzzy_matching(self, mock_find_object, mock_find_column):
        """Test that column names are fuzzy matched."""
        from utils.magic_functions import _validate_and_fuzzy_match_params

        mock_find_object.return_value = "public.patients"
        mock_find_column.return_value = "age"

        params = {"table": "patients", "column": "ag"}  # Slightly misspelled
        validated = _validate_and_fuzzy_match_params(params)

        assert validated["column"] == "age"
        mock_find_column.assert_called_once_with("patients", "ag")

    @patch("utils.magic_functions.find_closest_object_name")
    def test_fuzzy_match_failure_returns_original(self, mock_find_object):
        """Test that failed fuzzy matching returns original value."""
        from utils.magic_functions import _validate_and_fuzzy_match_params

        mock_find_object.side_effect = Exception("No match found")

        params = {"table": "nonexistent"}
        validated = _validate_and_fuzzy_match_params(params)

        assert validated["table"] == "nonexistent"
