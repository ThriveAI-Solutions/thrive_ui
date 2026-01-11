"""Semantic recognition service for magic commands.

This module provides LLM-based classification of natural language queries
to determine if they map to magic commands.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of semantic magic command classification."""

    command_pattern: str | None  # Regex pattern from MAGIC_RENDERERS
    command_name: str | None  # Human-readable name (e.g., "/distribution")
    extracted_params: dict = field(default_factory=dict)  # Extracted parameters (table, column, etc.)
    confidence: float = 0.0  # 0.0 to 1.0
    explanation: str = ""  # Brief explanation for UI
    alternatives: list = field(default_factory=list)  # Other possible commands with scores


class SemanticMagicService:
    """Service for semantic recognition of magic commands using user's LLM."""

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Cache for classification results (keyed by question hash)
    _cache: dict[str, ClassificationResult] = {}

    @classmethod
    def classify(cls, question: str, command_dict: dict) -> ClassificationResult:
        """
        Classify a natural language question as a magic command.

        Args:
            question: User's natural language input
            command_dict: MAGIC_RENDERERS dictionary

        Returns:
            ClassificationResult with command, params, and confidence
        """
        # Check cache first
        cache_key = cls._get_cache_key(question)
        if cache_key in cls._cache:
            logger.debug(f"Cache hit for question: {question[:50]}")
            return cls._cache[cache_key]

        # Build command catalog for LLM
        command_catalog = cls._build_command_catalog(command_dict)

        # Get available tables/columns for context
        available_objects = cls._get_available_objects()

        # Build LLM prompt
        system_prompt = cls._build_system_prompt(command_catalog, available_objects)
        user_prompt = cls._build_user_prompt(question)

        try:
            # Call LLM using user's configured provider
            result = cls._call_llm(system_prompt, user_prompt)

            # Parse LLM response
            classification = cls._parse_llm_response(result, command_dict)
        except Exception as e:
            logger.error(f"Semantic classification failed: {e}")
            classification = ClassificationResult(
                command_pattern=None,
                command_name=None,
                extracted_params={},
                confidence=0.0,
                explanation="Classification failed",
                alternatives=[],
            )

        # Cache result
        cls._cache[cache_key] = classification

        return classification

    @classmethod
    def _build_command_catalog(cls, command_dict: dict) -> list[dict]:
        """Build a catalog of commands for the LLM prompt."""
        catalog = []
        for pattern, meta in command_dict.items():
            # Skip help commands and internal commands
            func_name = getattr(meta.get("func"), "__name__", "")
            if func_name in ["_help", "_followup_help", "_clear", "_history_search"]:
                continue

            # Extract command name from pattern
            command_name = cls._extract_command_name(pattern)
            # Extract parameter names
            param_names = re.findall(r"\?P<(\w+)>", pattern)

            catalog.append(
                {
                    "pattern": pattern,
                    "name": command_name,
                    "description": meta.get("description", ""),
                    "category": meta.get("category", "Other"),
                    "parameters": param_names,
                    "sample_values": meta.get("sample_values", {}),
                }
            )
        return catalog

    @classmethod
    def _extract_command_name(cls, pattern: str) -> str:
        """Extract readable command name from regex pattern."""
        # e.g., r"^/distribution\s+..." -> "/distribution"
        match = re.search(r"\^/?(\w+)", pattern)
        if match:
            return f"/{match.group(1)}"
        return pattern[:20]

    @classmethod
    def _get_available_objects(cls) -> dict:
        """Get available tables and columns for context."""
        try:
            from utils.magic_functions import get_all_column_names, get_all_object_names

            tables_df = get_all_object_names()
            tables = tables_df["table_name"].tolist() if not tables_df.empty else []

            # Get columns for first few tables as examples
            columns_by_table = {}
            for table in tables[:5]:  # Limit to avoid long prompts
                try:
                    cols_df = get_all_column_names(table)
                    columns_by_table[table] = cols_df["column_name"].tolist()
                except Exception:
                    pass

            return {"tables": tables, "columns_by_table": columns_by_table}
        except Exception as e:
            logger.warning(f"Failed to get available objects: {e}")
            return {"tables": [], "columns_by_table": {}}

    @classmethod
    def _build_system_prompt(cls, catalog: list[dict], available_objects: dict) -> str:
        """Build the system prompt for LLM classification."""
        # Group by category for clarity
        by_category = {}
        for cmd in catalog:
            cat = cmd["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(cmd)

        catalog_text = []
        for category, commands in by_category.items():
            catalog_text.append(f"\n## {category}")
            for cmd in commands:
                params = ", ".join(cmd["parameters"]) if cmd["parameters"] else "none"
                catalog_text.append(f"- {cmd['name']}: {cmd['description']} (params: {params})")

        tables_text = ", ".join(available_objects.get("tables", [])[:20])

        # Sample columns for context
        columns_sample = []
        for table, cols in available_objects.get("columns_by_table", {}).items():
            columns_sample.append(f"{table}: {', '.join(cols[:10])}")
        columns_text = "; ".join(columns_sample) if columns_sample else "N/A"

        return f"""You are a magic command classifier for a data analysis chatbot.
Your task is to determine if a user's natural language question maps to one of the available magic commands.

AVAILABLE MAGIC COMMANDS:
{"".join(catalog_text)}

AVAILABLE TABLES: {tables_text}

SAMPLE COLUMNS: {columns_text}

RESPONSE FORMAT (JSON only, no markdown code blocks):
{{
  "command": "/command_name" or null,
  "table": "extracted_table_name" or null,
  "column": "extracted_column_name" or null,
  "column1": "first_column" or null,
  "column2": "second_column" or null,
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation",
  "alternatives": [{{"command": "/other", "confidence": 0.5}}]
}}

RULES:
1. Only suggest commands that match the user's intent
2. Extract table/column names from the user's query (use fuzzy matching mentally)
3. Set confidence based on how clearly the intent matches:
   - 0.85+ = very clear match ("show distribution of age" -> /distribution)
   - 0.50-0.84 = possible match but ambiguous ("analyze patients" -> /describe or /profile?)
   - below 0.50 = likely a data query, not a magic command
4. Include alternatives for ambiguous queries
5. Return null command if this looks like a data QUERY (e.g., "how many patients over 65?")
6. Keywords suggesting magic commands: describe, distribution, outliers, correlation, cluster, heatmap, wordcloud, boxplot, profile, missing, duplicates"""

    @classmethod
    def _build_user_prompt(cls, question: str) -> str:
        """Build the user prompt with the question."""
        return f'Classify this user input: "{question}"'

    @classmethod
    def _call_llm(cls, system_prompt: str, user_prompt: str) -> str:
        """Call the user's configured LLM."""
        try:
            from utils.vanna_calls import VannaService

            vanna_service = VannaService.from_streamlit_session()
            response = vanna_service.submit_prompt(system_prompt, user_prompt)
            return str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "{}"

    @classmethod
    def _parse_llm_response(cls, response: str, command_dict: dict) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        try:
            # Try to extract JSON from response
            # Handle markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                json_str = json_match.group(0) if json_match else "{}"

            data = json.loads(json_str)

            command_name = data.get("command")
            confidence = float(data.get("confidence", 0.0))

            # Find matching pattern in command_dict
            pattern = None
            if command_name:
                for p in command_dict.keys():
                    if cls._extract_command_name(p) == command_name:
                        pattern = p
                        break

            # Build extracted parameters
            params = {}
            for key in [
                "table",
                "column",
                "column1",
                "column2",
                "x",
                "y",
                "color",
                "num_rows",
                "percentage",
                "operation",
                "command",
            ]:
                if data.get(key):
                    params[key] = data[key]

            # Parse alternatives
            alternatives = []
            for alt in data.get("alternatives", []):
                if isinstance(alt, dict):
                    alternatives.append((alt.get("command", ""), alt.get("confidence", 0.0)))

            return ClassificationResult(
                command_pattern=pattern,
                command_name=command_name,
                extracted_params=params,
                confidence=confidence,
                explanation=data.get("explanation", ""),
                alternatives=alternatives,
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}, response: {response[:200]}")
            return ClassificationResult(
                command_pattern=None,
                command_name=None,
                extracted_params={},
                confidence=0.0,
                explanation="Failed to classify",
                alternatives=[],
            )

    @classmethod
    def _get_cache_key(cls, question: str) -> str:
        """Generate cache key from question."""
        return hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]

    @classmethod
    def clear_cache(cls):
        """Clear the classification cache."""
        cls._cache.clear()
