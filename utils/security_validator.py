"""
Security validation service for RAG training data.

Prevents forbidden table and column references from being added to the RAG model.
Loads forbidden references from the existing JSON config file.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates training data against forbidden references."""

    def __init__(self):
        self.forbidden_tables: set[str] = set()
        self.forbidden_columns: set[str] = set()
        self._load_forbidden_references()

    def _load_forbidden_references(self) -> None:
        """Load forbidden references from the JSON config file."""
        try:
            forbidden_file_path = Path(__file__).parent / "config/forbidden_references.json"
            with forbidden_file_path.open("r") as file:
                forbidden_data = json.load(file)

            self.forbidden_tables = {t.lower() for t in forbidden_data.get("tables", [])}
            self.forbidden_columns = {c.lower() for c in forbidden_data.get("columns", [])}

            logger.info(
                "Loaded %d forbidden tables and %d forbidden columns",
                len(self.forbidden_tables),
                len(self.forbidden_columns),
            )
        except Exception as e:
            logger.error("Error loading forbidden references: %s", e)
            self.forbidden_tables = set()
            self.forbidden_columns = set()

    def validate_sql_content(self, content: str) -> tuple[bool, list[str]]:
        """Validate SQL content for forbidden references.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        content_lower = content.lower()

        for table in self.forbidden_tables:
            patterns = [
                rf"\bfrom\s+{re.escape(table)}\b",
                rf"\bjoin\s+{re.escape(table)}\b",
                rf"\bupdate\s+{re.escape(table)}\b",
                rf"\binsert\s+into\s+{re.escape(table)}\b",
                rf"\bdelete\s+from\s+{re.escape(table)}\b",
                rf"\bcreate\s+table\s+{re.escape(table)}\b",
                rf"\balter\s+table\s+{re.escape(table)}\b",
                rf"\bdrop\s+table\s+{re.escape(table)}\b",
                rf"\b{re.escape(table)}\.",
            ]
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    violations.append(f"Forbidden table reference: {table}")
                    break

        for column in self.forbidden_columns:
            patterns = [
                rf"\b{re.escape(column)}\b",
                rf"\.{re.escape(column)}\b",
            ]
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    violations.append(f"Forbidden column reference: {column}")
                    break

        return len(violations) == 0, violations

    def validate_documentation(self, content: str) -> tuple[bool, list[str]]:
        """Validate documentation content for forbidden references.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        content_lower = content.lower()

        for table in self.forbidden_tables:
            if table in content_lower:
                violations.append(f"Forbidden table reference in documentation: {table}")

        for column in self.forbidden_columns:
            if column in content_lower:
                violations.append(f"Forbidden column reference in documentation: {column}")

        return len(violations) == 0, violations

    def validate_training_data_file(self, training_data: dict) -> tuple[bool, list[str]]:
        """Validate a complete training data file.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        all_violations = []

        sample_queries = training_data.get("sample_queries", [])
        for i, query_item in enumerate(sample_queries):
            if "query" in query_item:
                is_valid, violations = self.validate_sql_content(query_item["query"])
                if not is_valid:
                    all_violations.extend([f"Query {i + 1}: {v}" for v in violations])

            if "answer" in query_item:
                is_valid, violations = self.validate_sql_content(query_item["answer"])
                if not is_valid:
                    all_violations.extend([f"Answer {i + 1}: {v}" for v in violations])

            if "question" in query_item:
                is_valid, violations = self.validate_documentation(query_item["question"])
                if not is_valid:
                    all_violations.extend([f"Question {i + 1}: {v}" for v in violations])

        sample_documents = training_data.get("sample_documents", [])
        for i, doc_item in enumerate(sample_documents):
            if "documentation" in doc_item:
                is_valid, violations = self.validate_documentation(doc_item["documentation"])
                if not is_valid:
                    all_violations.extend([f"Document {i + 1}: {v}" for v in violations])

        return len(all_violations) == 0, all_violations

    def reload_forbidden_references(self) -> None:
        """Reload forbidden references from config file."""
        logger.info("Reloading forbidden references...")
        self._load_forbidden_references()

    def get_forbidden_references(self) -> dict[str, list[str]]:
        """Get current forbidden references for debugging/admin purposes."""
        return {
            "tables": list(self.forbidden_tables),
            "columns": list(self.forbidden_columns),
        }


# Global singleton
security_validator = SecurityValidator()
