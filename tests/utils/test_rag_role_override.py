from unittest.mock import patch

import pytest


class TestChromaRoleOverride:
    def test_chroma_role_override_allows_all(self, in_memory_chromadb_client):
        from utils.chromadb_vector import ThriveAI_ChromaDB

        class ConcreteThriveAIChroma(ThriveAI_ChromaDB):
            def system_message(self, message: str):
                return message

            def user_message(self, message: str):
                return message

            def assistant_message(self, message: str):
                return message

            def submit_prompt(self, prompt, **kwargs):
                return ""

            # Deterministic tiny embedding for tests (length 8)
            def generate_embedding(self, data: str, **kwargs):
                vec = [0.0] * 8
                for ch in data[:8]:
                    vec[ord(ch) % 8] += 1.0
                return vec

        with patch("streamlit.secrets", new={"restrict_rag_by_role": False}):
            # Use the same in-memory client across instances to share collections
            admin = ConcreteThriveAIChroma(user_role=0, client=in_memory_chromadb_client)
            nurse = ConcreteThriveAIChroma(user_role=2, client=in_memory_chromadb_client)
            patient_writer = ConcreteThriveAIChroma(user_role=3, client=in_memory_chromadb_client)

            # Seed entries written by different roles
            admin.add_question_sql("How many patients?", "SELECT COUNT(*) FROM patients;")
            nurse.add_ddl("CREATE TABLE nursing_notes (id INT, note TEXT);")
            patient_writer.add_documentation("Patient can download their own records via portal.")

            # Reader with highest role should see all when override is disabled
            patient_reader = ConcreteThriveAIChroma(user_role=3, client=in_memory_chromadb_client)
            df = patient_reader.get_training_data()

            assert len(df) == 3
            assert set(df["training_data_type"]) == {"sql", "ddl", "documentation"}


class TestMilvusRoleOverride:
    pymilvus = pytest.importorskip("pymilvus", reason="pymilvus not installed")

    def _store(self, tmp_path, role: int):
        from utils.milvus_vector import ThriveAI_Milvus

        return ThriveAI_Milvus(
            user_role=role,
            config={
                "mode": "lite",
                "text_dim": 64,
                "collection_prefix": "unit_override",
                "persist_path": str(tmp_path / "milvus_lite.db"),
            },
        )

    def test_milvus_role_override_allows_all(self, tmp_path):
        with patch("streamlit.secrets", new={"restrict_rag_by_role": False}):
            admin = self._store(tmp_path, role=0)
            nurse = self._store(tmp_path, role=2)
            patient_writer = self._store(tmp_path, role=3)

            # Seed entries written by different roles
            admin.add_question_sql("How many patients?", "SELECT COUNT(*) FROM patients;")
            nurse.add_ddl("CREATE TABLE nursing_notes (id INT, note TEXT);")
            patient_writer.add_documentation("Patient can download their own records via portal.")

            # Reader with highest role should see all when override is disabled
            patient_reader = self._store(tmp_path, role=3)
            df = patient_reader.get_training_data()

            assert len(df) == 3
            assert set(df["training_data_type"]) == {"sql", "ddl", "documentation"}


