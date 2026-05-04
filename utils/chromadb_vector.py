import json
from typing import Any

import pandas as pd
from chromadb.api import ClientAPI
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.legacy.utils import deterministic_uuid


from utils.quick_logger import get_logger

logger = get_logger(__name__)


class ThriveAI_ChromaDB(ChromaDB_VectorStore):
    def __init__(self, user_role: int, config=None, client: ClientAPI | None = None):
        # Prepare a new config dictionary to pass to the superclass
        super_config = config.copy() if config else {}

        # Set the collection metadata for cosine similarity.
        # This will be used by the superclass's __init__ method.
        super_config["collection_metadata"] = {"hnsw:space": "cosine"}

        if client:
            # If an external client is provided, use it.
            super_config["client"] = client
            super_config.pop("path", None)  # Avoid creating a persistent client
        elif super_config.get("in_memory"):
            # For in-memory, we tell the superclass to create an in-memory client.
            super_config["client"] = "in-memory"
            super_config.pop("path", None)
        # For the default case (persistent client), we don't need to do anything
        # extra as the superclass handles it. The path will be in the config.

        super().__init__(config=super_config)
        self.user_role = user_role
        if client:
            self.client = client

    def add_question_sql(self, question: str, sql: str, metadata: dict[str, Any] | None = None, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
            metadatas=self._prepare_metadata(metadata),
        )

        return id

    def add_ddl(self, ddl: str, metadata: dict[str, Any] | None = None, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
            metadatas=self._prepare_metadata(metadata),
        )
        return id

    def add_documentation(
        self, documentation: str, metadata: dict[str, Any] | None = None, id: str | None = None, **kwargs
    ) -> str:
        if id is None:
            id = deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
            metadatas=self._prepare_metadata(metadata),
        )
        return id

    # Override to use Ollama embeddings if configured
    def generate_embedding(self, data: str, **kwargs: Any) -> list[float]:
        try:
            import streamlit as st  # optional in tests

            ai_keys = st.secrets.get("ai_keys", {})
            embed_model = ai_keys.get("ollama_embed_model")
            if embed_model:
                import ollama

                host = ai_keys.get("ollama_host", "http://localhost:11434")
                client = ollama.Client(host)
                res = client.embeddings(model=embed_model, prompt=data)
                emb = res.get("embedding")
                if isinstance(emb, list) and emb:
                    return emb
        except Exception:
            pass
        # Fallback to superclass implementation
        return super().generate_embedding(data, **kwargs)

    def get_training_data(self, metadata: dict[str, Any] | None = None, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get(where=self._prepare_retrieval_metadata(metadata))

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get(where=self._prepare_retrieval_metadata(metadata))

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get(where=self._prepare_retrieval_metadata(metadata))

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        """Remove a training entry by ID from all collections."""
        found = False
        for collection in [self.sql_collection, self.ddl_collection, self.documentation_collection]:
            try:
                collection.delete(ids=[id])
                found = True
            except Exception:
                pass
        return found

    def get_similar_question_sql(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.sql_collection.query(
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_sql,
                where=self._prepare_retrieval_metadata(metadata),
            )
        )

    def get_related_ddl(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.ddl_collection.query(
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_ddl,
                where=self._prepare_retrieval_metadata(metadata),
            )
        )

    def get_related_documentation(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        retrieval_metadata = self._prepare_retrieval_metadata(metadata)
        logger.debug(f"Querying documentation_collection with metadata: {retrieval_metadata}")
        return self._extract_documents(
            self.documentation_collection.query(
                query_embeddings=[self.generate_embedding(question)],
                n_results=self.n_results_documentation,
                where=retrieval_metadata,
            )
        )

    def _prepare_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["user_role"] = self._get_effective_role()
        return metadata

    def _prepare_retrieval_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["user_role"] = {"$gte": self._get_effective_role()}
        return metadata

    def _is_role_restriction_enabled(self) -> bool:
        """Check secrets for role restriction toggle. Defaults to True when unavailable."""
        try:
            import streamlit as st  # optional in tests

            # Support both top-level and nested under security
            if "restrict_rag_by_role" in st.secrets:
                return bool(st.secrets.get("restrict_rag_by_role"))
            security = st.secrets.get("security", {})
            return bool(security.get("restrict_rag_by_role", True))
        except Exception:
            return True

    def _get_effective_role(self) -> int:
        """Return 0 when restriction disabled to allow all training; else the user role."""
        return 0 if not self._is_role_restriction_enabled() else int(self.user_role)
