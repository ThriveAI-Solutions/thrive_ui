import logging
import json
from typing import Any

import pandas as pd
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.utils import deterministic_uuid

from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)


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
            documents=ddl, embeddings=self.generate_embedding(ddl), ids=id, metadatas=self._prepare_metadata(metadata)
        )
        return id

    def add_documentation(self, documentation: str, metadata: dict[str, Any] | None = None, **kwargs) -> str:
        id = deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
            metadatas=self._prepare_metadata(metadata),
        )
        return id

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

    def get_similar_question_sql(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results_sql,
                where=self._prepare_retrieval_metadata(metadata),
            )
        )

    def get_related_ddl(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_results_ddl,
                where=self._prepare_retrieval_metadata(metadata),
            )
        )

    def get_related_documentation(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        retrieval_metadata = self._prepare_retrieval_metadata(metadata)
        logger.debug(f"Querying documentation_collection with metadata: {retrieval_metadata}")
        return self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
                where=retrieval_metadata,
            )
        )

    def _prepare_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["user_role"] = self.user_role
        return metadata

    def _prepare_retrieval_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if metadata is None:
            metadata = {}
        metadata["user_role"] = {"$gte": self.user_role}
        return metadata
