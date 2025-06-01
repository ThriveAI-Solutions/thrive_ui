import json
from typing import Any

import pandas as pd
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.utils import deterministic_uuid

import chromadb


class ThriveAI_ChromaDB(ChromaDB_VectorStore):
    def __init__(self, user_role: int, config=None, client: chromadb.ClientAPI | None = None):
        if client:
            self.client = client
            # If a client is provided, we assume collections will be managed externally or by super init with a config that doesn't specify a path
            # To ensure super() doesn't override with a persistent client, pass a config without a path
            super_config = config.copy() if config else {}
            super_config.pop("path", None)
            super().__init__(config=super_config)
        elif config and config.get("in_memory"):
            self.client = chromadb.Client()  # Create in-memory client
            # Pass a config to super that won't create a persistent client
            super_config = config.copy()
            super_config.pop("path", None)  # Ensure no path is sent to super if we want in-memory
            super().__init__(config=super_config)
        else:
            # Default behavior: use path from config or default path for PersistentClient
            # Let super().__init__ handle client creation in this case
            super().__init__(config=config)

        self.user_role = user_role
        # Ensure collections are created if not by super (e.g. if client was passed and super_config was minimal)
        # The super().__init__ already calls self._create_collections
        # However, if a client was passed, self.client was set BEFORE super().__init__.
        # We need to ensure _create_collections uses the client we intend.
        # ChromaDB_VectorStore._create_collections uses self.client, so it should be fine.

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
        return self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
                where=self._prepare_retrieval_metadata(metadata),
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
