import json
from typing import Any

import pandas as pd
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.utils import deterministic_uuid


class ThriveAI_ChromaDB(ChromaDB_VectorStore):
    def __init__(self, config=None):
        super().__init__(config=config)

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
            metadatas=metadata,
        )

        return id

    def add_ddl(self, ddl: str, metadata: dict[str, Any] | None = None, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
            metadatas=metadata
        )
        return id

    def add_documentation(self, documentation: str, metadata: dict[str, Any] | None = None, **kwargs) -> str:
        id = deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
            metadatas=metadata,
        )
        return id

    def get_training_data(self, metadata: dict[str, Any] | None = None, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get(where=metadata)

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

        ddl_data = self.ddl_collection.get(where=metadata)

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

        doc_data = self.documentation_collection.get(where=metadata)

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
                where=metadata,
            )
        )

    def get_related_ddl(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_results_ddl,
                where=metadata,
            )
        )

    def get_related_documentation(self, question: str, metadata: dict[str, Any] | None = None, **kwargs) -> list:
        return self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
                where=metadata,
            )
        )
