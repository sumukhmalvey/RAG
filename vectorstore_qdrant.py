from __future__ import annotations

import os
from typing import List, Any, Optional

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient  # Add this import

from embedding import EmbeddingPipeline


class QdrantVectorStoreWrapper:
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 700,
        chunk_overlap: int = 150,
    ):
        self.collection_name = collection_name

        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

        # Create Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        self.chunker = EmbeddingPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vectordb: Optional[QdrantVectorStore] = None

        print(f"[INFO] QdrantVectorStore initialized (collection='{collection_name}')")

    def build_from_documents(self, documents: List[Any]):
        chunks = self.chunker.chunk_documents(documents)

        self.vectordb = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
        )

        print("[INFO] Documents successfully indexed in Qdrant")

    def load(self):
        self.vectordb = QdrantVectorStore(
            client=self.client,  # Add client parameter
            embedding=self.embeddings,
            collection_name=self.collection_name,
        )

        print(f"[INFO] Loaded Qdrant collection '{self.collection_name}'")

    def query(self, query: str, top_k: int = 5):
        if self.vectordb is None:
            raise RuntimeError("Vector store not initialized")

        docs = self.vectordb.similarity_search(query, k=top_k)

        return [
            {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
            for d in docs
        ]