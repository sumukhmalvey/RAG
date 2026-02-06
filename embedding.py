from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingPipeline:
    """
    Handles document chunking only.
    Embedding is delegated to the vector store (Qdrant).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        print(
            f"[INFO] Text splitter initialized "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Splits documents into overlapping text chunks.
        """
        chunks = self.splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
