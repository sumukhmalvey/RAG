import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from data_loader import load_all_documents
from vectorstore_qdrant import QdrantVectorStoreWrapper

# Load environment variables from .env
load_dotenv()


class RAGSearch:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        # -------- Vector Store (Qdrant) --------
        self.vectorstore = QdrantVectorStoreWrapper(
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        # Try loading existing collection, otherwise build it
        try:
            self.vectorstore.load()
            print("[INFO] Loaded existing Qdrant collection")
        except Exception:
            print("[INFO] No existing collection found. Building new one...")
            documents = load_all_documents("data")
            self.vectorstore.build_from_documents(documents)

        # -------- LLM (Groq) --------
        # IMPORTANT: API key is read automatically from .env
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.2,
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_answer(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        context_chunks = [r["page_content"] for r in results]
        context = "\n\n---\n\n".join(context_chunks)

        if not context.strip():
            return "I don't know based on the provided manuals."

        prompt = f"""
You are an experienced car mechanic assistant.

Use ONLY the information from the vehicle service manuals below.
Be precise and step-by-step when procedures are involved.

If the answer is not present in the manuals, say:
"I don't know based on the provided manuals."

Question:
{query}

Manual Excerpts:
{context}

Answer:
""".strip()

        response = self.llm.invoke(prompt)

        # Handle both string and structured outputs safely
        if isinstance(response.content, str):
            return response.content

        return str(response.content)


# -------- Run directly --------
if __name__ == "__main__":
    rag = RAGSearch()

    answer = rag.search_and_answer(
        "how can i change the oil filter on the i10 2010 model refer to the manual provided and give me step by step instructions",
        top_k=3,
    )

    print("\nANSWER:\n")
    print(answer)


