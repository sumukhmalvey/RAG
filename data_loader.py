from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Recursively loads PDF, TXT, and DOCX files from data_dir
    and returns a list of LangChain Document objects.
    """

    data_path = Path(data_dir).resolve()
    print(f"[INFO] Data path: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    documents: List[Any] = []

    # -------- PDFs --------
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[INFO] Loaded {len(loaded)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # -------- TXTs --------
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[INFO] Found {len(txt_files)} TXT files")

    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[INFO] Loaded TXT file {txt_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # -------- DOCX --------
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"[INFO] Found {len(docx_files)} DOCX files")

    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[INFO] Loaded DOCX file {docx_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load DOCX {docx_file}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents
