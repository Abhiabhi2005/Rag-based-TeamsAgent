"""
Ingestion pipeline for the Meeting/Document Knowledge Assistant.

What this does:
  1. Loads every PDF in data/pdfs/
  2. Splits each document into overlapping chunks
  3. Caches the chunks (text + source metadata) to disk as JSON

No model downloads happen here -- retrieval is BM25 (keyword-based),
built fresh from the cached chunks at query time by retriever.py.

Run directly to (re)build the chunk cache after adding/changing PDFs:
    python src/ingest.py
"""
import json
import sys
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def load_pdf_text(pdf_path: Path) -> str:
    """Extract text from a single PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def load_all_documents(pdf_dir: Path) -> list[Document]:
    """Load every PDF in pdf_dir into LangChain Document objects, tagged with source filename."""
    documents = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}. Add some and re-run.")
        sys.exit(1)

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        text = load_pdf_text(pdf_path)
        if not text.strip():
            print(f"  Warning: no extractable text in {pdf_path.name} (skipped)")
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name},
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving source metadata on each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks


def cache_chunks(chunks: list[Document]) -> None:
    """Persist chunks (text + metadata) to disk as JSON, for retriever.py to load."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    serializable = [
        {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
    ]
    with open(config.CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Cached {len(chunks)} chunks to {config.CHUNKS_CACHE_PATH}")


def main():
    print("=== Ingestion pipeline starting ===")
    documents = load_all_documents(config.PDF_DIR)
    chunks = chunk_documents(documents)
    cache_chunks(chunks)
    print("=== Done. Chunk cache is ready for retrieval. ===")


if __name__ == "__main__":
    main()
