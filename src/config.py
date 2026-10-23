"""
Central configuration for the Meeting/Document Knowledge Assistant.
Keep paths and tunables here so nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
CHUNKS_CACHE_PATH = INDEX_DIR / "chunks.json"

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Retrieval ---
TOP_K = 4

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# --- LLM (Groq) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# --- Garnet Cache ---
GARNET_HOST = os.getenv("GARNET_HOST", "localhost")
GARNET_PORT = int(os.getenv("GARNET_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# --- SQLite ---
SQLITE_DB = PROJECT_ROOT / "data" / "chat.db"