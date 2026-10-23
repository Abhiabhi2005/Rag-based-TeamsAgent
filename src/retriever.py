#retriever.py
"""
BM25 retriever for the Meeting/Document Knowledge Assistant.

Loads the chunk cache built by ingest.py and builds an in-memory BM25 index
(rank_bm25). No model downloads, no vector database -- pure keyword-based
sparse retrieval, well suited for meeting notes where specific names, dates,
and terms matter.
"""
import json
import re
import redis
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config
cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer. Good enough for BM25 on short docs."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self._tokenized_corpus = [_tokenize(c.page_content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @classmethod
    def from_cache(cls, cache_path=None) -> "BM25Retriever":
        cache_path = cache_path or config.CHUNKS_CACHE_PATH
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No chunk cache found at {cache_path}. Run `python src/ingest.py` first."
            )
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        chunks = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw]
        return cls(chunks)

    def retrieve(self, query: str, top_k: int = None):

        cache_key = f"retrieve:{query}"

        cached = cache.get(cache_key)

        if cached:
            data = json.loads(cached)

            return [
                Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"]
                )
                for item in data
            ]

        top_k = top_k or config.TOP_K

        tokenized_query = _tokenize(query)

        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )

        docs = [
            doc
            for score, doc in ranked[:top_k]
            if score > 0
        ]

        serialized = [
            {
                "page_content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]

        cache.setex(
            cache_key,
            3600,
            json.dumps(serialized)
        )

        return docs

if __name__ == "__main__":
    # Quick manual test: run `python src/retriever.py` after ingest.py has run.
    retriever = BM25Retriever.from_cache()
    test_query = "what is the launch date"
    print(f"Query: {test_query!r}\n")
    for doc in retriever.retrieve(test_query):
        print(f"[{doc.metadata['source']}]")
        print(doc.page_content[:200].replace("\n", " "))
        print()
