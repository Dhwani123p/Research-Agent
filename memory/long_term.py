"""Long-term memory — persists research across sessions using ChromaDB."""

import os
import hashlib
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "research_sessions"


class LongTermMemory:
    """
    Stores and retrieves research notes across sessions.
    Uses ChromaDB with sentence-transformer embeddings for semantic search.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, query: str, content: str, source: str, tool: str):
        """Embed and persist a research note."""
        doc_id = hashlib.md5(f"{query}:{source}:{content[:50]}".encode()).hexdigest()
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{
                "query": query,
                "source": source,
                "tool": tool,
                "timestamp": datetime.now().isoformat(),
            }],
        )

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve the most relevant past notes for a query."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )
        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({
                "content": doc,
                "source": meta.get("source", ""),
                "tool": meta.get("tool", ""),
                "original_query": meta.get("query", ""),
                "timestamp": meta.get("timestamp", ""),
            })
        return retrieved

    def save_session_summary(self, query: str, report: str):
        """Save the final report summary for a completed session."""
        self.save(
            query=query,
            content=f"RESEARCH SUMMARY: {report[:1500]}",
            source="session_summary",
            tool="report_generator",
        )
