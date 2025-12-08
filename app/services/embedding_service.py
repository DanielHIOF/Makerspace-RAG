"""
Makerspace RAG - Embedding Service
Generates embeddings using Ollama for semantic search
"""

import os
import json
import numpy as np
import requests
from typing import List, Optional
from pathlib import Path


class EmbeddingService:
    """Service for generating and caching text embeddings."""

    def __init__(self, model: str = None, host: str = None):
        self.model = model or os.environ.get('EMBEDDING_MODEL', 'mxbai-embed-large')
        self.host = host or os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
        self.cache_file = Path('embeddings_cache.json')
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load embeddings cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                print(f"  [Embeddings] Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                print(f"  [Embeddings] Cache load error: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save embeddings cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f)
        except Exception as e:
            print(f"  [Embeddings] Cache save error: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text."""
        # Check cache first
        cache_key = hash(text) % (10 ** 10)  # Use hash as key
        cache_key_str = str(cache_key)

        if cache_key_str in self._cache:
            return self._cache[cache_key_str]

        try:
            response = requests.post(
                f'{self.host}/api/embeddings',
                json={'model': self.model, 'prompt': text},
                timeout=30
            )

            if response.status_code == 200:
                embedding = response.json().get('embedding')
                if embedding:
                    self._cache[cache_key_str] = embedding
                    return embedding
        except Exception as e:
            print(f"  [Embeddings] Error: {e}")

        return None

    def get_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        total = len(texts)
        new_count = 0

        for i, text in enumerate(texts):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

            if embedding and str(hash(text) % (10 ** 10)) not in self._cache:
                new_count += 1

            if show_progress and (i + 1) % 100 == 0:
                print(f"  [Embeddings] Progress: {i + 1}/{total}")

        # Save cache after batch
        if new_count > 0:
            self._save_cache()
            print(f"  [Embeddings] Cached {new_count} new embeddings")

        return embeddings

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, query: str, documents: List[str], document_embeddings: List[List[float]],
               top_k: int = 5) -> List[tuple]:
        """
        Search documents by semantic similarity.

        Returns list of (index, score) tuples sorted by score descending.
        """
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        scores = []
        for i, doc_emb in enumerate(document_embeddings):
            if doc_emb:
                score = self.cosine_similarity(query_embedding, doc_emb)
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def clear_cache(self):
        """Clear the embeddings cache."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("  [Embeddings] Cache cleared")

    def get_stats(self) -> dict:
        """Get embedding service statistics."""
        return {
            'model': self.model,
            'cached_embeddings': len(self._cache),
            'cache_file': str(self.cache_file)
        }


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
