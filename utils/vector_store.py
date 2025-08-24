import os
import pickle
import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    """
    Handles FAISS index + metadata persistence.
    Supports append-only updates for RAG.
    """
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata: List[Dict] = []

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str], sources: List[str]):
        """Append embeddings and corresponding metadata."""
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings.astype("float32"))

        for t, s in zip(texts, sources):
            self.metadata.append({"text": t, "source": s})

        self._save()

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        """Return indices and scores for top-k matches."""
        if self.index is None or len(self.metadata) == 0:
            return [], []

        norms = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12
        query_vec = query_vec / norms
        D, I = self.index.search(query_vec.astype("float32"), top_k)
        return D[0], I[0]

    def get_texts(self, indices: List[int]):
        """Retrieve metadata text by indices."""
        return [self.metadata[i]["text"] for i in indices if i < len(self.metadata)]

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
