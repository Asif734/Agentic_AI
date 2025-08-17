# agents/mental_health_agent.py
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from .llm_interface import local_llm


class MentalHealthAgent:
    """
    Fast mental-health retrieval agent.

    - Expects a CSV with a single column 'text' where each row looks like:
      "<HUMAN>: your question ... <ASSISTANT>: the response ..."

    - On first run:
        * Parses Q/A pairs
        * Builds normalized embeddings with SentenceTransformer
        * Saves:
            - FAISS index  -> index_path (e.g. data/mh_index.faiss)
            - Metadata     -> meta_path  (e.g. data/mh_meta.pkl)
    - On subsequent runs: loads both files instantly.

    - At query time:
        * Encodes the user message (1 vector)
        * Searches FAISS for top matches in milliseconds
        * If best score >= threshold, returns dataset answer
        * Else falls back to local_llm (optionally with small retrieved context)
    """

    def __init__(
        self,
        csv_file=r"C:\Users\Asif\VSCODE\University Chatbot\data\train.csv",
        index_path="data/mh_index.faiss",
        meta_path="data/mh_meta.pkl",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        threshold: float = 0.55,  # cosine similarity threshold (0..1)
        include_context_in_fallback: bool = True,
        fallback_context_k: int = 3,
    ):
        self.csv_file = csv_file
        self.index_path = index_path
        self.meta_path = meta_path
        self.top_k = max(1, top_k)
        self.threshold = float(threshold)
        self.include_context_in_fallback = include_context_in_fallback
        self.fallback_context_k = max(1, fallback_context_k)

        # Load embedding model once
        self.model = SentenceTransformer(embed_model_name)

        # Try to load prebuilt index + metadata; otherwise build them.
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load_index_and_meta()
            print("MentalHealthAgent: Loaded FAISS index and metadata.")
        else:
            print("MentalHealthAgent: Building index from CSV (first run)...")
            self._build_index_from_csv()
            print("MentalHealthAgent: Index built and saved.")

    # ---------- Public API ----------
    def generate_prompt(self, message: str) -> str:
        """
        Retrieve best matching response from dataset via FAISS.
        If similarity is below threshold, fall back to local LLM.
        """
        if not getattr(self, "pairs", None) or not getattr(self, "index", None):
            # Graceful fallback if something failed to load/build
            return local_llm(self._fallback_prompt(message, context_text=None))

        # Encode and normalize the query
        q = self._encode_and_normalize([message])  # shape (1, d)

        # Search FAISS (inner product on normalized vectors -> cosine similarity)
        D, I = self.index.search(q, self.top_k)    # D: scores, I: indices
        scores = D[0]
        idxs = I[0]

        # Best hit
        best_score = float(scores[0])
        best_idx = int(idxs[0])

        if best_idx >= 0 and best_score >= self.threshold:
            # High-confidence match from dataset
            return self.pairs[best_idx]["assistant"]

        # Fallback: LLM (optionally give small retrieved context)
        if self.include_context_in_fallback:
            context_text = self._format_context(idxs, scores, k=min(self.fallback_context_k, len(self.pairs)))
        else:
            context_text = None

        return local_llm(self._fallback_prompt(message, context_text))

    # ---------- Internal helpers ----------
    def _build_index_from_csv(self):
        df = pd.read_csv(self.csv_file)
        if "text" not in df.columns:
            raise ValueError("MentalHealthAgent: CSV must contain a 'text' column.")

        # Parse Q/A pairs
        pairs = []
        for raw in df["text"].dropna().astype(str).tolist():
            if "<HUMAN>:" in raw and "<ASSISTANT>:" in raw:
                human_part, rest = raw.split("<ASSISTANT>:", 1)
                human = human_part.replace("<HUMAN>:", "").strip()
                assistant = rest.strip()
                if human and assistant:
                    pairs.append({"human": human, "assistant": assistant})

        if not pairs:
            raise ValueError("MentalHealthAgent: No valid <HUMAN>/<ASSISTANT> pairs found.")

        self.pairs = pairs

        # Build embeddings (normalized for cosine similarity)
        questions = [p["human"] for p in self.pairs]
        emb = self._encode_and_normalize(questions)  # (N, d)

        # Build FAISS index: Inner Product (cosine since vectors are normalized)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb.astype(np.float32))

        # Save index + meta
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)

        meta = {
            "pairs": self.pairs,
            "dim": dim,
            "embed_model_name": self.model.get_sentence_embedding_dimension()
            if hasattr(self.model, "get_sentence_embedding_dimension")
            else dim,
        }
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Keep in memory
        self.index = index

    def _load_index_and_meta(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.pairs = meta["pairs"]

    def _encode_and_normalize(self, texts):
        """
        Encode list[str] -> (N, d) float32 L2-normalized embeddings.
        """
        vec = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if vec.ndim == 1:
            vec = vec[None, :]
        vec = vec.astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        return vec / norms

    def _format_context(self, idxs, scores, k=3):
        """
        Build a compact context block with top-k retrieved Q/A pairs.
        """
        items = []
        for rank in range(min(k, len(self.pairs))):
            idx = int(idxs[rank])
            if idx < 0:
                continue
            sim = float(scores[rank])
            q = self.pairs[idx]["human"]
            a = self.pairs[idx]["assistant"]
            items.append(f"[sim={sim:.2f}] Q: {q}\nA: {a}")
        return "\n\n".join(items) if items else None

    def _fallback_prompt(self, user_message: str, context_text: str | None):
        """
        Compose a supportive LLM prompt; optionally include retrieved context to help the LLM.
        """
        if context_text:
            return (
                "You are a supportive, empathetic mental health assistant. "
                "Use the following examples if relevant, but keep the reply concise, warm, and non-clinical. "
                "If risk is detected (self-harm, harm to others, or medical emergency), advise contacting local professionals.\n\n"
                f"Examples:\n{context_text}\n\n"
                f"User: {user_message}\nAssistant:"
            )
        else:
            return (
                "You are a supportive, empathetic mental health assistant. "
                "Respond concisely and kindly. If risk is detected (self-harm, harm to others, or medical emergency), "
                "advise contacting local professionals.\n\n"
            )