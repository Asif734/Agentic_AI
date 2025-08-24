from sentence_transformers import SentenceTransformer
from typing import List
from .llm_interface import local_llm
from utils.document_loader import pdf_to_text, chunk_text
from utils.vector_store import VectorStore

class PublicAgentRAG:
    """RAG-based public agent for text and PDF ingestion."""

    def __init__(
        self,
        index_path="data/public_index.faiss",
        meta_path="data/public_meta.pkl",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        top_k: int = 5
    ):
        self.chunk_size = chunk_size
        self.top_k = top_k

        # Embedding model
        self.model = SentenceTransformer(embed_model_name)

        # Vector store
        self.store = VectorStore(index_path, meta_path)

    # ---------- Ingestion ----------
    def add_text(self, text: str, source: str = "manual"):
        chunks = chunk_text(text, self.chunk_size)
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        self.store.add_embeddings(embeddings, chunks, [source]*len(chunks))

    def add_pdf(self, pdf_path: str):
        text = pdf_to_text(pdf_path)
        self.add_text(text, source=pdf_path)

    # ---------- Query ----------
    def respond(self, query: str) -> str:
        if not self.store.index or len(self.store.metadata) == 0:
            return local_llm(query)

        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, idxs = self.store.search(query_vec, self.top_k)
        context_chunks = self.store.get_texts(idxs)

        context_text = "\n".join(context_chunks)
        prompt = (
            "You are a knowledgeable university assistant. Use the context to answer the question:\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )
        return local_llm(prompt)
