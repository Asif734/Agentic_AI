import os
from pypdf import PdfReader
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from database import public_docs_collection

# Path to PDFs
PDF_FOLDER = "pdfs/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
model = SentenceTransformer(EMBED_MODEL_NAME)

async def load_pdfs():
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

            # Generate embeddings
            embedding = model.encode([text], convert_to_numpy=True)[0].tolist()

            # Store in MongoDB
            await public_docs_collection.insert_one({
                "doc_type": "pdf",
                "title": filename,
                "content": text,
                "embedding": embedding
            })
            print(f"Loaded: {filename}")

if __name__ == "__main__":
    asyncio.run(load_pdfs())
