"""
vector_store.py
Builds a FAISS vector store from paper chunks + related abstracts,
and provides a retrieval function.
"""

from pathlib import Path
import re
import pickle

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this project


def load_chunks_from_file(chunk_file: Path) -> list[str]:
    """Parse a '=== CHUNK N ===' formatted file back into a list of chunk strings."""
    text = chunk_file.read_text(encoding="utf-8")
    parts = re.split(r"=== CHUNK \d+ ===\n", text)
    return [p.strip() for p in parts if p.strip()]


def load_all_documents(chunks_dir: str, abstracts_dir: str) -> list[dict]:
    """
    Returns a list of dicts:
    {"text": ..., "source_type": "paper_chunk" | "related_abstract", "source_id": ..., "chunk_index": ...}
    """
    documents = []

    chunks_dir = Path(chunks_dir)
    for chunk_file in sorted(chunks_dir.glob("*_chunks.txt")):
        paper_id = chunk_file.stem.replace("_chunks", "")
        chunks = load_chunks_from_file(chunk_file)
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source_type": "paper_chunk",
                "source_id": paper_id,
                "chunk_index": i
            })

    abstracts_dir = Path(abstracts_dir)
    for abs_file in sorted(abstracts_dir.glob("*.txt")):
        text = abs_file.read_text(encoding="utf-8").strip()
        documents.append({
            "text": text,
            "source_type": "related_abstract",
            "source_id": abs_file.stem,
            "chunk_index": 0
        })

    return documents


def build_vector_store(chunks_dir: str, abstracts_dir: str, output_dir: str):
    """
    Embeds all documents and builds a FAISS index.
    Saves the index + document metadata to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    documents = load_all_documents(chunks_dir, abstracts_dir)
    print(f"Loaded {len(documents)} documents (paper chunks + abstracts)")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [doc["text"] for doc in documents]
    print("Embedding all documents (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product, not L2
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, str(output_dir / "faiss_index.bin"))
    with open(output_dir / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print(f"\nSaved FAISS index ({index.ntotal} vectors, dim={dimension}) to {output_dir}")
    return index, documents


class Retriever:
    """Loads a saved vector store and retrieves top-k relevant chunks for a query."""

    def __init__(self, store_dir: str):
        store_dir = Path(store_dir)
        self.index = faiss.read_index(str(store_dir / "faiss_index.bin"))
        with open(store_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, query: str, top_k: int = 5, paper_id_filter: str = None) -> list[dict]:
        """
        Retrieve top_k most relevant documents for the query.
        If paper_id_filter is set, only search within that paper's own chunks
        PLUS the related abstracts (mirrors the assignment's two evidence sources).
        """
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # Search a bit wider than top_k so filtering still leaves enough results
        search_k = top_k * 4 if paper_id_filter else top_k
        distances, indices = self.index.search(query_vec, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if paper_id_filter:
                is_own_paper = doc["source_type"] == "paper_chunk" and doc["source_id"] == paper_id_filter
                is_related_abstract = doc["source_type"] == "related_abstract"
                if not (is_own_paper or is_related_abstract):
                    continue
            results.append({**doc, "distance": float(dist)})
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    build_vector_store(
        chunks_dir="data/chunks",
        abstracts_dir="data/abstracts",
        output_dir="data/vector_store"
    )