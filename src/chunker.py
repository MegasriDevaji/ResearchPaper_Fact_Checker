"""
chunker.py
Splits extracted text into overlapping chunks for embedding,
using sentence-based grouping for more coherent chunks.
"""

from pathlib import Path
import re


def chunk_text(text: str, sentences_per_chunk: int = 4, overlap_sentences: int = 1) -> list[str]:
    """
    Split text into overlapping chunks of N sentences each.
    More coherent than raw word-count chunking for scientific text.
    """
    clean_text = re.sub(r"--- PAGE \d+ ---", "", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += sentences_per_chunk - overlap_sentences

    return chunks


def chunk_all_papers(extracted_text_dir: str, output_dir: str,
                      sentences_per_chunk: int = 4, overlap_sentences: int = 1) -> dict:
    """
    Chunk every .txt file in extracted_text_dir.
    Saves each paper's chunks as a numbered list in a text file.
    Returns dict: paper_id -> list of chunks.
    """
    extracted_dir = Path(extracted_text_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = {}
    for txt_file in sorted(extracted_dir.glob("*.txt")):
        paper_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")

        chunks = chunk_text(text, sentences_per_chunk=sentences_per_chunk,
                             overlap_sentences=overlap_sentences)
        all_chunks[paper_id] = chunks

        out_file = output_dir / f"{paper_id}_chunks.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i} ===\n{chunk}\n\n")

        print(f"{paper_id}: {len(chunks)} chunks -> {out_file}")

    return all_chunks


if __name__ == "__main__":
    chunk_all_papers(
        extracted_text_dir="data/extracted_text",
        output_dir="data/chunks",
        sentences_per_chunk=4,
        overlap_sentences=1
    )
    print("\nDone. Check data/chunks/ for chunked output.")