"""
pdf_processor.py
Extracts text from PDF files using PyMuPDF.
"""

import pymupdf
import unicodedata  # add this import
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text = unicodedata.normalize("NFKC", text)  # <-- fixes ligatures (ﬁ -> fi, etc.)
        full_text.append(f"\n--- PAGE {page_num} ---\n{text}")

    doc.close()
    return "".join(full_text)

def extract_all_papers(papers_dir: str, output_dir: str) -> dict:
    """
    Extract text from every PDF in papers_dir, save as .txt in output_dir.
    Returns a dict mapping paper_id -> extracted text.
    """
    papers_dir = Path(papers_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for pdf_file in sorted(papers_dir.glob("*.pdf")):
        paper_id = pdf_file.stem  # e.g. "lora" from "lora.pdf"
        print(f"Extracting: {pdf_file.name} ...")

        text = extract_text_from_pdf(pdf_file)
        results[paper_id] = text

        out_file = output_dir / f"{paper_id}.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"  -> saved {len(text)} characters to {out_file}")

    return results


if __name__ == "__main__":
    extract_all_papers(
        papers_dir="data/papers",
        output_dir="data/extracted_text"
    )
    print("\nDone. Check data/extracted_text/ for .txt files.")