"""Run improved local paper fact-checking without Gemini."""

import json
import re
from pathlib import Path

from graph import get_retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
RESULTS_DIR = PROJECT_ROOT / "results"

PAPER_ID = "lora"
MAX_CLAIMS_PER_RUN = 5
RETRIEVE_K = 20
FINAL_K = 5


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"---\s*PAGE\s*\d+\s*---", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.replace("- ", "")
    return sentence.strip()


def extract_local_claims(text: str) -> list[str]:
    """Extract useful sentences while ignoring PDF headers and metadata."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = []

    for raw_sentence in sentences:
        sentence = clean_sentence(raw_sentence)

        if len(sentence) < 40 or len(sentence) > 700:
            continue

        letters = re.findall(r"[A-Za-z]", sentence)
        uppercase_letters = re.findall(r"[A-Z]", sentence)

        if letters and len(uppercase_letters) / len(letters) > 0.65:
            continue

        if "@" in sentence or "arxiv:" in sentence.lower():
            continue

        if sentence not in claims:
            claims.append(sentence)

    return claims


def tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def rerank_documents(
    claim: str,
    documents: list[dict],
) -> list[dict]:
    """Combine semantic retrieval with lexical and numeric matching."""
    claim_tokens = tokens(claim)
    claim_numbers = numbers(claim)

    ranked = []

    for document in documents:
        text = document.get("text", "")
        text_tokens = tokens(text)
        text_numbers = numbers(text)

        word_coverage = len(claim_tokens & text_tokens) / max(
            len(claim_tokens),
            1,
        )

        numeric_coverage = (
            len(claim_numbers & text_numbers) / len(claim_numbers)
            if claim_numbers
            else 1.0
        )

        semantic_score = float(document.get("distance", 0.0))

        score = (
            0.45 * semantic_score
            + 0.40 * word_coverage
            + 0.15 * numeric_coverage
        )

        ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:FINAL_K]]


def verify_locally(claim: str, evidence_documents: list[dict]) -> dict:
    """Verify using lexical and numeric evidence matching."""
    evidence = " ".join(
        document.get("text", "") for document in evidence_documents
    )

    claim_tokens = tokens(claim)
    evidence_tokens = tokens(evidence)
    claim_numbers = numbers(claim)
    evidence_numbers = numbers(evidence)

    word_coverage = len(claim_tokens & evidence_tokens) / max(
        len(claim_tokens),
        1,
    )

    numbers_match = claim_numbers.issubset(evidence_numbers)

    if word_coverage >= 0.70 and numbers_match:
        verdict = "supported"
        reason = "The retrieved evidence matches the claim and its numerical values."
    else:
        verdict = "unaddressed"
        reason = (
            "The retrieved evidence does not sufficiently support all "
            "claim details."
        )

    return {
        "claim": claim,
        "verdict": verdict,
        "evidence_chunk": (
            evidence_documents[0].get("text", "")
            if evidence_documents
            else ""
        ),
        "confidence": round(
            min(word_coverage, 1.0) if numbers_match else word_coverage * 0.5,
            2,
        ),
        "reason": reason,
        "mode": "local",
    }


def run_paper(paper_id: str) -> list[dict]:
    text_path = TEXT_DIR / f"{paper_id}.txt"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing paper text: {text_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    text = text_path.read_text(encoding="utf-8")
    claims_path = RESULTS_DIR / f"{paper_id}_claims.json"
    output_path = RESULTS_DIR / f"{paper_id}_local_results.json"

    claims = extract_local_claims(text)

    claims_path.write_text(
        json.dumps(claims, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    claims = claims[:MAX_CLAIMS_PER_RUN]

    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        results = []

    completed = {result["claim"] for result in results}
    retriever = get_retriever()

    for index, claim in enumerate(claims, start=1):
        if claim in completed:
            print(f"Skipping claim {index}/{len(claims)}...")
            continue

        print(f"Checking claim {index}/{len(claims)}...", flush=True)

        documents = retriever.retrieve(
            query=claim,
            top_k=RETRIEVE_K,
            paper_id_filter=paper_id,
        )

        documents = rerank_documents(claim, documents)
        result = verify_locally(claim, documents)
        result["retrieved_evidence"] = documents

        results.append(result)

        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        print(f"Saved {len(results)} result(s).", flush=True)

    return results


if __name__ == "__main__":
    results = run_paper(PAPER_ID)
    print(f"\nCompleted {len(results)} local result(s).")