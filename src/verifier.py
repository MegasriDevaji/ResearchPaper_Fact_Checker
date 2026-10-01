"""Verify factual claims using only supplied evidence."""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


VERIFICATION_PROMPT = """
You are an evidence-grounded research-paper fact checker.

Use ONLY the supplied evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Definitions:
- supported: evidence directly supports the complete claim.
- contradicted: evidence directly conflicts with the claim.
- unaddressed: evidence is missing, irrelevant, partial, or insufficient.

Return ONLY valid JSON in this exact format:
{{
  "verdict": "supported | contradicted | unaddressed",
  "evidence_chunk": "the most relevant evidence, or empty string",
  "confidence": 0.0,
  "reason": "brief explanation based only on the evidence"
}}

Confidence must be a number between 0.0 and 1.0.
"""


def _create_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {ENV_PATH}.")

    return genai.Client(api_key=api_key)


def _parse_json(raw_output: str) -> dict:
    cleaned = raw_output.strip()

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        cleaned = match.group(1).strip()

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    verdict = str(result.get("verdict", "")).lower().strip()

    if verdict not in {"supported", "contradicted", "unaddressed"}:
        raise ValueError(f"Invalid verdict: {verdict}")

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "verdict": verdict,
        "evidence_chunk": str(result.get("evidence_chunk", "")),
        "confidence": confidence,
        "reason": str(result.get("reason", "")),
    }


def verify_claim(
    claim: str,
    evidence: str,
    model: str | None = None,
) -> dict:
    """Verify one claim against supplied evidence."""
    if not claim.strip():
        raise ValueError("Claim cannot be empty.")

    model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    prompt = VERIFICATION_PROMPT.format(claim=claim, evidence=evidence)
    client = _create_client()

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},
            )

            raw_output = (response.text or "").strip()

            if not raw_output:
                raise ValueError("The model returned an empty response.")

            return _parse_json(raw_output)

        except Exception as error:
            error_text = str(error)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini remained unavailable after three attempts."
                ) from error

            time.sleep(5 * (attempt + 1))

    raise RuntimeError("Verification failed.")


if __name__ == "__main__":
    test_claim = "Our method achieved 94% accuracy."
    test_evidence = "The results section reports that our method achieved 94% accuracy."

    result = verify_claim(test_claim, test_evidence)

    print(json.dumps(result, indent=2))