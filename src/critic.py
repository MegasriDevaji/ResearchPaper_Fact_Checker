"""Independently critique an initial claim-verification result."""

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


CRITIC_PROMPT = """
You are an independent critic reviewing a research-paper fact check.

Do not automatically agree with the initial verifier.
Use ONLY the supplied claim and evidence.

Claim:
{claim}

Evidence:
{evidence}

Initial verifier verdict:
{initial_verdict}

Initial verifier reasoning:
{initial_reason}

Check for:
- numerical mismatches
- partial or irrelevant evidence
- unsupported assumptions
- exaggerated conclusions
- missing conditions
- evidence that supports only part of the claim

Return ONLY valid JSON:
{{
  "decision": "accepted | challenged",
  "verdict": "supported | contradicted | unaddressed",
  "reason": "brief evidence-based critique",
  "confidence": 0.0
}}

If the evidence is insufficient, use "unaddressed".
Confidence must be between 0.0 and 1.0.
"""


def _create_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {ENV_PATH}.")
    return genai.Client(api_key=api_key)


def _parse_critique(raw_output: str) -> dict:
    cleaned = raw_output.strip()

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        cleaned = match.group(1).strip()

    result = json.loads(cleaned)

    decision = str(result.get("decision", "")).lower().strip()
    verdict = str(result.get("verdict", "")).lower().strip()

    if decision not in {"accepted", "challenged"}:
        raise ValueError(f"Invalid critic decision: {decision}")

    if verdict not in {"supported", "contradicted", "unaddressed"}:
        raise ValueError(f"Invalid critic verdict: {verdict}")

    confidence = float(result.get("confidence", 0.0))

    return {
        "decision": decision,
        "verdict": verdict,
        "reason": str(result.get("reason", "")),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def critique_claim(
    claim: str,
    evidence: str,
    initial_verdict: str,
    initial_reason: str,
    model: str | None = None,
) -> dict:
    """Critique an initial verification result."""
    if not claim.strip():
        raise ValueError("Claim cannot be empty.")

    model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    prompt = CRITIC_PROMPT.format(
        claim=claim,
        evidence=evidence,
        initial_verdict=initial_verdict,
        initial_reason=initial_reason,
    )

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
                raise ValueError("The critic returned an empty response.")

            return _parse_critique(raw_output)

        except Exception as error:
            error_text = str(error)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini remained unavailable after three attempts."
                ) from error

            time.sleep(5 * (attempt + 1))

    raise RuntimeError("Critique failed.")


if __name__ == "__main__":
    result = critique_claim(
        claim="Our method achieved 94% accuracy.",
        evidence=(
            "The results section reports that our method achieved "
            "94% accuracy."
        ),
        initial_verdict="supported",
        initial_reason="The evidence explicitly confirms the accuracy.",
    )

    print(json.dumps(result, indent=2))