"""Reconcile verifier and critic opinions into a final verdict."""

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

VALID_VERDICTS = {"supported", "contradicted", "unaddressed"}

JUDGE_PROMPT = """
You are the final judge in an evidence-grounded fact-checking system.

Use ONLY the claim and supplied evidence. Do not use outside knowledge.
Prioritize evidence over either agent's opinion.
If evidence is insufficient, choose unaddressed.

Claim:
{claim}

Evidence:
{evidence}

Initial verifier:
Verdict: {initial_verdict}
Reason: {initial_reason}
Confidence: {initial_confidence}

Independent critic:
Decision: {critic_decision}
Verdict: {critic_verdict}
Reason: {critic_reason}
Confidence: {critic_confidence}

Return ONLY valid JSON:
{{
  "verdict": "supported | contradicted | unaddressed",
  "evidence_chunk": "most relevant evidence, or empty string",
  "confidence": 0.0,
  "reason": "brief final explanation",
  "debate_changed_verdict": false
}}

Confidence must be between 0.0 and 1.0.
"""


def _create_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {ENV_PATH}.")
    return genai.Client(api_key=api_key)


def _parse_result(raw_output: str, claim: str) -> dict:
    cleaned = raw_output.strip()

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        cleaned = match.group(1).strip()

    result = json.loads(cleaned)
    verdict = str(result.get("verdict", "")).lower().strip()

    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid final verdict: {verdict}")

    confidence = float(result.get("confidence", 0.0))

    return {
        "claim": claim,
        "verdict": verdict,
        "evidence_chunk": str(result.get("evidence_chunk", "")),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(result.get("reason", "")),
        "debate_changed_verdict": bool(
            result.get("debate_changed_verdict", False)
        ),
    }


def judge_claim(
    claim: str,
    evidence: str,
    initial_result: dict,
    critic_result: dict,
    model: str | None = None,
) -> dict:
    """Produce the final verdict after verifier–critic debate."""
    if not claim.strip():
        raise ValueError("Claim cannot be empty.")

    model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    prompt = JUDGE_PROMPT.format(
        claim=claim,
        evidence=evidence,
        initial_verdict=initial_result["verdict"],
        initial_reason=initial_result.get("reason", ""),
        initial_confidence=initial_result.get("confidence", 0.0),
        critic_decision=critic_result["decision"],
        critic_verdict=critic_result["verdict"],
        critic_reason=critic_result.get("reason", ""),
        critic_confidence=critic_result.get("confidence", 0.0),
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
                raise ValueError("The judge returned an empty response.")

            return _parse_result(raw_output, claim)

        except Exception as error:
            error_text = str(error)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini remained unavailable after three attempts."
                ) from error

            time.sleep(5 * (attempt + 1))

    raise RuntimeError("Judging failed.")


if __name__ == "__main__":
    initial_result = {
        "verdict": "supported",
        "confidence": 1.0,
        "reason": "The evidence explicitly confirms the accuracy.",
    }

    critic_result = {
        "decision": "accepted",
        "verdict": "supported",
        "confidence": 1.0,
        "reason": "The evidence directly matches the claim.",
    }

    result = judge_claim(
        claim="Our method achieved 94% accuracy.",
        evidence="The results section reports that our method achieved 94% accuracy.",
        initial_result=initial_result,
        critic_result=critic_result,
    )

    print(json.dumps(result, indent=2))