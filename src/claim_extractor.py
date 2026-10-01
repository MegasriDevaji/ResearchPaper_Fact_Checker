"""Extract factual claims from research-paper text using Google Gemini."""

import json
import os
import re
from pathlib import Path

import time

from dotenv import load_dotenv
from google import genai

from prompts import CLAIM_EXTRACTION_PROMPT_V2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def _create_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY was not found in {ENV_PATH}. "
            "Add GEMINI_API_KEY=your_key to the .env file."
        )

    return genai.Client(api_key=api_key)


def _parse_claims(raw_output: str) -> list[str]:
    """Parse a JSON list, including Markdown-fenced JSON."""
    cleaned = raw_output.strip()

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        cleaned = match.group(1).strip()

    claims = json.loads(cleaned)

    if not isinstance(claims, list):
        raise ValueError("Expected a JSON list.")

    if not all(isinstance(claim, str) for claim in claims):
        raise ValueError("Every claim must be a string.")

    return [claim.strip() for claim in claims if claim.strip()]




def extract_claims(
    text: str,
    model: str | None = None,
) -> list[str]:
    """Extract specific factual claims from paper text."""
    if not text.strip():
        return []

    model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    client = _create_client()
    prompt = CLAIM_EXTRACTION_PROMPT_V2.format(text=text)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},
            )

            raw_output = (response.text or "").strip()

            if not raw_output:
                return []

            return _parse_claims(raw_output)

        except Exception as error:
            error_text = str(error)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Gemini is temporarily unavailable after 3 attempts. "
                    "Please wait a few minutes and try again."
                ) from error

            wait_seconds = 5 * (attempt + 1)
            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)



if __name__ == "__main__":
    sample_text = """
    Our method achieved 94% accuracy, outperforming the baseline by 6%.
    """

    claims = extract_claims(sample_text)

    print(f"\nExtracted {len(claims)} claims:\n")
    for index, claim in enumerate(claims, start=1):
        print(f"{index}. {claim}")