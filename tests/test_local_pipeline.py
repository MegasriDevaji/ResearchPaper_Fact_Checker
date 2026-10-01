import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from run_fact_checker import extract_local_claims, verify_locally


def test_extract_local_claims_ignores_headers():
    text = """
    --- PAGE 1 ---
    ABSTRACT
    LoRA reduces trainable parameters by 10000 times.
    """

    claims = extract_local_claims(text)

    assert len(claims) == 1
    assert "LoRA reduces" in claims[0]


def test_verify_supported_claim():
    result = verify_locally(
        "LoRA reduces trainable parameters by 10000 times.",
        [{"text": "LoRA reduces the number of trainable parameters by 10000 times."}],
    )

    assert result["verdict"] == "supported"
    assert result["confidence"] > 0


def test_verify_missing_number_as_unaddressed():
    result = verify_locally(
        "LoRA reduces trainable parameters by 10000 times.",
        [{"text": "LoRA reduces the number of trainable parameters for downstream tasks."}],
    )

    assert result["verdict"] == "unaddressed"


def test_verify_empty_evidence():
    result = verify_locally("LoRA improves performance.", [])

    assert result["verdict"] == "unaddressed"
    assert result["confidence"] == 0.0