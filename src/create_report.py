"""Create a Markdown report from local fact-checking results."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PAPER_ID = "lora"


def create_report(paper_id: str) -> Path:
    results_path = RESULTS_DIR / f"{paper_id}_local_results.json"
    report_path = RESULTS_DIR / f"{paper_id}_report.md"

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))

    counts = {
        "supported": 0,
        "contradicted": 0,
        "unaddressed": 0,
    }

    for result in results:
        verdict = result.get("verdict", "unaddressed")
        counts[verdict] = counts.get(verdict, 0) + 1

    lines = [
        f"# Fact-Checking Report: {paper_id}",
        "",
        "## Summary",
        "",
        f"- Total claims: **{len(results)}**",
        f"- Supported: **{counts['supported']}**",
        f"- Contradicted: **{counts['contradicted']}**",
        f"- Unaddressed: **{counts['unaddressed']}**",
        "- Verification mode: **local retrieval**",
        "",
        "## Claim Results",
        "",
    ]

    for index, result in enumerate(results, start=1):
        claim = result.get("claim", "")
        verdict = result.get("verdict", "unaddressed")
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "")
        evidence = result.get("evidence_chunk", "")

        lines.extend(
            [
                f"### {index}. {claim}",
                "",
                f"**Verdict:** `{verdict}`  ",
                f"**Confidence:** `{confidence}`",
                "",
                f"**Reason:** {reason}",
                "",
                "**Evidence:**",
                "",
                f"> {evidence.replace(chr(10), ' ')}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = create_report(PAPER_ID)
    print(f"Report created: {path}")
    