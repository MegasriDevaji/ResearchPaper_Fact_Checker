"""
prompts.py
Central place for all agent prompts. Keeping versions here
lets us document prompt iteration for the report.
"""

# ---------- CLAIM EXTRACTION ----------

CLAIM_EXTRACTION_PROMPT_V1 = """Extract factual claims from this research paper text.

TEXT:
{text}

List the claims."""


CLAIM_EXTRACTION_PROMPT_V2 = """You are a research paper analyst. Your task is to extract SPECIFIC, CHECKABLE factual claims from the text below.

A good claim contains a concrete number, comparison, or measurable result (e.g. "reduces parameters by 10,000x", "achieves 94% accuracy").
A bad claim is vague and unmeasurable (e.g. "the method is effective", "this is an important contribution").

Only extract claims that are explicitly stated in the text. Do not infer or add information.

TEXT:
{text}

Return your answer as a JSON list of strings, one claim per string. Return ONLY the JSON list, nothing else.
Example format: ["claim 1 text", "claim 2 text"]
"""