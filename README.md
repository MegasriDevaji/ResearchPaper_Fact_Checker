# Prompt Engineering Project

A research-paper fact-checking pipeline using PDF extraction, semantic retrieval, Gemini verification, and LangGraph orchestration.

## Pipeline

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
FAISS vector retrieval
 ↓
Claim extraction
 ↓
Verifier → Critic → Judge
 ↓
Fact-checking report
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

Create `.env` in the project root:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Never commit or share `.env`.

## Run local fact-checking

This mode does not use Gemini quota:

```powershell
python src\run_fact_checker.py
python src\create_report.py
```

Results are saved to:

```text
results\lora_local_results.json
results\lora_report.md
```

## Run tests

```powershell
python -m pytest tests -q
```

## Run the LangGraph pipeline

```powershell
python src\graph.py
```

The Gemini pipeline requires available API quota. Local mode can be used while quota is unavailable.

## Project structure

```text
data/
  extracted_text/
  chunks/
  vector_store/

src/
  claim_extractor.py
  chunker.py
  critic.py
  create_report.py
  graph.py
  judge.py
  pdf_processor.py
  run_fact_checker.py
  vector_store.py
  verifier.py

tests/
  test_local_pipeline.py

results/
```