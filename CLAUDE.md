# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Resume Screening web app. A landing page with a single upload button lets a user
upload any resume; the backend screens it against a built-in **SDE** role and
returns an explainable **FIT / UNFIT** report with a score.

It is a **hybrid LLM + embeddings + deterministic-scoring** system, not a keyword
matcher. **Google Gemini** only extracts structured data; embeddings do semantic
skill matching; **Python computes the score and verdict deterministically — the
verdict is never taken from the LLM.**

## Tech Stack

- Backend: Python + FastAPI (`backend/`)
- Frontend: HTML + Tailwind CDN (`frontend/index.html`)
- LLM: **Google Gemini** (`GEMINI_API_KEY`), abstracted in `llm_client.py`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)

## Pipeline (stage → file)

1. Parse — `resume_parser.py` (PDF/DOCX incl. tables/TXT; raises on empty/image-only)
2. Extract — `extraction.py` + `llm_client.py` (Gemini `response_schema`) → `CandidateProfile` (extract-only)
3. Normalize / anti-gaming — `normalize.py` (dedup, evidence confidence, stuffing, missing sections)
4. Semantic match — `semantic.py` (embedding cosine vs `role_profile.py` skills)
5. Score + verdict — `scoring.py` (pure Python, deterministic)
6. Report — `report.py` (grounded strengths/weaknesses/recommendation/reasoning)

Orchestrated by `screening.py::screen_resume`, which accepts injectable
`settings` / `llm` / `embedder` so tests can run without Ollama.

## Setup & Commands

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...         # or project-root .env
uvicorn main:app --reload        # http://127.0.0.1:8000
pytest -q                        # LLM mocked; embeddings + scoring run for real
```

## Conventions

- **The LLM extracts; Python decides.** Never let the verdict/score come from the
  model. Keep `scoring.py` deterministic and free of model calls.
- **No keyword/regex skill detection.** Skill equivalence is embedding-based in
  `semantic.py`. Do not reintroduce substring matching for skills.
- **All tunables in `config.py`** (env-overridable). Don't hardcode thresholds,
  weights, model names, or limits in logic.
- To retarget to another role, edit `role_profile.py` (weighted skills) only.
- Tests mock the LLM via `conftest.FakeLLM`; the embedder is a session fixture.

## Notes

- Requires `GEMINI_API_KEY` (env or `.env`). `GET /api/health` reports
  provider/model and whether the key is configured.
- First request loads the embedding model (~90 MB) once; first run downloads it.
- Without a valid key / network, `/api/screen` returns 503 with a clear message.
