# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Resume vs Job-Description screening web app. A split-screen landing page lets a
user paste **any job description** (left) and upload **any resume** (right); the
backend evaluates the resume against *that JD* and returns an explainable
**FIT / UNFIT** report with a score.

There is **no built-in role**. The role and all its requirements come from the
pasted JD. It is a **hybrid LLM + embeddings + deterministic-scoring** system,
not a keyword matcher. **Google Gemini** only extracts structured data (from both
the resume and the JD); embeddings do semantic skill matching; **Python computes
the score and verdict deterministically — the verdict is never taken from the
LLM.**

## Tech Stack

- Backend: Python + FastAPI (`backend/`)
- Frontend: HTML + Tailwind CDN — `frontend/login.html` (login gate) +
  `frontend/index.html` (split-screen JD + resume dashboard)
- Auth: session cookie (Starlette `SessionMiddleware`) + PBKDF2 password hashing
  in `auth.py`; `/api/screen*` are gated by a `current_user` dependency. Demo
  account `admin@test.com` / `admin123` is seeded into `users.json` (gitignored).
- LLM: **Google Gemini** (`GEMINI_API_KEY`), abstracted in `llm_client.py`.
  `MOCK_LLM=true` swaps in an offline rule-based provider (`mock_llm.py`) for
  extraction only — no Gemini calls/quota/key; matching & scoring are unchanged.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)

## Pipeline (stage → file)

1. Parse — `resume_parser.py` (PDF/DOCX incl. tables/TXT; raises on empty/image-only)
2. Extract — `extraction.py` + `llm_client.py` (Gemini `response_schema`) →
   `CandidateProfile` (resume) and `JobRequirements` (JD), both extract-only
3. Normalize / anti-gaming — `normalize.py` (dedup, evidence confidence, stuffing, missing sections)
4. Semantic match — `semantic.py` (embedding cosine: JD skills vs candidate skills)
5. Score + verdict — `scoring.py` (pure Python, deterministic; JD-derived weights + experience window)
6. Report — `report.py` (grounded strengths/weaknesses/recommendations/reasoning)

Orchestrated by `screening.py`: `screen(jd_text, resume_text)` (single) and
`screen_bulk(jd_text, resumes)` (many, ranked), both via `get_requirements()`
which **caches extracted `JobRequirements` by JD text** so one JD costs one Gemini
call across a whole batch. All accept injectable `settings` / `llm` / `embedder`
so tests run without calling Gemini. `clear_jd_cache()` resets the cache.

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
- **No hardcoded role.** Requirements (skills/experience/role title) come only
  from the pasted JD via `JobRequirements`. Do not reintroduce a fixed skill list
  or role profile.
- **No keyword/regex skill detection.** Skill equivalence is embedding-based in
  `semantic.py`. Do not reintroduce substring matching for skills. (The MockLLM's
  rule-based parsing lives in `mock_llm.py` and only emulates LLM *extraction* —
  never the matching stage.)
- **All tunables in `config.py`** (env-overridable). Don't hardcode thresholds,
  weights, model names, or limits in logic.
- Tests mock the LLM via `conftest.FakeLLM` (dispatches resume vs JD by response
  model); the embedder is a session fixture.

## Notes

- Requires `GEMINI_API_KEY` (env or `.env`). `GET /api/health` reports
  provider/model and whether the key is configured.
- First request loads the embedding model (~90 MB) once; first run downloads it.
- Without a valid key / network, `/api/screen` returns 503 with a clear message.
