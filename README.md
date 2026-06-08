# Resume Screening (LLM + Embeddings)

Upload any resume on the landing page; the backend screens it against a built-in
**SDE** role and returns an explainable **FIT / UNFIT** report with a score.

This is a **hybrid** system — not a keyword matcher:

1. **Parse** — extract text from PDF / DOCX (incl. tables) / TXT.
2. **Extract** — **Google Gemini** turns the text into structured JSON
   (name, email, skills, experience, projects, education, certifications,
   technologies) via native structured output. The LLM *only* extracts; it never
   scores or decides.
3. **Normalize / anti-gaming** — dedup skills, grade each skill by whether it's
   backed by real experience/project text, detect keyword stuffing, flag missing
   sections.
4. **Semantic match** — `sentence-transformers` embeddings match skills by meaning,
   so *Spring Framework ≈ Spring Boot*, *REST Services ≈ REST APIs*, *JPA ≈
   Hibernate*, *ReactJS ≈ React*.
5. **Score (pure Python)** — weighted skill coverage × experience-window fit ×
   (1 − stuffing penalty). The FIT/UNFIT verdict is computed deterministically in
   Python — **never taken from the LLM**.
6. **Report** — match %, matched/missing skills, strengths, weaknesses,
   recommendation, and reasoning, all grounded in the extracted data.

## Stack
- **Backend:** Python + FastAPI (modular pipeline, logging, config, tests)
- **Frontend:** HTML + Tailwind (single upload button)
- **LLM:** **Google Gemini** — provider-abstracted in `llm_client.py`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally)

## Prerequisites
- Python 3.12
- A **Gemini API key** from https://aistudio.google.com/apikey

## Setup
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pulls torch — large, one-time
export GEMINI_API_KEY=...                 # or put it in the project-root .env
uvicorn main:app --reload                # http://127.0.0.1:8000
```
First request loads the embedding model (~90 MB) once.

## Tests
```bash
cd backend && source .venv/bin/activate
pytest -q     # LLM is mocked; embeddings + scoring run for real
```

## Configuration
All tunables live in `backend/config.py` and are overridable via env vars or a
project-root `.env` (see `.env.example`): model names, similarity/FIT thresholds,
experience window, anti-stuffing limits, file-size cap. Nothing is hardcoded in
the business logic.

## Project layout
```
my-project/
├── backend/
│   ├── main.py            # FastAPI routes, logging, error handling
│   ├── config.py          # pydantic-settings config (env-overridable)
│   ├── role_profile.py    # the SDE reference (weighted required skills)
│   ├── models.py          # Pydantic models (CandidateProfile, ScreeningReport, ...)
│   ├── resume_parser.py   # PDF/DOCX(+tables)/TXT -> text
│   ├── llm_client.py      # provider abstraction (Google Gemini)
│   ├── extraction.py      # LLM structured extraction (extract-only)
│   ├── normalize.py       # dedup, evidence grounding, anti-stuffing
│   ├── semantic.py        # sentence-transformers skill matching
│   ├── scoring.py         # deterministic score + verdict (no LLM)
│   ├── report.py          # explainable report assembly
│   ├── screening.py       # pipeline orchestrator
│   ├── conftest.py        # test fixtures
│   ├── tests/             # pytest suite (semantic, scoring, normalize, parser, e2e)
│   └── requirements.txt
└── frontend/
    └── index.html         # landing page (Tailwind, one upload button)
```
