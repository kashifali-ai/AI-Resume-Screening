# JobFit — Resume vs Job Description (LLM + Embeddings)

Paste **any job description** and upload **any resume**. The backend evaluates
how well the resume fits *that specific job* and returns an explainable
**FIT / UNFIT** report with a score — so a candidate can check their fit before
applying.

```
Job Description + Resume  ->  FIT / UNFIT (+ score, gaps, recommendations)
```

There is **no built-in role**. The role, its required/preferred skills, its
technologies, and its experience requirement all come from the pasted job
description. This is a **hybrid** system — not a keyword matcher:

1. **Parse** — extract text from the resume (PDF / DOCX incl. tables / TXT).
2. **Extract (LLM, twice)** — **Google Gemini** turns free text into structured
   JSON via native structured output:
   - the **resume** → `CandidateProfile` (name, summary, skills, technologies,
     experience, projects, education, certifications);
   - the **job description** → `JobRequirements` (role title, required skills,
     preferred skills, technologies, min/max experience years, education
     requirements, responsibilities).
   The LLM *only extracts*. It never scores, ranks, compares, or decides.
3. **Normalize / anti-gaming** — dedup skills, grade each skill by whether it's
   backed by real experience/project text, detect keyword stuffing, flag missing
   sections.
4. **Semantic match** — `sentence-transformers` embeddings match the candidate's
   skills against the JD's skills *by meaning*, so *Spring Framework ≈ Spring
   Boot*, *REST Services ≈ REST APIs*, *ReactJS ≈ React*, *PostgreSQL ≈ Postgres*.
5. **Score (pure Python)** — weighted skill coverage × experience-window fit ×
   (1 − stuffing penalty). Required skills weigh more than preferred ones; the
   experience window comes from the JD. The FIT/UNFIT verdict is computed
   deterministically in Python — **never taken from the LLM**.
6. **Report** — match %, matched/missing skills, strengths, weaknesses,
   experience comparison, resume-improvement recommendations, and reasoning —
   all grounded in the extracted data.

Works for Software Engineer, Backend/Frontend Developer, Data Analyst, Data
Engineer, Product Manager, QA Engineer, DevOps Engineer, or any other role —
because the requirements are read from the JD, not hardcoded.

## Login / authentication

The whole app sits behind a login screen — you must sign in before any screening
feature is reachable.

**Demo account** (seeded automatically on first run):

| Email | Password |
|-------|----------|
| `admin@test.com` | `admin123` |

- Visiting any page while logged out redirects to **`/login`**.
- Sign in with email + password → redirected to the JobFit dashboard, where all
  features work (single screening, bulk screening, CSV export, MockLLM and Gemini
  modes).
- **Log out** (top-right) clears the session; protected APIs return `401` again.

**How it works**
- **Session-based auth** via a signed cookie (Starlette `SessionMiddleware` +
  `itsdangerous`); the cookie only holds the user's email.
- Passwords are hashed with **PBKDF2-HMAC-SHA256** + a per-user random salt
  (`auth.py`) and stored in `users.json` (gitignored) — **plaintext is never
  stored**.
- Protected endpoints (`/api/screen`, `/api/screen-bulk`, and the bulk CSV
  export they serve) require a valid session via a `current_user` dependency and
  return `401` otherwise. `/api/auth/login`, `/api/auth/logout`, `/api/auth/me`
  manage the session; `/api/health` stays public (status only, no data).

**Config** (env-overridable, in `config.py`): `SESSION_SECRET` (set a strong
value in production), `SESSION_COOKIE`, `USERS_DB`, `DEMO_EMAIL`, `DEMO_PASSWORD`.

```bash
# Add a user from a shell (after activating the venv, from backend/):
python -c "from auth import get_user_store; from config import get_settings; \
get_user_store(get_settings()).add_user('you@example.com','your-password')"
```

## Bulk screening (one JD → many resumes)

The same job description is often screened against dozens of resumes. To avoid
re-paying for JD extraction every time, the extracted `JobRequirements` (and its
embedding-built weighted requirement list) are **cached by JD text**:

- The first time a JD is seen, Gemini extracts it **once** and the result is cached.
- Every subsequent resume for that JD — in the same batch *or* a later request —
  **reuses the cache and does not call Gemini for the JD again**.
- Each resume is still extracted individually (they differ), screened
  independently (one bad file never aborts the batch), and the deterministic
  Python scoring is **unchanged**.

So screening *N* resumes against one JD costs **1 JD Gemini call + N resume
calls**, not *N* + *N*. Upload up to **50 resumes** at once; results come back as
a table **ranked by score (highest first)** with a **Download CSV** button.

Each ranked row shows: candidate name, match score, FIT/UNFIT, matched-skill
count, missing-skill count, experience, and recommendation.

## MockLLM mode (offline, zero Gemini calls)

For testing, demos, and CI you can run the **entire** pipeline — including bulk
screening of up to 50 resumes — without calling Gemini at all.

**Enable it** (one of):
```bash
# .env  (or export before launching)
MOCK_LLM=true
```
When `MOCK_LLM=true`:
- **No Gemini calls, no quota used, no `GEMINI_API_KEY` required.**
- A deterministic, rule-based `MockLLMClient` (`mock_llm.py`) replaces Gemini for
  **both** resume and JD extraction:
  - *Resume →* `CandidateProfile` parsed from the resume's own sections (name,
    email, summary, skills/technologies, dated experience, projects, education,
    certifications).
  - *JD →* `JobRequirements` parsed from the JD (role title, required/preferred
    skills, technologies, `N+`/`N-M years`, education, responsibilities).
- It logs `[MOCK_LLM] Resume extracted`, `[MOCK_LLM] JD extracted`, and
  `[MOCK_LLM] No Gemini call performed`.

**Everything else is identical.** Only the extraction provider changes — the
normalization, **embedding-based** semantic matching, deterministic Python
scoring, ranking, and report generation are unchanged. (The mock's rule-based
parsing is confined to *extraction*; skill *matching* is still done by embeddings
in `semantic.py`, never by substring rules.)

**When `MOCK_LLM=false` (default), Gemini is used exactly as before.**

| | `MOCK_LLM=true` (mock) | `MOCK_LLM=false` (Gemini) |
|---|---|---|
| Extraction | rule-based, offline | Google Gemini structured output |
| API key / quota | none needed | `GEMINI_API_KEY` required, uses quota |
| Determinism | fully deterministic | near-deterministic (`temperature=0`) |
| Extraction quality | good on clearly-sectioned text; weaker on messy/freeform resumes | strong on messy, real-world resumes |
| Best for | tests, demos, CI, bulk dry-runs | production screening |
| Matching & scoring | **identical** (embeddings + Python) | **identical** (embeddings + Python) |

```bash
# Demo bulk screening with no Gemini at all:
MOCK_LLM=true uvicorn main:app --reload     # then upload many resumes in the UI
```

## Architecture (stage → file)

| Stage | File | Responsibility |
|------|------|----------------|
| 1. Parse | `resume_parser.py` | PDF/DOCX(+tables)/TXT → text; raises on empty/image-only |
| 2. Extract | `extraction.py` + `llm_client.py` (+ `mock_llm.py`) | Gemini structured output → `CandidateProfile` and `JobRequirements` (extract-only); `MockLLMClient` provides the same shapes offline when `MOCK_LLM=true` |
| 3. Normalize | `normalize.py` | dedup, evidence confidence, anti-stuffing, missing-section flags |
| 4. Match | `semantic.py` | embedding cosine between JD skills and candidate skills |
| 5. Score | `scoring.py` | pure-Python deterministic score + FIT/UNFIT verdict |
| 6. Report | `report.py` | grounded strengths/weaknesses/recommendations/reasoning |
| — Orchestrate | `screening.py` | `get_requirements()` (JD-cached), `screen()` (single), `screen_bulk()` (many, ranked); injectable `settings`/`llm`/`embedder` |
| — Auth | `auth.py` + `main.py` | PBKDF2 password hashing, JSON user store, session cookie, `current_user` gate |
| — API | `main.py` | `POST /api/screen` (single), `POST /api/screen-bulk` (many) — both login-gated; auth + `GET /api/health` public |

### JD caching (how it works)

`screening.py` keeps a process-level cache keyed by `sha256(jd_text)`:
`get_requirements(jd_text, …)` returns `(JobRequirements, weighted_requirements,
cached)`. On a cache hit the LLM is **not** called. Both `screen()` and
`screen_bulk()` go through it, so single and bulk runs share the same cache.
`clear_jd_cache()` empties it (used by tests). The cache is in-memory per process
(resets on restart); swap in Redis/etc. behind `get_requirements` if you need it
shared or persistent.

### How Gemini is used (extract-only)

`llm_client.py` calls Gemini with `response_mime_type="application/json"` and a
Pydantic model as `response_schema`, at `temperature=0`. The model returns
schema-shaped JSON directly. It is given two narrow extraction prompts (one for
the resume, one for the JD) that **forbid** judging, scoring, or comparing.
Swapping providers is a one-file change behind the `LLMClient` protocol.

### How deterministic scoring works (no LLM)

`scoring.py` contains no model calls. Given the matched skills (with per-skill
weight and evidence confidence), the candidate's computed experience years, and
the stuffing penalty:

```
skill_coverage = Σ(weight × confidence for matched skills) / Σ(weight for all required+preferred)
experience_factor = 1.0 if within the JD's window
                    overqualified_factor if above the JD's max
                    0.0 if below the JD's min   (None min ⇒ not gated)
score = skill_coverage × experience_factor × (1 − stuffing_penalty) × 100
verdict = FIT  if score ≥ fit_threshold AND experience meets the JD minimum
          UNFIT otherwise
```

Same inputs always produce the same verdict. The LLM cannot change it.

## Stack
- **Backend:** Python + FastAPI (modular pipeline, logging, config, tests)
- **Frontend:** HTML + Tailwind — split screen: JD textarea (left), resume upload
  (right, multi-select), **Analyze Match** button. One resume → full single
  report; many → ranked table + **Download CSV** (built client-side).
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

Open http://127.0.0.1:8000 and sign in with the demo account
**`admin@test.com` / `admin123`** (see [Login / authentication](#login--authentication)).

## API

All `/api/screen*` endpoints require an authenticated session (else `401`).

### Auth
- `POST /api/auth/login` — form `email`, `password` → sets session cookie.
- `POST /api/auth/logout` — clears the session.
- `GET /api/auth/me` — current user, or `401`.

### `POST /api/screen` — single resume (`multipart/form-data`)
- `job_description` (text, required)
- `resume` (file: PDF/DOCX/TXT, required)

Returns the full `ScreeningReport` JSON: `verdict`, `role_title`, `score`,
`match_percentage`, `matched_skills`, `missing_skills`, `strengths`,
`weaknesses`, `recommendation`, `recommendations`, `reasoning`, `flags`,
`experience_years`, `experience_ok`, `experience_required`,
`experience_comparison`, `candidate_resume`, `job_requirements`, `skill_matches`.

### `POST /api/screen-bulk` — many resumes (`multipart/form-data`)
- `job_description` (text, required)
- `resumes` (repeated file field, 1–50 files)
- `response_format` (optional: `json` default, or `csv` for a downloadable CSV)

Returns a `BulkScreeningResponse`: `role_title`, `job_requirements`, `jd_cached`
(was the JD served from cache — i.e. Gemini was *not* called for it),
`total` / `succeeded` / `failed`, and `results` — a list **ranked by score
descending** of `BulkResumeResult` rows (`filename`, `ok`, `error`,
`candidate_name`, `score`, `verdict`, `matched_count`, `missing_count`,
`experience_years`, `recommendation`, full `report`). Failed/unreadable resumes
appear as `ok=false` rows at the bottom rather than failing the whole batch.

Without a valid key / network (or on Gemini rate limits), the endpoints return
**503** with a clear message; in bulk mode a per-resume LLM failure becomes a
failed row.

## Tests
```bash
cd backend && source .venv/bin/activate
pytest -q     # LLM is mocked; embeddings + scoring run for real
```
Covers multiple roles (Backend, Data Analyst, Product Manager, DevOps), semantic
synonym matching, missing skills, experience boundaries (incl. "no requirement"),
preferred-vs-required weighting, verdict determinism, and **bulk + JD caching**
(`test_bulk.py`: JD extracted exactly once for N resumes, cross-request cache
hit, ranking order, per-resume failure isolation). `test_mock_llm.py` covers the
offline **MockLLM** path: rule-based JD/resume extraction, a full 50-resume bulk
run with **zero Gemini calls**, ranking, and CSV export.

## Configuration
All tunables live in `backend/config.py` and are overridable via env vars or a
project-root `.env`: model names, similarity/FIT thresholds, required/preferred
skill weights, overqualified factor, anti-stuffing limits, file-size cap, the
resume/JD character caps, and `max_bulk_resumes` (default 50). Nothing
role-specific is hardcoded.

## Project layout
```
my-project/
├── backend/
│   ├── main.py            # FastAPI routes (JD + resume), logging, errors
│   ├── config.py          # pydantic-settings config (env-overridable)
│   ├── models.py          # CandidateProfile, JobRequirements, ScreeningReport, BulkScreeningResponse, ...
│   ├── resume_parser.py   # PDF/DOCX(+tables)/TXT -> text
│   ├── auth.py            # password hashing (PBKDF2) + JSON user store
│   ├── llm_client.py      # provider abstraction (Gemini; MockLLM when MOCK_LLM=true)
│   ├── mock_llm.py        # offline rule-based extraction provider (no Gemini)
│   ├── report_csv.py      # bulk results -> CSV
│   ├── extraction.py      # LLM structured extraction: resume + JD (extract-only)
│   ├── normalize.py       # dedup, evidence grounding, anti-stuffing
│   ├── semantic.py        # sentence-transformers skill matching
│   ├── scoring.py         # deterministic score + verdict (no LLM)
│   ├── report.py          # explainable report assembly
│   ├── screening.py       # orchestrator: get_requirements (JD cache), screen, screen_bulk
│   ├── conftest.py        # test fixtures (FakeLLM dispatches resume vs JD)
│   ├── tests/             # pytest suite
│   └── requirements.txt
└── frontend/
    ├── login.html         # login screen (first page)
    └── index.html         # split-screen JD + resume dashboard (login-gated)
```

## Limitations (honest)
- **Extraction quality bounds everything.** If Gemini lists a skill only under
  "experience" prose and not as a discrete skill, it may not match (we match
  skill-to-skill). Coverage reflects what was *extracted as skills*.
- **No explicit weights in a JD.** Real JDs don't state numeric importance, so
  required vs preferred is a two-tier weighting (config-tunable), not per-skill.
- **Experience is date-span based.** Undated or narrative-only experience can
  under-count years; the JD's experience requirement is whatever Gemini extracts.
- **Single fixed FIT threshold** across all roles; some roles may warrant
  different bars. It is config-tunable but not per-role.
- **No OCR.** Image-only/scanned PDFs are rejected with a clear error.
- **Embedding model is English-centric**; non-English skills match less reliably.
- **Gemini availability.** Upstream rate limits / 503s surface as a 503 to the
  client (no silent fallback, no fabricated results).
