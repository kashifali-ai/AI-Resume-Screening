"""Pipeline orchestrator.

Wires the stages together. Contains NO matching/scoring logic itself — each
concern lives in its own module. There is no hardcoded role: the requirements
come entirely from the pasted job description, extracted by the LLM.

Bulk mode evaluates many resumes against ONE job description. The JD is
extracted (and its weighted requirement list built) exactly once and cached by
JD text, so screening N resumes costs 1 JD Gemini call + N resume calls — not
N JD calls. Deterministic Python scoring is unchanged.
"""

import hashlib
import threading

from config import Settings, get_settings
from extraction import extract_profile, extract_requirements
from llm_client import LLMClient, get_llm_client
from logging_config import get_logger
from models import (
    BulkResumeResult,
    BulkScreeningResponse,
    JobRequirements,
    ScreeningReport,
)
from normalize import (
    dedup_skills,
    detect_stuffing,
    missing_section_flags,
    skill_confidence,
)
from report import build_report
from scoring import score
from semantic import Embedder, match_skills

log = get_logger(__name__)

# --- Job-description requirements cache ------------------------------------
# Keyed by a hash of the JD text. Stores the extracted JobRequirements AND the
# embedding-built weighted requirement list, so a repeated JD skips both the
# Gemini call and the embedding dedup work.
_jd_cache: dict[str, tuple[JobRequirements, list[dict]]] = {}
_jd_cache_lock = threading.Lock()


def _jd_key(jd_text: str) -> str:
    return hashlib.sha256(jd_text.strip().encode("utf-8")).hexdigest()


def clear_jd_cache() -> None:
    """Empty the JD cache (used by tests and available for ops)."""
    with _jd_cache_lock:
        _jd_cache.clear()


def _build_requirements(
    jd: JobRequirements, embedder: Embedder, settings: Settings
) -> list[dict]:
    """Turn extracted JD fields into a weighted requirement list.

    Required skills + technologies carry the required weight; preferred skills
    carry the (lower) preferred weight. Near-duplicates are collapsed via
    embeddings, and preferred terms that duplicate a required term are dropped.
    """
    required_terms = dedup_skills(
        jd.required_skills + jd.technologies, embedder, settings.dedup_threshold
    )
    preferred_terms = dedup_skills(
        jd.preferred_skills, embedder, settings.dedup_threshold
    )
    req_lower = {t.lower() for t in required_terms}
    preferred_terms = [t for t in preferred_terms if t.lower() not in req_lower]

    return [
        {"skill": s, "kind": "required", "weight": settings.required_skill_weight}
        for s in required_terms
    ] + [
        {"skill": s, "kind": "preferred", "weight": settings.preferred_skill_weight}
        for s in preferred_terms
    ]


def get_requirements(
    jd_text: str,
    llm: LLMClient,
    settings: Settings,
    embedder: Embedder,
) -> tuple[JobRequirements, list[dict], bool]:
    """Return (job_requirements, weighted_requirements, cached).

    On a cache hit the LLM is NOT called. This is what makes bulk screening cheap
    and what guarantees one JD is extracted only once."""
    key = _jd_key(jd_text)
    with _jd_cache_lock:
        hit = _jd_cache.get(key)
    if hit is not None:
        jd, requirements = hit
        log.info("JD cache hit (role=%s) — skipping Gemini JD extraction.", jd.role_title)
        return jd, requirements, True

    jd = extract_requirements(jd_text, llm, settings)
    requirements = _build_requirements(jd, embedder, settings)
    with _jd_cache_lock:
        _jd_cache[key] = (jd, requirements)
    return jd, requirements, False


def _screen_one(
    jd: JobRequirements,
    requirements: list[dict],
    resume_text: str,
    *,
    settings: Settings,
    llm: LLMClient,
    embedder: Embedder,
) -> ScreeningReport:
    """Score a single resume against already-extracted JD requirements."""
    # Stage 2: structured extraction of the resume (LLM).
    profile = extract_profile(resume_text, llm, settings)

    # Stage 3: normalize candidate skills + anti-gaming.
    candidate_skills = dedup_skills(
        profile.skills + profile.technologies, embedder, settings.dedup_threshold
    )
    profile.skills = candidate_skills
    confidence = skill_confidence(candidate_skills, profile, settings)
    stuffing_penalty, stuffing_flags = detect_stuffing(
        resume_text, candidate_skills, confidence, settings
    )
    flags = missing_section_flags(profile) + stuffing_flags
    if not requirements:
        flags.append(
            "No skill requirements could be extracted from the job description."
        )

    # Stage 4: semantic skill matching (embeddings).
    matches = match_skills(
        requirements, candidate_skills, embedder, settings.sim_threshold
    )

    # Stage 5: deterministic scoring + verdict (pure Python).
    scored = score(matches, confidence, profile, stuffing_penalty, jd, settings)

    # Stage 6: explainable report.
    return build_report(scored, profile, jd, flags, settings)


def screen(
    jd_text: str,
    resume_text: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> ScreeningReport:
    """Evaluate a single resume against a job description (existing workflow).

    Dependencies (settings / llm / embedder) are injectable so tests can supply
    fakes without calling Gemini or downloading the embedding model.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("The job description is empty — paste a job description.")
    if not resume_text or not resume_text.strip():
        raise ValueError("The resume appears to be empty — no text could be read.")

    settings = settings or get_settings()
    llm = llm or get_llm_client(settings)
    embedder = embedder or Embedder(settings.embedding_model)

    jd, requirements, _ = get_requirements(jd_text, llm, settings, embedder)
    report = _screen_one(
        jd, requirements, resume_text, settings=settings, llm=llm, embedder=embedder
    )
    log.info(
        "Screening complete: role=%s verdict=%s score=%d coverage=%.0f%% exp=%.1fyr",
        report.role_title, report.verdict, report.score,
        report.match_percentage, report.experience_years,
    )
    return report


def screen_bulk(
    jd_text: str,
    resumes: list[tuple[str, str]],
    *,
    prefailed: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> BulkScreeningResponse:
    """Evaluate many resumes against ONE job description.

    `resumes` is a list of (filename, resume_text). `prefailed` is an optional
    list of (filename, error_message) for files that could not even be parsed —
    they appear in the ranked output as failed rows.

    The JD is extracted once (cached); each resume is screened independently so
    one bad resume never aborts the batch. Results are ranked by score, highest
    first. Scoring/ranking is pure Python — Gemini only extracts.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("The job description is empty — paste a job description.")

    settings = settings or get_settings()
    llm = llm or get_llm_client(settings)
    embedder = embedder or Embedder(settings.embedding_model)

    # JD extracted ONCE for the whole batch.
    jd, requirements, cached = get_requirements(jd_text, llm, settings, embedder)

    results: list[BulkResumeResult] = []
    for filename, text in resumes:
        try:
            if not text or not text.strip():
                raise ValueError("The resume appears to be empty — no text could be read.")
            report = _screen_one(
                jd, requirements, text,
                settings=settings, llm=llm, embedder=embedder,
            )
            results.append(
                BulkResumeResult(
                    filename=filename,
                    ok=True,
                    candidate_name=report.candidate_resume.candidate_name,
                    score=report.score,
                    verdict=report.verdict,
                    matched_count=len(report.matched_skills),
                    missing_count=len(report.missing_skills),
                    experience_years=report.experience_years,
                    recommendation=report.recommendation,
                    report=report,
                )
            )
        except Exception as e:  # noqa: BLE001 — isolate per-resume failures
            log.warning("Resume '%s' failed: %s", filename, e)
            results.append(BulkResumeResult(filename=filename, ok=False, error=str(e)))

    for filename, err in (prefailed or []):
        results.append(BulkResumeResult(filename=filename, ok=False, error=err))

    # Rank: successful rows first, highest score first; failures last.
    results.sort(
        key=lambda r: (r.ok, r.score if r.score is not None else -1),
        reverse=True,
    )

    succeeded = sum(1 for r in results if r.ok)
    response = BulkScreeningResponse(
        role_title=jd.role_title,
        job_requirements=jd,
        jd_cached=cached,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )
    log.info(
        "Bulk screening complete: role=%s resumes=%d ok=%d failed=%d jd_cached=%s",
        jd.role_title, response.total, response.succeeded, response.failed, cached,
    )
    return response
