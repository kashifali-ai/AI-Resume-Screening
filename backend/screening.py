"""Pipeline orchestrator.

Wires the stages together. Contains NO matching/scoring logic itself — each
concern lives in its own module. There is no hardcoded role: the requirements
come entirely from the pasted job description, extracted by the LLM.
"""

from config import Settings, get_settings
from extraction import extract_profile, extract_requirements
from llm_client import LLMClient, get_llm_client
from logging_config import get_logger
from models import JobRequirements, ScreeningReport
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


def screen(
    jd_text: str,
    resume_text: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> ScreeningReport:
    """Run the full pipeline: evaluate a resume against a job description.

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

    # Stage 2: structured extraction (LLM) — resume AND job description.
    profile = extract_profile(resume_text, llm, settings)
    jd = extract_requirements(jd_text, llm, settings)

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

    # Build the weighted requirement list from the JD (no hardcoded role).
    requirements = _build_requirements(jd, embedder, settings)
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
    report = build_report(scored, profile, jd, flags, settings)

    log.info(
        "Screening complete: role=%s verdict=%s score=%d coverage=%.0f%% exp=%.1fyr",
        report.role_title, report.verdict, report.score,
        report.match_percentage, report.experience_years,
    )
    return report
