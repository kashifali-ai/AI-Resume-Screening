"""Pipeline orchestrator.

Wires the six stages together. Contains NO matching/scoring logic itself — each
concern lives in its own module. All keyword/regex skill detection from the old
implementation has been removed.
"""

from config import Settings, get_settings
from extraction import extract_profile
from llm_client import LLMClient, get_llm_client
from logging_config import get_logger
from models import ScreeningReport
from normalize import (
    dedup_skills,
    detect_stuffing,
    missing_section_flags,
    skill_confidence,
)
from report import build_report
from role_profile import required_skills, role_profile_dict
from scoring import score
from semantic import Embedder, match_required_skills

log = get_logger(__name__)


def screen_resume(
    resume_text: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> ScreeningReport:
    """Run the full pipeline on already-extracted resume text.

    Dependencies (settings / llm / embedder) are injectable so tests can supply
    fakes without touching Ollama or downloading the embedding model.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("The resume appears to be empty — no text could be read.")

    settings = settings or get_settings()
    llm = llm or get_llm_client(settings)
    embedder = embedder or Embedder(settings.embedding_model)

    # Stage 2: structured extraction (LLM)
    profile = extract_profile(resume_text, llm, settings)

    # Stage 3: normalize + anti-gaming
    candidate_skills = dedup_skills(
        profile.skills + profile.technologies, embedder, settings.dedup_threshold
    )
    profile.skills = candidate_skills
    confidence = skill_confidence(candidate_skills, profile, settings)
    stuffing_penalty, stuffing_flags = detect_stuffing(
        resume_text, candidate_skills, confidence, settings
    )
    flags = missing_section_flags(profile) + stuffing_flags

    # Stage 4: semantic skill matching (embeddings)
    matches = match_required_skills(
        required_skills(), candidate_skills, embedder, settings.sim_threshold
    )

    # Stage 5: deterministic scoring + verdict (pure Python)
    scored = score(matches, confidence, profile, stuffing_penalty, settings)

    # Stage 6: explainable report
    reference = role_profile_dict(settings.min_experience, settings.max_experience)
    report = build_report(scored, profile, reference, flags, settings)

    log.info(
        "Screening complete: verdict=%s score=%d coverage=%.0f%% exp=%.1fyr",
        report.verdict, report.score, report.match_percentage, report.experience_years,
    )
    return report
