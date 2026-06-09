"""Stage 2: LLM-based structured extraction (resume AND job description).

The LLM's ONLY job is to turn messy free text into structured data: a
CandidateProfile from the resume and a JobRequirements from the job description.
It does not score, rank, compare, or decide FIT/UNFIT. The system prompts forbid
inventing data and forbid judging the candidate.
"""

from config import Settings
from llm_client import LLMClient, LLMError
from logging_config import get_logger
from models import CandidateProfile, JobRequirements

log = get_logger(__name__)

_RESUME_SYSTEM_PROMPT = (
    "You are a precise resume information-extraction engine. Extract ONLY "
    "information that is explicitly present in the resume text. Never invent, "
    "infer, or embellish. If a field is absent, leave it empty/null. Do not "
    "rate, score, judge, or compare the candidate to any job — only extract. "
    "Skills and technologies must be concrete (e.g. 'Spring Boot', "
    "'PostgreSQL'), not soft skills. The candidate_summary must be a short, "
    "neutral 1-2 sentence factual summary of the candidate's background drawn "
    "only from the resume. Return JSON only."
)

_JD_SYSTEM_PROMPT = (
    "You are a precise job-description information-extraction engine. Extract "
    "ONLY requirements explicitly stated in the job description. Never invent or "
    "infer requirements that are not written. Distinguish required/must-have "
    "skills from preferred/nice-to-have ones. List concrete technologies "
    "separately. For experience, extract numeric years: 'min_experience_years' "
    "is the minimum years stated (e.g. '3+ years' -> 3), 'max_experience_years' "
    "only if an upper bound is stated; leave both null if no experience "
    "requirement is given. Do NOT evaluate or score any candidate — there is no "
    "candidate here, only a job posting. Return JSON only."
)


def extract_profile(
    resume_text: str, llm: LLMClient, settings: Settings
) -> CandidateProfile:
    """Run the LLM extraction on a resume and validate it into a CandidateProfile."""
    text = resume_text[: settings.max_resume_chars]

    user = (
        "Extract structured data from the following resume. Return JSON matching "
        "the schema exactly.\n\n<resume>\n" + text + "\n</resume>"
    )

    raw = llm.extract_json(_RESUME_SYSTEM_PROMPT, user, CandidateProfile)
    try:
        profile = CandidateProfile.model_validate(raw)
    except Exception as e:
        raise LLMError(f"Resume extraction did not match the schema: {e}") from e

    log.info(
        "Extracted profile: name=%s skills=%d experience=%d projects=%d",
        profile.candidate_name,
        len(profile.skills),
        len(profile.experience),
        len(profile.projects),
    )
    return profile


def extract_requirements(
    jd_text: str, llm: LLMClient, settings: Settings
) -> JobRequirements:
    """Run the LLM extraction on a job description into a JobRequirements."""
    text = jd_text[: settings.max_jd_chars]

    user = (
        "Extract the structured requirements from the following job description. "
        "Return JSON matching the schema exactly.\n\n<job_description>\n"
        + text
        + "\n</job_description>"
    )

    raw = llm.extract_json(_JD_SYSTEM_PROMPT, user, JobRequirements)
    try:
        jd = JobRequirements.model_validate(raw)
    except Exception as e:
        raise LLMError(f"Job-description extraction did not match the schema: {e}") from e

    log.info(
        "Extracted JD: role=%s required=%d preferred=%d tech=%d exp=%s-%s",
        jd.role_title,
        len(jd.required_skills),
        len(jd.preferred_skills),
        len(jd.technologies),
        jd.min_experience_years,
        jd.max_experience_years,
    )
    return jd
