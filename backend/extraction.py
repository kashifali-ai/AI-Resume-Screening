"""Stage 2: LLM-based structured extraction.

The LLM's ONLY job is to turn messy resume text into a CandidateProfile. It does
not score, rank, or decide FIT/UNFIT. The system prompt forbids inventing data.
"""

from config import Settings
from llm_client import LLMClient, LLMError
from logging_config import get_logger
from models import CandidateProfile

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise resume information-extraction engine. Extract ONLY "
    "information that is explicitly present in the resume text. Never invent, "
    "infer, or embellish. If a field is absent, leave it empty/null. Do not "
    "rate, score, or judge the candidate — only extract. Skills and "
    "technologies must be concrete (e.g. 'Spring Boot', 'PostgreSQL'), not "
    "soft skills. Return JSON only."
)


def extract_profile(
    resume_text: str, llm: LLMClient, settings: Settings
) -> CandidateProfile:
    """Run the LLM extraction and validate it into a CandidateProfile."""
    text = resume_text[: settings.max_resume_chars]

    user = (
        "Extract structured data from the following resume. Return JSON matching "
        "the schema exactly.\n\n<resume>\n" + text + "\n</resume>"
    )

    raw = llm.extract_json(_SYSTEM_PROMPT, user, CandidateProfile)
    try:
        profile = CandidateProfile.model_validate(raw)
    except Exception as e:
        raise LLMError(f"Extraction did not match the expected schema: {e}") from e

    log.info(
        "Extracted profile: name=%s skills=%d experience=%d projects=%d",
        profile.candidate_name,
        len(profile.skills),
        len(profile.experience),
        len(profile.projects),
    )
    return profile
