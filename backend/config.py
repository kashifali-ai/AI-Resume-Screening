"""Central configuration. Every tunable lives here and is overridable via env
vars or the project-root .env file — nothing is hardcoded in business logic."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (extraction only) ---
    llm_provider: str = "gemini"
    gemini_api_key: str = ""                 # from env GEMINI_API_KEY
    gemini_model: str = "gemini-2.5-flash"   # override via env GEMINI_MODEL
    # When true, a deterministic rule-based MockLLM replaces Gemini for BOTH
    # resume and JD extraction — no Gemini calls, no quota, no API key needed.
    # When false, Gemini is used exactly as before.
    mock_llm: bool = False                    # set MOCK_LLM=true to enable

    # --- Embeddings (semantic skill matching) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sim_threshold: float = 0.55          # cosine >= this counts as a skill match
    dedup_threshold: float = 0.85        # cosine >= this collapses duplicate skills

    # --- Scoring / verdict ---
    # The role and its required skills/experience come entirely from the pasted
    # job description — nothing role-specific is hardcoded here.
    fit_threshold: float = 60.0          # overall score >= this -> FIT
    required_skill_weight: float = 1.0   # weight of a JD "required" skill
    preferred_skill_weight: float = 0.5  # weight of a JD "preferred"/nice-to-have skill
    overqualified_factor: float = 0.9    # multiplier when exp exceeds the JD's max
    evidence_confidence: float = 0.5     # confidence for a skill with no supporting text

    # --- Anti keyword-stuffing ---
    stuffing_max_repeats: int = 8        # a skill term repeated more than this is suspicious
    stuffing_low_evidence_ratio: float = 0.6
    stuffing_min_skills_for_check: int = 10
    stuffing_penalty_cap: float = 0.5

    # --- I/O ---
    max_file_mb: int = 5
    max_resume_chars: int = 40_000       # cap resume text sent to the LLM
    max_jd_chars: int = 20_000           # cap job-description text sent to the LLM
    max_bulk_resumes: int = 50           # max resumes in one bulk screening request
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
