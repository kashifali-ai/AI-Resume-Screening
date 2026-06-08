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

    # --- Embeddings (semantic skill matching) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sim_threshold: float = 0.55          # cosine >= this counts as a skill match
    dedup_threshold: float = 0.85        # cosine >= this collapses duplicate skills

    # --- Scoring / verdict ---
    fit_threshold: float = 60.0          # overall score >= this -> FIT
    min_experience: float = 1.0
    max_experience: float = 5.0
    overqualified_factor: float = 0.9    # multiplier when exp > max_experience
    evidence_confidence: float = 0.5     # confidence for a skill with no supporting text

    # --- Anti keyword-stuffing ---
    stuffing_max_repeats: int = 8        # a skill term repeated more than this is suspicious
    stuffing_low_evidence_ratio: float = 0.6
    stuffing_min_skills_for_check: int = 10
    stuffing_penalty_cap: float = 0.5

    # --- I/O ---
    max_file_mb: int = 5
    max_resume_chars: int = 40_000       # cap text sent to the LLM
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
