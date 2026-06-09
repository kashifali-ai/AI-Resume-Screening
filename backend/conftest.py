"""Pytest config: put the backend dir on sys.path and share fixtures."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def embedder(settings):
    """Real embedding model, loaded once for the whole test session."""
    from semantic import Embedder

    return Embedder(settings.embedding_model)


class FakeLLM:
    """Stand-in LLM client that returns preset extraction dicts.

    The pipeline calls the LLM twice — once for the resume (CandidateProfile)
    and once for the job description (JobRequirements). This fake dispatches on
    the requested response model so tests can supply both payloads.
    """

    def __init__(self, resume: dict | None = None, jd: dict | None = None):
        self.resume = resume or {}
        self.jd = jd or {}

    def extract_json(self, system: str, user: str, response_model) -> dict:
        name = getattr(response_model, "__name__", "")
        if name == "JobRequirements":
            return self.jd
        return self.resume
