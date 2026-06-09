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


@pytest.fixture(autouse=True)
def _clear_jd_cache():
    """The JD requirements cache is process-global; clear it before each test so
    a JD string used in one test never leaks cached requirements into another."""
    from screening import clear_jd_cache

    clear_jd_cache()
    yield
    clear_jd_cache()


class FakeLLM:
    """Stand-in LLM client that returns preset extraction dicts and counts calls.

    The pipeline calls the LLM for the resume (CandidateProfile) and the job
    description (JobRequirements); this fake dispatches on the requested response
    model. Pass `resumes=[...]` to return a different payload per resume (in
    order), or `resume=...` for a single shared payload. `calls` tracks how many
    times each model was requested — used to prove the JD is extracted only once.
    """

    def __init__(
        self,
        resume: dict | None = None,
        jd: dict | None = None,
        resumes: list[dict] | None = None,
    ):
        self.jd = jd or {}
        self._resume = resume or {}
        self._resumes = list(resumes) if resumes else None
        self._i = 0
        self.calls: dict[str, int] = {"CandidateProfile": 0, "JobRequirements": 0}

    def extract_json(self, system: str, user: str, response_model) -> dict:
        name = getattr(response_model, "__name__", "")
        self.calls[name] = self.calls.get(name, 0) + 1
        if name == "JobRequirements":
            return self.jd
        if self._resumes is not None:
            payload = self._resumes[self._i % len(self._resumes)]
            self._i += 1
            return payload
        return self._resume
