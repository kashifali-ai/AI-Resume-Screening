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
    """Stand-in LLM client that returns a preset extraction dict."""

    def __init__(self, payload: dict):
        self.payload = payload

    def extract_json(self, system: str, user: str, schema: dict) -> dict:
        return self.payload
