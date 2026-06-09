"""LLM provider abstraction. The pipeline depends only on the LLMClient
protocol, so swapping providers is a one-file change.

Default provider is Google Gemini (structured output via response_schema)."""

import json
from typing import Protocol

from pydantic import BaseModel

from config import Settings
from logging_config import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns unparseable output."""


class LLMClient(Protocol):
    def extract_json(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> dict:
        """Return a JSON object matching `response_model`'s schema."""
        ...


class GeminiClient:
    """Structured extraction via Google Gemini.

    Uses Gemini's native structured-output support: a Pydantic model is passed as
    `response_schema` with `response_mime_type='application/json'`, so the model
    returns schema-shaped JSON directly.
    """

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Export it or put it in the project "
                "root .env (see .env.example)."
            )
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise LLMError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            ) from e

        self._genai = genai
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def extract_json(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> dict:
        from google.genai import types

        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=response_model,  # Pydantic model -> JSON schema
                    temperature=0.0,                  # deterministic extraction
                ),
            )
        except Exception as e:  # auth, quota, network, bad model name, etc.
            raise LLMError(
                f"Gemini call failed ({e}). Check GEMINI_API_KEY, the model name "
                f"'{self._model}', and network access."
            ) from e

        # Prefer the SDK-parsed object; fall back to parsing the raw text.
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()

        text = getattr(resp, "text", "") or ""
        if not text:
            raise LLMError("Gemini returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Gemini returned invalid JSON: {e}") from e


def get_llm_client(settings: Settings) -> LLMClient:
    # MOCK_LLM short-circuits everything: no Gemini, no quota, no API key needed.
    if settings.mock_llm:
        from mock_llm import MockLLMClient

        return MockLLMClient(settings)
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return GeminiClient(settings)
    raise LLMError(f"Unknown llm_provider '{settings.llm_provider}'.")
