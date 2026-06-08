"""Stage 4: semantic skill matching via sentence-transformers embeddings.

This replaces exact/regex token matching. Cosine similarity recognizes
equivalents like Spring Framework ~ Spring Boot, JPA ~ Hibernate,
REST Services ~ REST APIs, ReactJS ~ React.

The embedding model is loaded once (process singleton) and reused.
"""

from functools import lru_cache

import numpy as np

from logging_config import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model '%s' (first load may be slow)...", model_name)
    return SentenceTransformer(model_name)


class Embedder:
    """Thin wrapper around a sentence-transformers model with cosine helpers."""

    def __init__(self, model_name: str):
        self._model = _load_model(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def similarity_matrix(self, a: list[str], b: list[str]) -> np.ndarray:
        """Cosine similarity matrix between rows of `a` and rows of `b`."""
        if not a or not b:
            return np.zeros((len(a), len(b)))
        ea, eb = self.encode(a), self.encode(b)
        return ea @ eb.T  # vectors are normalized -> dot product == cosine


def match_required_skills(
    required: list[str],
    candidate_skills: list[str],
    embedder: Embedder,
    threshold: float,
) -> list[dict]:
    """For each required skill, find the best-matching candidate skill.

    Returns a list of dicts: {required, matched_to, similarity, matched}.
    """
    if not candidate_skills:
        return [
            {"required": r, "matched_to": None, "similarity": 0.0, "matched": False}
            for r in required
        ]

    sims = embedder.similarity_matrix(required, candidate_skills)
    results = []
    for i, req in enumerate(required):
        best_j = int(np.argmax(sims[i]))
        best_sim = float(sims[i][best_j])
        matched = best_sim >= threshold
        results.append(
            {
                "required": req,
                "matched_to": candidate_skills[best_j] if matched else None,
                "similarity": round(best_sim, 4),
                "matched": matched,
            }
        )
    return results
