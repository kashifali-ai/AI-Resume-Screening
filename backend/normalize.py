"""Stage 3: normalization and anti-gaming.

Protects against:
  * duplicate skills (React / ReactJS / React.js collapse to one)
  * false positives / unsupported claims (skill confidence is graded by whether
    the skill actually appears in the candidate's experience/project text)
  * keyword stuffing (term repetition + many skills with no supporting evidence)
  * missing resume sections
"""

import re

import numpy as np

from config import Settings
from logging_config import get_logger
from models import CandidateProfile
from semantic import Embedder

log = get_logger(__name__)


def dedup_skills(skills: list[str], embedder: Embedder, threshold: float) -> list[str]:
    """Collapse near-duplicate skill strings using embedding similarity.

    Keeps the first-seen representative of each cluster. Exact case-insensitive
    duplicates are removed first.
    """
    seen, unique = set(), []
    for s in skills:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(s.strip())
    if len(unique) <= 1:
        return unique

    sims = embedder.similarity_matrix(unique, unique)
    keep, dropped = [], set()
    for i in range(len(unique)):
        if i in dropped:
            continue
        keep.append(unique[i])
        for j in range(i + 1, len(unique)):
            if j not in dropped and sims[i][j] >= threshold:
                dropped.add(j)  # j is a near-duplicate of i
    return keep


def _evidence_text(profile: CandidateProfile) -> str:
    parts = []
    for e in profile.experience:
        parts += [e.title or "", e.company or "", e.description]
    parts += profile.projects + profile.education + profile.certifications
    return " \n ".join(parts).lower()


def skill_confidence(
    candidate_skills: list[str], profile: CandidateProfile, settings: Settings
) -> dict[str, float]:
    """1.0 if a skill is backed by experience/project text, else a reduced value.

    This is evidence grounding, not keyword scoring: a skill merely listed in a
    skills blob with no supporting narrative is treated as a weaker signal.
    """
    evidence = _evidence_text(profile)
    conf = {}
    for skill in candidate_skills:
        token = skill.strip().lower()
        present = bool(token) and (
            token in evidence
            or any(w in evidence for w in token.split() if len(w) > 3)
        )
        conf[skill] = 1.0 if present else settings.evidence_confidence
    return conf


def detect_stuffing(
    raw_text: str,
    candidate_skills: list[str],
    confidence: dict[str, float],
    settings: Settings,
) -> tuple[float, list[str]]:
    """Return (penalty in [0, cap], list of human-readable flags)."""
    flags: list[str] = []
    penalty = 0.0
    text = raw_text.lower()

    # 1) Term repetition: a real skill is mentioned a handful of times, not 20.
    max_repeats, worst = 0, None
    for skill in candidate_skills:
        token = re.escape(skill.strip().lower())
        if not token:
            continue
        count = len(re.findall(rf"\b{token}\b", text))
        if count > max_repeats:
            max_repeats, worst = count, skill
    if max_repeats > settings.stuffing_max_repeats:
        excess = max_repeats - settings.stuffing_max_repeats
        penalty += min(0.3, 0.03 * excess)
        flags.append(
            f"Possible keyword stuffing: '{worst}' repeated {max_repeats} times."
        )

    # 2) Many skills claimed, most without supporting evidence.
    if len(candidate_skills) >= settings.stuffing_min_skills_for_check:
        no_evidence = sum(1 for c in confidence.values() if c < 1.0)
        ratio = no_evidence / len(candidate_skills)
        if ratio >= settings.stuffing_low_evidence_ratio:
            penalty += 0.2
            flags.append(
                f"{no_evidence}/{len(candidate_skills)} listed skills have no "
                f"supporting experience or project text."
            )

    return min(penalty, settings.stuffing_penalty_cap), flags


def missing_section_flags(profile: CandidateProfile) -> list[str]:
    flags = []
    if not profile.skills and not profile.technologies:
        flags.append("Resume has no detectable skills section.")
    if not profile.experience:
        flags.append("Resume has no detectable work experience.")
    if not profile.education:
        flags.append("Resume has no detectable education section.")
    return flags
