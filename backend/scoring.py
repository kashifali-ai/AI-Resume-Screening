"""Stage 5: deterministic scoring and verdict (pure Python, no LLM).

Every number here is reproducible from the structured inputs (the extracted
resume + the extracted job requirements). The LLM never sees this stage — it
only supplied the extracted data. The FIT/UNFIT verdict is computed here, never
taken from the model.
"""

import re
from datetime import datetime

from config import Settings
from logging_config import get_logger
from models import CandidateProfile, Experience, JobRequirements, SkillMatch

log = get_logger(__name__)


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"(19|20)\d{2}", value)
    return int(m.group(0)) if m else None


def compute_experience_years(experiences: list[Experience]) -> float:
    """Total years from experience date spans, merging overlapping intervals so
    concurrent roles aren't double-counted. Uses the real current year."""
    now_year = datetime.now().year
    spans: list[tuple[int, int]] = []
    for e in experiences:
        start = _parse_year(e.start)
        end_raw = (e.end or "").strip().lower()
        if end_raw in ("present", "current", "now", ""):
            end = now_year if start else None
        else:
            end = _parse_year(e.end)
        if start and end and end >= start:
            spans.append((start, end))

    if not spans:
        return 0.0

    spans.sort()
    merged = [list(spans[0])]
    for s, en in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])
    total = sum(en - s for s, en in merged)
    return float(round(total, 1))


def experience_factor(
    years: float,
    min_req: float | None,
    max_req: float | None,
    settings: Settings,
) -> tuple[float, bool, bool]:
    """Return (factor, meets_minimum, within_window) using the JD's own window.

    When the job description states no experience requirement (min_req is None),
    experience is not a gating factor and does not penalize the score.
    """
    if min_req is None:
        return 1.0, True, True
    if years < min_req:
        return 0.0, False, False
    if max_req is not None and years > max_req:
        return settings.overqualified_factor, True, False
    return 1.0, True, True


def score(
    matches: list[dict],
    confidence: dict[str, float],
    profile: CandidateProfile,
    stuffing_penalty: float,
    jd: JobRequirements,
    settings: Settings,
) -> dict:
    """Combine semantic matches + evidence confidence + experience + penalties
    into a deterministic score and verdict.

    Each match dict carries its own `weight` and `kind` (required/preferred),
    derived from the job description — there is no hardcoded skill list."""
    skill_matches: list[SkillMatch] = []
    weighted_hit, weighted_total = 0.0, 0.0

    for m in matches:
        req = m["required"]
        w = float(m.get("weight", 1.0))
        conf = confidence.get(m["matched_to"], 1.0) if m["matched"] else 1.0
        contribution = w * conf if m["matched"] else 0.0
        weighted_hit += contribution
        weighted_total += w
        skill_matches.append(
            SkillMatch(
                required=req,
                kind=m.get("kind", "required"),
                matched_to=m["matched_to"],
                similarity=m["similarity"],
                matched=m["matched"],
                confidence=conf if m["matched"] else 0.0,
                weight=w,
            )
        )

    skill_coverage = (weighted_hit / weighted_total) if weighted_total else 0.0

    years = compute_experience_years(profile.experience)
    exp_factor, meets_min, within_window = experience_factor(
        years, jd.min_experience_years, jd.max_experience_years, settings
    )

    overall = skill_coverage * exp_factor * (1.0 - stuffing_penalty) * 100.0
    overall = round(overall, 1)

    # Without any extracted requirements there is nothing to screen against.
    has_requirements = weighted_total > 0
    is_fit = has_requirements and overall >= settings.fit_threshold and meets_min
    verdict = "FIT" if is_fit else "UNFIT"

    matched_skills = [m.required for m in skill_matches if m.matched]
    missing_skills = [m.required for m in skill_matches if not m.matched]

    return {
        "verdict": verdict,
        "score": int(round(overall)),
        "match_percentage": round(skill_coverage * 100, 1),
        "experience_years": years,
        "experience_ok": meets_min,
        "within_window": within_window,
        "has_requirements": has_requirements,
        "skill_coverage": round(skill_coverage, 4),
        "experience_factor": exp_factor,
        "stuffing_penalty": round(stuffing_penalty, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_matches": skill_matches,
    }
