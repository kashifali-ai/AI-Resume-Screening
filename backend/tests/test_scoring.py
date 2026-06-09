"""Deterministic scoring, experience math, and verdict boundaries.

The role/skills/experience window all come from a JobRequirements object — there
is no hardcoded role. Verdicts are computed in Python, never from an LLM.
"""

from datetime import datetime

import pytest

from config import Settings
from models import CandidateProfile, Experience, JobRequirements
from scoring import compute_experience_years, experience_factor, score


@pytest.fixture
def settings():
    return Settings()


def _jd(required, *, min_exp=1.0, max_exp=5.0, preferred=None):
    return JobRequirements(
        role_title="Test Role",
        required_skills=required,
        preferred_skills=preferred or [],
        min_experience_years=min_exp,
        max_experience_years=max_exp,
    )


def _matches(matched, all_reqs, *, kind="required", weight=1.0):
    return [
        {"required": r, "skill": r, "kind": kind, "weight": weight,
         "matched_to": (r if r in matched else None),
         "similarity": 0.9 if r in matched else 0.1,
         "matched": r in matched}
        for r in all_reqs
    ]


# --- experience math -------------------------------------------------------

def test_experience_merges_overlapping_intervals():
    exps = [
        Experience(start="2018", end="2021"),  # 3 yrs
        Experience(start="2020", end="2022"),  # overlaps -> merged span 2018-2022 = 4
    ]
    assert compute_experience_years(exps) == 4.0


def test_experience_present_uses_current_year():
    start = datetime.now().year - 2
    years = compute_experience_years([Experience(start=str(start), end="Present")])
    assert years == 2.0


def test_experience_factor_uses_jd_window(settings):
    assert experience_factor(0.5, 1.0, 5.0, settings) == (0.0, False, False)   # below min
    assert experience_factor(3, 1.0, 5.0, settings) == (1.0, True, True)       # within
    f, meets, within = experience_factor(20, 1.0, 5.0, settings)               # above max
    assert meets is True and within is False and f == settings.overqualified_factor


def test_experience_no_requirement_is_not_gated(settings):
    # JD with no experience requirement -> never penalizes, never gates.
    assert experience_factor(0.0, None, None, settings) == (1.0, True, True)


# --- scoring & verdict -----------------------------------------------------

def test_score_is_deterministic(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    jd = _jd(["Python", "Java"])
    matches = _matches({"Python", "Java"}, ["Python", "Java"])
    conf = {"Python": 1.0, "Java": 1.0}
    a = score(matches, conf, profile, 0.0, jd, settings)
    b = score(matches, conf, profile, 0.0, jd, settings)
    assert a["score"] == b["score"]


def test_full_match_in_window_is_fit(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])  # 3y
    jd = _jd(["Python", "Java"])
    matches = _matches({"Python", "Java"}, ["Python", "Java"])
    out = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, jd, settings)
    assert out["verdict"] == "FIT" and out["score"] == 100


def test_below_minimum_experience_is_unfit_even_with_all_skills(settings):
    profile = CandidateProfile(experience=[])  # 0 years
    jd = _jd(["Python", "Java"], min_exp=2.0)
    matches = _matches({"Python", "Java"}, ["Python", "Java"])
    out = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, jd, settings)
    assert out["verdict"] == "UNFIT"
    assert out["experience_ok"] is False


def test_missing_required_skills_lower_coverage(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    jd = _jd(["Python", "Java", "Kubernetes", "AWS"])
    matches = _matches({"Python", "Java"}, ["Python", "Java", "Kubernetes", "AWS"])
    out = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, jd, settings)
    assert out["match_percentage"] == 50.0
    assert set(out["missing_skills"]) == {"Kubernetes", "AWS"}


def test_preferred_skills_weigh_less_than_required(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    jd = _jd(["Python"], preferred=["Go"])
    # required matched, preferred missing -> coverage should stay high (preferred=0.5)
    matches = (
        _matches({"Python"}, ["Python"], kind="required", weight=1.0)
        + _matches(set(), ["Go"], kind="preferred", weight=0.5)
    )
    out = score(matches, {"Python": 1.0}, profile, 0.0, jd, settings)
    # 1.0 / (1.0 + 0.5) = 66.7%
    assert out["match_percentage"] == pytest.approx(66.7, abs=0.1)


def test_no_requirements_is_unfit(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    jd = _jd([], min_exp=None, max_exp=None)
    out = score([], {}, profile, 0.0, jd, settings)
    assert out["verdict"] == "UNFIT"
    assert out["has_requirements"] is False


def test_stuffing_penalty_reduces_score(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    jd = _jd(["Python", "Java"])
    matches = _matches({"Python", "Java"}, ["Python", "Java"])
    clean = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, jd, settings)
    penalized = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.5, jd, settings)
    assert penalized["score"] < clean["score"]
