"""Deterministic scoring, experience math, and verdict boundaries."""

from datetime import datetime

import pytest

from config import Settings
from models import CandidateProfile, Experience
from scoring import compute_experience_years, experience_factor, score


@pytest.fixture
def settings():
    return Settings()


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


def test_experience_factor_enforces_window(settings):
    assert experience_factor(0.5, settings) == (0.0, False, False)        # below min
    assert experience_factor(3, settings) == (1.0, True, True)            # within
    f, meets, within = experience_factor(20, settings)                    # above max
    assert meets is True and within is False and f == settings.overqualified_factor


def _matches(matched_reqs, all_reqs):
    return [
        {"required": r, "matched_to": (r if r in matched_reqs else None),
         "similarity": 0.9 if r in matched_reqs else 0.1,
         "matched": r in matched_reqs}
        for r in all_reqs
    ]


def test_score_is_deterministic(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    reqs = ["Python", "Java"]
    matches = _matches({"Python", "Java"}, reqs)
    conf = {"Python": 1.0, "Java": 1.0}
    a = score(matches, conf, profile, 0.0, settings)
    b = score(matches, conf, profile, 0.0, settings)
    assert a["score"] == b["score"]


def test_full_match_in_window_is_fit(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])  # 3y
    reqs = ["Python", "Java"]
    matches = _matches({"Python", "Java"}, reqs)
    out = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, settings)
    assert out["verdict"] == "FIT" and out["score"] == 100


def test_below_minimum_experience_is_unfit_even_with_all_skills(settings):
    profile = CandidateProfile(experience=[])  # 0 years
    reqs = ["Python", "Java"]
    matches = _matches({"Python", "Java"}, reqs)
    out = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, settings)
    assert out["verdict"] == "UNFIT"


def test_stuffing_penalty_reduces_score(settings):
    profile = CandidateProfile(experience=[Experience(start="2021", end="2024")])
    reqs = ["Python", "Java"]
    matches = _matches({"Python", "Java"}, reqs)
    clean = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.0, settings)
    penalized = score(matches, {"Python": 1.0, "Java": 1.0}, profile, 0.5, settings)
    assert penalized["score"] < clean["score"]
