"""Anti-gaming: dedup, evidence confidence, stuffing, missing sections."""

from models import CandidateProfile, Experience
from normalize import (
    dedup_skills,
    detect_stuffing,
    missing_section_flags,
    skill_confidence,
)

# `settings` and `embedder` come from conftest.py (session-scoped).


def test_dedup_removes_case_insensitive_duplicates(embedder):
    out = dedup_skills(["Python", "python", "  Python "], embedder, 0.95)
    assert len(out) == 1


def test_dedup_collapses_semantic_variants(embedder):
    out = dedup_skills(["React", "ReactJS", "React.js", "PostgreSQL"], embedder, 0.7)
    assert "PostgreSQL" in out
    assert len(out) < 4  # the React variants collapse


def test_skill_confidence_grounded_in_experience(settings):
    profile = CandidateProfile(
        skills=["Python", "Rust"],
        experience=[Experience(description="Built services in Python at scale.")],
    )
    conf = skill_confidence(["Python", "Rust"], profile, settings)
    assert conf["Python"] == 1.0                       # supported by experience
    assert conf["Rust"] == settings.evidence_confidence  # listed but unsupported


def test_detect_stuffing_flags_repetition(settings):
    text = ("Python " * 30)  # absurd repetition
    conf = {"Python": 1.0}
    penalty, flags = detect_stuffing(text, ["Python"], conf, settings)
    assert penalty > 0 and flags


def test_detect_stuffing_flags_unsupported_skill_dump(settings):
    skills = [f"Skill{i}" for i in range(12)]
    conf = {s: settings.evidence_confidence for s in skills}  # none supported
    text = " ".join(skills)
    penalty, flags = detect_stuffing(text, skills, conf, settings)
    assert penalty > 0
    assert any("no supporting" in f for f in flags)


def test_clean_resume_has_no_stuffing_penalty(settings):
    text = "Experienced engineer who used Python and Java to build APIs."
    conf = {"Python": 1.0, "Java": 1.0}
    penalty, flags = detect_stuffing(text, ["Python", "Java"], conf, settings)
    assert penalty == 0.0 and not flags


def test_missing_sections_detected():
    flags = missing_section_flags(CandidateProfile())
    assert any("skills" in f for f in flags)
    assert any("experience" in f for f in flags)
