"""End-to-end pipeline with a mocked LLM (real embedder, real scoring).

Proves the verdict is driven by extracted resume data, not by the LLM directly,
and that semantic matching makes synonyms count.
"""

from datetime import datetime

from conftest import FakeLLM
from screening import screen_resume


def _strong_payload():
    start = datetime.now().year - 3
    return {
        "candidate_name": "Aarav Sharma",
        "email": "aarav@example.com",
        # Deliberately uses SYNONYMS, not the exact required tokens:
        "skills": ["Python", "Java", "Spring Framework", "FastAPI",
                   "Large Language Models", "Git", "GitHub",
                   "Algorithms and Data Structures", "OOP"],
        "technologies": ["PostgreSQL", "Docker"],
        "experience": [{
            "company": "Acme", "title": "SDE", "start": str(start), "end": "Present",
            "description": ("Built FastAPI and Spring services in Python and Java; "
                            "integrated LLMs; strong data structures and OOP."),
        }],
        "projects": ["LLM support assistant"],
        "education": ["B.Tech CS"],
        "certifications": [],
    }


def _weak_payload():
    return {
        "candidate_name": "Riya Verma",
        "skills": ["HTML", "CSS", "JavaScript"],
        "technologies": [],
        "experience": [],
        "projects": [],
        "education": ["B.A."],
        "certifications": [],
    }


def test_strong_resume_is_fit(settings, embedder):
    report = screen_resume(
        "irrelevant raw text", settings=settings,
        llm=FakeLLM(_strong_payload()), embedder=embedder,
    )
    assert report.verdict == "FIT"
    # Synonyms matched the required canonical skills:
    assert "Spring Boot" in report.matched_skills
    assert "Data Structures and Algorithms" in report.matched_skills


def test_weak_resume_is_unfit(settings, embedder):
    report = screen_resume(
        "irrelevant raw text", settings=settings,
        llm=FakeLLM(_weak_payload()), embedder=embedder,
    )
    assert report.verdict == "UNFIT"
    assert report.experience_ok is False
    assert report.missing_skills  # required skills not met


def test_report_is_explainable(settings, embedder):
    report = screen_resume(
        "irrelevant raw text", settings=settings,
        llm=FakeLLM(_strong_payload()), embedder=embedder,
    )
    assert report.strengths and report.reasoning and report.recommendation
