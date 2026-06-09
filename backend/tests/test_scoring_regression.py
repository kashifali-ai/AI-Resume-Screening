"""Regression guard for the 'all scores 0' bug.

Root cause: in MockLLM mode a prose job description (no 'Required skills:' header)
yielded an empty requirement list, so every resume scored 0 with 0 matched / 0
missing. These tests use a header-LESS, prose JD — the exact shape that broke —
and assert real, differentiated, non-zero scores. Scoring is NOT mocked: real
embedder + real deterministic Python scoring; only extraction is the MockLLM.
"""

from datetime import datetime

from conftest import FakeLLM
from mock_llm import MockLLMClient, extract_requirements
from screening import clear_jd_cache, screen, screen_bulk

# Prose JD with NO explicit skill-section headers — the regression case.
JD_PROSE = """Software Development Engineer

We are looking for a Software Development Engineer to join our team.
You should have strong experience with Python, Java, FastAPI and Spring Boot.
Familiarity with REST APIs, PostgreSQL and Docker is expected.
2+ years of professional experience required.
"""


def _resume(name, skills, years=4):
    start = datetime.now().year - years
    return f"""{name}
Software Engineer
{name.split()[0].lower()}@example.com
Skills
- {', '.join(skills)}
Experience
Software Engineer, Acme Corp ({start} - Present)
- Built and shipped services using {', '.join(skills)}.
Education
B.Tech, Computer Science (2018)
"""


def test_prose_jd_extracts_required_skills():
    jd = extract_requirements(JD_PROSE)
    assert jd["required_skills"], "prose JD must yield non-empty required skills"
    lowered = {s.lower() for s in jd["required_skills"]}
    assert {"python", "java"} <= lowered


def test_prose_jd_scores_are_not_all_zero_and_ranked(settings, embedder):
    clear_jd_cache()
    resumes = [
        ("strong.txt", _resume("Strong Match",
            ["Python", "Java", "FastAPI", "Spring Boot", "REST APIs", "PostgreSQL", "Docker"], 5)),
        ("medium.txt", _resume("Medium Match", ["Python", "Java"], 3)),
        ("weak.txt", _resume("Weak Match", ["HTML", "CSS"], 2)),
    ]
    mock = MockLLMClient(settings)
    resp = screen_bulk(JD_PROSE, resumes, settings=settings, llm=mock, embedder=embedder)

    scores = [r.score for r in resp.results]
    assert any(s > 0 for s in scores), "scores must not all be zero"
    assert resp.results[0].matched_count > 0
    # ranked high -> low and genuinely differentiated
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) > 1
    assert resp.results[0].candidate_name == "Strong Match"
    # JD extracted once for the batch
    assert mock.calls["JobRequirements"] == 1
    assert mock.calls["CandidateProfile"] == 3


def test_single_and_bulk_consistent(settings, embedder):
    resume_text = _resume("Solo Candidate",
                          ["Python", "Java", "FastAPI", "Spring Boot"], 4)

    clear_jd_cache()
    single = screen(JD_PROSE, resume_text, settings=settings,
                    llm=MockLLMClient(settings), embedder=embedder)

    clear_jd_cache()
    bulk = screen_bulk(JD_PROSE, [("solo.txt", resume_text)], settings=settings,
                       llm=MockLLMClient(settings), embedder=embedder)

    assert single.score > 0
    assert bulk.results[0].score == single.score
    assert bulk.results[0].verdict == single.verdict
    assert bulk.results[0].matched_count == len(single.matched_skills)


def test_gemini_mode_still_works_unchanged(settings, embedder):
    # The non-mock path (FakeLLM stands in for Gemini) is unaffected by the fix.
    jd = {
        "role_title": "Backend Engineer",
        "required_skills": ["Python", "Java"], "preferred_skills": [],
        "technologies": [], "min_experience_years": 2, "max_experience_years": 8,
        "education_requirements": [], "responsibilities": [],
    }
    resume = {
        "candidate_name": "G User", "email": "g@x.com", "candidate_summary": "",
        "skills": ["Python", "Java"], "technologies": [],
        "experience": [{"company": "Acme", "title": "Eng",
                        "start": str(datetime.now().year - 4), "end": "Present",
                        "description": "Built services in Python and Java."}],
        "projects": ["proj"], "education": ["BS"], "certifications": [],
    }
    clear_jd_cache()
    rep = screen("Backend Engineer JD", "resume text", settings=settings,
                 llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert rep.score > 0 and rep.verdict in ("FIT", "UNFIT")
    assert "Python" in rep.matched_skills
