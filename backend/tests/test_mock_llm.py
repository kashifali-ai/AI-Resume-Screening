"""MockLLM mode: deterministic, offline extraction + full bulk workflow.

These tests make ZERO Gemini calls (no API key needed). They use the real
embedder and real deterministic scoring — only extraction is mocked.
"""

import pytest

from config import Settings
from llm_client import get_llm_client
from models import CandidateProfile, JobRequirements
from mock_llm import MockLLMClient, extract_requirements, extract_resume
from report_csv import bulk_to_csv
from screening import screen_bulk


# --- text builders ---------------------------------------------------------

SKILL_POOL = ["Python", "Java", "REST APIs", "SQL",
              "Docker", "Kubernetes", "PostgreSQL", "Git"]

JD_TEXT = """Backend Engineer

We are hiring a Backend Engineer to build scalable services.

Required skills: Python, Java, REST APIs, SQL
Preferred skills: Docker, Kubernetes
Technologies: PostgreSQL, Git

Requires 2-8 years of experience.

Responsibilities:
- Design and build REST APIs.
- Optimize database queries.

Education: Bachelor's degree in Computer Science.
"""


def resume_text(name, skills, start_year=2020, end="Present"):
    return f"""{name}
Software Engineer
{name.split()[0].lower()}@example.com

Summary
Backend engineer experienced with {', '.join(skills[:2])}.

Skills
- {', '.join(skills)}

Experience
Software Engineer, Acme Corp ({start_year} - {end})
- Built and shipped services using {', '.join(skills)}.

Projects
- A production project using {skills[0]}

Education
B.Tech, Computer Science (2018)
"""


# --- mock JD extraction ----------------------------------------------------

def test_mock_jd_extraction_rule_based():
    jd = extract_requirements(JD_TEXT)
    assert jd["role_title"] == "Backend Engineer"
    assert set(jd["required_skills"]) >= {"Python", "Java", "REST APIs", "SQL"}
    assert set(jd["preferred_skills"]) >= {"Docker", "Kubernetes"}
    assert set(jd["technologies"]) >= {"PostgreSQL", "Git"}
    assert jd["min_experience_years"] == 2.0
    assert jd["max_experience_years"] == 8.0
    assert any("Bachelor" in e for e in jd["education_requirements"])
    assert jd["responsibilities"]
    # Must validate against the real schema.
    JobRequirements.model_validate(jd)


def test_mock_jd_extraction_plus_years_only():
    jd = extract_requirements("Data Analyst\nRequired: SQL, Excel\nRequires 3+ years.")
    assert jd["role_title"] == "Data Analyst"
    assert "SQL" in jd["required_skills"]
    assert jd["min_experience_years"] == 3.0
    assert jd["max_experience_years"] is None


# --- mock resume extraction ------------------------------------------------

def test_mock_resume_extraction_parses_fields():
    text = resume_text("Aarav Sharma", ["Python", "Java", "REST APIs"], 2021)
    profile = extract_resume(text)
    assert profile["candidate_name"] == "Aarav Sharma"
    assert profile["email"] == "aarav@example.com"
    assert "Python" in profile["skills"]
    assert "REST APIs" in profile["skills"]            # multiword skill preserved
    assert profile["experience"] and profile["experience"][0]["start"] == "2021"
    assert profile["education"]
    CandidateProfile.model_validate(profile)


def test_mock_client_dispatches_on_model():
    mock = MockLLMClient(Settings(mock_llm=True))
    jd_user = f"<job_description>\n{JD_TEXT}\n</job_description>"
    res_user = f"<resume>\n{resume_text('Bob Lee', ['Python'])}\n</resume>"
    jd = mock.extract_json("sys", jd_user, JobRequirements)
    res = mock.extract_json("sys", res_user, CandidateProfile)
    assert jd["role_title"] == "Backend Engineer"
    assert res["candidate_name"] == "Bob Lee"
    assert mock.calls == {"JobRequirements": 1, "CandidateProfile": 1}


def test_mock_mode_needs_no_api_key():
    # Empty key + mock_llm -> MockLLMClient, no error (Gemini would have raised).
    client = get_llm_client(Settings(mock_llm=True, gemini_api_key=""))
    assert isinstance(client, MockLLMClient)


# --- bulk workflow (50 resumes) --------------------------------------------

def _bulk_50(settings, embedder):
    resumes = []
    for i in range(50):
        skills = SKILL_POOL[: (i % len(SKILL_POOL)) + 1]
        start = 2024 - (i % 6 + 1)
        name = f"Candidate {i:02d}"
        resumes.append((f"resume_{i:02d}.txt", resume_text(name, skills, start)))
    mock = MockLLMClient(settings)
    resp = screen_bulk(JD_TEXT, resumes, settings=settings, llm=mock, embedder=embedder)
    return resp, mock


def test_bulk_50_resumes_zero_gemini_calls(settings, embedder):
    resp, mock = _bulk_50(settings, embedder)
    assert resp.total == 50 and resp.succeeded == 50 and resp.failed == 0
    # JD extracted exactly once; one resume extraction per resume; all via mock.
    assert mock.calls["JobRequirements"] == 1
    assert mock.calls["CandidateProfile"] == 50
    assert resp.role_title == "Backend Engineer"


def test_bulk_ranking_is_sorted_desc(settings, embedder):
    resp, _ = _bulk_50(settings, embedder)
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    # Top candidate has the broadest, evidenced skill set.
    top = resp.results[0]
    assert top.score >= resp.results[-1].score
    assert top.verdict in ("FIT", "UNFIT")
    assert top.matched_count is not None and top.missing_count is not None
    assert top.recommendation


def test_bulk_missing_skills_and_report_fields(settings, embedder):
    # One weak resume must surface missing required skills + a full report.
    resumes = [
        ("weak.txt", resume_text("Weak One", ["HTML"], 2023)),
        ("strong.txt", resume_text("Strong One", SKILL_POOL, 2016)),
    ]
    mock = MockLLMClient(settings)
    resp = screen_bulk(JD_TEXT, resumes, settings=settings, llm=mock, embedder=embedder)
    by_name = {r.candidate_name: r for r in resp.results}
    weak = by_name["Weak One"]
    assert weak.missing_count and weak.missing_count > 0
    assert weak.report is not None
    assert weak.report.strengths and weak.report.weaknesses
    assert weak.report.recommendations


# --- CSV export ------------------------------------------------------------

def test_csv_export_matches_results(settings, embedder):
    resumes = [
        ("a.txt", resume_text("Alpha", SKILL_POOL, 2017)),
        ("b.txt", resume_text("Bravo", ["Python", "Java"], 2021)),
    ]
    mock = MockLLMClient(settings)
    resp = screen_bulk(JD_TEXT, resumes, settings=settings, llm=mock, embedder=embedder)
    csv_text = bulk_to_csv(resp)
    lines = csv_text.strip().splitlines()
    assert lines[0] == ("Rank,Candidate,Score,Verdict,Matched,Missing,"
                        "Experience (yr),Recommendation,File")
    assert len(lines) == 1 + resp.total                # header + one row per resume
    assert "Alpha" in csv_text and "Bravo" in csv_text
    # Rank 1 row reflects the top-ranked candidate.
    assert lines[1].startswith(f"1,{resp.results[0].candidate_name}")


def test_csv_export_handles_failed_rows(settings, embedder):
    mock = MockLLMClient(settings)
    resp = screen_bulk(
        JD_TEXT, [("ok.txt", resume_text("Okay", SKILL_POOL, 2018))],
        prefailed=[("scan.pdf", "No text could be extracted.")],
        settings=settings, llm=mock, embedder=embedder,
    )
    csv_text = bulk_to_csv(resp)
    assert "FAILED" in csv_text
    assert "scan.pdf" in csv_text
