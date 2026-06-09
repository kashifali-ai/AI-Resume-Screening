"""End-to-end pipeline with a mocked LLM (real embedder, real scoring).

Proves the verdict is driven by extracted resume + JD data, not by the LLM
directly, that semantic matching makes synonyms count, and that ANY role works —
the role and its requirements come entirely from the pasted job description.
"""

from datetime import datetime

from conftest import FakeLLM
from screening import screen


def _resume(skills, technologies=None, years=3, projects=None, education=None):
    start = datetime.now().year - years
    return {
        "candidate_name": "Test Candidate",
        "email": "test@example.com",
        "candidate_summary": "Engineer.",
        "skills": skills,
        "technologies": technologies or [],
        "experience": [{
            "company": "Acme", "title": "Engineer",
            "start": str(start), "end": "Present",
            "description": "Built and shipped production software. " + " ".join(skills),
        }] if years else [],
        "projects": projects or ["A shipped project"],
        "education": education or ["B.S. Computer Science"],
        "certifications": [],
    }


def _jd(role, required, *, preferred=None, technologies=None, min_exp=2, max_exp=6):
    return {
        "role_title": role,
        "required_skills": required,
        "preferred_skills": preferred or [],
        "technologies": technologies or [],
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
        "education_requirements": [],
        "responsibilities": [],
    }


def test_backend_developer_strong_resume_is_fit(settings, embedder):
    # Resume uses SYNONYMS of the JD's required skills.
    resume = _resume(
        ["Python", "Spring Framework", "REST APIs", "Postgres"],
        technologies=["Docker"], years=4,
    )
    jd = _jd("Backend Developer",
             ["Python", "Spring Boot", "REST Services"],
             technologies=["PostgreSQL"], min_exp=2, max_exp=8)
    report = screen("jd text", "resume text", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.role_title == "Backend Developer"
    assert report.verdict == "FIT"
    # Synonyms matched the JD's canonical skills:
    assert "Spring Boot" in report.matched_skills
    assert "REST Services" in report.matched_skills
    assert "PostgreSQL" in report.matched_skills


def test_data_analyst_role_works(settings, embedder):
    resume = _resume(["SQL", "Python", "Tableau", "Statistics"], years=3)
    jd = _jd("Data Analyst",
             ["SQL", "Data Visualization", "Statistics"],
             preferred=["Tableau"], min_exp=1, max_exp=5)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.role_title == "Data Analyst"
    assert report.verdict == "FIT"
    assert "SQL" in report.matched_skills


def test_product_manager_role_works(settings, embedder):
    resume = _resume(["Roadmapping", "Stakeholder Management", "Agile", "Analytics"],
                     years=5)
    jd = _jd("Product Manager",
             ["Product Roadmap", "Stakeholder Management", "Agile"],
             min_exp=3, max_exp=10)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.role_title == "Product Manager"
    assert report.verdict == "FIT"


def test_mismatched_resume_for_role_is_unfit(settings, embedder):
    # Frontend resume vs a DevOps job -> should miss the required skills.
    resume = _resume(["HTML", "CSS", "JavaScript"], years=2)
    jd = _jd("DevOps Engineer",
             ["Kubernetes", "Terraform", "AWS", "CI/CD"], min_exp=3, max_exp=8)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.verdict == "UNFIT"
    assert report.missing_skills  # required DevOps skills not met


def test_below_experience_requirement_is_unfit(settings, embedder):
    # Has the skills, but not the years the JD demands.
    resume = _resume(["Python", "Java"], years=0)  # no experience
    jd = _jd("Software Engineer", ["Python", "Java"], min_exp=3, max_exp=8)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.verdict == "UNFIT"
    assert report.experience_ok is False


def test_no_experience_requirement_does_not_gate(settings, embedder):
    # No work history (0 years), but skills are evidenced via projects, and the
    # JD states no experience requirement -> experience must not gate the verdict.
    resume = _resume(["Python", "Java"], years=0,
                     projects=["Built CLI tools in Python and Java"])
    jd = _jd("Junior Engineer", ["Python", "Java"], min_exp=None, max_exp=None)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    # Skills fully matched and experience is not a gate -> FIT.
    assert report.verdict == "FIT"


def test_report_is_explainable(settings, embedder):
    resume = _resume(["Python", "Spring Boot"], years=4)
    jd = _jd("Software Engineer", ["Python", "Spring Boot"], min_exp=2, max_exp=8)
    report = screen("jd", "resume", settings=settings,
                    llm=FakeLLM(resume=resume, jd=jd), embedder=embedder)
    assert report.strengths and report.reasoning and report.recommendation
    assert report.recommendations
    assert report.experience_comparison


def test_empty_job_description_raises(settings, embedder):
    import pytest
    with pytest.raises(ValueError):
        screen("   ", "resume text", settings=settings,
               llm=FakeLLM(resume=_resume(["Python"]), jd=_jd("X", ["Python"])),
               embedder=embedder)
