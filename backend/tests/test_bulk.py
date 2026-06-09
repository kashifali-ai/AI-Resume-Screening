"""Bulk screening: JD extracted once, resumes scored independently and ranked.

Real embedder + real deterministic scoring; only the LLM is faked. The fake
counts how many times each extraction model was requested, which is how we prove
one JD is extracted only once while many resumes are evaluated.
"""

from datetime import datetime

import pytest

from conftest import FakeLLM
from screening import clear_jd_cache, screen, screen_bulk


def _resume(name, skills, years=4, projects=None):
    start = datetime.now().year - years
    return {
        "candidate_name": name,
        "email": f"{name.split()[0].lower()}@example.com",
        "candidate_summary": "Engineer.",
        "skills": skills,
        "technologies": [],
        "experience": [{
            "company": "Acme", "title": "Engineer",
            "start": str(start), "end": "Present",
            "description": "Shipped production software. " + " ".join(skills),
        }] if years else [],
        "projects": projects or ["A shipped project"],
        "education": ["B.S. Computer Science"],
        "certifications": [],
    }


def _jd(role="Software Engineer", required=None, min_exp=2, max_exp=8):
    return {
        "role_title": role,
        "required_skills": required or ["Python", "Java"],
        "preferred_skills": [],
        "technologies": [],
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
        "education_requirements": [],
        "responsibilities": [],
    }


JD_TEXT = "Software Engineer. Requires Python and Java. 2+ years experience."


def test_jd_extracted_only_once_for_many_resumes(settings, embedder):
    resumes_payloads = [
        _resume("Alice", ["Python", "Java"]),
        _resume("Bob", ["Python", "Java"]),
        _resume("Carol", ["Python", "Java"]),
        _resume("Dave", ["Python", "Java"]),
        _resume("Eve", ["Python", "Java"]),
    ]
    files = [(f"r{i}.txt", "resume text") for i in range(5)]
    llm = FakeLLM(jd=_jd(), resumes=resumes_payloads)

    resp = screen_bulk(JD_TEXT, files, settings=settings, llm=llm, embedder=embedder)

    # PROOF: the JD was extracted exactly once; each resume extracted once.
    assert llm.calls["JobRequirements"] == 1
    assert llm.calls["CandidateProfile"] == 5
    assert resp.total == 5 and resp.succeeded == 5 and resp.failed == 0
    assert resp.jd_cached is False  # first time this JD is seen


def test_jd_cache_persists_across_requests(settings, embedder):
    clear_jd_cache()
    files = [(f"a{i}.txt", "resume text") for i in range(3)]

    llm1 = FakeLLM(jd=_jd(), resumes=[_resume("A", ["Python", "Java"])])
    r1 = screen_bulk(JD_TEXT, files, settings=settings, llm=llm1, embedder=embedder)
    assert llm1.calls["JobRequirements"] == 1
    assert r1.jd_cached is False

    # Second request, SAME JD text, brand-new llm — Gemini must NOT be called for JD.
    llm2 = FakeLLM(jd=_jd(), resumes=[_resume("B", ["Python", "Java"])])
    r2 = screen_bulk(JD_TEXT, files, settings=settings, llm=llm2, embedder=embedder)
    assert llm2.calls["JobRequirements"] == 0       # served from cache
    assert llm2.calls["CandidateProfile"] == 3      # resumes still extracted
    assert r2.jd_cached is True


def test_single_screen_shares_the_same_cache(settings, embedder):
    clear_jd_cache()
    # Prime the cache via the single-resume workflow.
    llm1 = FakeLLM(jd=_jd(), resume=_resume("Solo", ["Python", "Java"]))
    screen(JD_TEXT, "resume text", settings=settings, llm=llm1, embedder=embedder)
    assert llm1.calls["JobRequirements"] == 1

    # A bulk run with the same JD reuses it — no second JD extraction.
    llm2 = FakeLLM(jd=_jd(), resumes=[_resume("X", ["Python", "Java"])])
    resp = screen_bulk(JD_TEXT, [("x.txt", "t")], settings=settings,
                       llm=llm2, embedder=embedder)
    assert llm2.calls["JobRequirements"] == 0
    assert resp.jd_cached is True


def test_results_ranked_by_score_descending(settings, embedder):
    payloads = [
        _resume("Weak", ["HTML", "CSS"], years=1),          # unrelated -> low
        _resume("Strong", ["Python", "Java"], years=4),     # full match -> high
        _resume("Medium", ["Python"], years=3),             # half match -> middle
    ]
    files = [("weak.txt", "t"), ("strong.txt", "t"), ("medium.txt", "t")]
    llm = FakeLLM(jd=_jd(), resumes=payloads)

    resp = screen_bulk(JD_TEXT, files, settings=settings, llm=llm, embedder=embedder)

    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)        # highest first
    assert resp.results[0].candidate_name == "Strong"
    # Each row exposes the columns the ranked table needs.
    top = resp.results[0]
    assert top.verdict in ("FIT", "UNFIT")
    assert top.matched_count is not None and top.missing_count is not None
    assert top.experience_years is not None and top.recommendation


def test_one_bad_resume_does_not_abort_batch(settings, embedder):
    payloads = [
        _resume("Good", ["Python", "Java"]),
        {"candidate_name": "ignored"},  # used only if reached
    ]
    files = [("good.txt", "resume text"), ("empty.txt", "   ")]  # 2nd is empty
    llm = FakeLLM(jd=_jd(), resumes=payloads)

    resp = screen_bulk(JD_TEXT, files, settings=settings, llm=llm, embedder=embedder)

    assert resp.total == 2 and resp.succeeded == 1 and resp.failed == 1
    failed = [r for r in resp.results if not r.ok]
    assert failed and failed[0].filename == "empty.txt" and failed[0].error
    # The empty resume must NOT have triggered a Gemini resume call.
    assert llm.calls["CandidateProfile"] == 1
    # Failed rows rank last.
    assert resp.results[-1].ok is False


def test_prefailed_files_appear_as_failed_rows(settings, embedder):
    llm = FakeLLM(jd=_jd(), resumes=[_resume("Good", ["Python", "Java"])])
    resp = screen_bulk(
        JD_TEXT, [("good.txt", "resume text")],
        prefailed=[("scan.pdf", "No text could be extracted.")],
        settings=settings, llm=llm, embedder=embedder,
    )
    assert resp.total == 2 and resp.succeeded == 1 and resp.failed == 1
    assert any(r.filename == "scan.pdf" and not r.ok for r in resp.results)


def test_bulk_empty_jd_raises(settings, embedder):
    with pytest.raises(ValueError):
        screen_bulk("   ", [("r.txt", "t")], settings=settings,
                    llm=FakeLLM(jd=_jd(), resumes=[_resume("A", ["Python"])]),
                    embedder=embedder)


def test_bulk_works_for_any_role(settings, embedder):
    # A completely different role still works with no code changes.
    jd_text = "Data Analyst. Requires SQL, Tableau, Statistics. 1+ years."
    jd = _jd(role="Data Analyst", required=["SQL", "Tableau", "Statistics"], min_exp=1)
    payloads = [
        _resume("Analyst One", ["SQL", "Tableau", "Statistics"], years=3),
        _resume("Analyst Two", ["Excel"], years=2),
    ]
    llm = FakeLLM(jd=jd, resumes=payloads)
    resp = screen_bulk(jd_text, [("a.txt", "t"), ("b.txt", "t")],
                       settings=settings, llm=llm, embedder=embedder)
    assert resp.role_title == "Data Analyst"
    assert resp.results[0].candidate_name == "Analyst One"
    assert llm.calls["JobRequirements"] == 1
