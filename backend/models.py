"""Pydantic models for the screening pipeline.

CandidateProfile and JobRequirements are filled by the LLM (extraction only).
SkillMatch and ScreeningReport are produced by deterministic Python code.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start: Optional[str] = None  # e.g. "2021" or "2021-03"
    end: Optional[str] = None    # e.g. "2023", "Present"
    description: str = ""


class CandidateProfile(BaseModel):
    """Structured data extracted from the resume by the LLM. Extraction only —
    contains no scores or verdicts."""

    candidate_name: str = "Unknown"
    email: Optional[str] = None
    candidate_summary: str = ""
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class JobRequirements(BaseModel):
    """Structured requirements extracted from a pasted job description by the
    LLM. Extraction only — the LLM never scores the candidate or decides FIT.

    Experience requirements are extracted as numbers (e.g. "3+ years" -> min 3)
    and left null when the JD states no requirement."""

    role_title: str = "Unknown Role"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    min_experience_years: Optional[float] = None
    max_experience_years: Optional[float] = None
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class SkillMatch(BaseModel):
    required: str
    kind: Literal["required", "preferred"] = "required"
    matched_to: Optional[str] = None
    similarity: float = 0.0
    matched: bool = False
    confidence: float = 1.0  # evidence-grounded; lowers contribution when unsupported
    weight: float = 1.0


class ScreeningReport(BaseModel):
    verdict: Literal["FIT", "UNFIT"]
    role_title: str
    score: int
    match_percentage: float
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    recommendations: list[str]          # concrete resume-improvement suggestions
    reasoning: list[str]
    flags: list[str]
    experience_years: float             # candidate's computed experience
    experience_ok: bool
    experience_required: Optional[dict] = None  # {"min": x, "max": y} from the JD
    experience_comparison: str          # human-readable candidate-vs-required line
    candidate_resume: CandidateProfile
    job_requirements: JobRequirements
    skill_matches: list[SkillMatch]


class BulkResumeResult(BaseModel):
    """One row in a bulk screening run. Summary fields are filled for resumes
    that screened successfully; `ok=False` rows carry an `error` instead."""

    filename: str
    ok: bool = True
    error: Optional[str] = None
    # summary (present when ok) — what the ranked table shows
    candidate_name: Optional[str] = None
    score: Optional[int] = None
    verdict: Optional[Literal["FIT", "UNFIT"]] = None
    matched_count: Optional[int] = None
    missing_count: Optional[int] = None
    experience_years: Optional[float] = None
    recommendation: Optional[str] = None
    # full report for drill-down (optional to keep payloads lean)
    report: Optional[ScreeningReport] = None


class BulkScreeningResponse(BaseModel):
    """Result of evaluating many resumes against ONE job description.

    The JD is extracted once; `jd_cached` reports whether those requirements were
    served from the cache (i.e. Gemini was NOT called for the JD on this request).
    `results` is ranked by score, highest first."""

    role_title: str
    job_requirements: JobRequirements
    jd_cached: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkResumeResult]
