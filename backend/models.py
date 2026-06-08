"""Pydantic models for the screening pipeline.

CandidateProfile is filled by the LLM (extraction only). SkillMatch and
ScreeningReport are produced by deterministic Python code.
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
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class SkillMatch(BaseModel):
    required: str
    matched_to: Optional[str] = None
    similarity: float = 0.0
    matched: bool = False
    confidence: float = 1.0  # evidence-grounded; lowers contribution when unsupported
    weight: float = 1.0


class ScreeningReport(BaseModel):
    verdict: Literal["FIT", "UNFIT"]
    score: int
    match_percentage: float
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    reasoning: list[str]
    flags: list[str]
    experience_years: float
    experience_ok: bool
    candidate_resume: CandidateProfile
    reference_resume: dict
    skill_matches: list[SkillMatch]
