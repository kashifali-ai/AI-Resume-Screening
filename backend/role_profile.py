"""The target role definition (replaces the old reference_resume.py).

Skills are written as natural-language phrases, not terse tokens, because the
embedding model matches phrases far better than abbreviations. Each skill has a
weight reflecting its importance to the SDE role. This is domain configuration,
not analysis logic — change it to retarget the screener to another role.
"""

ROLE_NAME = "Software Development Engineer (SDE)"

# required skill -> weight (relative importance)
REQUIRED_SKILLS: dict[str, float] = {
    "Python": 1.0,
    "Java": 1.0,
    "FastAPI": 0.8,
    "Spring Boot": 0.8,
    "Large Language Models (LLM)": 0.7,
    "Git": 0.5,
    "GitHub": 0.5,
    "Data Structures and Algorithms": 1.0,
    "Object-Oriented Programming": 0.8,
}


def required_skills() -> list[str]:
    return list(REQUIRED_SKILLS)


def skill_weight(skill: str) -> float:
    return REQUIRED_SKILLS.get(skill, 1.0)


def role_profile_dict(min_experience: float, max_experience: float) -> dict:
    """The reference profile returned to the client for display."""
    return {
        "role": ROLE_NAME,
        "required_skills": required_skills(),
        "skill_weights": REQUIRED_SKILLS,
        "experience_years": {"min": min_experience, "max": max_experience},
    }
