"""Stage 6: build the explainable ScreeningReport.

Strengths, weaknesses, recommendation, and reasoning are derived deterministically
from the scored data — every line is grounded in the candidate's own resume and
the computed numbers. No free-form LLM prose, no fabrication.
"""

from config import Settings
from models import CandidateProfile, ScreeningReport, SkillMatch


def _strengths(scored: dict, profile: CandidateProfile, within_window: bool) -> list[str]:
    out = []
    strong = [
        m for m in scored["skill_matches"]
        if m.matched and m.confidence >= 1.0 and m.weight >= 0.8
    ]
    for m in sorted(strong, key=lambda x: -x.weight):
        out.append(
            f"Evidence-backed {m.required} (matched '{m.matched_to}', "
            f"similarity {m.similarity:.2f})."
        )
    if within_window:
        out.append(
            f"{scored['experience_years']:g} years of experience fits the target range."
        )
    if profile.projects:
        out.append(f"{len(profile.projects)} project(s) listed, showing applied work.")
    if profile.certifications:
        out.append(f"Holds {len(profile.certifications)} certification(s).")
    return out or ["No standout strengths detected for this role."]


def _weaknesses(scored: dict, settings: Settings) -> list[str]:
    out = []
    for skill in scored["missing_skills"]:
        out.append(f"No evidence of required skill: {skill}.")
    weak = [
        m for m in scored["skill_matches"]
        if m.matched and 0 < m.confidence < 1.0
    ]
    for m in weak:
        out.append(
            f"Claims {m.required} (via '{m.matched_to}') but it is not supported "
            f"by any experience or project."
        )
    years = scored["experience_years"]
    if not scored["experience_ok"]:
        out.append(
            f"Experience ({years:g} yrs) is below the {settings.min_experience:g}-year minimum."
        )
    elif not scored["within_window"]:
        out.append(
            f"Experience ({years:g} yrs) exceeds the {settings.max_experience:g}-year "
            f"target (possibly overqualified)."
        )
    return out or ["No significant weaknesses detected."]


def _recommendation(verdict: str, score: int) -> str:
    if verdict == "FIT" and score >= 80:
        return "Strong match — advance to a technical interview."
    if verdict == "FIT":
        return "Potential match — worth a screening call before deciding."
    if score >= 45:
        return "Borderline — does not currently meet the bar for this role."
    return "Not a fit for this SDE role."


def _reasoning(scored: dict, settings: Settings) -> list[str]:
    return [
        f"Weighted skill coverage: {scored['match_percentage']:.0f}% "
        f"(matched {len(scored['matched_skills'])} of "
        f"{len(scored['matched_skills']) + len(scored['missing_skills'])} required skills).",
        f"Experience factor: {scored['experience_factor']:g} "
        f"({scored['experience_years']:g} yrs vs target "
        f"{settings.min_experience:g}-{settings.max_experience:g}).",
        f"Keyword-stuffing penalty applied: {scored['stuffing_penalty'] * 100:.0f}%.",
        f"Final score = coverage × experience × (1 − penalty) = {scored['score']} "
        f"(FIT threshold {settings.fit_threshold:g}).",
    ]


def build_report(
    scored: dict,
    profile: CandidateProfile,
    reference: dict,
    extra_flags: list[str],
    settings: Settings,
) -> ScreeningReport:
    return ScreeningReport(
        verdict=scored["verdict"],
        score=scored["score"],
        match_percentage=scored["match_percentage"],
        matched_skills=scored["matched_skills"],
        missing_skills=scored["missing_skills"],
        strengths=_strengths(scored, profile, scored["within_window"]),
        weaknesses=_weaknesses(scored, settings),
        recommendation=_recommendation(scored["verdict"], scored["score"]),
        reasoning=_reasoning(scored, settings),
        flags=extra_flags,
        experience_years=scored["experience_years"],
        experience_ok=scored["experience_ok"],
        candidate_resume=profile,
        reference_resume=reference,
        skill_matches=scored["skill_matches"],
    )
