"""Stage 6: build the explainable ScreeningReport.

Strengths, weaknesses, recommendation(s), and reasoning are derived
deterministically from the scored data — every line is grounded in the
candidate's own resume and the requirements extracted from the pasted job
description. No free-form LLM prose, no fabrication, no hardcoded role.
"""

from config import Settings
from models import CandidateProfile, JobRequirements, ScreeningReport


def _required_matches(scored: dict):
    return [m for m in scored["skill_matches"] if m.kind == "required"]


def _strengths(scored: dict, profile: CandidateProfile, within_window: bool) -> list[str]:
    out = []
    strong = [
        m for m in scored["skill_matches"]
        if m.matched and m.confidence >= 1.0
    ]
    # Lead with required skills, then preferred; strongest weight first.
    strong.sort(key=lambda x: (x.kind != "required", -x.weight))
    for m in strong:
        label = "required" if m.kind == "required" else "preferred"
        out.append(
            f"Evidence-backed {label} skill {m.required} "
            f"(matched '{m.matched_to}', similarity {m.similarity:.2f})."
        )
    if within_window and scored["experience_years"] > 0:
        out.append(
            f"{scored['experience_years']:g} years of experience fits the role's range."
        )
    if profile.projects:
        out.append(f"{len(profile.projects)} project(s) listed, showing applied work.")
    if profile.certifications:
        out.append(f"Holds {len(profile.certifications)} certification(s).")
    return out or ["No standout strengths detected for this role."]


def _weaknesses(scored: dict, jd: JobRequirements, settings: Settings) -> list[str]:
    out = []
    for m in _required_matches(scored):
        if not m.matched:
            out.append(f"No evidence of required skill: {m.required}.")
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
    if jd.min_experience_years is not None and not scored["experience_ok"]:
        out.append(
            f"Experience ({years:g} yrs) is below the role's "
            f"{jd.min_experience_years:g}-year minimum."
        )
    elif not scored["within_window"] and jd.max_experience_years is not None:
        out.append(
            f"Experience ({years:g} yrs) exceeds the role's "
            f"{jd.max_experience_years:g}-year target (possibly overqualified)."
        )
    return out or ["No significant weaknesses detected."]


def _recommendation(verdict: str, score: int, role_title: str) -> str:
    if verdict == "FIT" and score >= 80:
        return f"Strong match for {role_title} — a competitive application."
    if verdict == "FIT":
        return f"Potential match for {role_title} — worth applying with a tailored resume."
    if score >= 45:
        return f"Borderline for {role_title} — close some gaps before applying."
    return f"Not currently a fit for {role_title}."


def _recommendations(scored: dict, jd: JobRequirements) -> list[str]:
    """Concrete, resume-improvement suggestions (what the candidate can do)."""
    out = []
    missing_required = [m.required for m in _required_matches(scored) if not m.matched]
    if missing_required:
        out.append(
            "Add concrete experience or projects demonstrating: "
            + ", ".join(missing_required)
            + "."
        )
    missing_preferred = [
        m.required for m in scored["skill_matches"]
        if m.kind == "preferred" and not m.matched
    ]
    if missing_preferred:
        out.append(
            "Strengthen the application with these preferred skills if you have them: "
            + ", ".join(missing_preferred)
            + "."
        )
    unsupported = [
        m.required for m in scored["skill_matches"]
        if m.matched and 0 < m.confidence < 1.0
    ]
    if unsupported:
        out.append(
            "Back up listed skills with specifics in your experience/projects: "
            + ", ".join(unsupported)
            + "."
        )
    if jd.min_experience_years is not None and not scored["experience_ok"]:
        out.append(
            f"Highlight more relevant experience — the role expects at least "
            f"{jd.min_experience_years:g} years."
        )
    if scored["stuffing_penalty"] > 0:
        out.append(
            "Reduce keyword repetition and tie every listed skill to real work."
        )
    return out or ["Resume already aligns well with this job description."]


def _experience_comparison(scored: dict, jd: JobRequirements) -> str:
    years = scored["experience_years"]
    if jd.min_experience_years is None:
        return (
            f"Candidate has {years:g} years of experience; the job lists no "
            f"specific experience requirement."
        )
    if jd.max_experience_years is not None:
        window = f"{jd.min_experience_years:g}-{jd.max_experience_years:g}"
    else:
        window = f"{jd.min_experience_years:g}+"
    status = "meets" if scored["experience_ok"] else "below"
    return (
        f"Candidate has {years:g} years vs the role's {window} years required "
        f"({status} the requirement)."
    )


def _reasoning(scored: dict, jd: JobRequirements, settings: Settings) -> list[str]:
    total = len(scored["matched_skills"]) + len(scored["missing_skills"])
    if jd.min_experience_years is None:
        exp_line = (
            f"Experience factor: {scored['experience_factor']:g} "
            f"({scored['experience_years']:g} yrs; no requirement stated)."
        )
    else:
        window = (
            f"{jd.min_experience_years:g}"
            + (f"-{jd.max_experience_years:g}" if jd.max_experience_years is not None else "+")
        )
        exp_line = (
            f"Experience factor: {scored['experience_factor']:g} "
            f"({scored['experience_years']:g} yrs vs required {window})."
        )
    return [
        f"Weighted skill coverage: {scored['match_percentage']:.0f}% "
        f"(matched {len(scored['matched_skills'])} of {total} job-derived skills).",
        exp_line,
        f"Keyword-stuffing penalty applied: {scored['stuffing_penalty'] * 100:.0f}%.",
        f"Final score = coverage × experience × (1 − penalty) = {scored['score']} "
        f"(FIT threshold {settings.fit_threshold:g}).",
    ]


def build_report(
    scored: dict,
    profile: CandidateProfile,
    jd: JobRequirements,
    extra_flags: list[str],
    settings: Settings,
) -> ScreeningReport:
    if jd.min_experience_years is None:
        experience_required = None
    else:
        experience_required = {
            "min": jd.min_experience_years,
            "max": jd.max_experience_years,
        }
    return ScreeningReport(
        verdict=scored["verdict"],
        role_title=jd.role_title,
        score=scored["score"],
        match_percentage=scored["match_percentage"],
        matched_skills=scored["matched_skills"],
        missing_skills=scored["missing_skills"],
        strengths=_strengths(scored, profile, scored["within_window"]),
        weaknesses=_weaknesses(scored, jd, settings),
        recommendation=_recommendation(scored["verdict"], scored["score"], jd.role_title),
        recommendations=_recommendations(scored, jd),
        reasoning=_reasoning(scored, jd, settings),
        flags=extra_flags,
        experience_years=scored["experience_years"],
        experience_ok=scored["experience_ok"],
        experience_required=experience_required,
        experience_comparison=_experience_comparison(scored, jd),
        candidate_resume=profile,
        job_requirements=jd,
        skill_matches=scored["skill_matches"],
    )
