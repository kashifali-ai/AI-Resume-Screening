"""Deterministic, offline MockLLM extraction provider.

When MOCK_LLM=true this stands in for Gemini so the whole pipeline (and bulk
screening of up to 50 resumes) can run with ZERO Gemini calls, zero quota, and
no API key — useful for tests, demos, and CI.

IMPORTANT scope note: this module emulates the *LLM extraction* stage only — it
turns raw resume / JD text into structured CandidateProfile / JobRequirements
data, which is exactly what Gemini does here. It is rule-based (section + token
parsing) purely to *extract* fields. It does NOT do skill matching: skill
equivalence is still decided downstream by embeddings in `semantic.py`, and the
score/verdict are still computed in `scoring.py`. The rest of the pipeline is
unchanged regardless of which provider produced the extracted data.
"""

import re

from logging_config import get_logger

log = get_logger(__name__)

# --- section header vocabularies -------------------------------------------

_RESUME_HEADERS = {
    "summary": ["summary", "profile", "objective", "about", "professional summary",
                "career objective"],
    "skills": ["skills", "technical skills", "core skills", "key skills",
               "skill set", "areas of expertise", "core competencies"],
    "technologies": ["technologies", "tech stack", "technical proficiencies",
                     "tools", "tools & technologies", "tools and technologies"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment", "employment history", "work history"],
    "projects": ["projects", "personal projects", "key projects",
                 "academic projects", "selected projects"],
    "education": ["education", "academic background", "qualifications", "academics"],
    "certifications": ["certifications", "certificates", "licenses",
                       "certification", "licences"],
}

_JD_HEADERS = {
    "required": ["required skills", "requirements", "required qualifications",
                 "must have", "must-have", "what you'll need", "what you need",
                 "minimum qualifications", "required", "key requirements",
                 "skills required"],
    "preferred": ["preferred skills", "preferred qualifications", "nice to have",
                  "nice-to-have", "bonus", "preferred", "good to have", "pluses",
                  "nice to haves"],
    "technologies": ["technologies", "tech stack", "tools", "technical skills"],
    "responsibilities": ["responsibilities", "what you'll do", "duties", "role",
                         "key responsibilities", "the role", "what you will do"],
    "education": ["education", "education requirements", "educational requirements"],
}

_BULLET_RE = re.compile(r"^[\-\*•●▪‣⁃∙●▪◦·•\s]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Inline cue phrases that introduce a skill list in prose JDs/resumes, e.g.
# "experience with Python, Java and FastAPI" or "proficiency in SQL".
_CUE_RE = re.compile(
    r"\b(?:experience (?:with|in|using)|proficien(?:t|cy) (?:in|with)|"
    r"knowledge of|familiar(?:ity)? with|expertise in|skilled in|skills? (?:in|with)|"
    r"strong (?:in|with)|background in|hands[- ]on (?:with|in)|"
    r"working knowledge of|competen(?:t|cy) (?:in|with)|build(?:ing)? with|"
    r"developed (?:in|with)|using)\b",
    re.I,
)
# Tokens that are clearly NOT skills (requirement sentences, education lines, …).
_NOISE_RE = re.compile(
    r"\b(years?|experience|require[sd]?|responsib\w*|degree|bachelor'?s?|"
    r"master'?s?|ph\.?d|ability|able to|including|etc|join|team|looking|"
    r"candidate|role|we are|you (?:should|will|have))\b",
    re.I,
)
_DATE_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:[-–—]|to)\s*((?:19|20)\d{2}|present|current|now|ongoing)",
    re.I,
)
_EXP_RANGE_RE = re.compile(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*\+?\s*years", re.I)
_EXP_MIN_RE = re.compile(r"(\d{1,2})\s*\+?\s*years", re.I)


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def _inner(text: str, tag: str) -> str:
    """Pull the content the extraction prompt wrapped in <tag>…</tag>."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return (m.group(1) if m else text).strip()


def _match_header(line: str, header_map: dict[str, list[str]]):
    """Return (canonical_section, inline_content) if `line` is a section header,
    else (None, None). Handles both 'Header' lines and 'Header: inline' lines."""
    if ":" in line:
        head, _, rest = line.partition(":")
        h = head.strip().lower()
        for canon, kws in header_map.items():
            if h in kws:
                return canon, rest.strip()
    stripped = line.strip().rstrip(":").lower()
    if len(stripped.split()) <= 4:
        for canon, kws in header_map.items():
            if stripped in kws:
                return canon, ""
    return None, None


def _split_sections(text: str, header_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Group lines under detected section headers. Lines before the first header
    land under '_preamble'."""
    sections: dict[str, list[str]] = {"_preamble": []}
    current = "_preamble"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        canon, inline = _match_header(line, header_map)
        if canon:
            current = canon
            sections.setdefault(canon, [])
            if inline:
                sections[canon].append(inline)
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def _skill_tokens(lines: list[str]) -> list[str]:
    """Split clean skill/technology section lines into individual skill phrases.

    Splits on commas, semicolons and pipes only — multiword skills like
    'REST APIs', 'Data Structures and Algorithms' or 'CI/CD' stay intact.
    Requirement-sentence noise ('5+ years of experience') is dropped."""
    out = []
    for line in lines:
        for part in re.split(r"[,;|]", _strip_bullet(line)):
            p = part.strip().strip(".").strip()
            if not p or len(p.split()) > 7 or _NOISE_RE.search(p):
                continue
            out.append(p)
    return _dedup(out)


def _split_skill_list(segment: str) -> list[str]:
    """Split a free-text fragment (e.g. a cue-phrase tail) into skills,
    breaking on commas/semicolons/pipes AND conjunctions (and/or/&)."""
    out = []
    for part in re.split(r",|;|\||&|\band\b|\bor\b", segment, flags=re.I):
        p = _strip_bullet(part).strip(" .").strip()
        # Trim trailing filler clauses: "Docker is expected" -> "Docker".
        p = re.sub(r"\s+(?:is|are|would be|will be|a)\b.*$", "", p, flags=re.I).strip()
        if p and len(p.split()) <= 6 and not _NOISE_RE.search(p):
            out.append(p)
    return out


def _cue_skills(text: str) -> list[str]:
    """Extract skills from inline cue phrases anywhere in the text, e.g.
    'experience with Python, Java and FastAPI' -> [Python, Java, FastAPI]."""
    out = []
    for m in _CUE_RE.finditer(text):
        segment = re.split(r"[.;\n]", text[m.end():], maxsplit=1)[0]
        out += _split_skill_list(segment)
    return _dedup(out)


def _clean_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        s = _strip_bullet(line)
        if s:
            out.append(s)
    return out


def _parse_experiences(lines: list[str]) -> list[dict]:
    """Build experience entries from dated lines. Years come from the date
    ranges; the full section text is attached as evidence so the downstream
    confidence/grounding logic works the same as with Gemini output."""
    section_text = " ".join(_strip_bullet(l) for l in lines).strip()
    exps: list[dict] = []
    for line in lines:
        m = _DATE_RANGE_RE.search(line)
        if m:
            exps.append({
                "title": _strip_bullet(line),
                "company": None,
                "start": m.group(1),
                "end": m.group(2),
                "description": "",
            })
    if not exps:
        if section_text:
            exps.append({"title": "", "company": None, "start": None,
                         "end": None, "description": section_text})
        return exps
    # All evidence text on the first entry — span math is per-entry and merged,
    # so this does not affect the computed years.
    exps[0]["description"] = section_text
    return exps


def _first_name_line(preamble: list[str]) -> str:
    for line in preamble:
        s = _strip_bullet(line)
        if s and "@" not in s and not _DATE_RANGE_RE.search(s):
            return s[:80]
    return "Unknown"


def extract_resume(resume_text: str) -> dict:
    """Rule-based CandidateProfile extraction from raw resume text."""
    sec = _split_sections(resume_text, _RESUME_HEADERS)
    preamble = sec.get("_preamble", [])

    skills = _skill_tokens(sec.get("skills", []))
    technologies = _skill_tokens(sec.get("technologies", []))
    # Fallback for resumes without a clean Skills section: mine inline cues
    # ("experience with…", "built … using …") from the whole resume.
    if not skills and not technologies:
        skills = _cue_skills(resume_text)

    email_m = _EMAIL_RE.search(resume_text)
    summary = " ".join(_clean_lines(sec.get("summary", [])))[:400]
    if not summary and len(preamble) > 1:
        summary = " ".join(_clean_lines(preamble[1:3]))[:400]

    return {
        "candidate_name": _first_name_line(preamble),
        "email": email_m.group(0) if email_m else None,
        "candidate_summary": summary,
        "skills": skills,
        "technologies": technologies,
        "experience": _parse_experiences(sec.get("experience", [])),
        "projects": _clean_lines(sec.get("projects", [])),
        "education": _clean_lines(sec.get("education", [])),
        "certifications": _clean_lines(sec.get("certifications", [])),
    }


def _parse_experience_requirement(text: str):
    """Return (min_years, max_years) extracted from JD text, or (None, None)."""
    rng = _EXP_RANGE_RE.search(text)
    if rng:
        return float(rng.group(1)), float(rng.group(2))
    m = _EXP_MIN_RE.search(text)
    if m:
        return float(m.group(1)), None
    return None, None


def _role_title(preamble: list[str], full_text: str) -> str:
    for line in preamble:
        s = _strip_bullet(line)
        if s:
            # Trim a long first sentence down to the title phrase.
            return re.split(r"[.\n]", s)[0].strip()[:80]
    first = next((l.strip() for l in full_text.splitlines() if l.strip()), "")
    return (first[:80] or "Unknown Role")


def extract_requirements(jd_text: str) -> dict:
    """Rule-based JobRequirements extraction from raw job-description text."""
    sec = _split_sections(jd_text, _JD_HEADERS)
    preamble = sec.get("_preamble", [])

    required = _skill_tokens(sec.get("required", []))
    preferred = _skill_tokens(sec.get("preferred", []))
    technologies = _skill_tokens(sec.get("technologies", []))

    # Robustness: most real JDs are prose without a 'Required skills:' header.
    # Mine inline cue phrases across the whole JD; anything found that isn't
    # already a preferred/technology term is treated as required. This is what
    # prevents prose JDs from producing an empty requirement list (and thus
    # all-zero scores).
    known = {s.lower() for s in preferred + technologies}
    cue = [s for s in _cue_skills(jd_text) if s.lower() not in known]
    required = _dedup(required + cue)

    min_exp, max_exp = _parse_experience_requirement(jd_text)

    education = _clean_lines(sec.get("education", []))
    if not education:
        education = [
            l.strip() for l in jd_text.splitlines()
            if re.search(r"\b(bachelor|master|degree|b\.?tech|b\.?sc|ph\.?d)\b", l, re.I)
        ]

    return {
        "role_title": _role_title(preamble, jd_text) or "Unknown Role",
        "required_skills": required,
        "preferred_skills": preferred,
        "technologies": technologies,
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
        "education_requirements": education,
        "responsibilities": _clean_lines(sec.get("responsibilities", [])),
    }


class MockLLMClient:
    """LLMClient-compatible provider that performs NO network calls.

    Dispatches on the requested response model: CandidateProfile -> resume
    extraction, JobRequirements -> JD extraction. Returns plain dicts matching
    each schema (validated downstream just like real Gemini output)."""

    def __init__(self, settings):
        self._settings = settings
        # Per-instance call counts — proves the JD is extracted once in bulk.
        self.calls: dict[str, int] = {"CandidateProfile": 0, "JobRequirements": 0}
        log.info("[MOCK_LLM] Mock LLM provider active — Gemini will not be called.")

    def extract_json(self, system: str, user: str, response_model) -> dict:
        name = getattr(response_model, "__name__", "")
        self.calls[name] = self.calls.get(name, 0) + 1
        log.info("[MOCK_LLM] No Gemini call performed")
        if name == "JobRequirements":
            data = extract_requirements(_inner(user, "job_description"))
            log.info("[MOCK_LLM] JD extracted (role=%s, required=%d, preferred=%d)",
                     data["role_title"], len(data["required_skills"]),
                     len(data["preferred_skills"]))
            return data
        data = extract_resume(_inner(user, "resume"))
        log.info("[MOCK_LLM] Resume extracted (name=%s, skills=%d, experience=%d)",
                 data["candidate_name"], len(data["skills"]), len(data["experience"]))
        return data
