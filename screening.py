"""Screening logic used by the matching agent's nodes and tools.

Round 1  initial_screen   RAG search over all indexed resumes → top 10        (Search Resumes)
Round 2  deep_analysis    LLM reads each full resume → score, strengths, gaps (Rank Candidates)
Round 3  recommend        hire / no-hire for each assessed candidate          (Rank Candidates)
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import (
    Assessment,
    CandidateDetails,
    InterviewQuestions,
    JobRequirements,
    Recommendations,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filesSystemTools.fs_tools import read_file, write_file  # noqa: E402
from job_matcher import job_matches  # noqa: E402

load_dotenv()

REPORT_DIR = PROJECT_ROOT / "reports"
INITIAL_SCREEN_SIZE = 10
BORDERLINE = (50, 75)
HIRE_SCORE = 70           # rule-based decisions: hire needs every must-have and at least this score
SCORE_NOISE = 3           # score changes this small are run-to-run variation, not a real change
LOCATION_TERMS = re.compile(
    r"\b(locat\w*|relocat\w*|visa|on-?site|remote|based in|time ?zones?|commut\w*)\b", re.IGNORECASE
)

llm = ChatOpenAI(model="gpt-5-nano")


def structured(schema):
    # gpt-5-nano occasionally returns malformed structured output; retry before giving up.
    return llm.with_structured_output(schema, method="function_calling").with_retry(stop_after_attempt=3)


def is_borderline(candidate: CandidateDetails) -> bool:
    return BORDERLINE[0] <= candidate.match_score <= BORDERLINE[1]


def _scoring_view(req: JobRequirements) -> str:
    """Requirements as shown to the scoring rounds. Location is left out unless the role is on-site only."""
    exclude = {"search_query"} if req.onsite_required else {"search_query", "location", "company"}
    return req.model_dump_json(indent=2, exclude=exclude)


# ---------------------------------------------------------------- job understanding

def load_job_description(job: str) -> str:
    """Return JD text from a file path, or the input itself if it is not a path."""
    job = job.strip()
    path = Path(job)
    if len(job) < 300 and path.suffix and path.exists():
        result = read_file(str(path))
        if not result["success"]:
            raise RuntimeError(f"Could not read JD {path}: {result.get('error')}")
        return result["content"]
    return job


def extract_requirements(
    job_description: str, previous: JobRequirements | None = None, feedback: str = ""
) -> JobRequirements:
    """Parse must-have vs nice-to-have requirements, optionally applying recruiter changes."""
    prompt = [
        SystemMessage(content=(
            "You are a technical recruiter. Extract the job requirements from the job description, "
            "separating must-have skills from nice-to-have skills. Use short canonical skill names "
            "(e.g. 'Python', 'AWS'). Only use facts stated in the JD or requested by the recruiter. "
            "A short query like 'React and 3+ years' is a valid JD: infer a sensible title, and list only "
            "the skills it names; never add skills that are merely typical for the role. Location is not "
            "a skill: set onsite_required only when the JD explicitly says the role is on-site only."
        )),
        HumanMessage(content=job_description),
    ]
    if previous and feedback:
        prompt.append(HumanMessage(content=(
            f"Current requirements:\n{previous.model_dump_json(indent=2)}\n\n"
            f"Apply these recruiter changes and return the full updated requirements. If they describe "
            f"a different role or a new search, replace the requirements entirely:\n{feedback}"
        )))
    return structured(JobRequirements).invoke(prompt)


def requirements_changed(before: JobRequirements, after: JobRequirements) -> bool:
    """True when a change affects who passes the screen or how they are scored. Free-text fields
    (title, responsibilities, search query) are ignored: the model rewords them between calls."""
    def criteria(req: JobRequirements) -> tuple:
        return (
            {s.lower() for s in req.required_skills},
            {s.lower() for s in req.nice_to_have_skills},
            req.min_experience_years,
            req.onsite_required,
            req.location.lower() if req.onsite_required else "",
        )
    return criteria(before) != criteria(after)


# ---------------------------------------------------------------- round 1

def initial_screen(req: JobRequirements, top_k: int = INITIAL_SCREEN_SIZE) -> tuple[list[dict], str]:
    """RAG search for the top k. Candidates passing the hard skill/experience filters come first;
    if fewer than k pass, the rest are filled from the unfiltered ranking."""
    query = (
        f"{req.title}. {req.search_query}\n"
        f"Required Skills: {', '.join(req.required_skills)}\n"
        f"{req.min_experience_years}+ years of experience."
    )
    matches = job_matches(query, top_k=top_k, strict=True)["top_matches"]
    if len(matches) >= top_k:
        return matches, "all pass skill/experience filters"
    seen = {m["resume_path"] for m in matches}
    relaxed = [m for m in job_matches(query, top_k=top_k * 2, strict=False)["top_matches"] if m["resume_path"] not in seen]
    return matches + relaxed[: top_k - len(matches)], f"{len(matches)} pass skill/experience filters, rest filled by similarity"


# ---------------------------------------------------------------- round 2

def _resume_text(resume_path: str) -> str:
    result = read_file(str(PROJECT_ROOT / resume_path))
    return result.get("content", "")[:4000] if result.get("success") else ""


def _words(skill: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", skill.lower())


def _contains_words(outer: list[str], inner: list[str]) -> bool:
    return bool(inner) and any(outer[i:i + len(inner)] == inner for i in range(len(outer) - len(inner) + 1))


def _must_haves_only(skills: list[str], required: list[str]) -> list[str]:
    """Keep only items that name a must-have skill, written the way the requirements write it.

    Matching is by whole words, so 'Amazon Web Services (AWS)' counts as 'AWS' but 'PostgreSQL' is not 'SQL'.
    """
    kept = []
    for skill in skills:
        words = _words(skill)
        match = next(
            (r for r in required if _contains_words(words, _words(r)) or _contains_words(_words(r), words)),
            None,
        )
        if match and match not in kept:
            kept.append(match)
    return kept


def _clean(assessment: Assessment, req: JobRequirements) -> Assessment:
    """Validate an assessment: skill lists name real must-haves; location is not a gap unless on-site only."""
    matched = _must_haves_only(assessment.matched_skills, req.required_skills)
    missing = [s for s in _must_haves_only(assessment.missing_skills, req.required_skills) if s not in matched]
    gaps = assessment.gaps if req.onsite_required else [g for g in assessment.gaps if not LOCATION_TERMS.search(g)]
    return assessment.model_copy(update={"matched_skills": matched, "missing_skills": missing, "gaps": gaps})


def deep_analysis(req: JobRequirements, matches: list[dict], feedback: str = "") -> list[CandidateDetails]:
    """Assess every round-1 candidate against the full resume, then rank by score."""
    system = SystemMessage(content=(
        "You are a technical recruiter doing a deep second-round screen of one candidate. Compare the "
        "full resume to the requirements. Score 0-100: missing must-have skills or too little experience "
        "must lower the score substantially. Use the full range: reserve 90+ for candidates with every "
        "must-have, most nice-to-haves and strong, quantified evidence of the responsibilities. "
        "matched_skills and missing_skills may only contain items from required_skills. Strengths and "
        "gaps must be specific and grounded in the resume; do not mention the candidate's location, "
        "relocation, visa or remote status. Excerpts must be verbatim. Give improvement_suggestions only "
        f"when the score is between {BORDERLINE[0]} and {BORDERLINE[1]} (borderline): what the candidate "
        "would need, or what to verify in interview, to become a fit."
        + (f"\n\nRecruiter guidance to apply when scoring:\n{feedback}" if feedback else "")
    ))
    prompts = [
        [
            system,
            HumanMessage(content=(
                f"Requirements:\n{_scoring_view(req)}\n\n"
                f"Candidate: {match['candidate_name']} ({match['experience_years']} years, "
                f"retrieval score {match['match_score']})\n\nResume:\n{_resume_text(match['resume_path'])}"
            )),
        ]
        for match in matches
    ]
    results = structured(Assessment).batch(prompts, return_exceptions=True) if prompts else []
    # A score of 0 alongside matched must-haves is a malformed answer: retry once, then derive the score.
    retry = [
        i for i, a in enumerate(results)
        if isinstance(a, Assessment) and a.match_score <= 0 and _clean(a, req).matched_skills
    ]
    if retry:
        for i, result in zip(retry, structured(Assessment).batch([prompts[i] for i in retry], return_exceptions=True)):
            if isinstance(result, Assessment):
                results[i] = result
    failed = Assessment(match_score=0, reasoning="Not assessed: the model returned an invalid answer. Re-run to retry.")

    candidates = []
    for match, result in zip(matches, results):
        assessment = _clean(result, req) if isinstance(result, Assessment) else failed
        if assessment.match_score <= 0 and assessment.matched_skills:
            coverage = len(assessment.matched_skills) / max(len(req.required_skills), 1)
            assessment = assessment.model_copy(update={"match_score": round(80 * coverage)})
        candidate = CandidateDetails(
            **assessment.model_dump(),
            candidate_id=Path(match["resume_path"]).stem,
            candidate_name=match["candidate_name"],
            resume_path=match["resume_path"],
            experience_years=match["experience_years"],
            retrieval_score=match["match_score"],
        )
        if not is_borderline(candidate):
            candidate.improvement_suggestions = []
        candidates.append(candidate)

    candidates.sort(key=lambda c: (c.match_score, c.retrieval_score), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


# ---------------------------------------------------------------- round 3

def _contradicts(candidate: CandidateDetails, rationale: str) -> bool:
    """True if a rationale calls a skill missing that round 2 found, or raises location."""
    text = rationale.lower()
    if LOCATION_TERMS.search(text):
        return True
    return any(
        re.search(rf"\b(missing|lacks?|lacking|without|no)\b[^.;]*\b{re.escape(skill.lower())}\b", text)
        for skill in candidate.matched_skills
    )


def _rule_decision(candidate: CandidateDetails) -> tuple[str, str]:
    if candidate.missing_skills:
        return "no_hire", f"Missing must-have: {', '.join(candidate.missing_skills)}; score {candidate.match_score:.0f}."
    if candidate.match_score < HIRE_SCORE:
        return "no_hire", f"All must-haves present but score {candidate.match_score:.0f} is below {HIRE_SCORE}."
    return "hire", f"All must-haves present; score {candidate.match_score:.0f}."


def recommend(req: JobRequirements, candidates: list[CandidateDetails]) -> list[CandidateDetails]:
    """Make a hire/no-hire call for each assessed candidate, consistent with round 2's findings."""
    summaries = [
        c.model_dump(include={
            "candidate_id", "candidate_name", "match_score", "experience_years",
            "matched_skills", "missing_skills", "strengths", "gaps",
        })
        for c in candidates
    ]
    recommendations = structured(Recommendations).invoke([
        SystemMessage(content=(
            "You are the hiring manager making final-round decisions. For every candidate decide hire "
            "(advance to interviews) or no_hire, judging against the requirements and relative to this "
            "pool. Base each decision only on the fields given: missing_skills is the complete list of "
            "missing must-haves, so never say a candidate lacks anything else, and never consider "
            "location. Missing two or more must-haves means no_hire; a strong candidate with one "
            "learnable gap can be hire with lower confidence. Use each candidate_id exactly as given."
        )),
        HumanMessage(content=(
            f"Requirements:\n{_scoring_view(req)}\n\n"
            f"Assessed candidates:\n{json.dumps(summaries, indent=2)}"
        )),
    ])
    decisions = {d.candidate_id: d for d in recommendations.decisions}

    updated = []
    for candidate in candidates:
        decision = decisions.get(candidate.candidate_id)
        if (
            decision is None
            or len(candidate.missing_skills) >= 2
            or candidate.match_score < HIRE_SCORE
            or candidate.experience_years < req.min_experience_years - 1
            or _contradicts(candidate, decision.rationale)
        ):
            verdict, reason = _rule_decision(candidate)
            if candidate.experience_years < req.min_experience_years - 1:
                verdict, reason = "no_hire", (
                    f"{candidate.experience_years} years of experience against a "
                    f"{req.min_experience_years}-year minimum; score {candidate.match_score:.0f}."
                )
            update = {"recommendation": verdict, "recommendation_confidence": "high",
                      "recommendation_reason": reason}
        else:
            update = {"recommendation": decision.decision, "recommendation_confidence": decision.confidence,
                      "recommendation_reason": decision.rationale}
        updated.append(candidate.model_copy(update=update))
    return updated


# ---------------------------------------------------------------- explainability

def find_candidate(candidates: list[CandidateDetails], ref: str) -> CandidateDetails | None:
    """Resolve a candidate by id, full/first name, or rank ('1', '#1')."""
    key = ref.strip().lower().lstrip("#")
    for candidate in candidates:
        name = candidate.candidate_name.lower()
        if key in {candidate.candidate_id, name, name.split()[0], str(candidate.rank)}:
            return candidate
    return next((c for c in candidates if key in c.candidate_name.lower()), None)


def explain_ranking_changes(old: list[CandidateDetails], new: list[CandidateDetails]) -> str:
    """Describe how the ranking moved between two screening runs."""
    if not old:
        return ""
    old_by_id = {c.candidate_id: c for c in old}
    new_ids = {c.candidate_id for c in new}
    lines = []
    for c in new:
        prev = old_by_id.get(c.candidate_id)
        if prev is None:
            lines.append(f"- NEW: {c.candidate_name} enters at #{c.rank} (score {c.match_score:.0f}). {c.reasoning}")
            continue
        if prev.rank != c.rank:
            direction = "up" if c.rank < prev.rank else "down"
            noise = abs(c.match_score - prev.match_score) <= SCORE_NOISE
            lines.append(
                f"- {c.candidate_name} moved {direction} #{prev.rank} → #{c.rank} "
                f"(score {prev.match_score:.0f} → {c.match_score:.0f}). "
                + ("Score change is within normal run-to-run variation; the move comes from others' scores."
                   if noise else c.reasoning)
            )
        if prev.recommendation != c.recommendation:
            lines.append(
                f"- {c.candidate_name} recommendation {prev.recommendation} → {c.recommendation}: "
                f"{c.recommendation_reason}"
            )
    for prev in old:
        if prev.candidate_id not in new_ids:
            lines.append(f"- DROPPED: {prev.candidate_name} (was #{prev.rank}) no longer in the top {len(new)}.")
    return "\n".join(lines) or "No changes in ranking or recommendations."


def summary_table(candidates: list[CandidateDetails]) -> str:
    lines = [
        "| # | Candidate | ID | Score | Exp | Recommendation | Missing must-haves |",
        "|---|-----------|----|-------|-----|----------------|--------------------|",
    ]
    for c in candidates:
        flag = " (borderline)" if is_borderline(c) else ""
        lines.append(
            f"| {c.rank} | {c.candidate_name} | {c.candidate_id} | {c.match_score:.0f}{flag} | "
            f"{c.experience_years} | {c.recommendation} | {', '.join(c.missing_skills) or '-'} |"
        )
    return "\n".join(lines)


def candidate_section(c: CandidateDetails) -> str:
    decision = {"hire": "HIRE", "no_hire": "NO HIRE", "pending": "PENDING"}[c.recommendation]
    lines = [
        f"## #{c.rank} {c.candidate_name} (`{c.candidate_id}`): {c.match_score:.0f}/100, {decision}"
        + (" (borderline)" if is_borderline(c) else ""),
        "",
        f"- Resume: `{c.resume_path}`",
        f"- Experience: {c.experience_years} years; retrieval score {c.retrieval_score:.0f}",
        f"- Must-have matched: {', '.join(c.matched_skills) or '-'}; missing: {', '.join(c.missing_skills) or '-'}",
        f"- Recommendation ({c.recommendation_confidence or 'n/a'} confidence): {c.recommendation_reason or '-'}",
        f"- Reasoning: {c.reasoning}",
        "",
        "**Strengths**",
        *[f"- {s}" for s in c.strengths or ["-"]],
        "",
        "**Gaps**",
        *[f"- {g}" for g in c.gaps or ["-"]],
    ]
    if c.improvement_suggestions:
        lines += ["", "**Improvement suggestions**", *[f"- {s}" for s in c.improvement_suggestions]]
    if c.relevant_excerpts:
        lines += ["", "**Evidence**", *[f"> {e}" for e in c.relevant_excerpts]]
    return "\n".join(lines)


def render_report(
    req: JobRequirements, candidates: list[CandidateDetails], iteration: int = 0, changes: str = ""
) -> str:
    """Detailed match report: summary table, ranking changes, per-candidate breakdown."""
    lines = [
        f"# Match Report: {req.title}",
        "",
        f"**Must-have:** {', '.join(req.required_skills) or 'n/a'}  ",
        f"**Nice-to-have:** {', '.join(req.nice_to_have_skills) or 'n/a'}  ",
        f"**Minimum experience:** {req.min_experience_years} years  ",
        f"**Refinement round:** {iteration}",
        "",
        summary_table(candidates),
    ]
    if changes:
        lines += ["", "## Ranking changes", "", changes]
    for c in candidates:
        lines += ["", candidate_section(c)]
    return "\n".join(lines) + "\n"


def save_report(req: JobRequirements, report: str, suffix: str = "match_report") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", req.title.lower()).strip("_") or "job"
    saved = write_file(str(REPORT_DIR / f"{slug}_{suffix}.md"), report)
    return saved.get("path", "")


# ---------------------------------------------------------------- comparison + interviews

def compare_candidates(req: JobRequirements, candidates: list[CandidateDetails]) -> str:
    """Head-to-head comparison as Markdown, ending with a verdict on the ordering."""
    response = llm.invoke([
        SystemMessage(content=(
            "Compare these candidates head-to-head for the role. Output Markdown: first a table with one "
            "column per candidate and rows for rank, score, experience, must-have coverage, key strengths, "
            "key gaps and recommendation; then a short verdict explaining why the higher-ranked candidate "
            "ranks higher, citing specific evidence from strengths, gaps and skills. Use only the data given."
        )),
        HumanMessage(content=(
            f"Requirements:\n{_scoring_view(req)}\n\n"
            f"Candidates:\n{json.dumps([c.model_dump() for c in candidates], indent=2)}"
        )),
    ])
    return response.text


def generate_interview_questions(req: JobRequirements, candidate: CandidateDetails) -> str:
    """Screening questions that verify strengths and probe gaps."""
    result: InterviewQuestions = structured(InterviewQuestions).invoke([
        SystemMessage(content=(
            "Write 6-8 screening interview questions for this candidate and role. About half should verify "
            "claimed strengths with specifics; the rest should probe gaps, missing skills and risks. "
            "Questions must be specific to this resume, not generic."
        )),
        HumanMessage(content=(
            f"Requirements:\n{_scoring_view(req)}\n\n"
            f"Candidate assessment:\n{candidate.model_dump_json(indent=2)}\n\n"
            f"Resume:\n{_resume_text(candidate.resume_path)}"
        )),
    ])
    lines = [f"# Screening questions: {candidate.candidate_name} for {req.title}", ""]
    for number, q in enumerate(result.questions, start=1):
        lines += [f"{number}. **{q.question}**", f"   - Probes: {q.focus}", f"   - Strong answer: {q.good_answer}"]
    return "\n".join(lines)
