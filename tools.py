"""Tools available to the matching agent.

- File-system tools from Milestone 1 (src/filesSystemTools/fs_tools.py)
- RAG search tool from Milestone 2 (src/job_matcher.py)
- extract_requirements, compare_candidates, generate_interview_questions
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import screening
from filesSystemTools import fs_tools
from job_matcher import job_matches


# ---------------------------------------------------------------- Milestone 1: file system

@tool
def read_file(filepath: str) -> dict:
    """Read a TXT, MD, RTF, PDF or DOCX file and return its text content and metadata."""
    return fs_tools.read_file(filepath)


@tool
def list_files(directory: str, extension: str | None = None) -> list[dict]:
    """List files under a directory, optionally filtered by extension (e.g. 'pdf')."""
    return fs_tools.list_files(directory, extension)


@tool
def search_in_file(filepath: str, keyword: str) -> dict:
    """Case-insensitive keyword search inside a file; returns matching lines with context."""
    return fs_tools.search_in_file(filepath, keyword)


@tool
def write_file(filepath: str, content: str) -> dict:
    """Write text content to a file, creating parent directories if needed."""
    return fs_tools.write_file(filepath, content)


# ---------------------------------------------------------------- Milestone 2: RAG

@tool
def search_resumes(job_description: str, top_k: int = 10, strict: bool = True) -> dict:
    """Semantic + keyword search over the indexed resumes in ChromaDB.

    Include a 'Required Skills: a, b, c' line and 'N+ years' in job_description to
    enable skill/experience scoring. strict=False keeps partial matches.
    """
    return job_matches(job_description, top_k=top_k, strict=strict)


# ---------------------------------------------------------------- additional tools

@tool
def extract_requirements(jd: str) -> str:
    """Parse a job description into must-have vs nice-to-have requirements.

    jd may be JD text, a path to a JD file, or a short query like 'React and 3+ years experience'.
    """
    return screening.extract_requirements(screening.load_job_description(jd)).model_dump_json(indent=2)


@tool
def compare_candidates(candidate_ids: list[str], state: Annotated[dict, InjectedState]) -> str:
    """Head-to-head comparison of shortlisted candidates, explaining why one ranks above another.

    candidate_ids may be ids (e.g. 'priya_patel'), names, or ranks ('1', '2', '3').
    """
    shortlist = state.get("shortlist", [])
    resolved = [screening.find_candidate(shortlist, ref) for ref in candidate_ids]
    unknown = [ref for ref, c in zip(candidate_ids, resolved) if c is None]
    if unknown:
        return f"Not in the current shortlist: {unknown}. Shortlisted ids: {[c.candidate_id for c in shortlist]}"
    return screening.compare_candidates(state["requirements"], sorted(resolved, key=lambda c: c.rank))


@tool
def generate_interview_questions(candidate_id: str, state: Annotated[dict, InjectedState]) -> str:
    """Create tailored screening questions for a shortlisted candidate that verify strengths and probe gaps."""
    candidate = screening.find_candidate(state.get("shortlist", []), candidate_id)
    if candidate is None:
        return f"Candidate '{candidate_id}' is not in the current shortlist."
    questions = screening.generate_interview_questions(state["requirements"], candidate)
    path = screening.save_report(state["requirements"], questions, f"{candidate.candidate_id}_questions")
    return f"{questions}\n\n(saved to {path})"


ALL_TOOLS = [
    read_file,
    list_files,
    search_in_file,
    write_file,
    search_resumes,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
]
