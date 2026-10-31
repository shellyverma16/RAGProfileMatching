"""State and schemas for the matching agent (matching_agent.py)."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.message import add_messages
from pydantic import BaseModel, BeforeValidator, Field

# LLMs sometimes send null for an empty list.
StrList = Annotated[list[str], BeforeValidator(lambda value: value or [])]


class JobRequirements(BaseModel):
    """The agent's structured understanding of the job description."""

    title: str = Field(description="Job title")
    company: str = Field(default="", description="Hiring company, if stated")
    location: str = Field(default="", description="Job location, if stated")
    onsite_required: bool = Field(
        default=False, description="True only if the JD explicitly says the role is on-site only (no remote)"
    )
    required_skills: StrList = Field(description="Must-have skills, one short canonical name per item")
    nice_to_have_skills: StrList = Field(default_factory=list, description="Preferred but optional skills")
    min_experience_years: int = Field(default=0, description="Minimum years of experience required")
    responsibilities: StrList = Field(default_factory=list, description="Key responsibilities")
    search_query: str = Field(description="Short natural-language query describing the ideal candidate")


class Assessment(BaseModel):
    """Round 2: the LLM's deep analysis of one candidate."""

    match_score: float = Field(description="0-100 fit for the role, higher is better")
    matched_skills: StrList = Field(default_factory=list, description="Must-have skills the candidate has")
    missing_skills: StrList = Field(default_factory=list, description="Must-have skills the candidate lacks")
    strengths: StrList = Field(default_factory=list, description="Concrete strengths for this role")
    gaps: StrList = Field(default_factory=list, description="Concrete gaps or risks for this role")
    relevant_excerpts: StrList = Field(default_factory=list, description="1-2 short verbatim resume quotes")
    reasoning: str = Field(description="Two or three sentences justifying the score")
    improvement_suggestions: StrList = Field(
        default_factory=list,
        description="Only for borderline candidates (score 50-75): what would make them a fit",
    )


class CandidateDetails(Assessment):
    """A screened candidate: retrieval data + assessment + recommendation."""

    candidate_id: str
    candidate_name: str
    resume_path: str
    experience_years: int = 0
    retrieval_score: float = 0
    rank: int = 0
    recommendation: Literal["hire", "no_hire", "pending"] = "pending"
    recommendation_confidence: Literal["high", "medium", "low", ""] = ""
    recommendation_reason: str = ""


class Decision(BaseModel):
    candidate_id: str
    decision: Literal["hire", "no_hire"]
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(description="One or two sentences")


class Recommendations(BaseModel):
    """Round 3: hire/no-hire decisions for the screened candidates."""

    decisions: list[Decision]


class InterviewQuestion(BaseModel):
    question: str
    focus: str = Field(description="Skill or gap this question probes")
    good_answer: str = Field(description="What a strong answer looks like")


class InterviewQuestions(BaseModel):
    questions: list[InterviewQuestion]


class FeedbackDecision(BaseModel):
    """How the agent interprets a recruiter message in the Human Feedback Loop."""

    # Approval is never inferred by the model: only an explicit approval word ends the loop
    # (APPROVALS in matching_agent.py), so a request can't be mistaken for sign-off.
    action: Literal["refine_requirements", "ask"] = Field(
        description=(
            "refine_requirements: the recruiter changes what the job requires (skills, experience, "
            "a different role or a new search), so candidates must be searched and re-ranked. "
            "ask: a question or request about the current shortlist (compare candidates, explain a "
            "ranking, interview questions, look up a file)."
        )
    )


class AgentState(TypedDict, total=False):
    # Conversation history: recruiter turns and the agent's replies and reasoning steps.
    messages: Annotated[list[AnyMessage], add_messages]

    # Job requirements understanding.
    job_input: str                 # raw input: a natural-language query, JD text or a JD file path
    job_description: str           # parsed JD text
    requirements: JobRequirements

    # Candidate shortlist and reasoning.
    search_results: list[dict]     # round 1: RAG top 10
    shortlist: list[CandidateDetails]           # rounds 2-3: assessed, ranked, hire/no-hire
    previous_shortlist: list[CandidateDetails]  # ranking before the latest refinement
    ranking_changes: str

    # Report and Human Feedback Loop.
    report: str
    report_path: str
    feedback: str                  # latest requirement change from the recruiter
    iteration: int                 # number of refinements


def make_checkpointer() -> InMemorySaver:
    """In-memory checkpointer that allows the Pydantic models stored in AgentState."""
    models = ("JobRequirements", "CandidateDetails")
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[(__name__, m) for m in models]))
