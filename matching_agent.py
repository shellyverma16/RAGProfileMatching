"""LangGraph candidate matching agent with a conversational CLI.

START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates
      → Generate Report → Human Feedback Loop → END

Search Resumes is the initial screen (RAG top 10 of 100 resumes). Rank Candidates runs the
second round (deep analysis of the top 10) and the final round (hire / no-hire).

The Human Feedback Loop is the chat: the recruiter can ask questions (the agent answers with
its tools and loops back for the next message), change requirements (back to Extract
Requirements, then re-rank and explain the changes), or approve (END).

Run from the project root after indexing resumes:
    uv run python matching_agent.py "Find me candidates with React and 3+ years experience"
    uv run python matching_agent.py dataset/job_descriptions/data_engineer.pdf
"""

import argparse
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import screening
import tools
from state import AgentState, FeedbackDecision, JobRequirements, make_checkpointer

APPROVALS = {"y", "yes", "ok", "approve", "approved", "lgtm", "done", "looks good", "approve shortlist"}
STATEFUL_TOOLS = {"compare_candidates", "generate_interview_questions"}
TOOLS_BY_NAME = {t.name: t for t in tools.ALL_TOOLS}
MAX_TOOL_STEPS = 6

ASK_PROMPT = """You are a recruiting assistant answering questions about the current candidate shortlist.

- Comparisons ("compare the top 3", "why did A rank higher than B"): always call compare_candidates
  with ids, names or ranks, then answer from its result.
- Interview questions: always call generate_interview_questions; never write questions yourself.
- Reading or searching resumes and other files: use the file-system tools; search_resumes queries the index.
- Strengths, gaps, improvement suggestions and hire/no-hire reasons are in the match report below.

Ground every claim in the match report or tool output. Refer to candidates by name and rank. Be concise."""

CLASSIFY_PROMPT = """Classify the recruiter's message about the current candidate shortlist.

refine_requirements only when the message changes the criteria: adds, removes or swaps a skill,
changes the minimum experience, or asks for a different role. Examples:
  "Make TypeScript a must-have" / "Lower experience to 2 years" / "Search for Django developers instead"
ask for everything about the current shortlist, including requests to produce something. Examples:
  "Generate interview questions for the top candidate" / "Compare the top 3" /
  "Why did #1 rank above #2?" / "Show the report for Daniel" / "List the JD files"

Current requirements: {requirements}"""

llm_with_tools =screening.llm.bind_tools(tools.ALL_TOOLS)


# ---------------------------------------------------------------- nodes

def parse_jd(state: AgentState) -> dict:
    """Load the JD from a file path, or treat the input as JD text or a natural-language query."""
    job_description = screening.load_job_description(state["job_input"])
    source = "file" if job_description != state["job_input"].strip() else "query"
    return {
        "job_description": job_description,
        "iteration": 0,
        "messages": [
            HumanMessage(content=state["job_input"]),
            AIMessage(content=f"Parsed job description from {source} ({len(job_description)} chars)."),
        ],
    }


def extract_requirements(state: AgentState) -> dict:
    """Turn the JD, plus any requirement change from the recruiter, into must-have / nice-to-have."""
    if state.get("feedback"):
        # human_feedback already applied the recruiter's change (it checks that something changed).
        requirements = state["requirements"]
    else:
        requirements = JobRequirements.model_validate_json(
            tools.extract_requirements.invoke({"jd": state["job_description"]})
        )
    return {
        "requirements": requirements,
        "messages": [AIMessage(content=(
            f"Requirements for '{requirements.title}': must-have {requirements.required_skills}, "
            f"nice-to-have {requirements.nice_to_have_skills}, "
            f"min {requirements.min_experience_years} years."
        ))],
    }


def search_resumes(state: AgentState) -> dict:
    """Initial screen: RAG search over all indexed resumes, keeping the top 10."""
    matches, mode = screening.initial_screen(state["requirements"])
    return {
        "search_results": matches,
        "messages": [AIMessage(content=f"Round 1 RAG screen: top {len(matches)} of all indexed resumes ({mode}).")],
    }


def rank_candidates(state: AgentState) -> dict:
    """Second round: deep analysis of the top 10. Final round: hire / no-hire for each."""
    assessed = screening.deep_analysis(state["requirements"], state["search_results"])
    shortlist = screening.recommend(state["requirements"], assessed)
    hires = [c.candidate_name for c in shortlist if c.recommendation == "hire"]
    return {
        "shortlist": shortlist,
        "messages": [
            AIMessage(content="Round 2 deep analysis: " + ", ".join(
                f"{c.candidate_name} ({c.match_score:.0f})" for c in shortlist
            )),
            AIMessage(content=f"Round 3 recommendations: hire {hires or 'none'}."),
        ],
    }


def generate_report(state: AgentState) -> dict:
    """Write the detailed match report and summarise it, with ranking changes after a refinement."""
    req = state["requirements"]
    changes = screening.explain_ranking_changes(state.get("previous_shortlist", []), state["shortlist"])
    report = screening.render_report(req, state["shortlist"], state.get("iteration", 0), changes)
    path = screening.save_report(req, report)
    summary = f"Shortlist for {req.title}:\n\n{screening.summary_table(state['shortlist'])}"
    if changes:
        summary += f"\n\nRanking changes:\n{changes}"
    summary += f"\n\nFull match report with strengths, gaps and improvement suggestions: {path}"
    return {
        "report": report,
        "report_path": path,
        "ranking_changes": changes,
        "messages": [AIMessage(content=summary)],
    }


def answer_question(state: AgentState, question: str) -> tuple[str, list[str]]:
    """Answer a recruiter question with the agent's tools. Returns the answer and the tools used."""
    conversation = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-20:]
    messages = [
        SystemMessage(content=f"{ASK_PROMPT}\n\nCurrent match report:\n{state['report']}"),
        *conversation,
        HumanMessage(content=question),
    ]
    used = []
    for _ in range(MAX_TOOL_STEPS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.text, used
        for call in response.tool_calls:
            args = {**call["args"], "state": state} if call["name"] in STATEFUL_TOOLS else call["args"]
            try:
                output = TOOLS_BY_NAME[call["name"]].invoke(args)
            except Exception as error:  # report tool failures to the model instead of crashing the chat
                output = f"Tool error: {error}"
            used.append(f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")
            messages.append(ToolMessage(str(output), tool_call_id=call["id"], name=call["name"]))
    return "I could not finish answering with the available tools. Please rephrase the question.", used


def human_feedback(state: AgentState) -> Command[Literal["extract_requirements", "human_feedback", "__end__"]]:
    """Wait for the recruiter, then approve, refine requirements, or answer and wait again."""
    last = state["messages"][-1]
    reply = str(interrupt({
        "message": last.content,
        "tools_used": last.additional_kwargs.get("tools_used", []),
        "report_path": state.get("report_path"),
    })).strip()

    if reply.lower().strip(" .!") in APPROVALS:
        action = "approve"
    elif not reply:  # a stray Enter must not approve the shortlist; ask again
        return Command(goto="human_feedback")
    else:
        action = screening.structured(FeedbackDecision).invoke([
            SystemMessage(content=CLASSIFY_PROMPT.format(requirements=state["requirements"].model_dump_json())),
            HumanMessage(content=reply),
        ]).action

    if action == "refine_requirements":
        refined = screening.extract_requirements(state["job_description"], state["requirements"], reply)
        if not screening.requirements_changed(state["requirements"], refined):
            action = "ask"  # misclassified: nothing a search or score depends on changed

    recruiter = HumanMessage(content=reply)
    if action == "approve":
        return Command(goto=END, update={"messages": [
            recruiter, AIMessage(content=f"Shortlist approved. Final report: {state.get('report_path')}"),
        ]})
    if action == "refine_requirements":
        return Command(goto="extract_requirements", update={
            "feedback": reply,
            "requirements": refined,
            "previous_shortlist": state["shortlist"],
            "iteration": state.get("iteration", 0) + 1,
            "messages": [recruiter, AIMessage(content="Updating requirements, then searching and re-ranking.")],
        })
    answer, used = answer_question(state, reply)
    return Command(goto="human_feedback", update={"messages": [
        recruiter, AIMessage(content=answer, additional_kwargs={"tools_used": used}),
    ]})


# ---------------------------------------------------------------- graph

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("parse_jd", parse_jd)
    builder.add_node("extract_requirements", extract_requirements)
    builder.add_node("search_resumes", search_resumes)
    builder.add_node("rank_candidates", rank_candidates)
    builder.add_node("generate_report", generate_report)
    builder.add_node("human_feedback", human_feedback)

    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "extract_requirements")
    builder.add_edge("extract_requirements", "search_resumes")
    builder.add_edge("search_resumes", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", "human_feedback")
    # human_feedback routes itself via Command: END, extract_requirements, or itself for the next question.

    return builder.compile(checkpointer=checkpointer or make_checkpointer())


graph = build_graph()


# ---------------------------------------------------------------- chat CLI

def run(graph_input, config) -> tuple[list[dict], dict]:
    """Run the graph until it pauses or ends, printing each node's reasoning step."""
    for update in graph.stream(graph_input, config, stream_mode="updates"):
        for node, values in update.items():
            if node in {"__interrupt__", "generate_report", "human_feedback"} or not isinstance(values, dict):
                continue
            for message in values.get("messages", []):
                if isinstance(message, AIMessage):
                    print(f"  [{node}] {message.content}")
    snapshot = graph.get_state(config)
    return [i.value for task in snapshot.tasks for i in task.interrupts], snapshot.values


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the candidate matching agent.")
    parser.add_argument("job", nargs="?", help="A query like 'React and 3+ years', JD text, or a JD file path.")
    args = parser.parse_args()

    print("Candidate matching agent. Ask questions, change requirements, or type 'approve' to finish.")
    job = (args.job or "").strip()
    while not job:
        job = input("\nWhat role are you hiring for? (a query, JD text or JD file path)\n> ").strip()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    pending, values = run({"job_input": job}, config)
    while pending:
        payload = pending[0]
        if payload["tools_used"]:
            print("\n  [tools] " + "; ".join(payload["tools_used"]))
        print("\n" + payload["message"])
        try:
            reply = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            reply = "approve"
        pending, values = run(Command(resume=reply), config)
    print("\n" + values["messages"][-1].content)


if __name__ == "__main__":
    main()
