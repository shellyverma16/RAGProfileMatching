# RAGProfileMatching


## Synthetic Dataset

The repository includes a synthetic, fictional corpus for local experimentation:

- 100 resumes as PDF: `dataset/resumes/`
- 12 job descriptions as PDF: `dataset/job_descriptions/`
- `dataset/manifest.json` with document metadata, skills and relative paths

Each resume has a summary, skills, a dated job history and education. Beyond the
35 original candidates, each job has strong, partial and weak fits (the `tier`
field in the manifest) plus distractor roles, so ranking can be evaluated.

All names, experience, education, skills, and achievements in the corpus are fictional.
To recreate the corpus on macOS, run:

```bash
python3 dataset/generate_synthetic_dataset.py
```

The generator uses only the Python standard library and is deterministic.
Re-index after regenerating (see below).

## Resume Index

`src/resume_rag.py` uses the shared `list_files` and `read_file` file-system
tools (preferring RTF if a PDF and RTF copy share a name). It preserves
sections as chunks, uses OpenAI to extract name, skills, experience years, and
education, and stores each chunk plus metadata in ChromaDB with OpenAI
embeddings.

```bash
export OPENAI_API_KEY="your-key"
uv sync
uv run python src/resume_rag.py
```

The local vector database is created in `chroma_db/`.

## Job Matching

After indexing, pass a quoted job description to the matcher. It retrieves
semantic matches, requires listed skills and minimum experience, then combines
semantic and keyword scores in the JSON response.

```bash
uv run python src/job_matcher.py "Data Engineer. Required Skills: Python, SQL. 4+ years."
```

Separate must-have skills with commas. Use `or` for alternatives, for example
`Python, SQL or Airflow`.

## Matching Agent (LangGraph)

`matching_agent.py` builds the agent graph and runs the chat CLI. `state.py`
holds the agent state (conversation history, job requirements, candidate
shortlist and reasoning), `tools.py` the tools, and `screening.py` the
screening logic.

```
START → Parse JD → Extract Requirements → Search Resumes → Rank Candidates
      → Generate Report → Human Feedback Loop → END
```

Multi-round screening:

1. **Initial screen** (Search Resumes): RAG search over all 100 resumes, keeping the top 10.
2. **Second round** (Rank Candidates): the LLM reads each of the 10 full resumes and
   returns a score, strengths, gaps, evidence, and improvement suggestions for
   borderline candidates (scores 50-75).
3. **Final round** (Rank Candidates): a hire or no-hire recommendation for each candidate.

Generate Report writes a detailed match report to `reports/`. The Human
Feedback Loop is the chat: ask questions and the agent answers with its tools;
change requirements and it goes back to Extract Requirements, re-ranks, and
explains how the ranking changed; type `approve` to finish.

```bash
uv run python matching_agent.py "Find me candidates with React and 3+ years experience"
> Compare the top 3 matches side by side
> Why did the #1 candidate rank higher than #2?
> Make TypeScript a must-have and re-rank
> Generate interview questions for the top candidate
> approve
```

You can also pass a JD file: `uv run python matching_agent.py dataset/job_descriptions/data_engineer.pdf`.

Tools (`tools.py`): the Milestone 1 file-system tools (`read_file`,
`list_files`, `search_in_file`, `write_file`), the Milestone 2 RAG search tool
(`search_resumes`), and `extract_requirements(jd)`,
`compare_candidates(candidate_ids)` and `generate_interview_questions(candidate_id)`.

** uv run python src/resume_rag.py
uv run python matching_agent.py 
Find me candidates with React and 3+ years experience
Compare the top 3 matches side by side, then Why did the #1 candidate rank higher than #2?
Make TypeScript and Accessibility must-haves and re-rank.
Generate interview questions for the top candidate.

uv run python matching_agent.py dataset/job_descriptions/data_engineer.pdf;

who is the highest-ranked borderline candidate, then Why is Farah ranked #1 but not recommended for hire?