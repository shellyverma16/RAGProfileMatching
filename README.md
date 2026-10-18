# RAGProfileMatching


## Synthetic Dataset

The repository includes a synthetic, fictional corpus for local experimentation:

- 35 resumes in both PDF and RTF: `dataset/resumes/`
- 6 job descriptions in both PDF and RTF: `dataset/job_descriptions/`
- `dataset/manifest.json` with document metadata and relative paths

All names, experience, education, skills, and achievements in the corpus are fictional.
To recreate the corpus on macOS, run:

```bash
python3 dataset/generate_synthetic_dataset.py
```

The generator uses only the Python standard library.

## Resume Index

`src/resume_rag.py` uses the shared `list_files` and `read_file` file-system
tools, preferring RTF when PDF and RTF copies share a name. It preserves
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
