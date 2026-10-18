"""Index local resumes in ChromaDB using OpenAI embeddings.

Set OPENAI_API_KEY, then run: uv run python src/resume_rag.py
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import chromadb
from openai import OpenAI

from filesSystemTools.fs_tools import list_files, read_file


RESUME_DIR = Path("dataset/resumes")
DB_DIR = Path("chroma_db")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTION_NAMES = ("Professional Summary", "Skills", "Experience", "Education")
METADATA_MODEL = "gpt-4o-mini"


def read_resume(path: Path) -> str:
    """Read a resume through the shared file-system tool."""
    result = read_file(str(path))
    if not result["success"]:
        raise RuntimeError(f"Could not read {path}: {result.get('error', 'unknown error')}")
    return str(result["content"])


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"Name": []}
    current = "Name"
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        if line in SECTION_NAMES:
            current = line
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {name: " ".join(lines) for name, lines in sections.items() if lines}


def extract_metadata(client: OpenAI, text: str, model: str = METADATA_MODEL) -> dict[str, str | int]:
    """Use an LLM to extract resume metadata as validated JSON."""
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract resume metadata. Return only JSON with name (string), "
                    "skills (array of strings), experience_years (integer), and education (string). "
                    "Use only facts stated in the resume."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Metadata extraction model returned an empty response.")
    try:
        extracted = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Metadata extraction model returned invalid JSON.") from error

    skills = extracted.get("skills")
    experience_years = extracted.get("experience_years")
    if (
        not isinstance(extracted.get("name"), str)
        or not isinstance(skills, list)
        or not all(isinstance(skill, str) for skill in skills)
        or not isinstance(experience_years, int)
        or isinstance(experience_years, bool)
        or not isinstance(extracted.get("education"), str)
    ):
        raise RuntimeError("Metadata extraction model returned incomplete or invalid fields.")
    return {
        "name": extracted["name"],
        "skills": ", ".join(skills),
        "experience_years": experience_years,
        "education": extracted["education"],
    }


def chunk_sections(sections: dict[str, str]) -> list[tuple[str, str]]:
    """Keep section names in every chunk so section meaning is searchable."""
    return [
        (section, f"{section}\n{content}")
        for section, content in sections.items()
        if section != "Name"
    ]


def resume_paths(directory: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for item in list_files(str(directory)):
        path = Path(item["path"])
        if path.suffix.lower() not in {".pdf", ".rtf"}:
            continue
        if path.suffix.lower() == ".rtf" or path.stem not in paths:
            paths[path.stem] = path
    return list(paths.values())


def project_relative_path(path: Path | str) -> str:
    """Return a path relative to the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve().relative_to(PROJECT_ROOT))


def embed(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def index_resumes(
    resume_dir: Path = RESUME_DIR,
    database_dir: Path = DB_DIR,
    model: str = "text-embedding-3-small",
    metadata_model: str = METADATA_MODEL,
) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before indexing resumes.")

    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    client = OpenAI()
    for path in resume_paths(resume_dir):
        text = read_resume(path)
        sections = split_sections(text)
        metadata = extract_metadata(client, text, metadata_model)
        for number, (section, chunk) in enumerate(chunk_sections(sections)):
            documents.append(chunk)
            metadatas.append(
                {
                    **metadata,
                    "source": project_relative_path(path),
                    "section": section,
                }
            )
            ids.append(f"{path.stem}-{number}")

    if not documents:
        raise RuntimeError(f"No .rtf or .pdf resumes found in {resume_dir}.")

    collection = chromadb.PersistentClient(path=str(database_dir)).get_or_create_collection(
        "resumes", metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed(client, documents, model),
    )
    return len(documents)


if __name__ == "__main__":
    print(f"Indexed {index_resumes()} resume sections in {DB_DIR}.")
