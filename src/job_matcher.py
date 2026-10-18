"""Find the best resume matches for a job description.

Run after indexing: uv run python src/job_matcher.py "Python data engineer"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import chromadb
from openai import OpenAI

from resume_rag import DB_DIR, embed, project_relative_path


KNOWN_SKILLS = (
    "Machine Learning",
    "Cloud Security",
    "Product Management",
    "TypeScript",
    "TensorFlow",
    "Kubernetes",
    "PostgreSQL",
    "Terraform",
    "Accessibility",
    "JavaScript",
    "Python",
    "PyTorch",
    "MLOps",
    "MLflow",
    "Django",
    "React",
    "Node.js",
    "Airflow",
    "Snowflake",
    "dbt",
    "AWS",
    "SQL",
    "Java",
)


def requirements(job_description: str) -> tuple[list[list[str]], int]:
    """Extract required skill groups and the highest stated minimum experience."""
    required_line = re.search(r"required skills\s*:?\s*(.*)", job_description, re.IGNORECASE)
    if required_line and required_line.group(1).strip():
        values = re.split(r"[,;]", required_line.group(1))
        skills = []
        for value in values:
            options = [
                option.strip().rstrip(".")
                for option in re.split(
                    r"\s+or\s+",
                    re.sub(r"\s+\d+\+?\s*years?\b.*", "", value),
                    flags=re.IGNORECASE,
                )
                if option.strip().rstrip(".")
            ]
            if options:
                skills.append(options)
    else:
        skills = [[skill]
            for skill in KNOWN_SKILLS
            if re.search(rf"\b{re.escape(skill)}\b", job_description, re.IGNORECASE)
        ]
    years = [int(value) for value in re.findall(r"\b(\d+)\s*\+?\s*years?\b", job_description, re.IGNORECASE)]
    return skills, max(years, default=0)


def matches_skills(skills: str, required_groups: list[list[str]]) -> list[str]:
    skills_lower = skills.lower()
    return [
        next((skill for skill in group if skill.lower() in skills_lower), "")
        for group in required_groups
    ]


def job_matches(
    job_description: str,
    top_k: int = 10,
    database_dir: Path = DB_DIR,
    model: str = "text-embedding-3-small",
) -> dict[str, object]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before matching resumes.")

    collection = chromadb.PersistentClient(path=str(database_dir)).get_collection("resumes")
    if collection.count() == 0:
        raise RuntimeError("No indexed resumes found. Run src/resume_rag.py first.")

    query_embedding = embed(OpenAI(), [job_description], model)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count(),
        include=["documents", "metadatas", "distances"],
    )
    required_groups, minimum_years = requirements(job_description)
    candidates: dict[str, dict[str, object]] = {}

    for document, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        source = project_relative_path(str(metadata["source"]))
        candidate = candidates.setdefault(
            source,
            {
                "candidate_name": metadata["name"],
                "resume_path": source,
                "skills": str(metadata["skills"]),
                "experience_years": int(metadata["experience_years"]),
                "semantic_score": 0.0,
                "sections": [],
                "excerpts": [],
            },
        )
        candidate["semantic_score"] = max(
            float(candidate["semantic_score"]), max(0.0, 1 - float(distance) / 2)
        )
        candidate["sections"].append(str(metadata["section"]))
        candidate["excerpts"].append(str(document)[:240])

    ranked: list[dict[str, object]] = []
    for candidate in candidates.values():
        matched_skills = matches_skills(str(candidate["skills"]), required_groups)
        if int(candidate["experience_years"]) < minimum_years:
            continue
        if required_groups and not all(matched_skills):
            continue

        keyword_score = len([skill for skill in matched_skills if skill]) / len(required_groups) if required_groups else 0
        match_score = round(100 * (0.7 * float(candidate["semantic_score"]) + 0.3 * keyword_score))
        matched_sections = sorted(set(candidate["sections"]))
        ranked.append(
            {
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": match_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": candidate["excerpts"][:2],
                "reasoning": (
                    f"Matched skills: {', '.join(matched_skills) or 'semantic match'}; "
                    f"relevant sections: {', '.join(matched_sections)}."
                ),
            }
        )

    ranked.sort(key=lambda candidate: int(candidate["match_score"]), reverse=True)
    return {"job_description": job_description, "top_matches": ranked[:top_k]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match resumes to a job description.")
    parser.add_argument("job_description", help="Job description text, quoted.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum matches to return.")
    args = parser.parse_args()
    print(json.dumps(job_matches(args.job_description, args.top_k), indent=2))
