"""llm_file_assistant.py

Small assistant that integrates `fs_tools` with an LLM (OpenAI function-calling).

It will attempt to use OpenAI if `OPENAI_API_KEY` is available, otherwise falls back
to a simple keyword-driven router.
"""
import os
import json
import re
from pathlib import Path
from typing import Any

from fs_tools import read_file, list_files, write_file, search_in_file

try:
    import openai
except Exception:
    openai = None


FUNCTIONS = [
    {
        "name": "read_file",
        "description": "Read a file and return text and metadata",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to file"}
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory, optionally filter by extension",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "extension": {"type": "string"},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "search_in_file",
        "description": "Search for a keyword inside a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "keyword": {"type": "string"},
            },
            "required": ["filepath", "keyword"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file path",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    {
        "name": "create_summary_file",
        "description": "Create a brief summary text file for a resume",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the resume file"},
                "output_path": {"type": "string", "description": "Optional output path for the summary file"},
            },
            "required": ["filepath"],
        },
    },
]


def _resolve_resume_directory(base_dir: str = "sample_resumes") -> str:
    """Resolve the resume folder from either the project root or the script folder."""
    candidate = Path(base_dir)
    if candidate.is_absolute():
        return str(candidate)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / candidate,
        Path.cwd() / candidate,
        script_dir,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(script_dir / candidate)


def _resolve_resume_path(file_hint: str, base_dir: str = "sample_resumes") -> str | None:
    """Find an existing resume file based on a user prompt or file name."""
    if not file_hint:
        return None

    if os.path.exists(file_hint):
        return file_hint

    base_path = Path(_resolve_resume_directory(base_dir))
    candidates = []
    for item in [file_hint, Path(file_hint).name, Path(file_hint).stem]:
        if not str(item):
            continue
        candidates.append(str(base_path / str(item)))
        for ext in [".txt", ".pdf", ".docx", ".md"]:
            candidates.append(str(base_path / f"{str(item).rstrip('.')}{ext}"))
        candidates.append(str(Path(__file__).resolve().parent / str(item)))
        for ext in [".txt", ".pdf", ".docx", ".md"]:
            candidates.append(str(Path(__file__).resolve().parent / f"{str(item).rstrip('.')}{ext}"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def create_summary_file(filepath: str, output_path: str = None) -> dict:
    """Create a simple summary file for a resume."""
    read_result = read_file(filepath)
    if not read_result.get("success"):
        return {"success": False, "error": read_result.get("error", "Unable to read resume")}

    content = read_result.get("content", "") or ""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        summary = "Summary: No content found.\n"
    else:
        name = lines[0]
        role = lines[1] if len(lines) > 1 else "Professional"
        skills = []
        for line in lines:
            if line.lower().startswith("skills:"):
                skills = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
                break
        summary = (
            f"Resume Summary: {name}\n"
            f"Current role: {role}\n\n"
            f"Key skills: {', '.join(skills) if skills else 'Not provided'}\n\n"
            f"Highlights:\n"
            f"- {lines[2] if len(lines) > 2 else 'Professional experience available in the source document.'}\n"
            f"- {lines[3] if len(lines) > 3 else 'Additional qualifications are listed in the resume.'}\n"
            f"\nSource file: {filepath}\n"
        )

    target = Path(output_path) if output_path else Path(filepath).with_name(f"{Path(filepath).stem}_summary.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(summary, encoding="utf-8")
    return {"success": True, "path": str(target.resolve()), "summary": summary}


def _call_tool(name: str, arguments: dict) -> Any:
    if name == "read_file":
        return read_file(arguments["filepath"])
    if name == "list_files":
        return list_files(arguments["directory"], arguments.get("extension"))
    if name == "search_in_file":
        return search_in_file(arguments["filepath"], arguments["keyword"])
    if name == "write_file":
        return write_file(arguments["filepath"], arguments["content"])
    if name == "create_summary_file":
        return create_summary_file(arguments["filepath"], arguments.get("output_path"))
    return {"error": "Unknown tool"}


def ask_with_openai(prompt: str, model: str = None) -> Any:
    if openai is None:
        return {"error": "openai package not installed"}
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        return {"error": "OPENAI_API_KEY not set"}

    model = model or os.environ.get("OPENAI_MODEL", "gpt-4-0613")
    messages = [{"role": "user", "content": prompt}]

    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto",
    )

    message = resp["choices"][0]["message"]
    if message.get("function_call"):
        fname = message["function_call"]["name"]
        args_text = message["function_call"].get("arguments") or "{}"
        try:
            args = json.loads(args_text)
        except Exception:
            args = {}
        tool_result = _call_tool(fname, args)
        return {"tool_called": fname, "arguments": args, "result": tool_result}

    return {"content": message.get("content")}


def simple_router(prompt: str, base_dir: str = "sample_resumes") -> Any:
    p = prompt.lower()
    resolved_base_dir = _resolve_resume_directory(base_dir)

    if "create a summary file" in p or "create summary file" in p or "summary file" in p:
        file_hint = None
        match = re.search(r"for\s+([A-Za-z0-9_\-.]+(?:\.[A-Za-z0-9]+)?)", p)
        if match:
            file_hint = match.group(1)
        else:
            file_hint = re.findall(r"([A-Za-z0-9_\-.]+(?:\.[A-Za-z0-9]+)?)", prompt)
            file_hint = file_hint[0] if file_hint else None

        if not file_hint:
            return {"success": False, "error": "No resume file name found in the prompt."}

        resolved_path = _resolve_resume_path(file_hint, resolved_base_dir)
        if not resolved_path:
            return {"success": False, "error": f"Resume file not found: {file_hint}"}

        result = create_summary_file(resolved_path)
        return result

    if "read all resumes" in p or "read all" in p:
        files = list_files(resolved_base_dir, extension=None)
        out = {}
        for f in files:
            r = read_file(f["path"])
            out[f["name"]] = {"success": r.get("success"), "snippet": (r.get("content") or "")[:800]}
        return out

    if "find resumes" in p or "find resume" in p or "mention" in p:
        m = re.search(r"mention(?:ing)?\s+([A-Za-z0-9_\-]+)", p)
        if m:
            keyword = m.group(1)
        else:
            keyword = p.split()[-1]
        files = list_files(resolved_base_dir, extension=None)
        matches = []
        for f in files:
            s = search_in_file(f["path"], keyword)
            if s.get("count", 0) > 0:
                matches.append({"file": f["name"], "matches": s["matches"]})
        return {"keyword": keyword, "results": matches}

    return {"error": "Could not interpret prompt. Try: 'Read all resumes', 'Find resumes mentioning Python', or 'Create a summary file for resume_john_doe.txt'"}


def main():
    print("LLM File Assistant Demo")
    print("Type a command (eg. 'Read all resumes in sample_resumes' or 'Find resumes mentioning Python'). Ctrl-C to exit.")
    while True:
        try:
            prompt = input('\n> ')
            if not prompt.strip():
                continue
            if os.environ.get("OPENAI_API_KEY"):
                out = ask_with_openai(prompt)
            else:
                out = simple_router(prompt)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except KeyboardInterrupt:
            print('\nExiting')
            break
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':
    main()
