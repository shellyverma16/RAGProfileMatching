"""fs_tools.py

Core file system tools for reading, listing, writing, and searching files.

Supports TXT, PDF, DOCX for reading. Returns structured dicts.
"""
from pathlib import Path
import datetime
import re
from typing import List, Dict, Any

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:    
    import docx
except Exception:
    docx = None


def _file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix.lower(),
    }


def _read_rtf(path: Path) -> str:
    text = path.read_text(encoding="ascii")
    text = re.sub(r"{\\fonttbl.*?}", "", text, flags=re.DOTALL)
    text = text.replace(r"\par", "\n")
    text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
    return text.replace("{", "").replace("}", "").strip()


def read_file(filepath: str) -> Dict[str, Any]:
    """Read a file (txt, rtf, pdf, docx) and return content + metadata.

    Returns: { success: bool, content: str, metadata: dict, error?: str }
    """
    path = Path(filepath)
    if not path.exists():
        return {"success": False, "error": "File not found", "path": str(path)}

    ext = path.suffix.lower()
    metadata = _file_metadata(path)

    try:
        if ext in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"success": True, "content": text, "metadata": metadata}

        if ext == ".rtf":
            return {"success": True, "content": _read_rtf(path), "metadata": metadata}

        if ext == ".pdf":
            if PyPDF2 is None:
                return {"success": False, "error": "PyPDF2 not installed"}
            try:
                reader = PyPDF2.PdfReader(str(path))
            except Exception as e:
                return {"success": False, "error": "Pdf read error", "details": str(e)}
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
            text = "\n".join(texts)
            return {"success": True, "content": text, "metadata": metadata}

        if ext in (".docx",):
            if docx is None:
                return {"success": False, "error": "python-docx not installed"}
            try:
                doc = docx.Document(str(path))
            except Exception as e:
                return {"success": False, "error": "Docx read error", "details": str(e)}
            paragraphs = [p.text for p in doc.paragraphs]
            text = "\n".join(paragraphs)
            return {"success": True, "content": text, "metadata": metadata}

        # fallback: try to read as text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"success": True, "content": text, "metadata": metadata}
        except Exception as e:
            return {"success": False, "error": f"Unsupported file type: {ext}", "details": str(e)}

    except Exception as e:
        return {"success": False, "error": "Read error", "details": str(e)}


def list_files(directory: str, extension: str = None) -> List[Dict[str, Any]]:
    """List files under `directory`. If `extension` provided, filter by that extension.

    Returns list of metadata dicts.
    """
    p = Path(directory)
    if not p.exists():
        return []

    ext = None
    if extension:
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext

    results = []
    for f in p.rglob('*'):
        if f.is_file():
            if ext and f.suffix.lower() != ext:
                continue
            results.append(_file_metadata(f))
    return results


def write_file(filepath: str, content: str) -> Dict[str, Any]:
    """Write `content` to `filepath`, creating parent dirs if needed.

    Returns: { success: bool, path: str, message: str }
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return {"success": True, "path": str(path.resolve()), "message": "Written"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> Dict[str, Any]:
    """Search for `keyword` in file content (case-insensitive).

    Returns: { success: bool, matches: [ {line_no, line, context} ], metadata }
    """
    res = read_file(filepath)
    if not res.get("success"):
        return {"success": False, "error": res.get("error", "read failed")}

    text = res.get("content", "")
    needle = keyword.lower()
    matches = []

    # line-based search for better context
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if needle in line.lower():
            idx = line.lower().find(needle)
            start = max(0, idx - context_chars)
            end = min(len(line), idx + len(needle) + context_chars)
            snippet = line[start:end]
            matches.append({"line_no": i, "line": line.strip(), "context": snippet.strip()})

    return {"success": True, "matches": matches, "count": len(matches), "metadata": res.get("metadata")}


__all__ = ["read_file", "list_files", "write_file", "search_in_file"]
