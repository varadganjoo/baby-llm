"""Document loading helpers (text + PDF) for feeding the baby model."""

from __future__ import annotations

from pathlib import Path
from typing import List


class DocumentError(RuntimeError):
    """Raised when a document cannot be parsed."""


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".cfg",
    ".ini",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def read_document(path: Path) -> str:
    """Return UTF-8 text for the provided document path."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return _read_text_file(path)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(
            f"{path} is not UTF-8 text. Convert it to UTF-8 before feeding."
        ) from exc


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise DocumentError(
            "PDF support requires the 'pypdf' package. Install it via `pip install pypdf`."
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - library error surface
        raise DocumentError(f"Failed to open PDF {path}: {exc}") from exc

    pages = []
    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            raise DocumentError(
                f"Could not extract text from page {page_index + 1} of {path}: {exc}"
            ) from exc
        pages.append(text)

    content = "\n\n".join(part.strip() for part in pages if part and part.strip())
    if not content:
        raise DocumentError(f"No extractable text found in {path}.")
    return content


def collect_documents(root: Path) -> List[Path]:
    """Recursively find supported documents under root."""
    paths: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"Folder {root} does not exist.")
    for candidate in root.rglob("*"):
        if candidate.is_file() and _is_supported(candidate):
            paths.append(candidate)
    if not paths:
        raise DocumentError(f"No supported documents found in {root}.")
    paths.sort()
    return paths


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" or path.suffix.lower() in TEXT_EXTENSIONS
