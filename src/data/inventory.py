from __future__ import annotations

from pathlib import Path
import shutil

from src.utils.io import ensure_dir, project_root, write_json


ALLOWED_SUFFIXES = {".csv", ".json", ".txt"}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "models",
    "reports",
    "output_ner_model",
    "results",
    "src",
    "scripts",
    "tests",
}
EXCLUDED_FILENAMES = {"requirements.txt", "dev.spacy", "train.spacy", "output.txt"}


def _iter_artifacts(root: Path) -> list[Path]:
    """List raw candidate artifacts while skipping generated/project-internal files."""
    paths: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if p.name in EXCLUDED_FILENAMES:
            continue
        low_name = p.name.lower()
        if (
            low_name.startswith("dict_")
            or "entities_found" in low_name
            or "encrypted" in low_name
            or "decrypted" in low_name
        ):
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        paths.append(p)
    return sorted(paths)


def _target_name(root: Path, src: Path) -> str:
    """Flatten relative path to deterministic filename in `data/raw`."""
    rel = src.relative_to(root)
    return "__".join(rel.parts)


def sync_raw_artifacts(move: bool = False) -> dict:
    """Copy (or move) discovered artifacts into `data/raw` and write manifest."""
    root = project_root()
    raw_dir = root / "data" / "raw"
    ensure_dir(raw_dir)

    artifacts = _iter_artifacts(root)
    # Rien à copier depuis la racine : on garde les données déjà dans data/raw.
    if not artifacts:
        files = [p.name for p in raw_dir.glob("*") if p.is_file() and p.name != "manifest.json"]
        return {"mode": "keep_existing", "total_artifacts": len(files), "copied": [], "moved": []}

    for old in raw_dir.glob("*"):
        if old.is_file():
            old.unlink()

    copied: list[str] = []
    moved: list[str] = []

    for src in artifacts:
        dst = raw_dir / _target_name(root, src)
        ensure_dir(dst.parent)
        if move:
            shutil.move(str(src), str(dst))
            moved.append(str(src.relative_to(root)))
        else:
            shutil.copy2(src, dst)
            copied.append(str(src.relative_to(root)))

    manifest = {
        "mode": "move" if move else "copy",
        "total_artifacts": len(artifacts),
        "copied": copied,
        "moved": moved,
        "raw_dir": "data/raw",
    }
    write_json(root / "data" / "raw" / "manifest.json", manifest)
    return manifest
