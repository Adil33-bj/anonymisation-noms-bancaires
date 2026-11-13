from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any


def project_root() -> Path:
    """Return repository root (`stage/`) based on this file location."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    """Create directory tree if missing."""
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON file."""
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON file with pretty indentation."""
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dict rows."""
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL (one JSON object per line)."""
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
