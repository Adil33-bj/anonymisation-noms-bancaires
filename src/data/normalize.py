from __future__ import annotations

import json
from pathlib import Path

from src.data.schema import Sample
from src.utils.io import project_root, write_jsonl


def _difficulty_from_text(text: str) -> str:
    """Heuristic difficulty level used for stratified reporting."""
    low = text.lower()
    if any(k in low for k in ["mme", "mr", "transacation", "salma", "el "]):
        return "hard"
    if any(k in low for k in ["virement", "paiement", "retrait", "crédit", "transfert"]):
        return "medium"
    return "easy"


def _sample_from_raw(idx: int, text: str, entities: list[tuple[int, int, str]], source: str) -> dict:
    """Normalize one sample into canonical Phase-1 schema."""
    row = Sample(
        id=f"{source}_{idx:05d}",
        text=text,
        entities=[{"start": s, "end": e, "label": l} for s, e, l in entities],
        source=source,
        difficulty=_difficulty_from_text(text),
    ).to_dict()
    # Keep v1 canonical schema strict and minimal.
    return {
        "id": row["id"],
        "text": row["text"],
        "entities": row["entities"],
        "source": row["source"],
        "language": row["language"],
        "difficulty": row["difficulty"],
        "split": row["split"],
    }


def load_annotated_names() -> list[str]:
    """Lire la liste des noms annotés à la main (une personne par ligne)."""
    path = project_root() / "data" / "annotations" / "noms_annotes.txt"
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def annotate_names(text: str, names: list[str]) -> list[tuple[int, int, str]]:
    """Trouver les noms de la liste dans le texte et renvoyer leurs positions.

    On cherche les noms les plus longs d'abord, pour que « Yassine El Mansouri »
    soit pris en entier et pas seulement « Yassine ».
    """
    entities = []
    for name in sorted(names, key=len, reverse=True):
        start = text.find(name)
        while start != -1:
            end = start + len(name)
            overlap = any(s < end and start < e for s, e, _ in entities)
            if not overlap:
                entities.append((start, end, "PERSON"))
            start = text.find(name, end)
    return sorted(entities)


def _load_text_lines(path: Path) -> list[str]:
    """Read non-empty lines from plain-text corpus file."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                lines.append(clean)
    return lines


def build_canonical_dataset() -> list[dict]:
    """Create one canonical jsonl-ready dataset from all supported sources."""
    root = project_root()

    rows: list[dict] = []

    # Source 1: spaCy handcrafted training examples
    train_data: list[tuple[str, dict]] = []
    try:
        from ner.train_data import TRAIN_DATA  # local project import

        train_data = TRAIN_DATA
    except Exception:
        # Keep Phase 1 pipeline runnable even if spaCy is not installed.
        train_data = []

    for i, (text, ann) in enumerate(train_data):
        entities = ann.get("entities", [])
        rows.append(_sample_from_raw(i, text, entities, source="ner_train_data"))

    # Source 2+: load text artifacts from data/raw inventory.
    raw_dir = root / "data" / "raw"
    names = load_annotated_names()
    base = len(rows)
    if raw_dir.exists():
        row_idx = 0
        for path in sorted(raw_dir.iterdir()):
            if not path.is_file():
                continue

            name = path.name.lower()
            if "manifest" in name:
                continue
            if path.suffix.lower() == ".txt":
                for line in _load_text_lines(path):
                    entities = annotate_names(line, names)
                    rows.append(_sample_from_raw(base + row_idx, line, entities, source=f"raw_txt:{path.stem}"))
                    row_idx += 1
            elif path.suffix.lower() in {".json", ".jsonl"}:
                try:
                    if path.suffix.lower() == ".jsonl":
                        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    else:
                        items = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue

                if isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    continue

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    entities = item.get("entities", [])
                    parsed: list[tuple[int, int, str]] = []
                    if isinstance(entities, list):
                        for ent in entities:
                            if not isinstance(ent, dict):
                                continue
                            s = ent.get("start")
                            e = ent.get("end")
                            l = ent.get("label", "PERSON")
                            if isinstance(s, int) and isinstance(e, int) and isinstance(l, str):
                                parsed.append((s, e, l))
                    # Pas d'annotation dans le fichier : on utilise la liste de noms.
                    if "entities" not in item:
                        parsed = annotate_names(text, names)
                    rows.append(_sample_from_raw(base + row_idx, text, parsed, source=f"raw_json:{path.stem}"))
                    row_idx += 1

    return rows


def save_canonical_dataset(rows: list[dict]) -> Path:
    """Persist canonical dataset as v1 jsonl."""
    root = project_root()
    out = root / "data" / "processed" / "v1" / "dataset.jsonl"
    write_jsonl(out, rows)
    return out


def main() -> None:
    rows = build_canonical_dataset()
    out = save_canonical_dataset(rows)
    print(f"✅ Canonical dataset written: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
