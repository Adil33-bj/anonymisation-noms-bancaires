"""Dataset validation and cleaning utilities.

The validator is strict on schema and entity spans, and also reports duplicate
content quality issues. A separate cleaning function keeps only valid/unique
rows for downstream training.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.utils.io import read_jsonl, write_json, write_jsonl


ALLOWED_LABELS = {"PERSON"}
REQUIRED_FIELDS = ["id", "text", "entities", "source", "language", "difficulty", "split"]


def _norm_text(text: str) -> str:
    """Normalize text for duplicate/leakage checks."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _near_sig(text: str) -> str:
    """Create a coarse near-duplicate signature."""
    stripped = re.sub(r"[^a-z0-9]", "", text)
    return hashlib.md5(stripped.encode()).hexdigest()[:12]


def _validate_row(row: dict) -> list[dict]:
    """Validate one row and return a list of error objects."""
    errors: list[dict] = []
    sid = row.get("id", "")
    text = row.get("text", "")
    entities = row.get("entities", [])

    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append({"id": sid, "type": "missing_field", "field": field})

    if not isinstance(text, str) or not text.strip():
        errors.append({"id": sid, "type": "empty_text"})
        return errors

    if not isinstance(entities, list):
        errors.append({"id": sid, "type": "entities_not_list"})
        return errors

    for ent in entities:
        if not isinstance(ent, dict):
            errors.append({"id": sid, "type": "entity_not_dict", "entity": ent})
            continue
        if not all(k in ent for k in ["start", "end", "label"]):
            errors.append({"id": sid, "type": "entity_missing_keys", "entity": ent})
            continue
        s, e, l = ent["start"], ent["end"], ent["label"]
        if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s or e > len(text):
            errors.append({"id": sid, "type": "entity_span_invalid", "entity": ent, "text_len": len(text)})
        if not isinstance(l, str) or l not in ALLOWED_LABELS:
            errors.append({"id": sid, "type": "invalid_label", "label": l})

    return errors


def validate_dataset(path: Path) -> dict:
    """Run full validation and duplicate detection report on a dataset."""
    rows = read_jsonl(path)
    errors: list[dict] = []
    seen_exact: dict[str, str] = {}
    seen_near: dict[str, str] = {}

    for row in rows:
        row_errors = _validate_row(row)
        errors.extend(row_errors)
        if row_errors:
            continue

        sid = row["id"]
        tnorm = _norm_text(row["text"])
        if tnorm in seen_exact:
            errors.append({"id": sid, "type": "duplicate_exact", "duplicate_of": seen_exact[tnorm]})
        else:
            seen_exact[tnorm] = sid

        sig = _near_sig(tnorm)
        if sig in seen_near and seen_near[sig] != sid:
            errors.append({"id": sid, "type": "duplicate_near", "duplicate_of": seen_near[sig]})
        else:
            seen_near[sig] = sid

    return {"total_rows": len(rows), "error_count": len(errors), "errors": errors}


def clean_dataset(path: Path, out_path: Path) -> dict:
    """Create a cleaned dataset by dropping invalid rows and exact duplicates."""
    rows = read_jsonl(path)
    clean_rows: list[dict] = []
    seen_text: set[str] = set()
    dropped_invalid = 0
    dropped_duplicates = 0

    for row in rows:
        row_errors = _validate_row(row)
        if row_errors:
            dropped_invalid += 1
            continue

        tnorm = _norm_text(row["text"])
        if tnorm in seen_text:
            dropped_duplicates += 1
            continue
        seen_text.add(tnorm)
        clean_rows.append(row)

    write_jsonl(out_path, clean_rows)
    return {
        "input_rows": len(rows),
        "clean_rows": len(clean_rows),
        "dropped_invalid": dropped_invalid,
        "dropped_exact_duplicates": dropped_duplicates,
        "out_path": out_path.name,
    }


def leakage_report(split_path: Path, dataset_path: Path | None = None) -> dict:
    """Check ID and text overlap across train/dev/test split sets."""
    import json

    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    train_ids = set(split.get("train", []))
    dev_ids = set(split.get("dev", []))
    test_ids = set(split.get("test", []))

    overlaps = {
        "train_dev": sorted(train_ids & dev_ids),
        "train_test": sorted(train_ids & test_ids),
        "dev_test": sorted(dev_ids & test_ids),
    }

    if dataset_path and dataset_path.exists():
        rows = read_jsonl(dataset_path)
        split_by_id: dict[str, str] = {}
        for split_name, ids in split.items():
            for sid in ids:
                split_by_id[sid] = split_name

        texts = {"train": {}, "dev": {}, "test": {}}
        for row in rows:
            sid = row.get("id", "")
            sname = split_by_id.get(sid, "")
            if sname not in texts:
                continue
            text_norm = _norm_text(str(row.get("text", "")))
            if not text_norm:
                continue
            texts[sname][text_norm] = sid

        td = sorted(set(texts["train"]) & set(texts["dev"]))
        tt = sorted(set(texts["train"]) & set(texts["test"]))
        dt = sorted(set(texts["dev"]) & set(texts["test"]))

        overlaps["text_train_dev_count"] = len(td)
        overlaps["text_train_test_count"] = len(tt)
        overlaps["text_dev_test_count"] = len(dt)
        overlaps["text_overlap_examples"] = {
            "train_dev": td[:10],
            "train_test": tt[:10],
            "dev_test": dt[:10],
        }

    overlaps["has_leakage"] = any(
        (isinstance(v, list) and len(v) > 0) or (isinstance(v, int) and v > 0)
        for v in overlaps.values()
    )
    return overlaps


def main() -> None:
    from src.utils.io import project_root

    root = project_root()
    dataset_path = root / "data" / "processed" / "v1" / "dataset.jsonl"
    validation = validate_dataset(dataset_path)
    write_json(root / "data" / "processed" / "validation_errors.json", validation)

    clean_path = root / "data" / "processed" / "v1" / "dataset_clean.jsonl"
    clean_info = clean_dataset(dataset_path, clean_path)
    write_json(root / "data" / "processed" / "cleaning_report.json", clean_info)

    split_path = root / "data" / "splits" / "split_v1.json"
    if split_path.exists():
        leak = leakage_report(split_path, clean_path)
        write_json(root / "data" / "processed" / "leakage_report.json", leak)

    print("✅ Validation and cleaning done")


if __name__ == "__main__":
    main()
