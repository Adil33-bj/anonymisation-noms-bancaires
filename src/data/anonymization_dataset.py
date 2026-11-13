"""Build paired anonymization evaluation dataset from canonical split rows."""

from __future__ import annotations

from pathlib import Path

from src.anonymization.mask import mask_spans
from src.utils.io import read_jsonl, write_jsonl


def build_anonymization_gold(dataset_path: Path, out_path: Path, split: str = "test") -> dict:
    """Create paired records: original text, expected masked text, gold entities."""
    rows = read_jsonl(dataset_path)
    pairs: list[dict] = []

    for row in rows:
        if row.get("split") != split:
            continue
        text = row.get("text", "")
        entities = row.get("entities", [])
        pairs.append(
            {
                "id": row.get("id", ""),
                "original_text": text,
                "expected_anonymized_text": mask_spans(text, entities),
                "expected_detected_entities": entities,
                "source": row.get("source", ""),
                "difficulty": row.get("difficulty", "medium"),
            }
        )

    write_jsonl(out_path, pairs)
    return {"split": split, "rows": len(pairs), "out_path": str(out_path)}
