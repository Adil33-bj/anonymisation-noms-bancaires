"""Deterministic dataset split utilities."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
import re

from src.utils.io import read_jsonl, write_jsonl


SEED = 42


def _sentence_pattern_category(text: str) -> str:
    """Categorize sentence patterns for stratification."""
    low = text.lower()
    patterns = [
        ("transfer", r"\b(virement|transfert|versement)\b"),
        ("payment", r"\b(paiement|facture|débit|credit|crédit)\b"),
        ("withdrawal", r"\b(retrait|guichet)\b"),
        ("admin", r"\b(service client|code client|agence|relev[ée])\b"),
    ]
    for label, expr in patterns:
        if re.search(expr, low):
            return label
    return "other"


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_split(dataset_path: Path, train_ratio: float = 0.7, dev_ratio: float = 0.15) -> dict:
    """Build deterministic split stratified by entity presence and sentence category.

    Duplicate normalized texts are grouped first to avoid text leakage across splits.
    """
    rows = read_jsonl(dataset_path)

    # Group identical normalized texts so one text family stays in one split.
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_norm_text(row.get("text", ""))].append(row)

    # Stratify groups by (has_entity, pattern_category).
    buckets: dict[tuple[bool, str], list[list[str]]] = defaultdict(list)
    for grouped_rows in groups.values():
        if not grouped_rows:
            continue
        exemplar = grouped_rows[0]
        key = (
            bool(exemplar.get("entities")),
            _sentence_pattern_category(exemplar.get("text", "")),
        )
        buckets[key].append([r["id"] for r in grouped_rows])

    rng = random.Random(SEED)
    split = {"train": [], "dev": [], "test": []}

    for id_groups in buckets.values():
        rng.shuffle(id_groups)
        n = len(id_groups)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)

        for grp in id_groups[:n_train]:
            split["train"].extend(grp)
        for grp in id_groups[n_train : n_train + n_dev]:
            split["dev"].extend(grp)
        for grp in id_groups[n_train + n_dev :]:
            split["test"].extend(grp)

    for split_name in split:
        split[split_name] = sorted(split[split_name])
    return split


def apply_split(dataset_path: Path, split: dict) -> None:
    """Apply split labels back into dataset rows in-place."""
    rows = read_jsonl(dataset_path)
    id_to_split: dict[str, str] = {}
    for split_name, ids in split.items():
        for sid in ids:
            id_to_split[sid] = split_name

    for row in rows:
        row["split"] = id_to_split.get(row["id"], "")
    write_jsonl(dataset_path, rows)


def main() -> None:
    from src.utils.io import project_root, write_json

    root = project_root()
    dataset_path = root / "data" / "processed" / "v1" / "dataset_clean.jsonl"
    split = build_split(dataset_path)
    write_json(root / "data" / "splits" / "split_v1.json", split)
    apply_split(dataset_path, split)
    print("✅ Split generated and applied")


if __name__ == "__main__":
    main()
