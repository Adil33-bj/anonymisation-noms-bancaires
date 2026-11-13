#!/usr/bin/env python3
"""Run data inventory, normalization, validation, splitting, and enrichment."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/run_data_pipeline.py` from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.anonymization_dataset import build_anonymization_gold
from src.data.augment import generate_augmented_train
from src.data.inventory import sync_raw_artifacts
from src.data.normalize import build_canonical_dataset, save_canonical_dataset
from src.data.split import apply_split, build_split
from src.data.validate import clean_dataset, leakage_report, validate_dataset
from src.data.versioning import append_data_changelog
from src.utils.io import project_root, write_json


def main() -> None:
    root = project_root()

    # 1) Sync raw artifacts into data/raw and keep a manifest.
    inventory = sync_raw_artifacts(move=False)

    # 2) Normalize all compatible sources into canonical jsonl (v1).
    rows = build_canonical_dataset()
    dataset_v1 = save_canonical_dataset(rows)

    # 3) Validate canonical dataset and output full error report.
    validation = validate_dataset(dataset_v1)
    write_json(root / "data" / "processed" / "validation_errors.json", validation)

    # 4) Build cleaned dataset for model training/evaluation.
    dataset_clean = root / "data" / "processed" / "v1" / "dataset_clean.jsonl"
    cleaning = clean_dataset(dataset_v1, dataset_clean)
    write_json(root / "data" / "processed" / "cleaning_report.json", cleaning)

    # 5) Deterministic split with stratification and no text leakage.
    split = build_split(dataset_clean)
    write_json(root / "data" / "splits" / "split_v1.json", split)
    apply_split(dataset_clean, split)

    leakage = leakage_report(root / "data" / "splits" / "split_v1.json", dataset_clean)
    write_json(root / "data" / "processed" / "leakage_report.json", leakage)

    # 6) Build enriched v2 dataset by train-only augmentation.
    dataset_v2 = root / "data" / "processed" / "v2" / "dataset_augmented.jsonl"
    augmentation = generate_augmented_train(dataset_clean, dataset_v2)

    # 7) Build anonymization gold set for quality checks.
    anon_gold = build_anonymization_gold(
        dataset_clean,
        root / "data" / "eval" / "anonymization_gold_test.jsonl",
        split="test",
    )
    write_json(root / "data" / "eval" / "anonymization_gold_meta.json", anon_gold)

    append_data_changelog(
        root / "data" / "processed" / "CHANGELOG.md",
        title="v1 -> v2 pipeline refresh",
        bullets=[
            f"raw artifacts synchronized: {inventory['total_artifacts']}",
            f"canonical rows (v1): {validation['total_rows']}",
            f"clean rows (v1 clean): {cleaning['clean_rows']}",
            f"v2 augmented rows: {augmentation['final_rows']}",
            f"split sizes: train={len(split['train'])}, dev={len(split['dev'])}, test={len(split['test'])}",
            f"text leakage detected: {leakage['has_leakage']}",
        ],
    )

    report = [
        "# Data Report",
        "",
        "## Inventory",
        f"- Raw artifacts synchronized: {inventory['total_artifacts']}",
        f"- Mode: {inventory['mode']}",
        "",
        "## Canonical Schema",
        "- Fields: id, text, entities, source, language, difficulty, split",
        f"- Canonical dataset path: {dataset_v1}",
        f"- Total rows v1: {validation['total_rows']}",
        "",
        "## Validation + Cleaning",
        f"- Validation errors: {validation['error_count']}",
        f"- Clean rows: {cleaning['clean_rows']}",
        f"- Dropped invalid: {cleaning['dropped_invalid']}",
        f"- Dropped exact duplicates: {cleaning['dropped_exact_duplicates']}",
        f"- Validation output: {root / 'data' / 'processed' / 'validation_errors.json'}",
        "",
        "## Split",
        "- Deterministic seed: 42",
        "- Stratification: entity presence + sentence pattern category",
        f"- Leakage detected: {leakage['has_leakage']}",
        f"- Split sizes: train={len(split['train'])}, dev={len(split['dev'])}, test={len(split['test'])}",
        f"- Frozen split file: {root / 'data' / 'splits' / 'split_v1.json'}",
        "",
        "## Versioning",
        f"- Changelog: {root / 'data' / 'processed' / 'CHANGELOG.md'}",
        "",
        "## Enrichment + Anonymization Gold",
        f"- Augmented dataset (v2): {dataset_v2}",
        f"- Augmentation report: {root / 'data' / 'processed' / 'v2' / 'augmentation_report.json'}",
        f"- Anonymization gold: {root / 'data' / 'eval' / 'anonymization_gold_test.jsonl'}",
    ]
    # Chemins relatifs dans le rapport (pas de chemin personnel de la machine).
    text = "\n".join(report).replace(str(root) + "/", "")
    (root / "reports" / "data_report.md").write_text(text, encoding="utf-8")
    print("✅ Data pipeline complete")


if __name__ == "__main__":
    main()
