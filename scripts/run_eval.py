#!/usr/bin/env python3
"""Evaluate baselines and spaCy model, then write metrics + report."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.dictionary_detector import DictionaryDetector
from src.detection.spacy_detector import SpacyDetector
from src.data.anonymization_dataset import build_anonymization_gold
from src.evaluation.evaluate_anonymization import (
    evaluate_anonymization_on_pairs,
    evaluate_anonymization_on_split,
)
from src.evaluation.evaluate_detection import eval_detector
from src.utils.io import project_root, write_json


def _ensure_results_dicts(root: Path) -> None:
    """Build `results/dict_*` files from raw CSVs if they are missing."""
    results_dir = root / "results"
    prenom_json = results_dir / "dict_prenom_to_id.json"
    nom_json = results_dir / "dict_nom_to_id.json"
    if prenom_json.exists() and nom_json.exists():
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "data" / "raw"
    prenoms_csv = raw_dir / "prenoms_marocains.csv"
    noms_csv = raw_dir / "noms_marocains.csv"
    if not prenoms_csv.exists() or not noms_csv.exists():
        raise RuntimeError("Missing raw name CSV files to build dictionary baselines.")

    def read_col(path: Path, preferred: list[str]) -> list[str]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return []
        cols = {c.lower(): c for c in rows[0].keys()}
        col = None
        for p in preferred:
            if p.lower() in cols:
                col = cols[p.lower()]
                break
        if col is None:
            col = next(iter(rows[0].keys()))
        return [str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()]

    prenoms = read_col(prenoms_csv, ["Prenom", "Prénom", "prenom"])
    noms = read_col(noms_csv, ["Nom", "nom"])
    write_json(prenom_json, {name: i for i, name in enumerate(sorted(set(prenoms)), 1)})
    write_json(nom_json, {name: i for i, name in enumerate(sorted(set(noms)), 1)})


def _top_error_categories(samples: list[dict], metric_key: str, top_k: int = 5) -> list[tuple[str, int]]:
    """Aggregate sample-level errors by `source|difficulty|pattern` category."""
    counter = Counter()
    for sample in samples:
        val = sample["metrics"].get(metric_key, 0)
        if val <= 0:
            continue
        category = f"{sample.get('source','')}|{sample.get('difficulty','')}|{sample.get('pattern_id','') or 'na'}"
        counter[category] += val
    return counter.most_common(top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dictionary and spaCy models on fixed split.")
    parser.add_argument("--dataset", default="data/processed/v2/dataset_augmented.jsonl")
    parser.add_argument("--model", default="models/ner_v2")
    parser.add_argument("--thresholds", nargs="*", type=int, default=[75, 80, 85, 90])
    args = parser.parse_args()

    root = project_root()
    dataset_path = root / args.dataset
    _ensure_results_dicts(root)
    results_dir = root / "results"

    # Baseline A: exact dictionary on test.
    exact = DictionaryDetector.from_results_dir(results_dir, use_fuzzy=False)
    exact_test, exact_samples = eval_detector(exact, dataset_path, split="test")

    # Baseline B: fuzzy dictionary threshold sweep on dev.
    best_t = None
    best_dev = None
    threshold_metrics: dict[str, dict] = {}
    for threshold in args.thresholds:
        fuzzy_dev = DictionaryDetector.from_results_dir(results_dir, use_fuzzy=True, fuzzy_threshold=threshold)
        dev_metrics, _ = eval_detector(fuzzy_dev, dataset_path, split="dev")
        threshold_metrics[str(threshold)] = dev_metrics
        if best_dev is None or dev_metrics["f1"] > best_dev["f1"]:
            best_dev = dev_metrics
            best_t = threshold
    if best_t is None:
        raise RuntimeError("Threshold sweep failed: no dev metric computed.")

    fuzzy = DictionaryDetector.from_results_dir(results_dir, use_fuzzy=True, fuzzy_threshold=best_t)
    fuzzy_test, fuzzy_samples = eval_detector(fuzzy, dataset_path, split="test")

    # Model C: trained spaCy NER.
    spacy_det = SpacyDetector(root / args.model)
    spacy_test, spacy_samples = eval_detector(spacy_det, dataset_path, split="test")

    # Anonymization quality from split and paired dataset.
    anon_gold_pairs = root / "data" / "eval" / "anonymization_gold_test.jsonl"
    if not anon_gold_pairs.exists():
        build_anonymization_gold(dataset_path, anon_gold_pairs, split="test")
    anon_dict_split = evaluate_anonymization_on_split(dataset_path, fuzzy, split="test")
    anon_spacy_split = evaluate_anonymization_on_split(dataset_path, spacy_det, split="test")
    anon_dict_pairs = evaluate_anonymization_on_pairs(anon_gold_pairs, fuzzy)
    anon_spacy_pairs = evaluate_anonymization_on_pairs(anon_gold_pairs, spacy_det)

    fp_categories = _top_error_categories(fuzzy_samples, "fp")
    fn_categories = _top_error_categories(fuzzy_samples, "fn")

    metrics = {
        "dataset": str(dataset_path),
        "dictionary_exact": exact_test,
        "dictionary_fuzzy_dev_sweep": threshold_metrics,
        "dictionary_fuzzy_best_threshold": best_t,
        "dictionary_fuzzy_test": fuzzy_test,
        "spacy_ner_test": spacy_test,
        "anonymization_split": {
            "dictionary_fuzzy": anon_dict_split,
            "spacy_ner": anon_spacy_split,
        },
        "anonymization_pairs": {
            "dictionary_fuzzy": anon_dict_pairs,
            "spacy_ner": anon_spacy_pairs,
        },
        "error_analysis": {
            "frequent_false_positives_by_category": fp_categories,
            "frequent_false_negatives_by_category": fn_categories,
            "sample_false_positives": [s for s in fuzzy_samples if s["metrics"]["fp"] > 0][:5],
            "sample_false_negatives": [s for s in fuzzy_samples if s["metrics"]["fn"] > 0][:5],
        },
    }
    write_json(root / "data" / "eval" / "metrics.json", metrics)

    lines = [
        "# Evaluation Report",
        "",
        "## Model comparison",
        "| Model | Precision | Recall | F1 | False Positive Rate |",
        "|---|---:|---:|---:|---:|",
        f"| Dictionary exact | {exact_test['precision']:.4f} | {exact_test['recall']:.4f} | {exact_test['f1']:.4f} | {exact_test['false_positive_rate']:.4f} |",
        f"| Dictionary fuzzy (t={best_t}) | {fuzzy_test['precision']:.4f} | {fuzzy_test['recall']:.4f} | {fuzzy_test['f1']:.4f} | {fuzzy_test['false_positive_rate']:.4f} |",
        f"| spaCy NER | {spacy_test['precision']:.4f} | {spacy_test['recall']:.4f} | {spacy_test['f1']:.4f} | {spacy_test['false_positive_rate']:.4f} |",
        "",
        "## Dev threshold sweep (dictionary fuzzy)",
    ]
    for threshold in args.thresholds:
        dev = threshold_metrics[str(threshold)]
        lines.append(
            f"- t={threshold}: P={dev['precision']:.4f}, R={dev['recall']:.4f}, F1={dev['f1']:.4f}, FPR={dev['false_positive_rate']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Anonymization metrics",
            f"- Split based (dictionary fuzzy): masking_recall={anon_dict_split['masking_recall']:.4f}, overmask_rate={anon_dict_split['overmask_rate']:.4f}",
            f"- Split based (spaCy): masking_recall={anon_spacy_split['masking_recall']:.4f}, overmask_rate={anon_spacy_split['overmask_rate']:.4f}",
            f"- Paired gold (dictionary fuzzy): masking_recall={anon_dict_pairs['masking_recall']:.4f}, overmask_rate={anon_dict_pairs['overmask_rate']:.4f}",
            f"- Paired gold (spaCy): masking_recall={anon_spacy_pairs['masking_recall']:.4f}, overmask_rate={anon_spacy_pairs['overmask_rate']:.4f}",
            "",
            "## Error analysis",
            "### Frequent false positives by category",
        ]
    )
    for cat, count in fp_categories:
        lines.append(f"- {cat}: {count}")
    lines.append("")
    lines.append("### Frequent false negatives by category")
    for cat, count in fn_categories:
        lines.append(f"- {cat}: {count}")

    lines.append("")
    lines.append("### Example false positives")
    for sample in metrics["error_analysis"]["sample_false_positives"]:
        lines.append(f"- [{sample['id']}] fp={sample['metrics']['fp']} :: {sample['text']}")
    lines.append("")
    lines.append("### Example false negatives")
    for sample in metrics["error_analysis"]["sample_false_negatives"]:
        lines.append(f"- [{sample['id']}] fn={sample['metrics']['fn']} :: {sample['text']}")

    (root / "reports" / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("✅ Evaluation complete")


if __name__ == "__main__":
    main()
