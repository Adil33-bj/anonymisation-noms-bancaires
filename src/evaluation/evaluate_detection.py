"""Evaluation loop for any detector implementing `detect(text)`."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.metrics import aggregate_detection, detection_metrics
from src.utils.io import read_jsonl


def eval_detector(detector, dataset_path: Path, split: str) -> tuple[dict, list[dict]]:
    """Evaluate a detector on one split and return (global_metrics, by_sample)."""
    rows = read_jsonl(dataset_path)
    by_sample: list[dict] = []
    totals: list[dict] = []

    for row in rows:
        if row.get("split") != split:
            continue
        text = row.get("text", "")
        gold = row.get("entities", [])
        pred_obj = detector.detect(text)
        pred = [
            {
                "start": p.start,
                "end": p.end,
                "label": p.label,
                "confidence": getattr(p, "confidence", None),
            }
            for p in pred_obj
        ]

        metrics = detection_metrics(gold, pred)
        totals.append({"tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"]})
        by_sample.append(
            {
                "id": row.get("id", ""),
                "source": row.get("source", ""),
                "difficulty": row.get("difficulty", "medium"),
                "is_augmented": bool(row.get("is_augmented", False)),
                "pattern_id": row.get("pattern_id", ""),
                "metrics": metrics,
                "text": text,
                "gold": gold,
                "pred": pred,
            }
        )

    return aggregate_detection(totals), by_sample
