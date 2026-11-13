"""Evaluate anonymization quality on split rows or paired gold dataset."""

from __future__ import annotations

from pathlib import Path

from src.anonymization.mask import mask_spans
from src.evaluation.metrics import anonymization_metrics
from src.utils.io import read_jsonl


def evaluate_anonymization_on_split(dataset_path: Path, detector, split: str = "test") -> dict:
    """Compute anonymization metrics from canonical split rows."""
    rows = read_jsonl(dataset_path)
    expected_masked: list[str] = []
    got_masked: list[str] = []

    for row in rows:
        if row.get("split") != split:
            continue
        text = row.get("text", "")
        gold = row.get("entities", [])
        pred_spans = [
            {"start": p.start, "end": p.end, "label": p.label}
            for p in detector.detect(text)
        ]

        expected_masked.append(mask_spans(text, gold))
        got_masked.append(mask_spans(text, pred_spans))

    return anonymization_metrics(expected_masked, got_masked)


def evaluate_anonymization_on_pairs(pairs_path: Path, detector) -> dict:
    """Compute anonymization metrics from paired evaluation dataset."""
    pairs = read_jsonl(pairs_path)
    expected_masked: list[str] = []
    got_masked: list[str] = []

    for row in pairs:
        text = row.get("original_text", "")
        expected = row.get("expected_anonymized_text", text)
        pred_spans = [
            {"start": p.start, "end": p.end, "label": p.label}
            for p in detector.detect(text)
        ]
        expected_masked.append(expected)
        got_masked.append(mask_spans(text, pred_spans))

    return anonymization_metrics(expected_masked, got_masked)
