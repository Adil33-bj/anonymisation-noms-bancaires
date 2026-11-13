#!/usr/bin/env python3
"""Run a detector on raw text and produce entities + anonymized output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anonymization.mask import mask_spans
from src.detection.dictionary_detector import DictionaryDetector
from src.detection.spacy_detector import SpacyDetector
from src.utils.io import project_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end detection + anonymization.")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--detector", choices=["dict", "spacy"], default="dict")
    parser.add_argument("--model", default="models/ner_v2")
    parser.add_argument("--fuzzy-threshold", type=int, default=80)
    parser.add_argument("--output-json", default="data/eval/end2end_entities.json")
    parser.add_argument("--output-text", default="data/eval/end2end_anonymized.txt")
    args = parser.parse_args()

    root = project_root()
    text = (root / args.input).read_text(encoding="utf-8")

    if args.detector == "dict":
        detector = DictionaryDetector.from_results_dir(
            root / "results",
            use_fuzzy=True,
            fuzzy_threshold=args.fuzzy_threshold,
        )
    else:
        detector = SpacyDetector(root / args.model)

    detected = detector.detect(text)
    spans = [
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "text": span.text,
            "confidence": getattr(span, "confidence", None),
        }
        for span in detected
    ]
    anonymized = mask_spans(text, spans)

    write_json(root / args.output_json, {"detector": args.detector, "spans": spans})
    (root / args.output_text).write_text(anonymized, encoding="utf-8")
    print("✅ End-to-end complete")


if __name__ == "__main__":
    main()
