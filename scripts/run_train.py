#!/usr/bin/env python3
"""Train spaCy NER model on processed split data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import project_root, read_jsonl, write_json


def _to_spacy_examples(rows: list[dict]) -> list[Example]:
    """Convert canonical rows to spaCy training examples."""
    nlp = spacy.blank("fr")
    examples: list[Example] = []
    for row in rows:
        doc = nlp.make_doc(row["text"])
        ann = {"entities": [(e["start"], e["end"], e["label"]) for e in row.get("entities", [])]}
        examples.append(Example.from_dict(doc, ann))
    return examples


def train_model(dataset_path: Path, model_out: Path, epochs: int = 20, seed: int = 42) -> dict:
    """Train a French blank spaCy model and save artifacts."""
    rows = read_jsonl(dataset_path)
    train_rows = [r for r in rows if r.get("split") == "train"]
    dev_rows = [r for r in rows if r.get("split") == "dev"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not train_rows:
        raise RuntimeError("No rows with split='train' in dataset.")

    spacy.util.fix_random_seed(seed)  # mêmes résultats à chaque lancement
    nlp = spacy.blank("fr")
    ner = nlp.add_pipe("ner")
    for row in train_rows:
        for entity in row.get("entities", []):
            ner.add_label(entity["label"])

    nlp.initialize()
    examples = _to_spacy_examples(train_rows)

    rng = random.Random(seed)
    epoch_losses: list[dict] = []
    for i in range(epochs):
        rng.shuffle(examples)
        losses: dict[str, float] = {}
        # On apprend par petits lots de 8 phrases (plusieurs mises à jour par époque).
        for batch in minibatch(examples, size=8):
            nlp.update(batch, drop=0.2, losses=losses)
        serializable_losses = {k: float(v) for k, v in losses.items()}
        epoch_losses.append({"epoch": i + 1, "losses": serializable_losses})
        print(f"Epoch {i + 1}/{epochs} losses={losses}")

    model_out.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_out)

    report = {
        "dataset": str(dataset_path),
        "model_out": str(model_out),
        "epochs": epochs,
        "seed": seed,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "epoch_losses": epoch_losses,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spaCy PERSON model on enriched dataset.")
    parser.add_argument("--dataset", default="data/processed/v2/dataset_augmented.jsonl")
    parser.add_argument("--model-out", default="models/ner_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = project_root()
    report = train_model(root / args.dataset, root / args.model_out, args.epochs, args.seed)
    write_json(root / "data" / "eval" / "train_report.json", report)
    print(f"✅ Model saved: {root / args.model_out}")


if __name__ == "__main__":
    main()
