from pathlib import Path

from src.data.validate import leakage_report, validate_dataset
from src.utils.io import write_json, write_jsonl


def test_invalid_offsets_flagged(tmp_path: Path) -> None:
    # Invalid entity end offset must be reported.
    dataset = tmp_path / "dataset.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "id": "x1",
                "text": "Ahmed Benali",
                "entities": [{"start": 0, "end": 99, "label": "PERSON"}],
                "source": "test",
                "language": "fr",
                "difficulty": "easy",
                "split": "train",
            }
        ],
    )
    report = validate_dataset(dataset)
    assert report["error_count"] > 0
    assert any(e["type"] == "entity_span_invalid" for e in report["errors"])


def test_duplicates_flagged(tmp_path: Path) -> None:
    # Exact duplicates must be flagged by validator.
    dataset = tmp_path / "dataset.jsonl"
    base = {
        "text": "Virement de Ahmed Benali vers le compte.",
        "entities": [{"start": 12, "end": 24, "label": "PERSON"}],
        "source": "test",
        "language": "fr",
        "difficulty": "medium",
        "split": "train",
    }
    write_jsonl(
        dataset,
        [
            {"id": "a1", **base},
            {"id": "a2", **base},
        ],
    )
    report = validate_dataset(dataset)
    assert any(e["type"] == "duplicate_exact" for e in report["errors"])


def test_leakage_by_text_detected(tmp_path: Path) -> None:
    # Same normalized text across train/dev should be detected as leakage.
    dataset = tmp_path / "dataset.jsonl"
    split_file = tmp_path / "split.json"
    rows = [
        {
            "id": "r1",
            "text": "Paiement pour Amal Bennani.",
            "entities": [{"start": 13, "end": 25, "label": "PERSON"}],
            "source": "test",
            "language": "fr",
            "difficulty": "easy",
            "split": "train",
        },
        {
            "id": "r2",
            "text": "Paiement pour amal bennani.",
            "entities": [{"start": 13, "end": 25, "label": "PERSON"}],
            "source": "test",
            "language": "fr",
            "difficulty": "easy",
            "split": "dev",
        },
    ]
    write_jsonl(dataset, rows)
    write_json(split_file, {"train": ["r1"], "dev": ["r2"], "test": []})
    leak = leakage_report(split_file, dataset)
    assert leak["has_leakage"] is True
    assert leak["text_train_dev_count"] == 1
