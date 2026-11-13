from pathlib import Path

from src.anonymization.mask import mask_spans
from src.data.anonymization_dataset import build_anonymization_gold
from src.utils.io import read_jsonl, write_jsonl


def test_mask_offsets_preserve_non_person_text() -> None:
    # Non-PERSON content must remain untouched.
    text = "Ahmed Benali a payé la facture."
    spans = [{"start": 0, "end": 12, "label": "PERSON"}]
    out = mask_spans(text, spans)
    assert out.startswith("[ANONYMIZED]")
    assert "a payé la facture." in out


def test_multiple_spans_masked() -> None:
    # Multiple PERSON spans should all be masked.
    text = "Ahmed Benali et Yassine Kharrouchi sont là."
    spans = [
        {"start": 0, "end": 12, "label": "PERSON"},
        {"start": 16, "end": 34, "label": "PERSON"},
    ]
    out = mask_spans(text, spans)
    assert out.count("[ANONYMIZED]") == 2


def test_overlapping_spans_mask_once() -> None:
    # Overlapping spans should be merged to keep replacement offset-safe.
    text = "Ahmed Benali"
    spans = [
        {"start": 0, "end": 5, "label": "PERSON"},
        {"start": 0, "end": 12, "label": "PERSON"},
    ]
    out = mask_spans(text, spans)
    assert out == "[ANONYMIZED]"


def test_build_anonymization_gold_pairs(tmp_path: Path) -> None:
    # Paired anonymization dataset should expose required fields.
    dataset = tmp_path / "dataset.jsonl"
    out = tmp_path / "pairs.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "id": "s1",
                "text": "Paiement pour Amal Bennani.",
                "entities": [{"start": 13, "end": 25, "label": "PERSON"}],
                "source": "unit",
                "language": "fr",
                "difficulty": "easy",
                "split": "test",
            }
        ],
    )
    info = build_anonymization_gold(dataset, out, split="test")
    rows = read_jsonl(out)
    assert info["rows"] == 1
    assert "original_text" in rows[0]
    assert "expected_anonymized_text" in rows[0]
    assert "expected_detected_entities" in rows[0]
