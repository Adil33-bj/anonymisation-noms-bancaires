from src.data.normalize import annotate_names
from src.evaluation.metrics import aggregate_detection


def test_annotate_names_takes_full_name() -> None:
    # Le nom complet doit être annoté, pas seulement le prénom.
    text = "Envoyer le relevé à Soukaina El Mansouri."
    entities = annotate_names(text, ["Soukaina", "Soukaina El Mansouri"])
    assert len(entities) == 1
    start, end, label = entities[0]
    assert text[start:end] == "Soukaina El Mansouri"
    assert label == "PERSON"


def test_aggregate_keeps_errors() -> None:
    # Les erreurs de plusieurs phrases ne doivent pas s'annuler.
    items = [{"tp": 0, "fp": 1, "fn": 0}, {"tp": 0, "fp": 0, "fn": 1}]
    result = aggregate_detection(items)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
