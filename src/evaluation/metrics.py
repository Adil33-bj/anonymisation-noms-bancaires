"""Evaluation metrics for detection and anonymization tasks."""

from __future__ import annotations


def span_set(spans: list[dict]) -> set[tuple[int, int, str]]:
    """Convert span list to hashable set representation."""
    return {(s["start"], s["end"], s.get("label", "PERSON")) for s in spans}


def detection_metrics(gold: list[dict], pred: list[dict]) -> dict:
    """Compute exact-span detection metrics."""
    g = span_set(gold)
    p = span_set(pred)
    return scores_from_counts(tp=len(g & p), fp=len(p - g), fn=len(g - p))


def scores_from_counts(tp: int, fp: int, fn: int) -> dict:
    """Calculer précision, rappel et F1 à partir des compteurs."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (tp + fp) if (tp + fp) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
    }


def aggregate_detection(items: list[dict]) -> dict:
    """Aggregate per-sample (tp/fp/fn) counts into one global metric object."""
    # On additionne simplement les compteurs de toutes les phrases.
    tp = sum(x["tp"] for x in items)
    fp = sum(x["fp"] for x in items)
    fn = sum(x["fn"] for x in items)
    return scores_from_counts(tp, fp, fn)


def anonymization_metrics(expected: list[str], got: list[str], marker: str = "[ANONYMIZED]") -> dict:
    """Compute masking recall and overmask rate.

    - Masking Recall = masked true PERSON / total true PERSON
    - Overmask Rate = masked non-PERSON / total masked
    """
    total_expected_masks = sum(text.count(marker) for text in expected)
    total_got_masks = sum(text.count(marker) for text in got)

    masked_true_person = 0
    for expected_text, got_text in zip(expected, got):
        # Conservative overlap by count (position-agnostic).
        masked_true_person += min(expected_text.count(marker), got_text.count(marker))

    masked_non_person = max(total_got_masks - total_expected_masks, 0)
    masking_recall = masked_true_person / total_expected_masks if total_expected_masks else 0.0
    overmask_rate = masked_non_person / total_got_masks if total_got_masks else 0.0

    return {
        "expected_masks": total_expected_masks,
        "got_masks": total_got_masks,
        "masked_true_person": masked_true_person,
        "masked_non_person": masked_non_person,
        "masking_recall": round(masking_recall, 4),
        "overmask_rate": round(overmask_rate, 4),
    }
