"""Offset-safe anonymization helpers."""

from __future__ import annotations


def _normalize_spans(spans: list[dict], text_len: int) -> list[tuple[int, int]]:
    """Validate and merge overlapping spans before masking."""
    valid: list[tuple[int, int]] = []
    for sp in spans:
        start = sp.get("start")
        end = sp.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > text_len:
            continue
        valid.append((start, end))

    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))

    merged: list[tuple[int, int]] = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start >= prev_end:
            merged.append((start, end))
        else:
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def mask_spans(text: str, spans: list[dict], token: str = "[ANONYMIZED]") -> str:
    """Replace target spans by token while preserving non-PERSON text exactly."""
    merged = _normalize_spans(spans, len(text))
    if not merged:
        return text

    out: list[str] = []
    last = 0
    for start, end in merged:
        out.append(text[last:start])
        out.append(token)
        last = end
    out.append(text[last:])
    return "".join(out)
