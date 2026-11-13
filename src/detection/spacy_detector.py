"""spaCy-based PERSON detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import spacy


@dataclass
class DetectedSpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float | None = None


class SpacyDetector:
    """Load a trained spaCy model and expose a unified `detect` API."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.nlp = spacy.load(model_dir)

    def detect(self, text: str) -> list[DetectedSpan]:
        doc = self.nlp(text)
        out: list[DetectedSpan] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                out.append(
                    DetectedSpan(
                        start=ent.start_char,
                        end=ent.end_char,
                        label="PERSON",
                        text=ent.text,
                        confidence=None,  # blank spaCy NER does not expose calibrated confidence by default
                    )
                )
        return out
