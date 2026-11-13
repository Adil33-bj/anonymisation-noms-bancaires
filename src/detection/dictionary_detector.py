"""Dictionary-based PERSON detector with optional fuzzy matching.

This detector supports:
- exact matching (baseline A)
- fuzzy matching with threshold sweep (baseline B)
- pair detection for `prenom nom` and `nom prenom`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz, process

from src.utils.io import read_json

STOPWORDS = {
    "est",
    "et",
    "de",
    "le",
    "la",
    "du",
    "un",
    "une",
    "ce",
    "cet",
    "cette",
    "mon",
    "ton",
    "son",
}


# Petits mots qui peuvent se trouver entre le prénom et le nom (ex : Yassine El Mansouri).
PARTICLES = {"el", "al", "ben", "ibn"}


@dataclass
class DetectedSpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float | None = None


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Tokenize words and keep character offsets for span-safe masking."""
    tokens: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def _fuzzy(tok_norm: str, candidates: Iterable[str], threshold: int) -> tuple[str | None, float]:
    """Return best candidate and score if fuzzy score >= threshold."""
    match = process.extractOne(tok_norm, list(candidates), scorer=fuzz.ratio)
    if not match:
        return None, 0.0
    cand, score, _ = match
    return (cand, float(score)) if score >= threshold else (None, float(score))


class DictionaryDetector:
    """Lookup detector using local Moroccan first/last name dictionaries."""

    def __init__(
        self,
        dict_prenom_to_id: dict,
        dict_nom_to_id: dict,
        fuzzy_threshold: int = 80,
        use_fuzzy: bool = True,
    ):
        self.pdict = {k.lower(): v for k, v in dict_prenom_to_id.items()}
        self.ndict = {k.lower(): v for k, v in dict_nom_to_id.items()}
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy

    @classmethod
    def from_results_dir(
        cls,
        results_dir: Path,
        fuzzy_threshold: int = 80,
        use_fuzzy: bool = True,
    ) -> "DictionaryDetector":
        pdict = read_json(results_dir / "dict_prenom_to_id.json")
        ndict = read_json(results_dir / "dict_nom_to_id.json")
        return cls(pdict, ndict, fuzzy_threshold=fuzzy_threshold, use_fuzzy=use_fuzzy)

    def _lookup_prenom(self, token: str) -> tuple[bool, float]:
        tok = token.lower()
        if tok in self.pdict:
            return True, 100.0
        if not self.use_fuzzy:
            return False, 0.0
        cand, score = _fuzzy(tok, self.pdict.keys(), self.fuzzy_threshold)
        return cand is not None, score

    def _lookup_nom(self, token: str) -> tuple[bool, float]:
        tok = token.lower()
        if tok in self.ndict:
            return True, 100.0
        if not self.use_fuzzy:
            return False, 0.0
        cand, score = _fuzzy(tok, self.ndict.keys(), self.fuzzy_threshold)
        return cand is not None, score

    def detect(self, text: str) -> list[DetectedSpan]:
        """Detect PERSON spans with pair-first logic, then fallback to single token."""
        tokens = tokenize_with_offsets(text)
        spans: list[DetectedSpan] = []
        i = 0

        while i < len(tokens):
            tok, start, end = tokens[i]
            if tok.lower() in STOPWORDS:
                i += 1
                continue

            is_prenom, score_prenom = self._lookup_prenom(tok)
            is_nom, score_nom = self._lookup_nom(tok)

            # Cas « Prénom + particule + Nom » : on prend les 3 mots ensemble.
            if is_prenom and i + 2 < len(tokens) and tokens[i + 1][0].lower() in PARTICLES:
                last_tok, _, last_end = tokens[i + 2]
                last_is_nom, last_score = self._lookup_nom(last_tok)
                if last_is_nom:
                    spans.append(
                        DetectedSpan(
                            start=start,
                            end=last_end,
                            label="PERSON",
                            text=text[start:last_end],
                            confidence=round(min(score_prenom, last_score) / 100.0, 4),
                        )
                    )
                    i += 3
                    continue

            # Try two-token entity first to maximize complete PERSON spans.
            if i + 1 < len(tokens):
                next_tok, _, next_end = tokens[i + 1]
                if next_tok.lower() not in STOPWORDS:
                    next_is_prenom, next_score_prenom = self._lookup_prenom(next_tok)
                    next_is_nom, next_score_nom = self._lookup_nom(next_tok)

                    if is_prenom and next_is_nom:
                        conf = min(score_prenom, next_score_nom) / 100.0
                        spans.append(
                            DetectedSpan(
                                start=start,
                                end=next_end,
                                label="PERSON",
                                text=text[start:next_end],
                                confidence=round(conf, 4),
                            )
                        )
                        i += 2
                        continue

                    if is_nom and next_is_prenom:
                        conf = min(score_nom, next_score_prenom) / 100.0
                        spans.append(
                            DetectedSpan(
                                start=start,
                                end=next_end,
                                label="PERSON",
                                text=text[start:next_end],
                                confidence=round(conf, 4),
                            )
                        )
                        i += 2
                        continue

            # Fallback to single-token detection.
            if is_prenom or is_nom:
                score = max(score_prenom, score_nom) / 100.0
                spans.append(
                    DetectedSpan(
                        start=start,
                        end=end,
                        label="PERSON",
                        text=text[start:end],
                        confidence=round(score, 4),
                    )
                )
            i += 1

        # Merge overlapping spans to keep offset-safe output.
        return _merge_overlaps(spans, text)


def _merge_overlaps(spans: list[DetectedSpan], text: str) -> list[DetectedSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[DetectedSpan] = [ordered[0]]
    for sp in ordered[1:]:
        prev = merged[-1]
        if sp.start >= prev.end:
            merged.append(sp)
            continue
        new_end = max(prev.end, sp.end)
        merged[-1] = DetectedSpan(
            start=prev.start,
            end=new_end,
            label="PERSON",
            text=text[prev.start:new_end],
            confidence=max(prev.confidence or 0.0, sp.confidence or 0.0),
        )
    return merged
