from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Entity:
    start: int
    end: int
    label: str = "PERSON"


@dataclass
class Sample:
    id: str
    text: str
    entities: list[dict]
    source: str
    language: str = "fr"
    difficulty: str = "medium"
    split: str = ""
    is_augmented: bool = False
    pattern_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
