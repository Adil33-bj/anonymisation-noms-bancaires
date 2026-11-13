import json
from pathlib import Path

from src.detect_names import fuzzy_match, STOPWORDS

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "results"

with open(RESULT_DIR / "dict_nom_to_id.json", "r", encoding="utf-8") as f:
    dict_nom_to_id = json.load(f)
with open(RESULT_DIR / "dict_prenom_to_id.json", "r", encoding="utf-8") as f:
    dict_prenom_to_id = json.load(f)

dict_nom_to_id_lower = {k.lower(): v for k, v in dict_nom_to_id.items()}
dict_prenom_to_id_lower = {k.lower(): v for k, v in dict_prenom_to_id.items()}


def test_exact_match_prenom():
    assert dict_prenom_to_id_lower.get("ahmed") == 125


def test_exact_match_nom():
    assert dict_nom_to_id_lower.get("benali") == 89


def test_fuzzy_match_prenom_ahmedi():
    match = fuzzy_match("ahmedi", dict_prenom_to_id_lower)
    assert match is not None
    assert match[1] == 125


def test_fuzzy_match_nom_souaad():
    match = fuzzy_match("souaad", dict_nom_to_id_lower)
    assert match is not None
    assert match[1] in [338, 355]


def test_fuzzy_match_prenom_yassin():
    match = fuzzy_match("yassin", dict_prenom_to_id_lower)
    assert match is not None
    assert match[1] == 1797


def test_stopword_ignored():
    assert "est" in STOPWORDS
