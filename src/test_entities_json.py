import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "results"

with open(RESULT_DIR / "entities_found.json", "r", encoding="utf-8") as f:
    entities = json.load(f)
with open(RESULT_DIR / "dict_nom_to_id.json", "r", encoding="utf-8") as f:
    dict_nom_to_id = json.load(f)
with open(RESULT_DIR / "dict_prenom_to_id.json", "r", encoding="utf-8") as f:
    dict_prenom_to_id = json.load(f)


def test_json_valid_structure():
    for entity in entities:
        assert "type" in entity
        if entity["type"].startswith("PRENOM"):
            assert "prenom" in entity and "id_prenom" in entity
        elif entity["type"].startswith("NOM"):
            assert "nom" in entity and "id_nom" in entity
        elif entity["type"].startswith("ENTITE"):
            assert "prenom" in entity and "nom" in entity
            assert "id_prenom" in entity and "id_nom" in entity
        else:
            raise AssertionError(f"Type inconnu: {entity['type']}")


def test_ids_valid_in_dicts():
    prenom_ids = set(dict_prenom_to_id.values())
    nom_ids = set(dict_nom_to_id.values())
    for entity in entities:
        if "id_prenom" in entity:
            assert entity["id_prenom"] in prenom_ids
        if "id_nom" in entity:
            assert entity["id_nom"] in nom_ids


def test_no_empty_fields():
    for entity in entities:
        for key in entity:
            assert entity[key] is not None
            if isinstance(entity[key], str):
                assert entity[key].strip() != ""


def test_fuzzy_entity_logic():
    for entity in entities:
        if entity["type"] == "ENTITE_fuzzy":
            assert entity["id_prenom"] in dict_prenom_to_id.values()
            assert entity["id_nom"] in dict_nom_to_id.values()
