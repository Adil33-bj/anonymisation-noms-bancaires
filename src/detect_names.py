#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from rapidfuzz import process, fuzz

STOPWORDS = {
    "est", "et", "de", "le", "la", "du", "un", "une", "ce", "cet", "cette", "mon", "ton", "son"
}

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "results"

with open(RESULT_DIR / "dict_nom_to_id.json", "r", encoding="utf-8") as f:
    dict_nom_to_id = json.load(f)
with open(RESULT_DIR / "dict_prenom_to_id.json", "r", encoding="utf-8") as f:
    dict_prenom_to_id = json.load(f)

dict_nom_to_id_lower = {k.lower(): v for k, v in dict_nom_to_id.items()}
dict_prenom_to_id_lower = {k.lower(): v for k, v in dict_prenom_to_id.items()}

entities = []


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", " ", text)
    return [tok for tok in cleaned.split() if tok]


def fuzzy_match(tok_norm: str, candidates: dict[str, int], threshold: int = 80):
    if not tok_norm or tok_norm in STOPWORDS:
        return None
    subset = {k: v for k, v in candidates.items() if k.startswith(tok_norm[0])}
    if not subset:
        return None
    match, score, _ = process.extractOne(tok_norm, subset.keys(), scorer=fuzz.ratio)
    return (match, candidates[match]) if score >= threshold else None


def detect_entities(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    tokens = tokenize(content)
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        tok_norm = tok.lower()

        if tok_norm in STOPWORDS:
            i += 1
            continue

        if tok_norm in dict_prenom_to_id_lower:
            id_prenom = dict_prenom_to_id_lower[tok_norm]

            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                next_norm = next_tok.lower()

                if next_norm in STOPWORDS:
                    i += 1
                    continue

                if next_norm in dict_nom_to_id_lower:
                    id_nom = dict_nom_to_id_lower[next_norm]
                    entities.append({
                        "type": "ENTITE",
                        "prenom": tok,
                        "nom": next_tok,
                        "id_prenom": id_prenom,
                        "id_nom": id_nom,
                    })
                    i += 2
                    continue

                fm_nom = fuzzy_match(next_norm, dict_nom_to_id_lower)
                if fm_nom:
                    _, id_nom = fm_nom
                    entities.append({
                        "type": "ENTITE_fuzzy",
                        "prenom": tok,
                        "nom": next_tok,
                        "id_prenom": id_prenom,
                        "id_nom": id_nom,
                    })
                    i += 2
                    continue

            entities.append({"type": "PRENOM_seul", "prenom": tok, "id_prenom": id_prenom})
            i += 1
            continue

        if tok_norm in dict_nom_to_id_lower:
            id_nom = dict_nom_to_id_lower[tok_norm]

            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                next_norm = next_tok.lower()
                if next_norm in dict_prenom_to_id_lower:
                    id_prenom = dict_prenom_to_id_lower[next_norm]
                    entities.append({
                        "type": "ENTITE_inversee",
                        "prenom": next_tok,
                        "nom": tok,
                        "id_prenom": id_prenom,
                        "id_nom": id_nom,
                    })
                    i += 2
                    continue

            entities.append({"type": "NOM_seul", "nom": tok, "id_nom": id_nom})
            i += 1
            continue

        fm_prenom = fuzzy_match(tok_norm, dict_prenom_to_id_lower)
        if fm_prenom:
            _, id_prenom = fm_prenom

            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                next_norm = next_tok.lower()

                if next_norm in STOPWORDS:
                    i += 1
                    continue

                if next_norm in dict_nom_to_id_lower:
                    id_nom = dict_nom_to_id_lower[next_norm]
                    entities.append({
                        "type": "ENTITE_fuzzy",
                        "prenom": tok,
                        "nom": next_tok,
                        "id_prenom": id_prenom,
                        "id_nom": id_nom,
                    })
                    i += 2
                    continue

                fm_nom = fuzzy_match(next_norm, dict_nom_to_id_lower)
                if fm_nom:
                    _, id_nom = fm_nom
                    entities.append({
                        "type": "ENTITE_fuzzy",
                        "prenom": tok,
                        "nom": next_tok,
                        "id_prenom": id_prenom,
                        "id_nom": id_nom,
                    })
                    i += 2
                    continue

            entities.append({"type": "PRENOM_fuzzy", "prenom": tok, "id_prenom": id_prenom})
            i += 1
            continue

        fm_nom = fuzzy_match(tok_norm, dict_nom_to_id_lower)
        if fm_nom:
            _, id_nom = fm_nom
            entities.append({"type": "NOM_fuzzy", "nom": tok, "id_nom": id_nom})

        i += 1

    output_path = RESULT_DIR / "entities_found.json"
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(entities, f_out, ensure_ascii=False, indent=2)

    print(f"✅ Entités exportées dans {output_path} ({len(entities)} trouvées)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python src/detect_names.py chemin/vers/input.txt")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    detect_entities(str(input_path))
