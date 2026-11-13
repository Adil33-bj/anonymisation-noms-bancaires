#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "results"

NOMS_SOURCE = ROOT_DIR / "noms_marocains.csv"
PRENOMS_SOURCE = ROOT_DIR / "prenoms_marocains.csv"


def ensure_result_dir():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_and_save_dicts():
    if not NOMS_SOURCE.is_file():
        raise FileNotFoundError(f"'{NOMS_SOURCE.name}' introuvable.")
    if not PRENOMS_SOURCE.is_file():
        raise FileNotFoundError(f"'{PRENOMS_SOURCE.name}' introuvable.")

    df_noms = pd.read_csv(NOMS_SOURCE, header=0)
    df_prenoms = pd.read_csv(PRENOMS_SOURCE, header=0)
    df_noms = df_noms[df_noms["Nom"] != "Nom"]
    df_prenoms = df_prenoms[df_prenoms["Prenom"] != "Prenom"]

    noms_list = df_noms["Nom"].tolist()
    dict_id_to_nom = {i + 1: noms_list[i] for i in range(len(noms_list))}
    dict_nom_to_id = {noms_list[i]: i + 1 for i in range(len(noms_list))}

    prenoms_list = df_prenoms["Prenom"].tolist()
    dict_id_to_prenom = {i + 1: prenoms_list[i] for i in range(len(prenoms_list))}
    dict_prenom_to_id = {prenoms_list[i]: i + 1 for i in range(len(prenoms_list))}

    mappings = [
        (dict_nom_to_id, "dict_nom_to_id.json"),
        (dict_id_to_nom, "dict_id_to_nom.json"),
        (dict_prenom_to_id, "dict_prenom_to_id.json"),
        (dict_id_to_prenom, "dict_id_to_prenom.json"),
    ]
    for d, filename in mappings:
        out_path = RESULT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"→ JSON créé : results/{filename}")


if __name__ == "__main__":
    ensure_result_dir()
    build_and_save_dicts()
    print("✔ Construction des dicts terminée.")
