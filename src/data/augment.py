"""Data augmentation utilities for Moroccan PERSON detection.

This module generates:
1) Positive synthetic banking samples with PERSON entities.
2) Hard negatives in the same banking domain with no PERSON entities.

The augmentation is controlled and deterministic:
- only new rows with `split="train"` are added
- metadata is tracked (`is_augmented`, `pattern_id`)
- generation uses a fixed random seed for reproducibility
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from src.utils.io import read_jsonl, write_json, write_jsonl


SEED = 42


def _read_name_column(path: Path, preferred_cols: list[str]) -> list[str]:
    """Read a CSV file and return the first matching non-empty name column."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []

    available = {c.lower(): c for c in rows[0].keys()}
    selected = None
    for col in preferred_cols:
        if col.lower() in available:
            selected = available[col.lower()]
            break
    if selected is None:
        selected = rows[0].keys().__iter__().__next__()

    values = []
    for row in rows:
        v = (row.get(selected) or "").strip()
        if v:
            values.append(v)
    return values


def _load_names(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Load first names and last names from raw CSV files."""
    prenoms = _read_name_column(raw_dir / "prenoms_marocains.csv", ["Prenom", "Prénom", "prenom"])
    noms = _read_name_column(raw_dir / "noms_marocains.csv", ["Nom", "nom"])
    return prenoms, noms


def _make_typo(name: str, rng: random.Random) -> str:
    """Create a lightweight typo variant (insertion, deletion, swap)."""
    if len(name) < 4:
        return name
    mode = rng.choice(["insert", "delete", "swap"])
    idx = rng.randrange(1, len(name) - 1)
    if mode == "insert":
        return name[:idx] + rng.choice("aeiou") + name[idx:]
    if mode == "delete":
        return name[:idx] + name[idx + 1 :]
    chars = list(name)
    chars[idx - 1], chars[idx] = chars[idx], chars[idx - 1]
    return "".join(chars)


def _name_variant(prenom: str, nom: str, rng: random.Random) -> tuple[str, str]:
    """Generate Moroccan-flavored name variants and return (text_name, pattern_id)."""
    forms = [
        ("full_standard", f"{prenom} {nom}"),
        ("full_upper_lower", f"{prenom.upper()} {nom.lower()}"),
        ("el_prefix", f"{prenom} El {nom}"),
        ("al_prefix", f"{prenom} Al {nom}"),
        ("double_space", f"{prenom}  {nom}"),
        ("typo_prenom", f"{_make_typo(prenom, rng)} {nom}"),
        ("typo_nom", f"{prenom} {_make_typo(nom, rng)}"),
    ]
    return rng.choice(forms)


def _positive_templates() -> list[tuple[str, str]]:
    """Return realistic banking contexts containing one PERSON placeholder."""
    return [
        ("tx_virement", "Virement de {name} vers le compte 1234."),
        ("tx_paiement", "Paiement destiné à {name} pour la facture 9021."),
        ("tx_cheque", "Le chèque de {name} a été encaissé ce matin."),
        ("tx_retrait", "Retrait effectué par {name} à l'agence de Rabat."),
        ("tx_reclamation", "Réclamation client ouverte au nom de {name}."),
        ("tx_verification", "Vérification KYC validée pour {name}."),
    ]


def _negative_templates() -> list[tuple[str, str]]:
    """Return hard negatives from the same domain without PERSON entities."""
    return [
        ("neg_solde", "Le solde du compte est suffisant pour cette opération."),
        ("neg_releve", "Demande de relevé mensuel enregistrée avec succès."),
        ("neg_systeme", "Transaction validée automatiquement par le système bancaire."),
        ("neg_support", "Veuillez contacter le service client pour assistance."),
        ("neg_agence", "Mise à jour des horaires de l'agence de Casablanca."),
        ("neg_cb", "Opposition sur carte bancaire effectuée sans incident."),
    ]


def _append_positive_rows(
    rows: list[dict],
    prenoms: list[str],
    noms: list[str],
    n_extra: int,
    rng: random.Random,
) -> list[dict]:
    """Generate PERSON-positive synthetic rows for train split only."""
    templates = _positive_templates()
    new_rows: list[dict] = []
    start_index = len(rows)

    for i in range(n_extra):
        prenom = rng.choice(prenoms)
        nom = rng.choice(noms)
        name_variant, variant_tag = _name_variant(prenom, nom, rng)
        template_tag, template = rng.choice(templates)
        text = template.format(name=name_variant)
        start = text.find(name_variant)
        end = start + len(name_variant)
        if start < 0:
            continue

        new_rows.append(
            {
                "id": f"aug_pos_{start_index + i:06d}",
                "text": text,
                "entities": [{"start": start, "end": end, "label": "PERSON"}],
                "source": "augmentation",
                "language": "fr",
                "difficulty": "hard",
                "split": "train",
                "is_augmented": True,
                "pattern_id": f"{template_tag}|{variant_tag}",
            }
        )
    return new_rows


def _append_negative_rows(
    rows: list[dict],
    n_extra: int,
    rng: random.Random,
) -> list[dict]:
    """Generate hard negatives for train split only."""
    templates = _negative_templates()
    new_rows: list[dict] = []
    start_index = len(rows)

    for i in range(n_extra):
        pattern_id, text = rng.choice(templates)
        new_rows.append(
            {
                "id": f"aug_neg_{start_index + i:06d}",
                "text": text,
                "entities": [],
                "source": "augmentation",
                "language": "fr",
                "difficulty": "hard",
                "split": "train",
                "is_augmented": True,
                "pattern_id": pattern_id,
            }
        )
    return new_rows


def _deduplicate_by_text(rows: list[dict]) -> list[dict]:
    """Keep the first sample for each normalized text to limit redundancy."""
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = " ".join(str(row.get("text", "")).strip().lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def generate_augmented_train(dataset_path: Path, out_path: Path, n_extra: int = 200) -> dict:
    """Create v2 dataset from v1 by adding controlled train-only augmentation."""
    rng = random.Random(SEED)
    rows = read_jsonl(dataset_path)
    root = dataset_path.resolve().parents[3]
    raw_dir = root / "data" / "raw"
    prenoms, noms = _load_names(raw_dir)
    if not prenoms or not noms:
        raise RuntimeError("Name dictionaries are empty; check data/raw/*.csv")

    pos_count = int(n_extra * 0.7)
    neg_count = n_extra - pos_count
    positives = _append_positive_rows(rows, prenoms, noms, pos_count, rng)
    negatives = _append_negative_rows(rows + positives, neg_count, rng)

    augmented = list(rows) + positives + negatives
    augmented = _deduplicate_by_text(augmented)
    write_jsonl(out_path, augmented)

    report = {
        "base_rows": len(rows),
        "generated_positive": len(positives),
        "generated_negative": len(negatives),
        "final_rows": len(augmented),
        "out_path": out_path.name,
    }
    write_json(out_path.parent / "augmentation_report.json", report)
    return report


def main() -> None:
    from src.utils.io import project_root

    root = project_root()
    in_path = root / "data" / "processed" / "v1" / "dataset_clean.jsonl"
    out_path = root / "data" / "processed" / "v2" / "dataset_augmented.jsonl"
    report = generate_augmented_train(in_path, out_path)
    print(f"✅ Augmented dataset generated: {report['final_rows']} rows")


if __name__ == "__main__":
    main()
