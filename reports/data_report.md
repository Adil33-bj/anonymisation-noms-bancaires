# Data Report

## Inventory
- Raw artifacts synchronized: 6
- Mode: keep_existing

## Canonical Schema
- Fields: id, text, entities, source, language, difficulty, split
- Canonical dataset path: data/processed/v1/dataset.jsonl
- Total rows v1: 1266

## Validation + Cleaning
- Validation errors: 1371
- Clean rows: 584
- Dropped invalid: 0
- Dropped exact duplicates: 682
- Validation output: data/processed/validation_errors.json

## Split
- Deterministic seed: 42
- Stratification: entity presence + sentence pattern category
- Leakage detected: False
- Split sizes: train=402, dev=84, test=98
- Frozen split file: data/splits/split_v1.json

## Versioning
- Changelog: data/processed/CHANGELOG.md

## Enrichment + Anonymization Gold
- Augmented dataset (v2): data/processed/v2/dataset_augmented.jsonl
- Augmentation report: data/processed/v2/augmentation_report.json
- Anonymization gold: data/eval/anonymization_gold_test.jsonl