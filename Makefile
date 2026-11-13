.PHONY: install data train eval e2e test all

PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PY) scripts/run_data_pipeline.py

train:
	$(PY) scripts/run_train.py --model-out models/ner_v2 --epochs 20

eval:
	$(PY) scripts/run_eval.py --dataset data/processed/v2/dataset_augmented.jsonl --model models/ner_v2

e2e:
	$(PY) scripts/run_end2end.py --input input.txt --detector dict

test:
	$(PY) -m pytest -q

all: data train eval e2e test
