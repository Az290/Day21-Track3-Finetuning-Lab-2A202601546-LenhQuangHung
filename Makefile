# Lab 21 — Fine-tuning LLMs (Track 3)
.DEFAULT_GOAL := help
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help setup setup-cpu smoke nb1 nb2 nb3 nb4 nb5 nb6 pipeline pipeline-full test verify colab data clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Full GPU install (torch first — see requirements.txt header)
	$(PY) -m venv $(VENV) && $(BIN)/pip install -U pip && $(BIN)/pip install -r requirements.txt
	@echo "Now: cp .env.example .env && make smoke"

setup-cpu:  ## CPU-only install — enough for NB1 + the whole test suite, no GPU
	$(PY) -m venv $(VENV) && $(BIN)/pip install -U pip && $(BIN)/pip install -r requirements-cpu.txt

smoke:  ## Imports + data + unit tests. No GPU, no model download.
	$(BIN)/python scripts/verify.py --smoke

data:  ## Regenerate the seed corpus (deterministic)
	$(BIN)/python scripts/make_seed_data.py

nb1:  ## NB1 — data, chat template, loss mask     (CPU, ~2 min)
	$(BIN)/python notebooks/01_data_and_mask.py
nb2:  ## NB2 — freeze eval + three baselines      (GPU, ~10 min)
	$(BIN)/python notebooks/02_baselines.py
nb3:  ## NB3 — train the correct configuration    (GPU, ~25 min T4)
	$(BIN)/python notebooks/03_train_correct.py
nb4:  ## NB4 — misconfiguration autopsy, 3 runs   (GPU, ~35 min T4)
	$(BIN)/python notebooks/04_misconfig_autopsy.py
nb5:  ## NB5 — four-group eval + verdict          (GPU, ~10 min)
	$(BIN)/python notebooks/05_evaluate_and_verdict.py
nb6:  ## NB6 — merge + adapter hot-swap (OPTIONAL)(GPU, ~10 min)
	$(BIN)/python notebooks/06_merge_and_serve.py

pipeline: nb1 nb2 nb3 nb4 nb5  ## CORE: NB1 -> NB5 (~80 min on a T4)
pipeline-full: pipeline nb6    ## Core + the optional merge/serve notebook

test:  ## Unit tests only
	$(BIN)/python -m pytest tests/ -q

verify:  ## Pre-submission gatekeeper — run this before you zip
	$(BIN)/python scripts/verify.py

colab:  ## Regenerate colab/*.ipynb from notebooks/*.py
	$(BIN)/python scripts/build_colab.py

clean:  ## Remove generated artifacts (keeps the seed corpus)
	rm -rf adapters/*/ data/split results/*.json results/*.csv gguf __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
