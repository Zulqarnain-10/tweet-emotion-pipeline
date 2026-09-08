# Convenience targets. Every recipe is a single command that also runs as-is in PowerShell.
# Override the interpreter with: make PYTHON=.venv/Scripts/python.exe <target>
# PYTHON only picks the interpreter that launches each tool. The DVC targets (repro, dag,
# metrics) need the venv activated, or its Scripts or bin directory first on PATH, because
# the stage commands in dvc.yaml call python by name and would otherwise run on the system
# interpreter.

PYTHON ?= python
IMAGE ?= ghcr.io/zulqarnain-10/tweet-emotion-pipeline:latest
API_URL ?= http://127.0.0.1:8000
HF_USER ?= syedzulqarnainh
LOADTEST_HOST ?= local uvicorn, Windows 11, Python 3.12, single process

.PHONY: setup nltk repro dag metrics test lint format serve loadtest docker-build docker-run sync-readme social-preview space-dry-run clean

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt -e .

nltk:
	$(PYTHON) -m tweet_emotion.setup_nltk

repro:
	$(PYTHON) -m dvc repro

dag:
	$(PYTHON) -m dvc dag

metrics:
	$(PYTHON) -m dvc metrics show

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

serve:
	$(PYTHON) -m tweet_emotion.api

loadtest:
	$(PYTHON) -m tweet_emotion.loadtest --url $(API_URL) --requests 300 --concurrency 10 --host "$(LOADTEST_HOST)"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

sync-readme:
	$(PYTHON) -m tweet_emotion.sync_readme

social-preview:
	$(PYTHON) scripts/make_social_preview.py

space-dry-run:
	$(PYTHON) scripts/deploy_space.py --hf-user $(HF_USER) --dry-run

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.space_build', 'htmlcov', 'build', 'dist'] + [str(q) for d in ('src', 'tests', 'scripts') for q in pathlib.Path(d).rglob('__pycache__')]]; [pathlib.Path(f).unlink(missing_ok=True) for f in ('.coverage', 'coverage.xml')]"
