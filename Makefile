.PHONY: install test lint clean data train experiments figures all

install:
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	ruff check cbm_variable_stars/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true

data:
	python scripts/01_download_gaia.py
	python scripts/02_download_ogle.py
	python scripts/03_crossmatch.py
	python scripts/04_extract_features.py
	python scripts/05_build_dataset.py

train:
	python scripts/06_train_models.py

experiments:
	python scripts/07_run_experiments.py

figures:
	python scripts/08_generate_figures.py

all: install data train experiments figures
