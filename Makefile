.PHONY: setup lint test manifest splits preprocess train eval table app

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests scripts app
	black --check src tests scripts app
	mypy src

test:
	pytest -q

manifest:
	$(PYTHON) scripts/build_manifest.py

# RUN ONCE, then commit splits/ and never regenerate - see CLAUDE.md §9.
splits:
	$(PYTHON) scripts/make_splits.py

preprocess:
	$(PYTHON) scripts/preprocess.py

train:
	$(PYTHON) scripts/train.py --config $(EXP)

eval:
	$(PYTHON) scripts/evaluate.py --run $(RUN)

table:
	$(PYTHON) scripts/build_results_table.py

app:
	streamlit run app/main.py
