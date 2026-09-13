# Glioma Biomarker Prediction from 3D MRI

Final-year research project predicting IDH mutation status and MGMT promoter methylation
status from preoperative multiparametric 3D MRI (UCSF-PDGM), comparing a 3D CNN, a hybrid
3D CNN-ViT, and a spiking neural network under one identical evaluation protocol.

This is a research codebase, not a product. See `CLAUDE.md` for the full protocol, and
`docs/DATASET.md`, `docs/METHODOLOGY.md`, `docs/PAPER_REVIEW.md` for the dataset reference,
the evaluation/leakage rules, and the critical review of the two base papers this project
positions itself against.

## Status

Phase 0 (setup/scaffold). See `PROGRESS.md` for current state.

## Setup

```bash
make setup   # pip install -e ".[dev]", pre-commit install
make lint    # ruff + black --check + mypy
make test    # pytest (tests/test_no_leakage.py is mandatory)
```

See `docs/ENVIRONMENT.md` for the resolved Python/dependency versions and compute notes.
