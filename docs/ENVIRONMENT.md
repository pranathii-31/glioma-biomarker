# Environment

Recorded on first setup, per CLAUDE.md §5 ("do not assume the versions in this file are current").

## Discrepancy vs. CLAUDE.md §5

CLAUDE.md specifies **Python 3.11**. The only interpreter available on this machine
(MacBook Pro M4, macOS, arm64) is **Python 3.10.10** (`/Library/Frameworks/Python.framework/
Versions/3.10`, python.org build, native arm64 - not Rosetta). No 3.11 interpreter, `pyenv`, or
`uv` is installed. Per the disk-wins rule in CLAUDE.md §9, `pyproject.toml` targets
`>=3.10,<3.12` instead of pinning 3.11. All pinned dependencies below support 3.10.

**Action for you:** install Python 3.11 (`brew install python@3.11` or `pyenv install 3.11`) if
you want to match CLAUDE.md exactly, or update CLAUDE.md §5 to say 3.10. Nothing in the pinned
dependency set requires 3.11 specifically.

## Resolved dependency versions (pinned in pyproject.toml, checked against PyPI 2026-09-13)

`numpy`, `pandas`, `scipy` and `scikit-learn` are pinned one or two minor versions below their
absolute PyPI latest because their newest releases (numpy 2.5.3, pandas 3.0.5, scipy 1.18.1,
scikit-learn 1.9.1) already require Python >=3.11/3.12 - confirmed by `pip index versions` in the
`.venv` created for this scaffold, which showed only the highest **3.10-compatible** release for
each. Everything else's latest release still supports 3.10.

| Package | Version |
|---|---|
| torch | 2.14.0 |
| monai | 1.6.0 |
| nibabel | 5.4.2 |
| SimpleITK | 2.5.6 |
| numpy | 2.2.6 (latest supporting py3.10; 2.5.3 needs >=3.12) |
| pandas | 2.3.3 (latest supporting py3.10; 3.0.5 needs >=3.11) |
| scipy | 1.15.3 (latest supporting py3.10; 1.18.1 needs >=3.11) |
| scikit-learn | 1.7.2 (latest supporting py3.10; 1.9.1 needs >=3.11) |
| pyradiomics | 3.1.0 |
| lightgbm | 4.7.0 |
| hydra-core | 1.3.6 |
| omegaconf | 2.3.1 |
| mlflow | 3.16.0 |
| streamlit | 1.63.0 |
| snntorch (optional `[snn]` extra) | 1.0.0 |
| pytest / ruff / black / mypy / pre-commit (dev) | 9.1.1 / 0.16.7 / 26.5.1 / 2.3.1 / 4.6.2 |

This is a second, independent reason (beyond CLAUDE.md's stated target) to install Python 3.11:
it would let numpy/pandas/scipy/scikit-learn track current releases instead of being capped by
the interpreter. Only the `dev` toolchain (pytest/ruff/black/mypy/pre-commit) was actually
installed and exercised in this session, to verify `make lint` and `make test` on the empty
scaffold - the full heavy/GPU-oriented dependency set (torch, monai, etc.) was not installed here,
by design, to avoid a multi-GB download before there is any code to run against it.

## Experiment tracking

CLAUDE.md §5 leaves this as "W&B *or* MLflow". Defaulted to **local MLflow** (`mlruns/`,
gitignored) so Phase 0-1 work needs no external account. See `docs/adr/001-experiment-tracking.md`.
Revisit before Phase 5 if you'd rather use W&B for the multi-seed / multi-architecture sweep.

## Compute

MacBook Pro, Apple M4, 16 GB unified memory. No discrete/CUDA GPU - training will use MPS or CPU.
Implications to keep in mind for later phases (not acted on yet, since no training code exists):
- 16 GB unified memory is tight for 3D volumes at 96^3 x 4 channels with batch size 4-8; may need
  smaller batches, gradient checkpointing, or the `whole_brain` 128^3 regime traded for a smaller
  crop.
- `torch.compile` / AMP bf16 support on MPS is less mature than on CUDA - budget time to verify
  before relying on it in Phase 5+.
- The SNN arm (`snntorch`) is a stretch goal per CLAUDE.md §8 and is kept in an optional
  `[snn]` extra for exactly this reason.
