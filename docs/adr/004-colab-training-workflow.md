# ADR 004: Move 3D ResNet grid training to Colab GPU, with resumable checkpoints

## Context

The 3D ResNet-18 pilot (1 fold, 1 seed, 5-epoch cap) on the local M4/MPS machine completed
epochs 0-3 with sane, non-leaky val AUCs, but per-epoch wall-clock degraded badly (30 -> 78 ->
41 -> 83 min) and epoch 4 stalled for 3h42m+ under severe host memory thrashing (free RAM ~74MB,
swap 80% full). It was killed (SIGTERM) on explicit instruction once diagnosed as thrashing, not
a code deadlock. It never wrote `metrics.json`, so there is no reportable pilot result from that
run (CLAUDE.md §9 rule 8) - see PROGRESS.md's Phase 5 section for the full account.

The full protocol needs 5 folds x 3 seeds x {resnet18, resnet34} = 30 runs, each up to 100
epochs (patience 15). Repeating the M4 attempt at that scale is not viable given the observed
thrashing, and CUDA on a Colab GPU is the approved alternative (user-confirmed).

## Decisions

1. **Full trainer-state checkpoints, not weights-only** (`src/glioma/train/loop.py`). Each
   `epoch_N.pt` now holds model state, optimizer state, the epoch index, best-AUC bookkeeping,
   the early-stopping patience counter, and RNG state. `resume_from` restores all of it, so a
   Colab session disconnect and a later `--resume` continue the exact same run - same cosine LR
   schedule position, same patience count - rather than silently restarting them from scratch.
   Weights-only resume was considered and rejected: it would change the run's outcome without
   any visible signal, which CLAUDE.md §9 ("long jobs... make them resumable") reads as requiring
   more than "the file exists."

2. **`epoch_limit` is a separate concept from `max_epochs`.** `max_epochs` fixes the cosine
   schedule's total length and must stay identical across every call of one logical run.
   `epoch_limit` caps how many epochs *this invocation* runs before stopping (without marking
   convergence), to fit a Colab session's time/idle budget. Conflating the two (i.e. lowering
   `max_epochs` to represent "how far we got this session") would have shifted the LR schedule
   and made a resumed run diverge from an uninterrupted one - caught by
   `tests/test_train_loop.py::test_resume_from_checkpoint_reproduces_an_uninterrupted_run`.

3. **Run directory names are stable** (`{architecture}_{fold}_seed{seed}`, no timestamp), a
   narrow, flagged deviation from docs/METHODOLOGY.md §10's `..._{YYYYMMDD-HHMM}` convention.
   A timestamped directory name would break `--resume`'s checkpoint lookup across sessions
   (each new invocation would get a new "unique" directory with no history). Start/end
   timestamps are recorded inside `metrics.json` instead, so run provenance is not lost.

4. **AMP dtype is auto-selected per device, not hardcoded to `configs/train/default.yaml`'s
   `bf16`.** *(The capability check described below was superseded by
   [ADR 005](005-phase5-throughput-and-checkpoint-retention.md): `is_bf16_supported()` also
   returns True on Turing, where bf16 is emulated. The classification of AMP dtype as infra
   rather than protocol, and everything else in this section, still stands.)* `bf16` needs Ampere+ tensor cores; a free-tier Colab GPU is commonly a T4 (Turing,
   fp16-only). `_select_amp_dtype` in `scripts/train_resnet_baseline.py` picks bf16 on
   bf16-capable CUDA, fp16 on other CUDA, and disables autocast entirely on MPS/CPU - matching
   what the M4 pilot actually ran, so an M4 rerun (if ever needed) stays comparable. This is an
   infra/hardware selection, not a change to the scientific protocol (docs/METHODOLOGY.md §7's
   "same optimizer family... epoch budget... loss" fairness requirement is about training
   hyperparameters, not numerical precision plumbing).
   - `train_with_early_stopping` wires in a `torch.amp.GradScaler`, enabled only when
     `device.type == "cuda" and amp_dtype == torch.float16`. fp16's narrow exponent range can
     silently underflow small gradients to zero with no error and no NaN; bf16 shares fp32's
     exponent range and doesn't need it. `GradScaler(enabled=False)` is a documented no-op
     passthrough, so it's left wired in unconditionally rather than branching the loop on dtype.
   - `_evaluate_auc` casts logits to `.float()` before `.numpy()`: under bf16 autocast, model
     output tensors stay bfloat16 through `.cpu()`/`.cat()`, and NumPy has no bfloat16 dtype at
     all - `.numpy()` on a raw bf16 tensor raises `TypeError`. Found by code review (this
     session's local `pytest` is blocked by an unrelated macOS/scipy issue, so it could not be
     caught by a CUDA/bf16 test run - there is no GPU available to actually exercise this path
     locally).

5. **Training data reaches Colab via Google Drive, never via git.** `data/processed/tumor_crop_*`
   (3.1GB of gitignored `.npy` cache) is uploaded to a Drive folder out-of-band and mounted into
   the Colab session; `notebooks/colab_pilot.ipynb` only clones the (data-free) git repo and
   shells out to `scripts/train_resnet_baseline.py` - no training logic lives in the notebook
   itself (CLAUDE.md §6: "no logic in notebooks... import from `src/`").

6. **Actual GPU execution happens outside this session.** Claude Code has no browser/Colab/
   cloud-GPU tool access in this environment; the notebook is prepared here but must be run by
   the user in an actual Colab GPU runtime, with the resulting `experiments/*/metrics.json`
   files brought back (via `git pull`/manual copy) for validation and inclusion in
   `results/baselines.md`.

## Consequences

- The M4 pilot's lack of a result is not retroactively fabricated or approximated; the Colab
  pilot (run by the user, from `notebooks/colab_pilot.ipynb`) is the first real timing signal for
  sizing the full grid, per the standing "no full grid without a completed, timed pilot" rule.
- `find_latest_checkpoint()` (`src/glioma/train/loop.py`) is the one place that decides which
  checkpoint a `--resume` continues from - by highest epoch number in the filename, not mtime,
  so a partially-synced Drive folder (files present but stale mtimes from the sync process)
  cannot pick the wrong checkpoint.
- `scripts/train_resnet_baseline.py`'s run_id format now differs from the `..._{YYYYMMDD-HHMM}`
  suffix used by the already-committed cheap-baseline experiments (`majority_idh_seed-na_...`,
  `radiomics_gbm_idh_seed0_...`) - `scripts/build_results_table.py` must not assume that suffix
  is always present when it reads `experiments/*/metrics.json`.
