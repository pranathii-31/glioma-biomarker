# ADR 005: Phase 5 throughput fixes - native-bf16 gate, cuDNN autotuning, checkpoint retention

Supersedes part of [ADR 004](004-colab-training-workflow.md) §4 (the bf16 capability check only;
every other decision in ADR 004 stands).

## Context

The Phase 5 grid (5 folds x 3 seeds x {resnet18, resnet34} = 30 runs) was running far slower and
consuming far more storage than the protocol requires. Profiling the actual code and hardware
produced three findings, none of them about the scientific protocol:

1. **The work is per-patient, not per-cohort.** A single forward pass on one 4-channel 96^3
   volume is **328 GFLOP** (resnet18) / **481 GFLOP** (resnet34), measured with
   `torch.utils.flop_counter`. Training is ~3x that. One epoch over the 317-patient training
   split plus 79-patient validation fold is ~340 TFLOP. Shrinking the cohort would not help and
   would break the frozen Phase 2 splits; the per-sample cost is the thing that is large.

2. **`torch.cuda.is_bf16_supported()` returns True on Turing**, where bf16 is emulated in
   software rather than executed on tensor cores. On the Colab free tier's T4 this made
   `_select_amp_dtype` choose bf16, measured at **202 s/epoch vs 65.5 s/epoch under fp16** for
   the same run - a 3.1x penalty for no numerical benefit.

3. **Checkpoints were retained for every epoch.** A full trainer-state checkpoint (model +
   AdamW moments + `best_state`) is ~532 MB for resnet18 and ~1017 MB for resnet34. Across the
   30-run grid that is **0.9-2.3 TB** resident on the Google Drive mount they are written to,
   which no practical Drive tier holds, and the sustained FUSE write pressure is the most likely
   cause of the `Transport endpoint is not connected` failures that cost whole Colab sessions.

For completeness, two things profiled as *not* being the problem and were deliberately left
alone: the data pipeline (0.014 s to load a cached volume, 0.07 s for the full MONAI
augmentation chain - ~6 s/epoch across 4 workers, against a 65-200 s epoch), and the local
M4/MPS path (measured 39-104 s per batch-of-6 step with **11.23 GB of 17.2 GB system RAM** held
by the MPS allocator - ~35 min/epoch, confirming ADR 004's thrashing diagnosis as a hardware
limit rather than a fixable bug).

## Decisions

1. **Gate bf16 on compute capability >= 8.0 (`_has_native_bf16`), not
   `torch.cuda.is_bf16_supported()`.** sm_80+ (Ampere, Ada, Hopper) has real bf16 tensor cores;
   Turing does not. fp16 is chosen everywhere else on CUDA, where the `GradScaler` already wired
   into `train_with_early_stopping` (ADR 004 §4) covers fp16's narrower dynamic range. This
   keeps ADR 004 §4's classification intact: AMP dtype is infra/hardware selection, not a
   scientific parameter - docs/METHODOLOGY.md §7's fairness requirement covers optimiser family,
   epoch budget and loss, not numerical-precision plumbing. `--amp-dtype` still overrides.

2. **Enable `torch.backends.cudnn.benchmark` on CUDA.** Every step sees an identical input shape
   `(batch_size, 4, 96, 96, 96)`, so cuDNN can autotune its 3D convolution algorithms once and
   reuse that choice rather than re-picking heuristically per call. Benchmark mode may select
   non-deterministic kernels, so the setting is recorded as `cudnn_benchmark` in each run's
   `metrics.json` - docs/METHODOLOGY.md §10 requires determinism-affecting settings to be
   recorded when enabled, not silently set.

3. **Retain only the newest 2 checkpoints per run** (`keep_last_checkpoints=2`, pruned after
   each epoch's save). `find_latest_checkpoint` only ever resumes from the newest, so older ones
   are dead weight. Keeping **two** rather than one is deliberate: a Drive FUSE mount can drop
   mid-write, leaving the newest checkpoint truncated, and an intact predecessor turns that from
   a lost run into a lost epoch. `keep < 1` raises rather than silently disabling resumability.

## Consequences

- Expected grid cost falls from ~83 GPU-h to roughly 13-22 GPU-h at 40 epochs/run (decision 1 is
  a measured 3.1x; decision 2 is an estimated 1.2-2x and is **not yet verified on this
  workload**). Drive residency falls from ~0.9-2.3 TB to ~47 GB.
- **`resnet18_fold_0_seed0` was trained under emulated bf16; every run after this change uses
  fp16.** Per ADR 004 §4 this is an infra difference rather than a protocol one, but it is a
  real difference in numerical precision across folds of one grid. Either re-run that fold under
  the new default, or report the mixed precision explicitly. `metrics.json` records `amp_dtype`
  per run, so which runs used which is always recoverable - this must not be left implicit.
- Checkpoint pruning is destructive and irreversible within a run. `tests/test_train_loop.py`
  covers both that retention is bounded to 2 and that a pruned run still resumes to bit-identical
  results (`test_pruned_run_still_resumes_from_the_newest_checkpoint`).
- Pre-ADR-004 checkpoints (e.g. `experiments/resnet18_pilot_fold_0_seed0/checkpoints/*.pt`) are
  weights-only, with no `optimizer_state` key, and will `KeyError` if passed to `--resume`. They
  are the archived M4 pilot and are not in the grid's path; nothing here changes that.
