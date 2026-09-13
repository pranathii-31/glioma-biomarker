# ADR 001: Experiment tracking with local MLflow

## Context

CLAUDE.md §5 lists "Weights & Biases *or* MLflow" without choosing. With 3 architectures x
several configurations x >=3 seeds (CLAUDE.md §2 rule 9), we will have 100+ runs by Phase 6-7
and need each `experiments/<run_id>/` to be traceable to a tracked run.

## Decision

Use **local MLflow** (`mlruns/`, gitignored) rather than Weights & Biases for now.

## Consequences

- No external account or API key needed to start Phase 0/1 work.
- Tracking data lives only on this machine unless explicitly exported; no built-in collaboration
  view. Acceptable for a single-author final-year project.
- If remote access or richer sweep tooling becomes useful (e.g. before the Phase 6/7 architecture
  comparison), switching to W&B is a config-level change (`configs/train/default.yaml` does not
  hard-code the tracker), not a rewrite.
- Revisit this decision explicitly before Phase 5, when the volume of runs makes a spreadsheet or
  ad hoc `mlruns/` browsing painful (see ACTION_PLAN.md Part C item 7).
