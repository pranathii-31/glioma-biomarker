# ADR 002: Split stratification key is IDH x grade-bucket, not IDH x grade x MGMT-availability

## Context

docs/METHODOLOGY.md §2 specifies stratifying the lock-box test split and the 5-fold CV by
"(IDH status x WHO grade x MGMT availability)". Checked against the actual metadata CSV
(495 patients, post follow-up-exclusion), the literal joint distribution is:

| IDH | Grade | MGMT available | n |
|---|---|---|---|
| wildtype | 3 | No | 1 |
| mutant | 2 | Yes | 3 |
| wildtype | 2 | No | 5 |
| wildtype | 2 | Yes | 5 |
| mutant | 3 | Yes | 9 |
| wildtype | 3 | Yes | 13 |
| wildtype | 4 | No | 19 |
| mutant | 3 | No | 20 |
| mutant | 4 | Yes | 28 |
| mutant | 2 | No | 43 |
| wildtype | 4 | Yes | 349 |

(11 non-empty cells of 12 possible, summing to 495 as expected; the 12th, mutant x grade 4 x
MGMT-unavailable, is empty.)

`StratifiedKFold(n_splits=5)` requires every stratum to have at least 5 members (and
`train_test_split(..., stratify=...)` requires at least 2). Several cells here have 1, 3, or 5 -
the literal 3-way key cannot be used for 5-fold CV at all, and is marginal even for the 20%
lock-box split.

## Decision

Use a 2-way key: **IDH status x grade-bucket**, where grade-bucket is `g4` (WHO grade 4) vs
`g23` (WHO grade 2 or 3 combined). This is mechanically enforced via
`sklearn.model_selection.train_test_split` / `StratifiedKFold`. The four resulting cells have a
minimum size of 24 patients (IDH-wildtype x g23), safe for both the test split and 5-fold CV
with real margin.

MGMT-availability is **not** a mechanical stratification axis. It is computed and reported per
split/fold in `results/splits_stratification.md` (via `build_stratification_table`) instead, so
any imbalance is visible rather than silently assumed away.

Grade-bucketing (rather than dropping grade from the key entirely) was kept because it directly
serves the mandatory grade-4-only / grade-2-3-only subgroup analysis in docs/METHODOLOGY.md §5 -
every fold and the test set now has a guaranteed, roughly proportional mix of both grade
regimes, which is the more consequential property for that later analysis than exact
IDH x grade x MGMT joint balance would have been.

## Alternatives considered

- **Literal 3-way stratification**: infeasible as shown above.
- **Custom multi-way/iterative balancer** (e.g. a hand-rolled greedy assignment, or a library
  such as `iterstrat`'s `IterativeStratification`) to approximately balance all three axes
  simultaneously without requiring large joint cells: more faithful to the doc's literal
  wording, but more code to write, test, and maintain, and not backed by the well-tested
  `sklearn` path already used everywhere else in this project. Presented to and declined by the
  project owner in favour of the simpler, standard-library approach.

## Consequences

- `docs/METHODOLOGY.md` §2 updated to describe the actual key used, with a pointer to this ADR.
- `results/splits_stratification.md` (generated once, alongside `splits/`) is the place to check
  whether MGMT-availability ended up reasonably balanced across folds. If a later phase finds a
  fold conspicuously MGMT-light, that is itself a reportable finding, not necessarily a bug.
- If a future phase needs a different split (e.g. because the download reveals patients whose
  grade or IDH the CSV got wrong), that is a "never regenerate `splits/`" situation - stop and
  ask, per CLAUDE.md §9, rather than silently rerunning `make splits`.
