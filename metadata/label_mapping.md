# Label mapping — frozen

Implemented in `src/glioma/data/labels.py`. Do not change any rule below without updating both
this file and the implementation, and recording an ADR (`docs/adr/`) - CLAUDE.md §9.

## Exclusions (applied first, before any other processing)

The 6 follow-up duplicate patients (docs/DATASET.md §2) are dropped from the metadata CSV before
label derivation or splitting. Each is identifiable by either its pre-v3 bare id or its post-v3
`_FUxxxd` suffix, depending on the release version - both spellings are recognised:

| Spelling(s) seen on disk / in the CSV | Baseline patient it duplicates |
|---|---|
| `UCSF-PDGM-0315` or `UCSF-PDGM-0433_FU007d` | `UCSF-PDGM-433` |
| `UCSF-PDGM-0278` or `UCSF-PDGM-0431_FU001d` | `UCSF-PDGM-431` |
| `UCSF-PDGM-0175` or `UCSF-PDGM-0396_FU175d` | `UCSF-PDGM-396` |
| `UCSF-PDGM-0138` or `UCSF-PDGM-0429_FU003d` | `UCSF-PDGM-429` |
| `UCSF-PDGM-0181` or `UCSF-PDGM-0409_FU001d` | `UCSF-PDGM-409` |
| `UCSF-PDGM-0289` or `UCSF-PDGM-0391_FU016d` | `UCSF-PDGM-391` |

## IDH mutation status

```
idh = 0 if IDH_column.strip().lower() == "wildtype" else 1   # 1 = mutant
```

Rationale: the `IDH` column uses many mutation-string spellings for the same mutation (e.g.
`IDH1 p.R132H` vs `IDH1 p.Arg132His` - docs/DATASET.md §5 point 2), so anything that is not
exactly `wildtype` is mapped to mutant rather than enumerating variants.

**Cross-check (must pass, not advisory):** every mutant label must correspond to a diagnosis
string containing `IDH-mutant`, and every wildtype label to one containing `IDH-wildtype`.
Any disagreement raises `ValueError` in `assert_idh_matches_diagnosis` rather than silently
producing a wrong label. `Final pathologic diagnosis` is used only for this check - it is never
a model input (CLAUDE.md §2 rule 4).

Missing `IDH` values remain `<NA>` (none expected - IDH is available for all 495 non-excluded
patients per docs/DATASET.md §5 - but not enforced as an assertion here, since the disk is the
source of truth if that ever changes).

## MGMT promoter methylation status

Primary definition (matches Elyassirad et al., the closest prior work on this dataset - see
docs/PAPER_REVIEW.md):

```
mgmt = 1 if MGMT_index > 0 else 0     # index == 0 -> unmethylated
mgmt = <NA> if MGMT_index is missing  # dropped from the MGMT loss term, not from the dataset
```

`MGMT index` is the source of truth, not `MGMT status`. Rows where the index is missing get
`<NA>` regardless of what `MGMT status` says (including rows where status is `positive` or
`negative` but index is `unknown`) - the masked multitask loss (docs/DATASET.md §7) uses this
`<NA>` to skip the MGMT term for that patient while still training the IDH head on it.

**Disagreements are logged, not silently resolved.** `find_mgmt_disagreements` flags every row
where `MGMT status` is `positive`/`negative` *and* `MGMT index` is known, but the two imply
different binary labels (e.g. status `positive` with index `0`). These rows still get a label
from the index-only rule above; the log is for auditing, per docs/DATASET.md §5 point 3.

## Forbidden input columns

`Final pathologic diagnosis (WHO 2021)`, `WHO CNS Grade`, `1p/19q` - never used as model
features (CLAUDE.md §2 rule 4). Listed in `labels.FORBIDDEN_INPUT_COLUMNS` so feature-building
code can assert against it.

## `who_grade` in the output table (added Phase 2)

`build_labels` also returns a `who_grade` column (2/3/4, nullable Int64). It exists **only**
for split stratification (docs/METHODOLOGY.md §2) and subgroup analysis (docs/METHODOLOGY.md
§5) - it is still a forbidden input column per the section above, and Phase 4's dataset/feature
code must exclude it explicitly rather than relying on it being absent from the table.
