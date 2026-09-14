# MGMT permutation test

Training labels shuffled within each fold's training set only (never the held-out fold), radiomics+GBM retrained per fold, pooled OOF AUC computed against the **real** held-out labels - docs/METHODOLOGY.md §3.

- AUC: 0.537 (95% CI 0.467-0.609)
- n = 306
- Verdict: **PASS (CI includes 0.5)**
