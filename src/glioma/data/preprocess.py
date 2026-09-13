"""Normalise, crop and resample raw NIfTI volumes into a cached float16 .npy array.

Per-patient, per-modality z-score inside the brain mask only; idempotent and resumable -
see docs/DATASET.md §7-8. Hash the preprocessing config into the cache directory name so a
config change forces a rebuild rather than silently reusing a stale cache.
"""
