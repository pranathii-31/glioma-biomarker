"""Build metadata/master_metadata.csv by globbing patient folders under data/raw.

Glob by file suffix inside each patient folder; never construct paths by string
concatenation of the folder name - see docs/DATASET.md §3.
"""
