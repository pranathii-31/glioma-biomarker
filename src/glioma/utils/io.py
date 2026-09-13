"""Path helpers built on pathlib.Path.

Never build paths by string concatenation, especially for the UCSF file-name quirks
(T1c vs t1ce, folder/file digit-count mismatches) - see docs/DATASET.md §3 and CLAUDE.md §6.
"""
