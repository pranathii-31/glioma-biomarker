"""Inference boundary for the research prototype (CLAUDE.md §4 `app/`, ADR 006).

Deliberately free of `monai`, `scipy` and `torch` imports at module level so the prototype's
validation layer and its tests stay importable without the training stack. A real predictor
loading a trained checkpoint imports those lazily, inside its own factory - see
`glioma.inference.predictor`.
"""
