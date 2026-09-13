"""Hybrid 3D CNN-ViT: a conv stem token grid feeding a small transformer.

Keep it small (2-4 layers, dim 256-384) with strong regularisation - a from-scratch ViT
does not work at this data scale, which is the core methodological error of Base Paper 1
(docs/PAPER_REVIEW.md). See CLAUDE.md §8.
"""
