"""Attention-in-tumour fraction and the shuffled-label saliency sanity check.

Proportion of top-k% saliency mass falling inside the tumour segmentation. If a model
trained on shuffled labels also lands its saliency on the tumour, the saliency method is
detecting the tumour, not the biomarker - see docs/METHODOLOGY.md §9.
"""
