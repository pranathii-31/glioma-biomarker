"""Training loop: AdamW, cosine schedule with warmup, AMP, early stopping.

Early stopping on validation AUC (patience 15), never on the lock-box test set -
see CLAUDE.md §2 rule 2 and §8.
"""
