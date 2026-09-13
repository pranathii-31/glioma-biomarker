"""Masked multitask loss.

Patients missing an MGMT label still contribute to the IDH head instead of being
dropped - see docs/DATASET.md §7.
"""
