"""MONAI Dataset/CacheDataset returning masked multitask labels.

Labels are masked, not dropped: MGMT is missing for most grade-2 patients - see
docs/DATASET.md §7 for the masked multitask loss this loader must support.
"""
