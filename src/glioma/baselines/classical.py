"""Majority-class, age-only and age+sex logistic-regression baselines.

Mandatory reference points (docs/METHODOLOGY.md §3): a model that doesn't beat age-only for IDH
is not doing anything interesting - IDH-mutant mean age is 38.8 vs 61.6 for wildtype
(docs/DATASET.md §5), and a deep model can otherwise silently be reproducing that trivially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from glioma.baselines.cv import FitPredict, run_cv


def _majority_fit_predict(labels: pd.Series) -> FitPredict:
    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        majority = float(labels.loc[train_ids].mean() >= 0.5)
        return {pid: majority for pid in eval_ids}

    return fit_predict


def run_majority_baseline(
    labels_df: pd.DataFrame, folds: dict[str, list[str]], task: str
) -> dict[str, float]:
    """Always predict the training fold's majority class - the trivial CLAUDE.md §2 rule 3 floor."""
    labels = labels_df.set_index("patient_id")[task]
    labelled_ids = set(labels.dropna().index)
    return run_cv(_majority_fit_predict(labels), folds, labelled_ids)


def _logistic_regression_fit_predict(labels: pd.Series, features: pd.DataFrame) -> FitPredict:
    def fit_predict(train_ids: list[str], eval_ids: list[str]) -> dict[str, float]:
        x_train = features.loc[train_ids].to_numpy(dtype=float)
        y_train = labels.loc[train_ids].to_numpy(dtype=float)
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        x_eval = features.loc[eval_ids].to_numpy(dtype=float)
        probs = model.predict_proba(x_eval)[:, 1]
        return dict(zip(eval_ids, probs.tolist(), strict=True))

    return fit_predict


def run_age_only_baseline(
    labels_df: pd.DataFrame, folds: dict[str, list[str]], task: str = "idh"
) -> dict[str, float]:
    """Logistic regression on age alone - docs/METHODOLOGY.md §3's mandatory clinical baseline."""
    indexed = labels_df.set_index("patient_id")
    labels = indexed[task]
    features = indexed[["age"]]
    labelled_ids = set(labels.dropna().index) & set(features.dropna().index)
    return run_cv(_logistic_regression_fit_predict(labels, features), folds, labelled_ids)


def run_age_sex_baseline(
    labels_df: pd.DataFrame, folds: dict[str, list[str]], task: str = "idh"
) -> dict[str, float]:
    """Logistic regression on age + sex - the second clinical-only reference (METHODOLOGY.md §3)."""
    indexed = labels_df.set_index("patient_id")
    labels = indexed[task]
    features = pd.DataFrame(
        {"age": indexed["age"], "sex_f": (indexed["sex"] == "F").astype(float)},
        index=indexed.index,
    )
    features.loc[indexed["sex"].isna(), "sex_f"] = np.nan
    labelled_ids = set(labels.dropna().index) & set(features.dropna().index)
    return run_cv(_logistic_regression_fit_predict(labels, features), folds, labelled_ids)
