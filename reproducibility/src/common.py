# -*- coding: utf-8 -*-
"""Shared configuration, data loading and model construction.

Everything downstream imports from here so that the analytic dataset, the random
seed and the model definitions have exactly one definition in the codebase.
"""
import os
import pickle
import platform
import sys

import numpy as np

# --------------------------------------------------------------------------
# One seed governs the stratified 70:30 split, the cross-validation fold
# construction, every model fit and every bootstrap resampling.
# --------------------------------------------------------------------------
SEED = 42
N_BOOT = 500
N_SPLITS = 5
N_REPEATS = 20
TEST_SIZE = 0.3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "output")
EXPECTED_DIR = os.path.join(ROOT, "expected")
os.makedirs(OUT_DIR, exist_ok=True)

# The analytic dataset. Override with the KLOSA_ANALYTIC_DATA environment
# variable if the file lives elsewhere. See README.md for how it is built and
# for the KLoSA data-access conditions -- the individual-level file cannot be
# redistributed with this code.
DATA_PATH = os.environ.get(
    "KLOSA_ANALYTIC_DATA",
    os.path.join(ROOT, "data", "analytic_dataset.pkl"),
)

# Hyperparameters selected by the Optuna search (Supplementary Table S2). They
# are fixed inputs to this re-analysis, not outputs of it: the search is not
# repeated, because re-running it would change the models rather than reproduce
# them.
HYPERPARAMS = {
    ("MCI", "SVC"): {"C": 0.01602005751543212, "gamma": 0.3130320104515628, "kernel": "sigmoid"},
    ("Dementia", "RF"): {"n_est": 110, "max_depth": 4, "min_split": 6, "min_leaf": 5},
}

FINAL_MODEL = {"MCI": "SVC", "Dementia": "RF"}

MODEL_ORDER = ["LASSO_LR", "SVC", "RF", "CatBoost", "XGBoost", "LightGBM", "GBM", "AdaBoost"]

DISPLAY_NAME = {
    "LASSO_LR": "L1 Penalized Logistic Regression",
    "SVC": "Support Vector Machine",
    "RF": "Random Forest",
    "CatBoost": "Categorical Boosting",
    "XGBoost": "Extreme Gradient Boosting",
    "LightGBM": "Light Gradient-Boosting Machine",
    "GBM": "Gradient Boosting Machine",
    "AdaBoost": "Adaptive Boosting",
}

SHORT_NAME = {
    "LASSO_LR": "LASSO LR", "SVC": "SVM", "RF": "RF", "CatBoost": "CatBoost",
    "XGBoost": "XGBoost", "LightGBM": "LightGBM", "GBM": "GBM", "AdaBoost": "AdaBoost",
}

COLOR = {
    "LASSO_LR": "#7f7f7f", "LightGBM": "#e377c2", "GBM": "#8c564b", "AdaBoost": "#ff7f0e",
    "CatBoost": "#9467bd", "XGBoost": "#2ca02c", "RF": "#1f77b4", "SVC": "#c0392b",
}

PINNED = {
    "python": "3.11.4", "numpy": "2.2.1", "scipy": "1.15.1", "pandas": "3.0.3",
    "scikit-learn": "1.9.0", "xgboost": "3.2.0", "lightgbm": "4.6.0",
    "catboost": "1.2.10", "statsmodels": "0.14.6",
}


def environment():
    """Return the versions actually in use, for the reproducibility record."""
    import catboost, lightgbm, pandas, scipy, sklearn, statsmodels, xgboost
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
        "statsmodels": statsmodels.__version__,
    }


def check_environment(strict=False):
    """Warn (or fail) when the running versions differ from the pinned ones."""
    env = environment()
    bad = {k: (v, env[k]) for k, v in PINNED.items() if env.get(k) != v}
    if bad:
        lines = ["Environment differs from the pinned versions:"]
        lines += ["  %-14s pinned %-10s running %s" % (k, p, r) for k, (p, r) in bad.items()]
        lines.append("  Random forest and XGBoost results are version-sensitive; "
                     "install requirements.txt to reproduce the published values.")
        msg = "\n".join(lines)
        if strict:
            raise SystemExit(msg)
        print(msg + "\n")
    else:
        print("Environment matches the pinned versions.\n")
    return env


def load_data():
    """Load the analytic dataset.

    Returns {outcome: (X, y)} with 22 baseline predictors per participant and a
    single binary outcome over the fixed six-year horizon.
    """
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            "Analytic dataset not found at:\n  %s\n"
            "Set KLOSA_ANALYTIC_DATA to its location, or rebuild it from the KLoSA\n"
            "release as described in README.md (the individual-level file cannot be\n"
            "redistributed with this code)." % DATA_PATH)
    with open(DATA_PATH, "rb") as fh:
        pre = pickle.load(fh)
    return {
        "MCI": (np.asarray(pre["X_mci"], float), np.asarray(pre["y_mci"]).astype(int)),
        "Dementia": (np.asarray(pre["X_dem"], float), np.asarray(pre["y_dem"]).astype(int)),
    }


def build(model_key, params, y_train):
    """Construct an untrained estimator with balanced class weighting.

    Inverse class weighting is applied through each algorithm's own mechanism:
    class_weight='balanced' where available, auto_class_weights for CatBoost,
    scale_pos_weight for XGBoost, and sample weights at fit time for the
    gradient boosting machine and AdaBoost (see fit_estimator).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  AdaBoostClassifier)
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    if model_key == "LASSO_LR":
        return Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=params["C"], penalty="l1", solver="saga",
                                                   max_iter=5000, class_weight="balanced",
                                                   random_state=SEED))])
    if model_key == "SVC":
        return Pipeline([("sc", StandardScaler()),
                         ("svc", SVC(C=params["C"], gamma=params["gamma"],
                                     kernel=params["kernel"], probability=True,
                                     class_weight="balanced", random_state=SEED))])
    if model_key == "RF":
        return RandomForestClassifier(n_estimators=params["n_est"], max_depth=params["max_depth"],
                                      min_samples_split=params["min_split"],
                                      min_samples_leaf=params["min_leaf"],
                                      class_weight="balanced", random_state=SEED, n_jobs=1)
    if model_key == "CatBoost":
        return CatBoostClassifier(iterations=params["iterations"], depth=params["depth"],
                                  learning_rate=params["lr"], l2_leaf_reg=params["l2"],
                                  auto_class_weights="Balanced", random_seed=SEED,
                                  verbose=0, thread_count=1,
                                  allow_writing_files=False)
    if model_key == "XGBoost":
        pos = int(np.sum(y_train))
        neg = int(len(y_train) - pos)
        return XGBClassifier(n_estimators=params["n_est"], max_depth=params["max_depth"],
                             learning_rate=params["lr"], subsample=params["subsample"],
                             colsample_bytree=params["colsample"],
                             scale_pos_weight=float(neg / pos), random_state=SEED,
                             eval_metric="auc", verbosity=0, n_jobs=1)
    if model_key == "LightGBM":
        return LGBMClassifier(n_estimators=params["n_est"], max_depth=params["max_depth"],
                              learning_rate=params["lr"], num_leaves=params["num_leaves"],
                              min_child_samples=params["min_child"], class_weight="balanced",
                              random_state=SEED, verbose=-1, n_jobs=1)
    if model_key == "GBM":
        return GradientBoostingClassifier(n_estimators=params["n_est"],
                                          max_depth=params["max_depth"],
                                          learning_rate=params["lr"],
                                          min_samples_split=params["min_split"],
                                          min_samples_leaf=params["min_leaf"],
                                          subsample=params["subsample"], random_state=SEED)
    if model_key == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier as _Ada
        kw = {"algorithm": "SAMME"} if "algorithm" in _Ada().get_params() else {}
        return AdaBoostClassifier(n_estimators=params["n_est"], learning_rate=params["lr"],
                                  random_state=SEED, **kw)
    raise ValueError("unknown model: %s" % model_key)


def fit_estimator(model_key, estimator, X, y):
    """Fit, supplying balanced sample weights to the two algorithms that lack a
    class_weight parameter."""
    from sklearn.utils.class_weight import compute_sample_weight
    if model_key in ("GBM", "AdaBoost"):
        estimator.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        estimator.fit(X, y)
    return estimator


def load_tuned_hyperparameters(path=None):
    """Read the tuned hyperparameters of all eight algorithms for both outcomes.

    Expects a CSV with columns: model, outcome, best_params (a Python dict
    literal). A copy is shipped in expected/tuned_hyperparameters.csv.
    """
    import ast
    import pandas as pd
    path = path or os.path.join(EXPECTED_DIR, "tuned_hyperparameters.csv")
    df = pd.read_csv(path)
    df["key"] = df["model"].str.replace("_tuned", "", regex=False)
    df["params"] = df["best_params"].apply(ast.literal_eval)
    return df


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def bootstrap(y, prob, metric, n_boot=N_BOOT, seed=SEED):
    """Bootstrap a metric over resamples of the held-out test set."""
    from sklearn.utils import resample
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], prob[idx]))
    vals = np.asarray(vals)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
