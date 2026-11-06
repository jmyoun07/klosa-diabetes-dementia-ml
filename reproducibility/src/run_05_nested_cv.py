# -*- coding: utf-8 -*-
"""Fully nested cross-validation -- the PRIMARY estimate of the expected
performance of the model-development pipeline.

Both the Optuna hyperparameter search and the choice among the eight algorithms
are confined to an inner stratified 5-fold cross-validation of each outer
training fold; the selected model is then evaluated once on the corresponding
outer fold. This is the only procedure in which no part of the estimate has seen
the data used to make the selection, so it is the one reported as primary.

The number of Optuna trials is reduced from the 150 used for the published
hyperparameter search to 50 per algorithm, to make the nested loop feasible
(5 outer folds x 8 algorithms x 50 trials x 5 inner folds = 10,000 fits per
outcome). Search spaces, sampler, class weighting and objective are otherwise
identical to the original search.

This script is SLOW -- roughly 30-60 minutes on 6 cores -- and is therefore not
part of run_all.py. Run it directly:

    python run_05_nested_cv.py            # 50 trials, as reported
    N_TRIALS=5 python run_05_nested_cv.py # quick smoke test, not the reported values

Writes output/nested_cv_results.csv and output/nested_cv_folds.csv.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import common as C

warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.environ.get("N_TRIALS", 50))
OUTER_SPLITS = 5
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
ALGOS = ["LASSO_LR", "SVC", "RF", "GBM", "AdaBoost", "LightGBM", "XGBoost", "CatBoost"]


def sample_params(model_type, trial):
    """The search space of the original hyperparameter optimisation."""
    if model_type == "LASSO_LR":
        return {"C": trial.suggest_float("C", 0.001, 100, log=True)}
    if model_type == "SVC":
        return {"C": trial.suggest_float("C", 0.01, 100, log=True),
                "gamma": trial.suggest_float("gamma", 0.0001, 1, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])}
    if model_type == "RF":
        return {"n_est": trial.suggest_int("n_est", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_split": trial.suggest_int("min_split", 2, 20),
                "min_leaf": trial.suggest_int("min_leaf", 1, 10)}
    if model_type == "GBM":
        return {"n_est": trial.suggest_int("n_est", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "lr": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "min_split": trial.suggest_int("min_split", 2, 20),
                "min_leaf": trial.suggest_int("min_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0)}
    if model_type == "AdaBoost":
        return {"n_est": trial.suggest_int("n_est", 50, 500),
                "lr": trial.suggest_float("lr", 0.01, 2.0, log=True)}
    if model_type == "LightGBM":
        return {"n_est": trial.suggest_int("n_est", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "lr": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child": trial.suggest_int("min_child", 5, 50)}
    if model_type == "XGBoost":
        return {"n_est": trial.suggest_int("n_est", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "lr": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample": trial.suggest_float("colsample", 0.5, 1.0)}
    if model_type == "CatBoost":
        return {"iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 8),
                "lr": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "l2": trial.suggest_float("l2", 1, 10)}
    raise ValueError(model_type)


def inner_cv_auc(model_type, params, X, y, seed):
    """Mean AUC over an inner stratified 5-fold cross-validation."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        if len(np.unique(y[va])) < 2:
            continue
        try:
            est = clone(C.build(model_type, params, y[tr]))
            C.fit_estimator(model_type, est, X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], est.predict_proba(X[va])[:, 1]))
        except Exception:
            aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def tune(model_type, X, y, n_trials, seed):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: inner_cv_auc(model_type, sample_params(model_type, t), X, y, seed),
                   n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _task(outcome, fold, train_idx, algo, X, y, n_trials):
    t0 = time.time()
    params, value = tune(algo, X[train_idx], y[train_idx], n_trials, seed=C.SEED + fold)
    print("    [%s fold %d] %-9s inner CV AUC %.4f (%.0fs)"
          % (outcome, fold, algo, value, time.time() - t0), flush=True)
    return (fold, algo, params, value)


def main():
    C.check_environment()
    data = C.load_data()
    print("nested cross-validation: %d outer folds x %d algorithms x %d Optuna trials x 5 inner "
          "folds\nrunning on %d workers\n" % (OUTER_SPLITS, len(ALGOS), N_TRIALS, N_JOBS))

    summary, fold_rows = [], []
    for outcome, (X, y) in data.items():
        print("=" * 78)
        print("%s (n=%d, events=%d)" % (outcome, len(y), y.sum()))
        print("=" * 78, flush=True)

        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=C.SEED)
        splits = list(outer.split(X, y))

        jobs = [delayed(_task)(outcome, k, tr, algo, X, y, N_TRIALS)
                for k, (tr, _) in enumerate(splits, 1) for algo in ALGOS]
        results = Parallel(n_jobs=N_JOBS, backend="loky")(jobs)

        by_fold = {}
        for fold, algo, params, value in results:
            by_fold.setdefault(fold, {})[algo] = (params, value)

        outer_aucs, inner_best = [], []
        for k, (tr, te) in enumerate(splits, 1):
            algo = max(by_fold[k], key=lambda a: by_fold[k][a][1])
            params, inner_auc = by_fold[k][algo]
            est = C.fit_estimator(algo, C.build(algo, params, y[tr]), X[tr], y[tr])
            outer_auc = roc_auc_score(y[te], est.predict_proba(X[te])[:, 1])
            outer_aucs.append(outer_auc)
            inner_best.append(inner_auc)
            row = dict(outcome=outcome, fold=k, n_train=len(tr), events_train=int(y[tr].sum()),
                       n_test=len(te), events_test=int(y[te].sum()), selected_algorithm=algo,
                       inner_cv_auc=inner_auc, outer_auc=outer_auc)
            row.update({"inner_%s" % a: by_fold[k][a][1] for a in ALGOS})
            fold_rows.append(row)
            print("  fold %d: selected %-9s inner %.4f -> outer %.4f"
                  % (k, algo, inner_auc, outer_auc), flush=True)

        chosen = pd.Series([r["selected_algorithm"] for r in fold_rows
                            if r["outcome"] == outcome]).value_counts()
        summary.append(dict(
            outcome=outcome, n_outer_folds=OUTER_SPLITS, n_trials=N_TRIALS,
            nested_auc_mean=float(np.mean(outer_aucs)),
            nested_auc_sd=float(np.std(outer_aucs, ddof=1)),
            nested_auc_min=float(np.min(outer_aucs)), nested_auc_max=float(np.max(outer_aucs)),
            inner_cv_auc_mean=float(np.mean(inner_best)),
            selection_tuning_optimism=float(np.mean(inner_best) - np.mean(outer_aucs)),
            selected_algorithms="; ".join("%s x%d" % (a, n) for a, n in chosen.items())))
        s = summary[-1]
        print("\n  NESTED (primary) mean outer AUC %.4f (SD %.4f; range %.4f to %.4f)"
              % (s["nested_auc_mean"], s["nested_auc_sd"], s["nested_auc_min"], s["nested_auc_max"]))
        print("  mean inner CV AUC (selection quantity) %.4f -> selection/tuning optimism %.4f\n"
              % (s["inner_cv_auc_mean"], s["selection_tuning_optimism"]), flush=True)

    pd.DataFrame(summary).to_csv(os.path.join(C.OUT_DIR, "nested_cv_results.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(C.OUT_DIR, "nested_cv_folds.csv"), index=False)
    print("wrote output/nested_cv_results.csv and output/nested_cv_folds.csv")

    if N_TRIALS == 50:
        print("\n--- reproduction check (nested cross-validation, primary estimate) ---")
        published = {"MCI": 0.6879, "Dementia": 0.6803}
        for s in summary:
            delta = abs(s["nested_auc_mean"] - published[s["outcome"]])
            print("  %-9s got %.4f, published %.4f  %s"
                  % (s["outcome"], s["nested_auc_mean"], published[s["outcome"]],
                     "match" if delta < 5e-4 else "DIFFERS by %.4f" % delta))


if __name__ == "__main__":
    main()
