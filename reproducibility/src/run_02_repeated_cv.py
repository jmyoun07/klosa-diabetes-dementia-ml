# -*- coding: utf-8 -*-
"""Table 5 -- repeated stratified 5-fold cross-validation, 20 repeats (PRIMARY).

Each algorithm is evaluated at its tuned hyperparameters. Within each repeat the
out-of-fold predictions are pooled and a single AUC, AUPRC and Brier score are
computed, giving 20 estimates per algorithm. Algorithms are ranked within every
repeat, and the proportion of repeats in which each ranks first is recorded.

Writes output/table5_repeated_cv.csv and output/repeated_cv_raw.csv.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import common as C

warnings.filterwarnings("ignore")

N_JOBS = max(1, (os.cpu_count() or 4) - 2)


def one_repeat(outcome, key, params, X, y, repeat, splits):
    oof = np.zeros(len(y))
    for train_idx, test_idx in splits:
        est = C.fit_estimator(key, C.build(key, params, y[train_idx]),
                              X[train_idx], y[train_idx])
        oof[test_idx] = est.predict_proba(X[test_idx])[:, 1]
    return (outcome, key, repeat, roc_auc_score(y, oof),
            average_precision_score(y, oof), brier_score_loss(y, oof))


def main():
    C.check_environment()
    data = C.load_data()
    tuned = C.load_tuned_hyperparameters()

    jobs = []
    for outcome, (X, y) in data.items():
        rskf = RepeatedStratifiedKFold(n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS,
                                       random_state=C.SEED)
        all_splits = list(rskf.split(X, y))
        per_repeat = [all_splits[i * C.N_SPLITS:(i + 1) * C.N_SPLITS]
                      for i in range(C.N_REPEATS)]
        for _, r in tuned[tuned["outcome"] == outcome].iterrows():
            for repeat in range(C.N_REPEATS):
                jobs.append(delayed(one_repeat)(outcome, r["key"], r["params"],
                                                X, y, repeat, per_repeat[repeat]))

    print("running %d (algorithm x repeat) tasks on %d workers ..." % (len(jobs), N_JOBS))
    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, backend="loky")(jobs)
    print("done in %.0fs\n" % (time.time() - t0))

    raw = pd.DataFrame(out, columns=["outcome", "key", "repeat", "auc", "auprc", "brier"])
    raw.to_csv(os.path.join(C.OUT_DIR, "repeated_cv_raw.csv"), index=False)

    rows = []
    for outcome in data:
        sub = raw[raw["outcome"] == outcome]
        wide = sub.pivot(index="repeat", columns="key", values="auc")
        ranks = wide.rank(axis=1, ascending=False)
        for key in wide.columns:
            a = wide[key].values
            rows.append(dict(
                outcome=outcome, model=C.DISPLAY_NAME[key], key=key,
                is_final=(key == C.FINAL_MODEL[outcome]),
                mean_auc=a.mean(), sd_auc=a.std(ddof=1),
                pct2_5=np.percentile(a, 2.5), pct97_5=np.percentile(a, 97.5),
                mean_auprc=sub[sub["key"] == key]["auprc"].mean(),
                mean_brier=sub[sub["key"] == key]["brier"].mean(),
                mean_rank=ranks[key].mean(), rank_min=int(ranks[key].min()),
                rank_max=int(ranks[key].max()),
                pct_ranked_first=100.0 * (ranks[key] == 1).mean()))

    df = pd.DataFrame(rows).sort_values(["outcome", "mean_auc"], ascending=[True, False])
    out_path = os.path.join(C.OUT_DIR, "table5_repeated_cv.csv")
    df.to_csv(out_path, index=False)
    for outcome in data:
        print("--- %s ---" % outcome)
        for _, r in df[df["outcome"] == outcome].iterrows():
            print("  %-34s mean AUC %.4f (SD %.4f)  rank %.2f  first %3.0f%%%s"
                  % (r["model"], r["mean_auc"], r["sd_auc"], r["mean_rank"],
                     r["pct_ranked_first"], "   <- final model" if r["is_final"] else ""))
    print("\nwrote %s" % out_path)
    compare(df)


def compare(df):
    exp_path = os.path.join(C.EXPECTED_DIR, "expected_results.csv")
    if not os.path.exists(exp_path):
        return
    exp = pd.read_csv(exp_path)
    exp = exp[exp["analysis"] == "repeated_cv"]
    merged = df.merge(exp, on=["outcome", "key"], how="inner")
    bad = merged[(merged["mean_auc"] - merged["expected_mean_auc"]).abs() > 5e-4]
    print("\n--- reproduction check (Table 5, primary analysis) ---")
    if len(bad) == 0:
        print("  all %d model-by-outcome combinations match the published values" % len(merged))
    else:
        for _, r in bad.iterrows():
            print("  MISMATCH %-9s %-34s got %.4f expected %.4f"
                  % (r["outcome"], r["model"], r["mean_auc"], r["expected_mean_auc"]))


if __name__ == "__main__":
    main()
