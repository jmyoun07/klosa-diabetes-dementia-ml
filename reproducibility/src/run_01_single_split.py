# -*- coding: utf-8 -*-
"""Table 3 -- single stratified 70:30 split (secondary, descriptive).

Fits all eight algorithms for both outcomes at their tuned hyperparameters and
reports, on the held-out test set: the point AUC, the bootstrap median AUC and
AUPRC with 95% bootstrap intervals, the Brier and scaled Brier scores, and the
operating point defined by the Youden index with Wilson intervals.

Writes output/table3_single_split.csv and compares against expected/.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)

import common as C

warnings.filterwarnings("ignore")


def main():
    C.check_environment()
    data = C.load_data()
    tuned = C.load_tuned_hyperparameters()

    rows = []
    for outcome, (X, y) in data.items():
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
        prevalence = float(yte.mean())
        print("\n%s: train %d (%d events) / test %d (%d events), prevalence %.4f"
              % (outcome, len(ytr), ytr.sum(), len(yte), yte.sum(), prevalence))

        for _, r in tuned[tuned["outcome"] == outcome].iterrows():
            key = r["key"]
            est = C.fit_estimator(key, C.build(key, r["params"], ytr), Xtr, ytr)
            prob = est.predict_proba(Xte)[:, 1]

            auc = roc_auc_score(yte, prob)
            brier = brier_score_loss(yte, prob)
            auc_m, auc_lo, auc_hi = C.bootstrap(yte, prob, roc_auc_score)
            ap_m, ap_lo, ap_hi = C.bootstrap(yte, prob, average_precision_score)

            fpr, tpr, thr = roc_curve(yte, prob)
            cut = thr[int(np.argmax(tpr - fpr))]
            pred = (prob >= cut).astype(int)
            tp = int(((pred == 1) & (yte == 1)).sum())
            fp = int(((pred == 1) & (yte == 0)).sum())
            fn = int(((pred == 0) & (yte == 1)).sum())
            tn = int(((pred == 0) & (yte == 0)).sum())

            rows.append(dict(
                outcome=outcome, model=C.DISPLAY_NAME[key], key=key,
                is_final=(key == C.FINAL_MODEL[outcome]),
                auc_point=auc, auc_boot_median=auc_m, auc_lo=auc_lo, auc_hi=auc_hi,
                auprc_boot_median=ap_m, auprc_lo=ap_lo, auprc_hi=ap_hi,
                auprc_baseline=prevalence,
                brier=brier, scaled_brier=1 - brier / (prevalence * (1 - prevalence)),
                threshold=float(cut),
                sensitivity=tp / (tp + fn), sens_lo=C.wilson_ci(tp, tp + fn)[0],
                sens_hi=C.wilson_ci(tp, tp + fn)[1],
                specificity=tn / (tn + fp), spec_lo=C.wilson_ci(tn, tn + fp)[0],
                spec_hi=C.wilson_ci(tn, tn + fp)[1],
                ppv=tp / (tp + fp) if (tp + fp) else np.nan,
                ppv_lo=C.wilson_ci(tp, tp + fp)[0], ppv_hi=C.wilson_ci(tp, tp + fp)[1],
                npv=tn / (tn + fn) if (tn + fn) else np.nan,
                npv_lo=C.wilson_ci(tn, tn + fn)[0], npv_hi=C.wilson_ci(tn, tn + fn)[1]))
            print("  %-34s point AUC %.4f | boot %.3f (%.3f-%.3f) | Brier %.4f%s"
                  % (C.DISPLAY_NAME[key], auc, auc_m, auc_lo, auc_hi, brier,
                     "   <- final model" if rows[-1]["is_final"] else ""))

    df = pd.DataFrame(rows).sort_values(
        ["outcome", "auc_boot_median"], ascending=[True, False])
    out = os.path.join(C.OUT_DIR, "table3_single_split.csv")
    df.to_csv(out, index=False)
    print("\nwrote %s" % out)
    compare(df)


def compare(df):
    """Check the reproduction against the values reported in the manuscript."""
    exp_path = os.path.join(C.EXPECTED_DIR, "expected_results.csv")
    if not os.path.exists(exp_path):
        return
    exp = pd.read_csv(exp_path)
    exp = exp[exp["analysis"] == "single_split"]
    merged = df.merge(exp, on=["outcome", "key"], how="inner", suffixes=("", "_exp"))
    bad = merged[(merged["auc_boot_median"] - merged["expected_auc_boot_median"]).abs() > 5e-4]
    print("\n--- reproduction check (Table 3) ---")
    if len(bad) == 0:
        print("  all %d model-by-outcome fits match the published values" % len(merged))
    else:
        for _, r in bad.iterrows():
            print("  MISMATCH %-9s %-34s got %.4f expected %.4f"
                  % (r["outcome"], r["model"], r["auc_boot_median"],
                     r["expected_auc_boot_median"]))


if __name__ == "__main__":
    main()
