# -*- coding: utf-8 -*-
"""Supplementary Table S8 and Figure 5 -- out-of-fold calibration of the two
final models.

Calibration is assessed on 5-fold cross-validated out-of-fold predictions across
the full sample, because the small number of events makes test-set calibration
unstable. Reports the calibration slope, the calibration-in-the-large, the mean
predicted probability against the observed prevalence, the expected calibration
error, the Brier and scaled Brier scores, and the decile table that underlies the
statement about over- or under-estimation across the prediction range.

Writes output/calibration_final_models.csv and output/calibration_deciles_*.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

import common as C

warnings.filterwarnings("ignore")


def out_of_fold(key, params, X, y):
    skf = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    oof = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        est = C.fit_estimator(key, C.build(key, params, y[train_idx]),
                              X[train_idx], y[train_idx])
        oof[test_idx] = est.predict_proba(X[test_idx])[:, 1]
    return oof


def main():
    C.check_environment()
    data = C.load_data()
    tuned = C.load_tuned_hyperparameters()

    rows = []
    for outcome, (X, y) in data.items():
        key = C.FINAL_MODEL[outcome]
        params = tuned[(tuned["outcome"] == outcome) & (tuned["key"] == key)].iloc[0]["params"]
        oof = out_of_fold(key, params, X, y)

        eps = 1e-6
        clipped = np.clip(oof, eps, 1 - eps)
        linear_pred = np.log(clipped / (1 - clipped))
        slope_fit = sm.Logit(y, sm.add_constant(linear_pred)).fit(disp=0)
        citl_fit = sm.Logit(y, np.ones((len(y), 1)), offset=linear_pred).fit(disp=0)

        decile = pd.qcut(oof, 10, labels=False, duplicates="drop")
        dec = (pd.DataFrame({"decile": decile, "observed": y, "predicted": oof})
               .groupby("decile")
               .agg(n=("observed", "size"), observed=("observed", "mean"),
                    predicted=("predicted", "mean")))
        dec["gap"] = dec["predicted"] - dec["observed"]
        dec.to_csv(os.path.join(C.OUT_DIR, "calibration_deciles_%s.csv" % outcome))

        brier = brier_score_loss(y, oof)
        rows.append(dict(
            outcome=outcome, model=C.DISPLAY_NAME[key],
            auc_out_of_fold=roc_auc_score(y, oof),
            slope=slope_fit.params[1],
            slope_lo=slope_fit.conf_int()[1][0], slope_hi=slope_fit.conf_int()[1][1],
            citl=citl_fit.params[0],
            citl_lo=citl_fit.conf_int()[0][0], citl_hi=citl_fit.conf_int()[0][1],
            mean_predicted=oof.mean(), observed=y.mean(),
            ece=float(np.average((dec["observed"] - dec["predicted"]).abs(), weights=dec["n"])),
            brier=brier, scaled_brier=1 - brier / (y.mean() * (1 - y.mean())),
            lowest_decile_predicted=dec["predicted"].iloc[0],
            lowest_decile_observed=dec["observed"].iloc[0],
            highest_decile_predicted=dec["predicted"].iloc[-1],
            highest_decile_observed=dec["observed"].iloc[-1],
            deciles_over_predicting=int((dec["gap"] > 0).sum())))

        r = rows[-1]
        print("\n--- %s (%s) ---" % (outcome, C.DISPLAY_NAME[key]))
        print("  calibration slope        %.3f (%.2f to %.2f)"
              % (r["slope"], r["slope_lo"], r["slope_hi"]))
        print("  calibration-in-the-large %+.3f (%+.2f to %+.2f)"
              % (r["citl"], r["citl_lo"], r["citl_hi"]))
        print("  mean predicted %.4f vs observed %.4f | ECE %.4f | Brier %.4f"
              % (r["mean_predicted"], r["observed"], r["ece"], r["brier"]))
        print("  deciles: predicted %.3f-%.3f vs observed %.3f-%.3f; %d of %d deciles over-predict"
              % (dec["predicted"].min(), dec["predicted"].max(),
                 dec["observed"].min(), dec["observed"].max(),
                 r["deciles_over_predicting"], len(dec)))

    out = os.path.join(C.OUT_DIR, "calibration_final_models.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
