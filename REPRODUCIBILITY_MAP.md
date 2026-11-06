# What reproduces each final table and figure

This document maps every final table and figure to the script that produces it, and
records the seed scheme and the pinned package versions used to obtain the reported
results, so that the analysis can be reproduced end to end.

The table and figure numbers below follow the published numbering.

---

## Pipeline order

| Step | Script | Produces |
|---|---|---|
| 1 | `preprocessing/klosa_preprocessing.py` | `preprocessed_data.pkl` — the 22 baseline predictors and the binary six-year outcome, from the raw KLoSA wave files (the reproducibility bundle reads this file; point `KLOSA_ANALYTIC_DATA` at it) |
| 2 | `preprocessing/generate_codebook.py` | Supplementary Table S1 (predictor list, coding, missingness) |
| 3 | `training/klosa_all_models_tuning.py` | tuned hyperparameters → Supplementary Table S2 |
| 4 | `reproducibility/src/run_all.py` | Table 4, Supplementary Table S18, Supplementary Table S8, Figure 2 |
| 5 | `reproducibility/src/run_05_nested_cv.py` | nested cross-validation — the **primary** estimate |
| 6 | `reproducibility/src/run_06_competing_risk_death.py` | competing-risk analysis with death from the exit surveys |

Steps 1–3 are run once. The hyperparameters from step 3 are **fixed inputs** to steps
4–6; re-running the Optuna search changes the models rather than reproducing them.

---

## Main tables and figures

| Output | Script |
|---|---|
| Table 1, Table 2 — baseline characteristics | `figures_tables/generate_baseline_tables.py` |
| Table 3 — operating characteristics at alternative thresholds | `sensitivity_analyses/threshold_dca.py` |
| Table 4 — repeated cross-validation across the eight algorithms | `reproducibility/src/run_02_repeated_cv.py` |
| Figure 1 — study flow chart | `figures_tables/make_flowchart.py` |
| Figure 2 — ROC curves (both cohorts) | `reproducibility/src/run_04_figure2.py` |
| Figure 3, Figure 4 — SHAP summary plots | `figures_tables/generate_final_model_figures.py` |
| Figure 5 — calibration, before and after recalibration | `sensitivity_analyses/calibration.py` (metrics also in `run_03_calibration.py`) |
| Figure 6 — decision curve analysis with bootstrap bands | `sensitivity_analyses/dca_ci.py` |
| Figure 7 — learning curves | `sensitivity_analyses/learning_curve.py` |
| Figure 8 — precision–recall curves | `sensitivity_analyses/auprc_brier.py` |

Figure 2 (ROC curves) is computed directly from the fitted models by
`reproducibility/src/run_04_figure2.py`, which regenerates the curves from the model
predictions on the held-out test set rather than from any saved image.

## Supplementary tables

| Output | Script |
|---|---|
| S1 predictors · S2 hyperparameters | `preprocessing/generate_codebook.py` · `training/klosa_all_models_tuning.py` |
| S3 missing-data sensitivity | `sensitivity_analyses/missingness_sensitivity.py` |
| S4 time-to-event · S5 competing risks | `sensitivity_analyses/survival_analysis.py` · `run_06_competing_risk_death.py` |
| S6 stratification and nested CV | `reproducibility/src/run_05_nested_cv.py` |
| S7 AUPRC and Brier · S8 calibration | `sensitivity_analyses/auprc_brier.py` · `run_03_calibration.py` |
| S9, Figure S1 SHAP stability | `sensitivity_analyses/shap_stability.py` |
| S10 K-MMSE sensitivity | `sensitivity_analyses/mmse_sensitivity.py` |
| S11 attrition · S12 leak-free pipeline | `sensitivity_analyses/attrition_sensitivity.py` · `leakfree_pipeline.py` |
| S13 net benefit · S14 survey-weighted | `sensitivity_analyses/dca_ci.py` · `survey_weighted.py` |
| S15 one-hot encoding | `sensitivity_analyses/onehot_sensitivity.py` |
| S16 K-MMSE variable audit | `sensitivity_analyses/kmmse_availability_audit.py` |
| S17 provenance of estimates | compiled from the outputs above; no separate script |
| S18 single-split performance | `reproducibility/src/run_01_single_split.py` |

---

## Seeds

A single integer, **42**, is the only seed set by hand. Everything else derives from it,
so the whole analysis is determined by that one value:

| Where | How the seed is used |
|---|---|
| train/test split | `train_test_split(test_size=0.3, stratify=y, random_state=42)` |
| repeated CV | `RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)` — the 20 repeats are consecutive slices of this single stream |
| calibration folds | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| nested CV | outer `StratifiedKFold(5, shuffle=True, random_state=42)`; inner folds and the Optuna `TPESampler` for outer fold *k* use `42 + k` |
| every estimator | `random_state=42` (`random_seed` for CatBoost) |
| bootstrap | `np.random.RandomState(42)`, 500 resamples |

## Package versions

`requirements.txt` pins every version. This is not optional: the random forest and
XGBoost fits depend on library-internal random number streams that changed between
releases, and the final dementia model is a random forest.

In the reproducibility bundle (`reproducibility/src/`) every model is fitted
**single-threaded** (`n_jobs=1`, `thread_count=1`). XGBoost returns different results
according to the number of threads it is given, so single-threaded fitting is required
to reproduce the reported values on a machine with a different core count. The
exploratory training and figure scripts elsewhere in the repository may use `n_jobs=-1`.

`environment.yml` gives a conda equivalent; `requirements-full.txt` adds the packages
needed for the SHAP, survival and competing-risk analyses.

## Self-check

Every script in `reproducibility/src/` compares its own output with the published
values and prints a pass/fail line per model and outcome. On the pinned environment
all 32 checks in `run_all.py` pass, and `run_05_nested_cv.py` reproduces the primary
estimates (0.688 for MCI, 0.680 for dementia).
