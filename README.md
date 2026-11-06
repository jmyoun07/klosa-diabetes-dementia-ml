# Prediction Models for MCI and Dementia among People with Treated Diabetes (KLoSA)

Analysis code for the study *"Development and Internal Validation of Exploratory
Prediction Models for Self-Reported Physician-Diagnosed Mild Cognitive Impairment
and Dementia among People with Treated Diabetes: A Prediction Modeling Study Using
Longitudinal Cohort Data from the KLoSA."* The models are trained and internally
validated on the Korean Longitudinal Study of Aging (KLoSA).

## Overview

Starting from the seventh KLoSA wave (2018), adults aged 45 years or older
receiving treatment for diabetes are identified and followed for incident,
self-reported physician-diagnosed mild cognitive impairment (MCI) and dementia.
Eight machine-learning algorithms are compared — L1-penalized (LASSO) logistic
regression, LightGBM, XGBoost, CatBoost, random forest, gradient boosting,
AdaBoost, and a support vector machine — each tuned with Optuna (TPE) and
evaluated by stratified cross-validation and bootstrap resampling on a held-out
test set. Variable importance is summarized with SHAP.

- **MCI model:** tuned support vector machine
- **Dementia model:** tuned random forest

## Which script reproduces each table and figure

See **[REPRODUCIBILITY_MAP.md](REPRODUCIBILITY_MAP.md)** for a one-to-one mapping
from every final table and figure (Tables 1–4, Figures 1–8, Supplementary
Tables S1–S18 and Figure S1) to the script that produces it, together with the
seed scheme and pinned package versions.

## Repository structure

```
preprocessing/          Build the analytic cohort and the variable codebook from raw KLoSA waves
  klosa_preprocessing.py       Cohort construction, predictor derivation, imputation
  generate_codebook.py         Predictor codebook / missingness summary (Supplementary S1)

training/               Model training, imbalance handling, and hyperparameter tuning
  klosa_ml_training.py         Class-imbalance strategy comparison + baseline training
  klosa_all_models_tuning.py   Optuna tuning of all eight models (Supplementary S2)
  klosa_baseline_tuning.py     Tree-model baseline tuning
  klosa_ml_smote.py            SMOTE variant (imbalance sensitivity analysis)

figures_tables/         Manuscript tables and figures
  generate_baseline_tables.py      Baseline characteristics (Tables 1–2)
  make_flowchart.py                Study flow chart (Figure 1)
  generate_performance_tables.py   Model-performance comparison + figures
  generate_final_model_figures.py  Final-model SHAP figures (Figures 3–4)

sensitivity_analyses/   Robustness and validation analyses
  threshold_dca.py, calibration.py, dca_ci.py, learning_curve.py, auprc_brier.py,
  internal_validation.py, nested_cv.py, repeated_cv.py, leakfree_pipeline.py,
  shap_stability.py, shap_stability_plot.py, missingness_sensitivity.py,
  onehot_sensitivity.py, mmse_sensitivity.py, kmmse_availability_audit.py,
  attrition_sensitivity.py, followup_attrition.py, competing_risk.py,
  survival_analysis.py, survey_weighted.py

reproducibility/        Self-contained, pinned driver that regenerates the headline
                        estimates (Table 4, Supplementary S18/S8, Figure 2, nested CV,
                        competing risk) and checks each result against the published
                        values. See reproducibility/README.md.
```

## Data availability

The KLoSA microdata are **not** included in this repository. They are
access-controlled and can be obtained, free of charge, from the Korea Employment
Information Service through the KLoSA portal:
<https://survey.keis.or.kr/klosa/klosa01.jsp>.

After obtaining the data, place the SPSS (`.sav`) wave files in a local `data/`
directory (git-ignored). By default the scripts read from `./data`; override this
with the `KLOSA_DATA_DIR` environment variable. The file names in
`preprocessing/klosa_preprocessing.py` (the `FILENAMES` map) and the `w*_*.sav`
globs may need to be updated to match your download.

Two derived, individual-level files are also required by specific scripts and are
likewise not redistributable — you build them from the KLoSA release:

- The analytic dataset read by the reproducibility bundle (22 baseline predictors
  and the binary six-year outcome). `preprocessing/klosa_preprocessing.py` writes it
  as `preprocessed_data.pkl`; point `KLOSA_ANALYTIC_DATA` at that file (the bundle
  otherwise looks for `reproducibility/data/analytic_dataset.pkl`).
- `reproducibility/src/cohort_death_linkage.csv` — `pid`, `death_year`,
  `death_month`, `seen_w10`, derived from the KLoSA exit survey. Required only by
  `run_06_competing_risk_death.py`.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11.4 (reference platform)
pip install -r requirements.txt
```

Package versions are pinned. The scikit-learn and XGBoost pins in particular are
required to reproduce the published values, because the random forest and XGBoost
fits depend on library-internal random-number streams that changed between
releases. In the reproducibility bundle all models are fitted single-threaded
(`n_jobs=1`, `thread_count=1`); the exploratory training and figure scripts may use
`n_jobs=-1`.

## Seeds

A single integer, **42**, is the only hand-set seed; every split, fold,
estimator, and bootstrap derives from it, so the whole analysis is determined by
that one value. The exact seed scheme is documented in
[REPRODUCIBILITY_MAP.md](REPRODUCIBILITY_MAP.md).

## Reproducing the analysis

```bash
# 1. Preprocess: build the cohort and predictors -> preprocessed_data.pkl
python preprocessing/klosa_preprocessing.py

# 2. Train and tune the eight models
python training/klosa_all_models_tuning.py

# 3. Regenerate the headline tables/figures with the pinned, self-checking driver
cd reproducibility && pip install -r requirements.txt && cd src && python run_all.py

# 4. Tables, figures, and sensitivity analyses (see REPRODUCIBILITY_MAP.md)
python figures_tables/generate_baseline_tables.py
python figures_tables/make_flowchart.py
# ... etc.
```

Every script in `reproducibility/src/` compares its output with the published
values and prints a pass/fail line per model and outcome; on the pinned
environment all 32 checks pass.

## Citation

If you use this code, please cite the paper (volume, issue, and DOI to be added
on publication):

> Lee K-S, Park BJ, Youn B-Y. Development and Internal Validation of Exploratory
> Prediction Models for Self-Reported Physician-Diagnosed Mild Cognitive
> Impairment and Dementia among People with Treated Diabetes: A Prediction
> Modeling Study Using Longitudinal Cohort Data from the KLoSA. *BMC Public
> Health*. 2026.

## License

Released under the MIT License — see [LICENSE](LICENSE). Use of the KLoSA data is
governed separately by the terms of the KLoSA data provider.
