# Reproducibility bundle

Code and pinned environment for reproducing every performance estimate reported in:

> **Development and Internal Validation of Exploratory Prediction Models for Self-Reported
> Physician-Diagnosed Mild Cognitive Impairment and Dementia among People with Treated
> Diabetes: A Prediction Modeling Study Using Longitudinal Cohort Data from the KLoSA**

---

## Quick start

**pip** (reference platform: CPython 3.11.4, Windows 11 x86-64)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux / macOS
python -m pip install -r requirements.txt
cd src && python run_all.py
```

**conda** (any platform)

```bash
conda env create -f environment.yml
conda activate klosa-mci-dementia
cd src && python run_all.py
```

Results are written to `output/`. Each script also checks its results against the
values reported in the manuscript and prints a pass/fail line for every model and
outcome. On the pinned environment all 32 checks pass.

Total runtime is roughly 5 minutes on 6 cores; the repeated cross-validation
(320 algorithm-by-repeat fits) dominates.

### Which requirements file

| File | Contents | When |
|---|---|---|
| `requirements.txt` | 10 packages + pinned transitive deps | running this bundle — **this is all you need** |
| `requirements-full.txt` | adds Optuna, SHAP, scikit-survival, lifelines, pyreadstat | reproducing the SHAP, survival and competing-risk analyses reported elsewhere in the paper |
| `environment.yml` | conda equivalent of `requirements.txt` | when pip wheels are unavailable for your platform |

The split is deliberate. The core driver imports only numpy, scipy, pandas,
scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, matplotlib and joblib.
SHAP — and therefore numba and llvmlite, the least portable part of the stack —
is not on the critical path, so a failure to install it cannot block the
reproduction of any reported performance estimate.

---

## What each script reproduces

| Script | Reproduces | Runtime |
|---|---|---|
| `run_01_single_split.py` | **Supplementary Table S18** — single stratified 70:30 split, all eight algorithms, both outcomes | ~1 min |
| `run_02_repeated_cv.py` | **Table 4** — repeated stratified 5-fold CV, 20 repeats (comparison among algorithms) | ~3 min |
| `run_03_calibration.py` | **Table S8 / Figure 5** — out-of-fold calibration of the two final models, with the decile table | ~30 s |
| `run_04_figure2.py` | **Figure 2** — ROC curves for both cohorts | ~1 min |
| `run_all.py` | all of the above, plus `output/reproducibility_record.json` | ~5 min |
| `run_05_nested_cv.py` | **fully nested cross-validation** — the primary estimate of expected pipeline performance | 30–60 min |

`run_05_nested_cv.py` is not part of `run_all.py` because it is far slower than
the rest: 5 outer folds × 8 algorithms × 50 Optuna trials × 5 inner folds =
10,000 model fits per outcome. Run it directly when you want to reproduce the
primary estimate:

```bash
python run_05_nested_cv.py              # 50 trials, as reported
N_TRIALS=5 python run_05_nested_cv.py   # quick smoke test — NOT the reported values
```

`src/common.py` holds the single definition of the seed, the analytic dataset,
and every model, so no estimator is defined twice.

---

## One dataset, one seed

A single random seed, **42**, governs the stratified 70:30 split, the
cross-validation fold construction, every model fit, and every bootstrap
resampling. There is one analytic dataset (`data/analytic_dataset.pkl`),
containing the 22 baseline predictors and the binary six-year outcome for the
MCI cohort (n = 1,246; 33 events) and the dementia cohort (n = 1,255; 50 events).

Fold construction is reproduced exactly as originally specified:

* single split — `train_test_split(test_size=0.3, stratify=y, random_state=42)`
* repeated CV — `RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)`
* calibration — `StratifiedKFold(5, shuffle=True, random_state=42)`

---

## Why the versions are pinned, and why everything is single-threaded

Twelve of the sixteen algorithm-by-outcome fits are bit-reproducible across
library versions. The exceptions are the **random forest** and **extreme gradient
boosting**, whose fitting draws on library-internal random number streams that
have changed between releases of scikit-learn and XGBoost. Because the final
dementia model is a random forest, reproducing its published values requires the
pinned versions. `common.check_environment()` warns when the running versions
differ.

Separately, the XGBoost implementation returns **different results according to
the number of threads it is allowed to use**, so pinning versions alone would not
have made it reproducible on a machine with a different core count. Every model
is therefore fitted single-threaded (`n_jobs=1`, `thread_count=1`). Only the
outer parallelism of the repeated cross-validation uses multiple cores, and it
distributes whole independent fits, so it does not affect any result. Random
forest, CatBoost and LightGBM were verified to be thread-insensitive; the
constraint is retained for all of them for uniformity.

Hyperparameter optimisation is **not** repeated. The tuned hyperparameters in
`expected/tuned_hyperparameters.csv` (Supplementary Table S2) are fixed inputs to
this re-analysis rather than outputs of it; re-running the Optuna search would
change the models rather than reproduce them.

---

## Reading the numbers

Estimates that differ do so because they come from different validation
procedures, not because they describe different models. For the final dementia
random forest:

| Value | What it is |
|---|---|
| **0.680** | **mean outer-fold AUC of the fully nested cross-validation — the primary estimate of expected pipeline performance** |
| 0.719 | mean AUC of repeated cross-validation, conditional on the tuned hyperparameters (comparison among algorithms) |
| 0.741 | point AUC on the single 70:30 split |
| 0.743 | bootstrap median of that same quantity (500 resamples) |
| 0.737 | AUC of the out-of-fold predictions used for calibration |
| 0.770 | concordance index of a **different model** — the random survival forest |

Only the nested cross-validation confines both the hyperparameter search and the
choice among algorithms to data the estimate never sees, which is why it is the
primary figure and why it is the lowest. The repeated cross-validation exceeds it
by about 0.02 for MCI and 0.04 for dementia — that gap is the tuning-related
optimism, not a difference between models.

Supplementary Table S17 of the paper lists all of these with the procedure that
produced each.

---

## Data availability

`data/analytic_dataset.pkl` is derived from the Korean Longitudinal Study of
Aging (KLoSA), waves 7–10. KLoSA microdata are distributed by the Korea
Employment Information Service and are subject to its data-use agreement.

**The individual-level file in `data/` must not be redistributed.** It is
excluded by `.gitignore`; remove it before publishing this bundle, and direct
users to obtain the KLoSA release themselves from
<https://survey.keis.or.kr/klosa/klosadownload/List.jsp>, then point
`KLOSA_ANALYTIC_DATA` at their own build:

```bash
set KLOSA_ANALYTIC_DATA=C:\path\to\analytic_dataset.pkl     # Windows
export KLOSA_ANALYTIC_DATA=/path/to/analytic_dataset.pkl    # Linux/macOS
```

Note that the cognitive-ability and MMSE variables formerly supplied with waves
1–9 have been withdrawn by the data provider and will not be re-released, so the
K-MMSE is available only at wave 10 (Supplementary Table S16).

---

## Layout

```
reproducibility/
  README.md
  requirements.txt           core: everything run_all.py needs
  requirements-full.txt      adds Optuna, SHAP, scikit-survival, lifelines
  environment.yml            conda equivalent of requirements.txt
  .gitignore
  src/
    common.py                  seed, data loading, model definitions
    run_01_single_split.py     Table 3
    run_02_repeated_cv.py      Table 5   (primary)
    run_03_calibration.py      Table S8 / Figure 5
    run_04_figure2.py          Figure 2
    run_all.py
  expected/
    tuned_hyperparameters.csv  Supplementary Table S2
    expected_results.csv       published values, for the automatic check
  data/
    analytic_dataset.pkl       not redistributable — see above
  output/                      generated
```

---

## Code availability

The wording below is identical in the manuscript (Availability of data and materials), in the TRIPOD+AI checklist (items 18e and 18f) and here:

> The analytic code, the preprocessing scripts, the model specifications (the final hyperparameters, also given in Supplementary Table S2) and a reproducibility package with pinned software versions are available at https://github.com/iynsu457-png/klosa-diabetes-dementia-ml.

Protocol, registration and patient/public involvement, as stated in the manuscript: no protocol was published and none was prepared in advance of the analysis; the study was not registered; patients or members of the public were not involved in the design, conduct, reporting or dissemination plans of this study.

