# -*- coding: utf-8 -*-
"""
Comparison of the primary preprocessing pipeline with a leak-free variant.

In the primary pipeline the median/mode imputation is computed on the full feature matrix
before the train-test split (klosa_preprocessing.py, handle_missing()), so test-set
information can enter imputation; scaling (StandardScaler) and SVC Platt calibration are
already fitted on training data only. Three of the 22 predictors have any missingness
(income 0.5%, BMI category 2.3%, grip-strength change ~33%). This repeats the analysis with
imputation fitted inside the training set and compares the two pipelines.

Leak-free pipeline (all steps fitted on training data only):
    ColumnTransformer( median-impute for continuous/ordinal, most-frequent for binary )
    -> StandardScaler (MCI/SVM only)  -> final model (with its published hyperparameters)

Two comparisons are produced for each final model:
  (A) the single stratified 70:30 split (seed 42), leaky vs leak-free
  (B) repeated stratified 5-fold cross-validation, 20 repeats, leaky vs leak-free

Outputs -> leakfree_pipeline_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.utils import resample

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, B = 42, 500

# continuous + ordinal predictors get the median; binary predictors get the mode
CONT_ORDINAL = ['age', 'marital', 'edu', 'living', 'income', 'adl_score', 'iadl_score',
                'bmi_cat', 'cesd_total', 'hearing', 'subjective_health']
BINARY = ['sex', 'rural', 'grip_decreased', 'drinking', 'smoking', 'exercise', 'fall_exp',
          'work_participation', 'stroke', 'heart_disease', 'hypertension']

with open(os.path.join(HERE, 'preimp_data.pkl'), 'rb') as f:
    raw = pickle.load(f)
with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    imp = pickle.load(f)

FEAT = list(raw['X_mci_raw'].columns)
CONT_IDX = [FEAT.index(c) for c in CONT_ORDINAL]
BIN_IDX = [FEAT.index(c) for c in BINARY]

DATA = {
    'MCI': dict(Xraw=raw['X_mci_raw'].values.astype(float),
                Ximp=np.asarray(imp['X_mci'], float),
                y=np.asarray(imp['y_mci']).astype(int), kind='svm'),
    'Dementia': dict(Xraw=raw['X_dem_raw'].values.astype(float),
                     Ximp=np.asarray(imp['X_dem'], float),
                     y=np.asarray(imp['y_dem']).astype(int), kind='rf'),
}


class OrderPreservingImputer(BaseEstimator, TransformerMixin):
    """Median for continuous/ordinal columns, mode for binary columns, fitted on the
    training data only and applied in place so the original column order is preserved.
    This isolates the effect of the imputation LOCATION from any column reordering."""

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.fill_ = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if j in BIN_IDX:
                vals, counts = np.unique(col, return_counts=True)
                self.fill_[j] = vals[np.argmax(counts)]
            else:
                self.fill_[j] = np.median(col)
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.fill_[j]
        return X


def imputer():
    return OrderPreservingImputer()


def final_model(kind, leakfree):
    if kind == 'svm':
        steps = [('sc', StandardScaler()),
                 ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433, kernel='sigmoid',
                             probability=True, class_weight='balanced', random_state=SEED))]
    else:
        steps = [('rf', RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                               min_samples_leaf=5, class_weight='balanced',
                                               random_state=SEED, n_jobs=-1))]
    if leakfree:
        steps = [('imp', imputer())] + steps
    return Pipeline(steps)


def boot(y, p, fn):
    rng = np.random.RandomState(SEED); v = []
    for _ in range(B):
        i = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], p[i]))
    v = np.array(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rows = []
for outcome, d in DATA.items():
    y, kind = d['y'], d['kind']
    print(f'\n{"="*94}\n{outcome} ({kind.upper()}): n={len(y)}, events={y.sum()}\n{"="*94}')

    # how different are the imputation constants computed on full data vs a training fold?
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(d['Xraw'], y, test_size=0.3, stratify=y,
                                                  random_state=SEED)
    for col in ['income', 'bmi_cat', 'grip_decreased']:
        j = FEAT.index(col)
        full = np.nanmedian(d['Xraw'][:, j]) if col != 'grip_decreased' else \
            pd.Series(d['Xraw'][:, j]).mode()[0]
        trn = np.nanmedian(Xtr_raw[:, j]) if col != 'grip_decreased' else \
            pd.Series(Xtr_raw[:, j]).mode()[0]
        print(f'  imputation constant for {col:16s}: full data {full:.3f}  vs  training fold {trn:.3f}')

    # (A) single 70:30 split
    print('\n  (A) Single stratified 70:30 split (seed 42):')
    Xtr_imp, Xte_imp, _, _ = train_test_split(d['Ximp'], y, test_size=0.3, stratify=y,
                                              random_state=SEED)
    for tag, leakfree, Xtr, Xte in [
            ('leaky (imputed on full data, current manuscript)', False, Xtr_imp, Xte_imp),
            ('leak-free (imputation fitted in training only)', True, Xtr_raw, Xte_raw)]:
        m = final_model(kind, leakfree).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        auc, alo, ahi = boot(yte, p, roc_auc_score)
        ap, plo, phi = boot(yte, p, average_precision_score)
        bs = brier_score_loss(yte, p)
        print(f'    {tag:52s} AUC {auc:.3f} ({alo:.3f}-{ahi:.3f}) | AUPRC {ap:.3f} | Brier {bs:.3f}')
        rows.append(dict(outcome=outcome, analysis='single split', pipeline=tag.split(' (')[0],
                         auc=auc, auc_lo=alo, auc_hi=ahi, auprc=ap, brier=bs))

    # (B) repeated stratified 5-fold CV, 20 repeats
    print('\n  (B) Repeated stratified 5-fold cross-validation, 20 repeats (pooled out-of-fold AUC):')
    for tag, leakfree, X in [('leaky (imputed on full data)', False, d['Ximp']),
                             ('leak-free (imputation in training folds)', True, d['Xraw'])]:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
        splits = list(rskf.split(X, y))
        aucs = []
        for r in range(20):
            oof = np.full(len(y), np.nan)
            for tr, te in splits[r * 5:(r + 1) * 5]:
                mm = final_model(kind, leakfree).fit(X[tr], y[tr])
                oof[te] = mm.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y, oof))
        aucs = np.array(aucs)
        print(f'    {tag:44s} mean AUC {aucs.mean():.4f} (SD {aucs.std(ddof=1):.4f}; '
              f'{np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f})')
        rows.append(dict(outcome=outcome, analysis='repeated CV', pipeline=tag.split(' (')[0],
                         auc=aucs.mean(), auc_lo=np.percentile(aucs, 2.5),
                         auc_hi=np.percentile(aucs, 97.5), auprc=np.nan,
                         brier=np.nan, auc_sd=aucs.std(ddof=1)))

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'leakfree_pipeline_results.csv'), index=False,
                          encoding='utf-8-sig')
print('\nSaved -> leakfree_pipeline_results.csv')
