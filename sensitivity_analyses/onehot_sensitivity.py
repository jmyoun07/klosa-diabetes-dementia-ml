# -*- coding: utf-8 -*-
"""
One-hot encoding sensitivity analysis for the nominal categorical predictors.

Several categorical predictors were entered as ordinal integer-coded variables, which is
defensible for the genuinely ordered variables (educational attainment, household income
quintile) but less appropriate for the nominal / non-monotonic ones: marital status
(married / widowed / other), living arrangement (alone / with spouse only / with >=2 family
members), and body mass index category (underweight / normal / overweight / obese, whose
relationship with risk is not monotonic).

The two final models are refit, at their published hyperparameters, after one-hot encoding
those three variables (dropping the first level as the reference), leaving the ordered and
binary predictors unchanged, and discrimination is compared with the primary ordinal-coded
models on:
  (A) the single stratified 70:30 split (seed 42), and
  (B) repeated stratified 5-fold cross-validation (20 repeats, pooled out-of-fold AUC).

Outputs -> onehot_sensitivity_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.utils import resample

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, B = 42, 500
NOMINAL = ['marital', 'living', 'bmi_cat']            # one-hot these
LEVELS = {'marital': [1, 2, 3], 'living': [1, 2, 3], 'bmi_cat': [1, 2, 3, 4]}

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)


def one_hot(Xdf):
    """One-hot encode the nominal variables (drop first level = reference); keep the rest."""
    X = Xdf.copy()
    parts = []
    for c in X.columns:
        if c in NOMINAL:
            for lev in LEVELS[c][1:]:                # drop first level as reference
                parts.append(pd.Series((X[c] == lev).astype(float), name=f'{c}_{lev}'))
        else:
            parts.append(X[c].astype(float))
    return pd.concat(parts, axis=1)


DATA = {'MCI': (pre['X_mci'], np.asarray(pre['y_mci']).astype(int), 'svm'),
        'Dementia': (pre['X_dem'], np.asarray(pre['y_dem']).astype(int), 'rf')}


def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', probability=True,
                                 class_weight='balanced', random_state=SEED))])


def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)


def boot(y, p, fn):
    rng = np.random.RandomState(SEED); v = []
    for _ in range(B):
        i = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], p[i]))
    v = np.array(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def single_split(make, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    m = make().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return boot(yte, p, roc_auc_score), boot(yte, p, average_precision_score), brier_score_loss(yte, p)


def repeated_cv(make, X, y):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    splits = list(rskf.split(X, y))
    aucs = []
    for r in range(20):
        oof = np.full(len(y), np.nan)
        for tr, te in splits[r * 5:(r + 1) * 5]:
            m = make().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    a = np.array(aucs)
    return a.mean(), a.std(ddof=1), np.percentile(a, 2.5), np.percentile(a, 97.5)


rows = []
for outcome, (Xdf, y, kind) in DATA.items():
    make = mci_model if kind == 'svm' else dem_model
    Xo = X_ord = np.asarray(Xdf, float)
    Xh = np.asarray(one_hot(Xdf), float)
    print(f'\n{"="*90}\n{outcome} ({kind.upper()}): n={len(y)}, events={y.sum()}   '
          f'ordinal {Xo.shape[1]} features -> one-hot {Xh.shape[1]} features\n{"="*90}')
    for tag, X in [('ordinal integer coding (primary)', Xo),
                   ('one-hot encoding of marital, living, BMI category', Xh)]:
        (auc, alo, ahi), (ap, plo, phi), bs = single_split(make, X, y)
        cvm, cvsd, cvlo, cvhi = repeated_cv(make, X, y)
        print(f'  {tag:52s}')
        print(f'      single split : AUC {auc:.3f} ({alo:.3f}-{ahi:.3f})  AUPRC {ap:.3f}  Brier {bs:.3f}')
        print(f'      repeated CV  : AUC {cvm:.3f} (SD {cvsd:.3f}; {cvlo:.3f}-{cvhi:.3f})')
        rows.append(dict(outcome=outcome, coding=tag.split(' (')[0], n_features=X.shape[1],
                         single_auc=auc, single_auc_lo=alo, single_auc_hi=ahi, single_auprc=ap,
                         single_brier=bs, cv_auc=cvm, cv_sd=cvsd, cv_lo=cvlo, cv_hi=cvhi))

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'onehot_sensitivity_results.csv'), index=False,
                          encoding='utf-8-sig')
print('\nSaved -> onehot_sensitivity_results.csv')
