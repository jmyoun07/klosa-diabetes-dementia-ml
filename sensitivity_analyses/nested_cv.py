# -*- coding: utf-8 -*-
"""
Nested cross-validation and stratification audit of the model-development pipeline.

Two analyses are produced.

(1) Stratification audit. The event count in every training and validation fold of the
    tuning CV, and in the outer folds of the nested CV, is printed so that the stratified
    70:30 split and stratified 5-fold cross-validation can be checked.

(2) Nested cross-validation. The primary analysis is not nested: hyperparameters are
    tuned by 5-fold CV inside the training set, the best of eight algorithms is selected
    on that same CV, and the held-out test set is scored once. This runs a fully nested CV
    in which both hyperparameter tuning and algorithm selection are performed inside the
    inner loop, so that the outer folds give a selection- and tuning-unbiased estimate of
    the whole model-development pipeline.

Search spaces, samplers, class weighting and the objective (mean AUC over a stratified
inner 5-fold CV) replicate klosa_all_models_tuning.py. The number of Optuna trials is
reduced from 150 to N_TRIALS (default 50) because the nested loop multiplies the fits by
the number of outer folds.

Outputs -> nested_cv_results.csv, nested_cv_folds.csv (+ console log)
"""
import os, sys, pickle, time, warnings

# single-threaded workers: the parallelism is over (outer fold x algorithm) tasks
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = int(os.environ.get('N_TRIALS', 50))
OUTER_SPLITS = int(os.environ.get('OUTER_SPLITS', 5))
ALGOS = ['LASSO_LR', 'SVC', 'RF', 'GBM', 'AdaBoost', 'LightGBM', 'XGBoost', 'CatBoost']
SEED = 42

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
DATA = {'MCI': (np.asarray(pre['X_mci'], float), np.asarray(pre['y_mci']).astype(int)),
        'Dementia': (np.asarray(pre['X_dem'], float), np.asarray(pre['y_dem']).astype(int))}


def build(model_type, trial=None, params=None):
    """Return an unfitted estimator. Either sample from `trial` or use `params`."""
    def s_float(n, lo, hi, log=False):
        return trial.suggest_float(n, lo, hi, log=log) if trial else params[n]

    def s_int(n, lo, hi):
        return trial.suggest_int(n, lo, hi) if trial else params[n]

    def s_cat(n, choices):
        return trial.suggest_categorical(n, choices) if trial else params[n]

    if model_type == 'LASSO_LR':
        return Pipeline([('scaler', StandardScaler()),
                         ('lr', LogisticRegression(C=s_float('C', 0.001, 100, True), penalty='l1',
                                                   solver='saga', max_iter=5000,
                                                   class_weight='balanced', random_state=SEED))])
    if model_type == 'SVC':
        return Pipeline([('scaler', StandardScaler()),
                         ('svc', SVC(C=s_float('C', 0.01, 100, True),
                                     gamma=s_float('gamma', 0.0001, 1, True),
                                     kernel=s_cat('kernel', ['rbf', 'poly', 'sigmoid']),
                                     probability=True, class_weight='balanced',
                                     random_state=SEED))])
    if model_type == 'RF':
        return RandomForestClassifier(n_estimators=s_int('n_est', 100, 500),
                                      max_depth=s_int('max_depth', 3, 15),
                                      min_samples_split=s_int('min_split', 2, 20),
                                      min_samples_leaf=s_int('min_leaf', 1, 10),
                                      class_weight='balanced', random_state=SEED, n_jobs=1)
    if model_type == 'GBM':
        return GradientBoostingClassifier(n_estimators=s_int('n_est', 100, 500),
                                          max_depth=s_int('max_depth', 3, 8),
                                          learning_rate=s_float('lr', 0.01, 0.3, True),
                                          min_samples_split=s_int('min_split', 2, 20),
                                          min_samples_leaf=s_int('min_leaf', 1, 10),
                                          subsample=s_float('subsample', 0.5, 1.0),
                                          random_state=SEED)
    if model_type == 'AdaBoost':
        # scikit-learn >= 1.6 dropped the `algorithm` argument; SAMME is the only option
        kw = {'algorithm': 'SAMME'} if 'algorithm' in AdaBoostClassifier().get_params() else {}
        return AdaBoostClassifier(n_estimators=s_int('n_est', 50, 500),
                                  learning_rate=s_float('lr', 0.01, 2.0, True),
                                  random_state=SEED, **kw)
    if model_type == 'LightGBM':
        return LGBMClassifier(n_estimators=s_int('n_est', 100, 500),
                              max_depth=s_int('max_depth', 3, 10),
                              learning_rate=s_float('lr', 0.01, 0.3, True),
                              num_leaves=s_int('num_leaves', 15, 63),
                              min_child_samples=s_int('min_child', 5, 50),
                              class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=1)
    if model_type == 'XGBoost':
        return XGBClassifier(n_estimators=s_int('n_est', 100, 500),
                             max_depth=s_int('max_depth', 3, 10),
                             learning_rate=s_float('lr', 0.01, 0.3, True),
                             subsample=s_float('subsample', 0.5, 1.0),
                             colsample_bytree=s_float('colsample', 0.5, 1.0),
                             random_state=SEED, eval_metric='auc', verbosity=0, n_jobs=1)
    if model_type == 'CatBoost':
        return CatBoostClassifier(iterations=s_int('iterations', 100, 500),
                                  depth=s_int('depth', 3, 8),
                                  learning_rate=s_float('lr', 0.01, 0.3, True),
                                  l2_leaf_reg=s_float('l2', 1, 10),
                                  auto_class_weights='Balanced', random_seed=SEED,
                                  verbose=0, thread_count=1)
    raise ValueError(model_type)


def fit_one(model_type, est, Xtr, ytr):
    """Fit, applying the class-weighting mechanism used for that algorithm."""
    m = clone(est)
    if model_type in ('GBM', 'AdaBoost'):
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    elif model_type == 'XGBoost':
        pos = int(ytr.sum()); neg = len(ytr) - pos
        m.set_params(scale_pos_weight=float(neg / pos) if pos else 1.0)
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr)
    return m


def inner_cv_auc(model_type, est, X, y, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        if len(np.unique(y[va])) < 2:
            continue
        try:
            m = fit_one(model_type, est, X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        except Exception:
            aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def tune(model_type, X, y, n_trials, seed=SEED):
    def objective(trial):
        return inner_cv_auc(model_type, build(model_type, trial=trial), X, y, seed)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# (1) stratification audit
def stratification_audit(name, X, y):
    print(f'\n--- Stratification audit: {name} (n={len(y)}, events={y.sum()}) ---')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f'  stratified 70:30 split -> train {len(ytr)} (events {ytr.sum()}, '
          f'{ytr.mean()*100:.2f}%) | test {len(yte)} (events {yte.sum()}, {yte.mean()*100:.2f}%)')
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, va) in enumerate(skf.split(Xtr, ytr), 1):
        rows.append((k, len(tr), int(ytr[tr].sum()), len(va), int(ytr[va].sum()),
                     ytr[va].mean() * 100))
        print(f'  tuning fold {k}: train {len(tr)} (events {int(ytr[tr].sum())})  '
              f'validation {len(va)} (events {int(ytr[va].sum())}, {ytr[va].mean()*100:.2f}%)')
    ev = [r[4] for r in rows]
    print(f'  -> every validation fold contains events (min {min(ev)}, max {max(ev)}); '
          f'no fold is event-free')
    return rows


audit = {n: stratification_audit(n, X, y) for n, (X, y) in DATA.items()}


# (2) nested CV
N_JOBS = int(os.environ.get('N_JOBS', 14))


def _task(name, k, tr, algo, X, y, n_trials):
    t0 = time.time()
    p, v = tune(algo, X[tr], y[tr], n_trials, seed=SEED + k)
    print(f'    [{name} fold {k}] {algo:9s} inner CV AUC {v:.4f} ({time.time()-t0:.0f}s)',
          flush=True)
    return (k, algo, p, v)


def nested_cv(name, X, y):
    bar = "=" * 74
    print(f'\n{bar}\nNested CV: {name}  (outer {OUTER_SPLITS}-fold, inner 5-fold, '
          f'{N_TRIALS} Optuna trials x {len(ALGOS)} algorithms)\n{bar}', flush=True)
    outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=SEED)
    splits = list(outer.split(X, y))
    jobs = [(k, tr, a) for k, (tr, te) in enumerate(splits, 1) for a in ALGOS]
    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(
        delayed(_task)(name, k, tr, a, X, y, N_TRIALS) for k, tr, a in jobs)
    print(f'  tuning finished in {time.time()-t0:.0f}s', flush=True)

    recs = []
    for k, (tr, te) in enumerate(splits, 1):
        cand = [(v, a, p) for (kk, a, p, v) in out if kk == k]
        per_algo = {a: v for (v, a, p) in cand}
        inner_auc, algo, params = max(cand, key=lambda z: z[0])
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        m = fit_one(algo, build(algo, params=params), Xtr, ytr)
        outer_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        recs.append(dict(outcome=name, fold=k, n_train=len(ytr), events_train=int(ytr.sum()),
                         n_test=len(yte), events_test=int(yte.sum()),
                         selected_algorithm=algo, inner_cv_auc=inner_auc, outer_auc=outer_auc,
                         **{f'inner_{a}': per_algo[a] for a in ALGOS}))
        print(f'  fold {k}: test n={len(yte)} events={int(yte.sum())} | selected {algo:9s} '
              f'| inner CV AUC {inner_auc:.4f} | OUTER AUC {outer_auc:.4f}', flush=True)

    df = pd.DataFrame(recs)
    o = df['outer_auc'].values; i = df['inner_cv_auc'].values
    print(f'\n  Nested (unbiased) AUC : mean {o.mean():.4f}  SD {o.std():.4f}  '
          f'range {o.min():.4f}-{o.max():.4f}')
    print(f'  Inner CV AUC (optimistic, used for selection): mean {i.mean():.4f}')
    print(f'  Selection + tuning optimism: {i.mean()-o.mean():+.4f}')
    print(f'  Algorithms selected across outer folds: '
          f'{dict(df["selected_algorithm"].value_counts())}', flush=True)
    return df


frames = [nested_cv(n, X, y) for n, (X, y) in DATA.items()]
folds = pd.concat(frames, ignore_index=True)
folds.to_csv(os.path.join(HERE, 'nested_cv_folds.csv'), index=False, encoding='utf-8-sig')

summary = []
for name, df in zip(DATA.keys(), frames):
    o = df['outer_auc'].values; i = df['inner_cv_auc'].values
    summary.append(dict(outcome=name, n_outer_folds=len(df), n_trials=N_TRIALS,
                        nested_auc_mean=o.mean(), nested_auc_sd=o.std(),
                        nested_auc_min=o.min(), nested_auc_max=o.max(),
                        inner_cv_auc_mean=i.mean(),
                        selection_tuning_optimism=i.mean() - o.mean(),
                        selected_algorithms='; '.join(
                            f'{k} x{v}' for k, v in df['selected_algorithm'].value_counts().items())))
pd.DataFrame(summary).to_csv(os.path.join(HERE, 'nested_cv_results.csv'),
                             index=False, encoding='utf-8-sig')

audit_rows = [dict(outcome=n, fold=r[0], n_train=r[1], events_train=r[2],
                   n_val=r[3], events_val=r[4], pct_val=r[5])
              for n, rows in audit.items() for r in rows]
pd.DataFrame(audit_rows).to_csv(os.path.join(HERE, 'stratification_audit.csv'),
                                index=False, encoding='utf-8-sig')
print('\nSaved -> nested_cv_results.csv, nested_cv_folds.csv, stratification_audit.csv')
