# -*- coding: utf-8 -*-
"""
Attrition / informative-censoring sensitivity analyses.

Participants who died or were lost to follow-up without a recorded diagnosis are classified
as non-events, and those lost tend to be older and less educated, so the binary fixed-horizon
outcome may under-ascertain events among higher-risk participants. The following analyses
address this:

  (0) Primary analysis (full cohort, the manuscript's models).
  (1) Excluding participants with no follow-up at all.
  (2) Restricting to participants with complete follow-up (observed at waves 8, 9 and 10).
  (2b) Participants whose 6-year status is fully ascertained (event observed, or seen at the
       final wave), the set on which the outcome is actually known at the horizon.
  (3) Inverse probability-of-censoring weighting (IPCW).
  (4) Time-to-event methods -- performed separately in survival_revision.py and
      competing_risk_revision.py; the headline results are re-printed here for comparison.

For IPCW the ascertainment indicator is R = 1 if the 6-year outcome is known (an event was
observed, or the participant was assessed at wave 10). A logistic model of R on the 22
baseline predictors gives pi = P(R = 1 | X); stabilized weights P(R = 1) / pi are applied
both when refitting the final models and when computing the performance metrics and the
IPCW-corrected cumulative incidence.

Outputs -> attrition_sensitivity_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SEED, B = 42, 500
WY = {'w08': 2.0, 'w09': 4.0, 'w10': 6.0}


def rd(f, cols):
    df, _ = pyreadstat.read_sav(os.path.join(ROOT, f), usecols=cols)
    return df


# cohort + follow-up pattern
df7 = rd('w07_20260227.sav', ['pid', 'w07A002_age', 'w07C014', 'w07Cadd_01', 'w07gender1', 'w07F001'])
c = df7[df7['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]
for w, f in [('w08', 'w08_20260227.sav'), ('w09', 'w09_20260227.sav'), ('w10', 'w10_20260317.sav')]:
    t = rd(f, ['pid', f'{w}Cadd_01']).rename(columns={f'{w}Cadd_01': f'cad_{w}'})
    c = c.merge(t, on='pid', how='left')

mci_pids = c[c['w07Cadd_01'] == 5]['pid'].tolist()
dem_pids = c[c['w07Cadd_01'].isin([3, 5])]['pid'].tolist()

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
FEAT = list(pre['feature_cols'])
DATA = {
    'MCI': (np.asarray(pre['X_mci'], float), np.asarray(pre['y_mci']).astype(int),
            c[c['pid'].isin(mci_pids)].set_index('pid').loc[mci_pids].reset_index()),
    'Dementia': (np.asarray(pre['X_dem'], float), np.asarray(pre['y_dem']).astype(int),
                 c[c['pid'].isin(dem_pids)].set_index('pid').loc[dem_pids].reset_index()),
}


def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', probability=True,
                                 class_weight='balanced', random_state=SEED))])


def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)


def fit_with_weights(make, X, y, w=None):
    m = make()
    if w is None:
        return m.fit(X, y)
    if isinstance(m, Pipeline):
        return m.fit(X, y, svc__sample_weight=w)
    return m.fit(X, y, sample_weight=w)


def boot(y, p, fn, w=None):
    rng = np.random.RandomState(SEED); v = []
    for _ in range(B):
        i = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], p[i]) if w is None else fn(y[i], p[i], sample_weight=w[i]))
    v = np.array(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def evaluate(label, X, y, make, w=None):
    if w is None:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        wtr = wte = None
    else:
        Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, w, test_size=0.3, stratify=y,
                                                        random_state=SEED)
    m = fit_with_weights(make, Xtr, ytr, wtr)
    p = m.predict_proba(Xte)[:, 1]
    auc, alo, ahi = boot(yte, p, roc_auc_score, wte)
    ap, plo, phi = boot(yte, p, average_precision_score, wte)
    bs = brier_score_loss(yte, p, sample_weight=wte)
    rate = y.mean() if w is None else float(np.average(y, weights=w))
    lo, hi = proportion_confint(int(y.sum()), len(y), method='wilson')
    mf = fit_with_weights(make, X, y, w)
    pi = permutation_importance(mf, X, y, n_repeats=10, random_state=SEED,
                                scoring='roc_auc', n_jobs=-1).importances_mean
    top = '; '.join(FEAT[j] for j in np.argsort(pi)[::-1][:5])
    print(f'\n  {label}')
    print(f'    n={len(y):5d}  events={int(y.sum()):3d}  event rate {rate*100:5.2f}% '
          f'({lo*100:.2f}-{hi*100:.2f}%)   test events={int(yte.sum())}')
    print(f'    bootstrap AUC {auc:.3f} ({alo:.3f}-{ahi:.3f}) | AUPRC {ap:.3f} ({plo:.3f}-{phi:.3f}) | '
          f'Brier {bs:.3f}')
    print(f'    top predictors: {top}')
    return dict(analysis=label, n=len(y), events=int(y.sum()), event_rate=rate,
                rate_lo=lo, rate_hi=hi, auc=auc, auc_lo=alo, auc_hi=ahi,
                auprc=ap, auprc_lo=plo, auprc_hi=phi, brier=bs, top_predictors=top)


rows = []
for outcome, (X, y, cc) in DATA.items():
    make = mci_model if outcome == 'MCI' else dem_model
    n_fu = sum(cc[f'cad_{w}'].notna().astype(int) for w in ['w08', 'w09', 'w10'])
    any_fu = (n_fu >= 1).values
    complete_fu = (n_fu == 3).values
    seen_final = cc['cad_w10'].notna().values
    ascertained = (y == 1) | seen_final

    print('\n' + '=' * 88)
    print(f'{outcome}: n={len(y)}, events={y.sum()} ({y.mean()*100:.2f}%)')
    print('=' * 88)
    print(f'  no follow-up at all            : {(~any_fu).sum():4d} ({(~any_fu).mean()*100:.1f}%)')
    print(f'  complete follow-up (3 waves)   : {complete_fu.sum():4d} ({complete_fu.mean()*100:.1f}%)')
    print(f'  6-year status fully ascertained: {ascertained.sum():4d} ({ascertained.mean()*100:.1f}%)')

    # IPCW: model the probability that the 6-year status is ascertained
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pi_hat = cross_val_predict(Pipeline([('sc', StandardScaler()),
                                         ('lr', LogisticRegression(max_iter=2000))]),
                               X, ascertained.astype(int), cv=skf, method='predict_proba')[:, 1]
    auc_cens = roc_auc_score(ascertained.astype(int), pi_hat)
    pi_hat = np.clip(pi_hat, 0.05, 1.0)
    stab = ascertained.mean() / pi_hat
    print(f'\n  censoring model (P[ascertained | X]) cross-validated AUC {auc_cens:.3f}; '
          f'stabilized weights: mean {stab[ascertained].mean():.3f}, '
          f'range {stab[ascertained].min():.2f}-{stab[ascertained].max():.2f}')

    rows.append(dict(outcome=outcome, **evaluate(
        '(0) Primary analysis, full cohort', X, y, make)))
    rows.append(dict(outcome=outcome, **evaluate(
        '(1) Excluding participants with no follow-up', X[any_fu], y[any_fu], make)))
    rows.append(dict(outcome=outcome, **evaluate(
        '(2) Restricted to complete follow-up (3 waves)', X[complete_fu], y[complete_fu], make)))
    rows.append(dict(outcome=outcome, **evaluate(
        '(2b) 6-year status fully ascertained', X[ascertained], y[ascertained], make)))
    rows.append(dict(outcome=outcome, **evaluate(
        '(3) IPCW on the fully ascertained set', X[ascertained], y[ascertained], make,
        w=stab[ascertained])))

    naive = y.mean()
    ipcw_rate = float(np.average(y[ascertained], weights=stab[ascertained]))
    unweighted_asc = y[ascertained].mean()
    print(f'\n  6-year cumulative incidence: primary (all as non-events) {naive*100:.2f}% | '
          f'fully ascertained, unweighted {unweighted_asc*100:.2f}% | IPCW {ipcw_rate*100:.2f}%')
    rows.append(dict(outcome=outcome, analysis='(4) Cumulative incidence comparison',
                     n=len(y), events=int(y.sum()), event_rate=naive,
                     rate_lo=np.nan, rate_hi=np.nan, auc=np.nan, auc_lo=np.nan, auc_hi=np.nan,
                     auprc=np.nan, auprc_lo=np.nan, auprc_hi=np.nan, brier=np.nan,
                     top_predictors=f'primary {naive*100:.2f}%; ascertained {unweighted_asc*100:.2f}%; '
                                    f'IPCW {ipcw_rate*100:.2f}%; censoring-model AUC {auc_cens:.3f}'))

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'attrition_sensitivity_results.csv'),
                          index=False, encoding='utf-8-sig')
print('\n' + '=' * 88)
print('(4) Time-to-event methods, from survival_revision.py (for comparison)')
print('=' * 88)
print('  Random survival forest, cross-validated C-index: 0.761 (SD 0.075) MCI; '
      '0.770 (SD 0.051) dementia')
print('  Cox proportional hazards, cross-validated C-index: 0.778 (SD 0.094) MCI; '
      '0.733 (SD 0.042) dementia')
print('\nSaved -> attrition_sensitivity_results.csv')
