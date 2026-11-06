# -*- coding: utf-8 -*-
"""
Sensitivity analysis using the K-MMSE cognitive assessment available at wave 10.

Waves 7-9 of the KLoSA release used here contain no cognitive test. Wave 10 (2024)
contains the 19 items of the Korean Mini-Mental State Examination (w10C401-w10C419),
scored to the standard 30-point K-MMSE total. Item coding: a value of 5 means no
sub-item correct (0 points); values 1-4 are the number of sub-items answered
correctly; -8/-9 are refusals and are treated as missing. The K-MMSE is available
only at the final wave, so an MMSE-defined incident outcome cannot be constructed
(there is no baseline MMSE). Two analyses are performed:

 (A) Agreement between the self-reported physician diagnosis and the K-MMSE at wave 10
     (sensitivity, specificity, predictive values, Cohen's kappa).

 (B) The two final models, with their published hyperparameters and the same 22
     baseline (wave-7) predictors, refitted to predict a K-MMSE-defined outcome at
     wave 10 among participants cognitively normal by self-report at baseline:
        - cognitive impairment      : K-MMSE <= 23
        - dementia-level impairment : K-MMSE <= 17
     Discrimination, AUPRC, Brier score and permutation importance are compared with
     the self-report-based models.

Outputs -> mmse_agreement.csv, mmse_sensitivity_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SEED, B = 42, 500

ITEMS = [f'w10C4{i:02d}' for i in range(1, 20)]
MAXP = {'w10C401': 3, 'w10C402': 3, 'w10C403': 1, 'w10C404': 1, 'w10C405': 4, 'w10C406': 1,
        'w10C407': 3, 'w10C408': 1, 'w10C409': 1, 'w10C410': 1, 'w10C411': 1, 'w10C412': 1,
        'w10C413': 1, 'w10C414': 1, 'w10C415': 1, 'w10C416': 3, 'w10C417': 1, 'w10C418': 1,
        'w10C419': 1}
assert sum(MAXP.values()) == 30


def rd(f, cols):
    df, _ = pyreadstat.read_sav(os.path.join(ROOT, f), usecols=cols)
    return df


# cohort + K-MMSE
df7 = rd('w07_20260227.sav', ['pid', 'w07A002_age', 'w07C014', 'w07Cadd_01', 'w07gender1', 'w07F001'])
c = df7[df7['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]

w10 = rd('w10_20260317.sav', ['pid', 'w10Cadd_01'] + ITEMS)
S = pd.DataFrame(index=w10.index)
for it in ITEMS:
    v = w10[it].where(w10[it] > 0)
    S[it] = np.minimum(np.where(v == 5, 0, v), MAXP[it])
w10['mmse'] = S.sum(axis=1, min_count=len(ITEMS))
c = c.merge(w10[['pid', 'w10Cadd_01', 'mmse']], on='pid', how='left')

mci_pids = c[c['w07Cadd_01'] == 5]['pid'].tolist()
dem_pids = c[c['w07Cadd_01'].isin([3, 5])]['pid'].tolist()

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
FEAT = list(pre['feature_cols'])
X_mci = np.asarray(pre['X_mci'], float)
X_dem = np.asarray(pre['X_dem'], float)
y_mci = np.asarray(pre['y_mci']).astype(int)
y_dem = np.asarray(pre['y_dem']).astype(int)
assert X_mci.shape[0] == len(mci_pids) and X_dem.shape[0] == len(dem_pids)

c_mci = c[c['pid'].isin(mci_pids)].set_index('pid').loc[mci_pids].reset_index()
c_dem = c[c['pid'].isin(dem_pids)].set_index('pid').loc[dem_pids].reset_index()

print('=' * 84)
print('K-MMSE availability at wave 10')
print('=' * 84)
for lab, cc in [('MCI cohort (baseline normal)', c_mci), ('Dementia cohort', c_dem)]:
    ok = cc['mmse'].notna()
    print(f'  {lab:32s} n={len(cc)}  with K-MMSE at wave 10: {ok.sum()} ({ok.mean()*100:.1f}%)  '
          f'mean {cc.loc[ok, "mmse"].mean():.1f} (SD {cc.loc[ok, "mmse"].std():.1f})')

# (A) agreement
print('\n' + '=' * 84)
print('(A) Agreement between the self-reported physician diagnosis and the K-MMSE at wave 10')
print('=' * 84)
sub = c_dem[c_dem['mmse'].notna() & c_dem['w10Cadd_01'].notna()].copy()
print(f'  participants with both measures: {len(sub)}')
print('\n  K-MMSE by self-reported wave-10 status:')
for v, name in [(1, 'dementia'), (3, 'MCI'), (5, 'no diagnosis')]:
    s = sub[sub['w10Cadd_01'] == v]['mmse']
    if len(s):
        print(f'    {name:14s} n={len(s):4d}  mean {s.mean():5.1f} (SD {s.std():4.1f})  median {s.median():.0f}')

agree_rows = []
for thr, tname in [(23, 'K-MMSE <= 23 (cognitive impairment)'), (17, 'K-MMSE <= 17 (dementia level)')]:
    ref = (sub['mmse'] <= thr).astype(int).values
    for pos, pname in [([1, 3], 'self-reported MCI or dementia'), ([1], 'self-reported dementia')]:
        idx = sub['w10Cadd_01'].isin(pos).astype(int).values
        tn, fp, fn, tp = confusion_matrix(ref, idx, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        k = cohen_kappa_score(ref, idx)
        sl, sh = proportion_confint(tp, tp + fn, method='wilson')
        print(f'\n  Reference {tname}  vs index "{pname}"')
        print(f'    prevalence(reference) {ref.mean()*100:5.1f}%   flagged by self-report {idx.mean()*100:5.1f}%')
        print(f'    sensitivity {sens:.3f} ({sl:.3f}-{sh:.3f})  specificity {spec:.3f}  '
              f'PPV {ppv:.3f}  NPV {npv:.3f}  kappa {k:.3f}')
        print(f'    missed by self-report: {fn} of {tp+fn} ({fn/(tp+fn)*100:.1f}%)')
        agree_rows.append(dict(reference=tname, index=pname, n=len(sub),
                               ref_prevalence=ref.mean(), index_positive=idx.mean(),
                               sensitivity=sens, sens_lo=sl, sens_hi=sh, specificity=spec,
                               ppv=ppv, npv=npv, kappa=k, missed_n=int(fn), ref_positive=int(tp + fn)))
pd.DataFrame(agree_rows).to_csv(os.path.join(HERE, 'mmse_agreement.csv'), index=False,
                                encoding='utf-8-sig')

# self-reported incident cases with a low wave-10 K-MMSE
inc = c_mci['mmse'].notna().values & (y_mci == 1)
print(f'\n  Of the {y_mci.sum()} self-reported incident MCI cases, {inc.sum()} have a wave-10 K-MMSE; '
      f'{(c_mci.loc[inc, "mmse"] <= 23).sum()} of these score <= 23 '
      f'({(c_mci.loc[inc, "mmse"] <= 23).mean()*100:.0f}%)')
incd = c_dem['mmse'].notna().values & (y_dem == 1)
print(f'  Of the {y_dem.sum()} self-reported incident dementia cases, {incd.sum()} have a wave-10 K-MMSE; '
      f'{(c_dem.loc[incd, "mmse"] <= 17).sum()} of these score <= 17 '
      f'({(c_dem.loc[incd, "mmse"] <= 17).mean()*100:.0f}%)')


# (B) refit models
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
    return np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


def evaluate(label, X, y, make):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    m = make().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc, alo, ahi = boot(yte, p, roc_auc_score)
    ap, plo, phi = boot(yte, p, average_precision_score)
    bs = brier_score_loss(yte, p)
    prev = yte.mean()
    mf = make().fit(X, y)
    pi = permutation_importance(mf, X, y, n_repeats=10, random_state=SEED,
                               scoring='roc_auc', n_jobs=-1).importances_mean
    top = [FEAT[j] for j in np.argsort(pi)[::-1][:6]]
    print(f'\n  {label}')
    print(f'    n={len(y)}  events={int(y.sum())} ({y.mean()*100:.1f}%)  test events={int(yte.sum())}')
    print(f'    bootstrap AUC {auc:.3f} ({alo:.3f}-{ahi:.3f}) | AUPRC {ap:.3f} ({plo:.3f}-{phi:.3f}), '
          f'baseline {prev:.3f} | Brier {bs:.3f}')
    print(f'    top predictors: {"; ".join(top)}')
    return dict(analysis=label, n=len(y), events=int(y.sum()), event_rate=y.mean(),
                auc=auc, auc_lo=alo, auc_hi=ahi, auprc=ap, auprc_lo=plo, auprc_hi=phi,
                auprc_baseline=prev, brier=bs, top_predictors='; '.join(top))


print('\n' + '=' * 84)
print('(B) Sensitivity analysis: K-MMSE-defined outcome at wave 10, baseline (wave-7) predictors')
print('=' * 84)
rows = []
keep = c_mci['mmse'].notna().values
Xk, mm = X_mci[keep], c_mci.loc[keep, 'mmse'].values
rows.append(evaluate('MCI cohort, K-MMSE <= 23 at wave 10 (SVM)', Xk, (mm <= 23).astype(int), mci_model))
rows.append(evaluate('MCI cohort, K-MMSE <= 17 at wave 10 (SVM)', Xk, (mm <= 17).astype(int), mci_model))

keepd = c_dem['mmse'].notna().values
Xd, mmd = X_dem[keepd], c_dem.loc[keepd, 'mmse'].values
rows.append(evaluate('Dementia cohort, K-MMSE <= 17 at wave 10 (RF)', Xd, (mmd <= 17).astype(int), dem_model))
rows.append(evaluate('Dementia cohort, K-MMSE <= 23 at wave 10 (RF)', Xd, (mmd <= 23).astype(int), dem_model))

print('\n  For reference, the self-report-based primary models:')
rows.append(evaluate('MCI cohort, self-reported incident MCI (SVM)', X_mci, y_mci, mci_model))
rows.append(evaluate('Dementia cohort, self-reported incident dementia (RF)', X_dem, y_dem, dem_model))

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'mmse_sensitivity_results.csv'), index=False,
                          encoding='utf-8-sig')
print('\nSaved -> mmse_agreement.csv, mmse_sensitivity_results.csv')
