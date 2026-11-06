# -*- coding: utf-8 -*-
"""
Learning curves (training AUC vs validation AUC across increasing sample sizes) for
the two final models, indicating whether performance has converged or remains
data-limited under a low events-per-variable ratio and class imbalance.

  * For each of 50 repeats, a stratified 70:30 train/validation split is drawn.
  * Within each repeat, the training set is sub-sampled with stratification to a
    series of fractions, preserving the outcome prevalence at every size.
  * The two final models (SVM for MCI, random forest for dementia) are refitted
    with their published hyperparameters at each size, and the training AUC and the
    validation AUC are recorded on the same held-out validation set.
  * AUC is computed from decision_function (SVM) or predict_proba (RF); both are
    monotonic in the score, so the AUC is unaffected.

Outputs -> learning_curve_results.csv, figures/Figure7_LearningCurve.{png,pdf}
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))
OUT = os.path.join(HERE, 'learning_curve_results.csv')

SEED = 42
N_REPEATS = 50
FRACTIONS = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
X_mci = np.asarray(pre['X_mci'], float); y_mci = np.asarray(pre['y_mci']).astype(int)
X_dem = np.asarray(pre['X_dem'], float); y_dem = np.asarray(pre['y_dem']).astype(int)


def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', class_weight='balanced',
                                 random_state=SEED))])


def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)


def score(m, X, y):
    if hasattr(m, 'decision_function'):
        try:
            return roc_auc_score(y, m.decision_function(X))
        except Exception:
            pass
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def learning_curve(X, y, make, label):
    print(f'\n===== {label}: n={len(y)}, events={y.sum()} =====')
    sizes, tr_all, va_all = None, [], []
    outer = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=0.3, random_state=SEED)
    for rep, (tr, va) in enumerate(outer.split(X, y)):
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
        tr_row, va_row, size_row = [], [], []
        for f in FRACTIONS:
            if f < 1.0:
                sub = StratifiedShuffleSplit(n_splits=1, train_size=float(f),
                                             random_state=SEED + rep)
                idx, _ = next(sub.split(Xtr, ytr))
            else:
                idx = np.arange(len(ytr))
            xs, ys = Xtr[idx], ytr[idx]
            if len(np.unique(ys)) < 2 or ys.sum() < 2:
                tr_row.append(np.nan); va_row.append(np.nan); size_row.append(len(ys))
                continue
            m = make(); m.fit(xs, ys)
            tr_row.append(score(m, xs, ys))
            va_row.append(score(m, Xva, yva))
            size_row.append(len(ys))
        tr_all.append(tr_row); va_all.append(va_row)
        if sizes is None:
            sizes = np.array(size_row)
    tr_all = np.array(tr_all, float); va_all = np.array(va_all, float)

    tr_m = np.nanmean(tr_all, 0); va_m = np.nanmean(va_all, 0)
    tr_lo = np.nanpercentile(tr_all, 2.5, 0); tr_hi = np.nanpercentile(tr_all, 97.5, 0)
    va_lo = np.nanpercentile(va_all, 2.5, 0); va_hi = np.nanpercentile(va_all, 97.5, 0)
    events = np.round(sizes * y.mean()).astype(int)

    print(f'{"n_train":>8} {"events":>7} {"train AUC":>20} {"validation AUC":>22} {"gap":>7}')
    for i, s in enumerate(sizes):
        print(f'{s:8d} {events[i]:7d}   {tr_m[i]:.3f} [{tr_lo[i]:.3f}-{tr_hi[i]:.3f}]  '
              f'{va_m[i]:.3f} [{va_lo[i]:.3f}-{va_hi[i]:.3f}]  {tr_m[i]-va_m[i]:+.3f}')

    # convergence diagnostics
    d_last = va_m[-1] - va_m[-3]              # change over the final 20% of data
    d_half = va_m[-1] - va_m[len(va_m) // 2]
    print(f'  validation AUC change over final 20% of training data : {d_last:+.4f}')
    print(f'  validation AUC change from 50% to 100% of the data     : {d_half:+.4f}')
    print(f'  optimism (train - validation) at full training size    : {tr_m[-1]-va_m[-1]:+.4f}')
    print(f'  optimism at the smallest training size                 : {tr_m[0]-va_m[0]:+.4f}')

    return dict(sizes=sizes, events=events, tr_m=tr_m, va_m=va_m,
                tr_lo=tr_lo, tr_hi=tr_hi, va_lo=va_lo, va_hi=va_hi,
                d_last=float(d_last), d_half=float(d_half),
                gap_full=float(tr_m[-1] - va_m[-1]), gap_small=float(tr_m[0] - va_m[0]))


r_mci = learning_curve(X_mci, y_mci, mci_model, 'MCI (support vector machine)')
r_dem = learning_curve(X_dem, y_dem, dem_model, 'Dementia (random forest)')

# figure
TRAIN_C, VAL_C = '#c0392b', '#1f77b4'
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax, r, title in [(axes[0], r_mci, 'A  Mild cognitive impairment (SVM)'),
                     (axes[1], r_dem, 'B  Dementia (random forest)')]:
    ax.plot(r['sizes'], r['tr_m'], '-o', color=TRAIN_C, ms=4, lw=1.6, label='Training AUC')
    ax.fill_between(r['sizes'], r['tr_lo'], r['tr_hi'], color=TRAIN_C, alpha=0.13, lw=0)
    ax.plot(r['sizes'], r['va_m'], '--s', color=VAL_C, ms=4, lw=1.6, label='Validation AUC')
    ax.fill_between(r['sizes'], r['va_lo'], r['va_hi'], color=VAL_C, alpha=0.13, lw=0)
    ax.axhline(0.5, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('Training set size (n)')
    ax.set_ylabel('Area under the ROC curve')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    sec = ax.secondary_xaxis('top')
    sec.set_xticks(r['sizes'][::2])
    sec.set_xticklabels(r['events'][::2])
    sec.set_xlabel('Incident events in the training set', fontsize=9)
    sec.tick_params(labelsize=8)

fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
for ext in ['png', 'pdf']:
    p = os.path.join(FIGDIR, f'Figure7_LearningCurve.{ext}')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print('wrote', p)

rows = []
for lab, r in [('MCI (SVM)', r_mci), ('Dementia (RF)', r_dem)]:
    for i, s in enumerate(r['sizes']):
        rows.append(dict(model=lab, n_train=int(s), events_train=int(r['events'][i]),
                         train_auc=r['tr_m'][i], train_lo=r['tr_lo'][i], train_hi=r['tr_hi'][i],
                         val_auc=r['va_m'][i], val_lo=r['va_lo'][i], val_hi=r['va_hi'][i],
                         gap=r['tr_m'][i] - r['va_m'][i]))
pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
print('\nSaved ->', OUT)
