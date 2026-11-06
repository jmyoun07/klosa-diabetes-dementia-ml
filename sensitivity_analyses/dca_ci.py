# -*- coding: utf-8 -*-
"""
Decision-curve analysis with bootstrap confidence intervals for net benefit.

Recalibration procedure:
  * Predictions are 5-fold cross-validated out-of-fold probabilities.
  * They are logistically recalibrated: a logistic model P(y=1) = sigmoid(a + b*logit(p))
    is fitted on those out-of-fold predictions, and the fitted a, b are applied.
  * This recalibration is internal (estimated in the same sample), so the decision
    curve reflects apparent, internally recalibrated net benefit, not net benefit
    that would be observed in a new sample.

Confidence intervals:
  * 500 bootstrap resamples of the participants. Within each resample the logistic
    recalibration is refitted (so the interval includes recalibration uncertainty),
    and the net benefit of the model, of "screen all", and of "screen none" (= 0)
    is computed across the screening-relevant threshold range.
  * 2.5th-97.5th percentile bands are reported and plotted.

Outputs -> dca_netbenefit_ci.csv, figures/Figure6_DCA_v2.{png,pdf}
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))
SEED, NB, EPS = 42, 500, 1e-6

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
DATA = {'MCI': (np.asarray(pre['X_mci'], float), np.asarray(pre['y_mci']).astype(int)),
        'Dementia': (np.asarray(pre['X_dem'], float), np.asarray(pre['y_dem']).astype(int))}


def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', probability=True,
                                 class_weight='balanced', random_state=SEED))])


def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)


def cv_oof(make, X, y, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = make(); m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def recalibrate(y, p, lp=None):
    lp = logit(p) if lp is None else lp
    m = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    a, b = m.params
    return 1 / (1 + np.exp(-(a + b * lp)))


def net_benefit(y, p, pt):
    pred = (p >= pt).astype(int); n = len(y)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return tp / n - fp / n * (pt / (1 - pt))


GRID = np.linspace(0.005, 0.20, 60)
rows = []
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
for ax, (outcome, title) in zip(axes, [('MCI', 'A  Mild cognitive impairment (SVM)'),
                                       ('Dementia', 'B  Dementia (random forest)')]):
    X, y = DATA[outcome]
    make = mci_model if outcome == 'MCI' else dem_model
    p = cv_oof(make, X, y)
    p_rc = recalibrate(y, p)
    prev = y.mean()

    nb_model = np.array([net_benefit(y, p_rc, t) for t in GRID])
    nb_all = np.array([prev - (1 - prev) * (t / (1 - t)) for t in GRID])

    # bootstrap: refit recalibration within each resample
    rng = np.random.RandomState(SEED)
    lp = logit(p)
    B_model = np.empty((NB, len(GRID)))
    B_all = np.empty((NB, len(GRID)))
    for b in range(NB):
        idx = rng.randint(0, len(y), len(y))
        yb, lpb = y[idx], lp[idx]
        if yb.sum() < 2:
            B_model[b] = np.nan; B_all[b] = np.nan; continue
        try:
            pb = recalibrate(yb, None, lp=lpb)
        except Exception:
            pb = p_rc[idx]
        pv = yb.mean()
        B_model[b] = [net_benefit(yb, pb, t) for t in GRID]
        B_all[b] = [pv - (1 - pv) * (t / (1 - t)) for t in GRID]

    lo = np.nanpercentile(B_model, 2.5, axis=0)
    hi = np.nanpercentile(B_model, 97.5, axis=0)
    lo_all = np.nanpercentile(B_all, 2.5, axis=0)
    hi_all = np.nanpercentile(B_all, 97.5, axis=0)

    ax.plot(GRID, nb_model, '-', color='#1f77b4', lw=1.8, label='Model (recalibrated)')
    ax.fill_between(GRID, lo, hi, color='#1f77b4', alpha=0.18, lw=0, label='95% bootstrap interval')
    ax.plot(GRID, nb_all, '--', color='#999999', lw=1.2, label='Screen all')
    ax.fill_between(GRID, lo_all, hi_all, color='#999999', alpha=0.12, lw=0)
    ax.axhline(0, color='black', lw=1.0, label='Screen none')
    ax.set_xlim(0, 0.20)
    ax.set_ylim(min(-0.01, nb_all.min()), prev + 0.01)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=7.6)

    print(f'\n=== {outcome} (prevalence {prev:.4f}) ===')
    for t in (0.02, 0.03, 0.05, 0.10):
        k = int(np.argmin(np.abs(GRID - t)))
        nb_all_t = prev - (1 - prev) * (t / (1 - t))
        print(f'  pt={t:.2f}: model NB {nb_model[k]:.4f} (95% CI {lo[k]:.4f} to {hi[k]:.4f}); '
              f'screen-all NB {nb_all_t:.4f}; screen-none 0')
        rows.append(dict(outcome=outcome, threshold=t, prevalence=prev,
                         nb_model=nb_model[k], nb_model_lo=lo[k], nb_model_hi=hi[k],
                         nb_screen_all=nb_all_t, nb_screen_none=0.0,
                         model_gt_all=nb_model[k] > nb_all_t,
                         ci_excludes_all=lo[k] > nb_all_t,
                         ci_excludes_zero=lo[k] > 0))

fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
for ext in ['png', 'pdf']:
    q = os.path.join(FIGDIR, f'Figure6_DCA_v2.{ext}')
    fig.savefig(q, dpi=300, bbox_inches='tight')
    print('wrote', q)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'dca_netbenefit_ci.csv'), index=False,
                          encoding='utf-8-sig')
from PIL import Image
print('png size:', Image.open(os.path.join(FIGDIR, 'Figure6_DCA_v2.png')).size)
print('Saved -> dca_netbenefit_ci.csv')
