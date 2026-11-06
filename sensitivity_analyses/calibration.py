# -*- coding: utf-8 -*-
"""
Calibration assessment of the MCI (SVM) and dementia (RF) risk models.

Predictions are 5-fold cross-validated out-of-fold probabilities on the full
sample (test-set calibration is unstable with 10 and 15 events). For each model
the calibration slope (coefficient of logit(p) in logit(P(y=1)) = a + b*logit(p))
and calibration-in-the-large (intercept of logit(P(y=1)) = a + offset(logit(p)),
with the slope fixed at 1) are computed with 95% Wald confidence intervals. A
lowess calibration belt with a 95% band from 500 bootstrap resamples is overlaid
with observed-vs-predicted deciles (95% Wilson intervals). Recalibration by Platt
scaling (logistic on logit(p)) and isotonic regression is fitted out of fold
(cross_val_predict) so post-recalibration metrics are not optimistic.

Outputs -> calibration_revision2_results.csv, calibration_deciles.csv,
           figures/Figure5_Calibration_v2.{png,pdf}
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))
SEED, NB, EPS = 42, 500, 1e-6

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.titlesize': 10.5,
                     'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})

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


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_metrics(y, p):
    """calibration slope, calibration-in-the-large, both with 95% CI, plus accuracy."""
    lp = logit(p)
    sl = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    slope = sl.params[1]
    slope_ci = sl.conf_int()[1]
    citl_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                      offset=lp).fit()
    citl = citl_fit.params[0]
    citl_ci = citl_fit.conf_int()[0]
    bs = brier_score_loss(y, p)
    prev = y.mean()
    ref = prev * (1 - prev)
    return dict(slope=slope, slope_lo=slope_ci[0], slope_hi=slope_ci[1],
                citl=citl, citl_lo=citl_ci[0], citl_hi=citl_ci[1],
                mean_pred=p.mean(), observed=prev, brier=bs, scaled_brier=1 - bs / ref,
                auc=roc_auc_score(y, p))


def deciles(y, p, k=10):
    """Observed vs predicted within k equal-count bins of predicted risk."""
    order = np.argsort(p)
    bins = np.array_split(order, k)
    rows = []
    for i, idx in enumerate(bins, 1):
        obs = y[idx].mean()
        lo, hi = proportion_confint(y[idx].sum(), len(idx), method='wilson')
        rows.append(dict(bin=i, n=len(idx), events=int(y[idx].sum()),
                         mean_pred=p[idx].mean(), observed=obs, obs_lo=lo, obs_hi=hi))
    d = pd.DataFrame(rows)
    ece = float(np.sum(np.abs(d.observed - d.mean_pred) * d.n) / len(y))
    return d, ece


def belt(y, p, grid, seed=SEED):
    """lowess calibration curve + 95% bootstrap band, evaluated on `grid`."""
    rng = np.random.RandomState(seed)
    curves = []
    for _ in range(NB):
        idx = rng.randint(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        sm_ = lowess(y[idx], p[idx], frac=0.75, it=0, return_sorted=True)
        curves.append(np.interp(grid, sm_[:, 0], sm_[:, 1]))
    C = np.clip(np.array(curves), 0, 1)
    base = lowess(y, p, frac=0.75, it=0, return_sorted=True)
    return (np.clip(np.interp(grid, base[:, 0], base[:, 1]), 0, 1),
            np.percentile(C, 2.5, axis=0), np.percentile(C, 97.5, axis=0))


rows, dec_rows, panels = [], [], {}
for outcome, (X, y) in DATA.items():
    make = mci_model if outcome == 'MCI' else dem_model
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_raw = cross_val_predict(make(), X, y, cv=skf, method='predict_proba')[:, 1]

    lp = logit(p_raw).reshape(-1, 1)
    p_platt = cross_val_predict(LogisticRegression(max_iter=1000), lp, y, cv=skf,
                                method='predict_proba')[:, 1]
    p_iso = cross_val_predict(IsotonicRegression(out_of_bounds='clip'), p_raw, y, cv=skf)
    p_iso = np.clip(p_iso, EPS, 1 - EPS)

    print(f'\n{"="*80}\n{outcome}: n={len(y)}, events={y.sum()}, prevalence={y.mean():.4f}\n{"="*80}')
    for label, p in [('Uncalibrated', p_raw), ('Platt scaling', p_platt),
                     ('Isotonic regression', p_iso)]:
        m = cal_metrics(y, p)
        d, ece = deciles(y, p)
        m.update(outcome=outcome, method=label, ece=ece)
        rows.append(m)
        d.insert(0, 'method', label); d.insert(0, 'outcome', outcome)
        dec_rows.append(d)
        print(f'  {label:20s} slope {m["slope"]:.2f} [{m["slope_lo"]:.2f}, {m["slope_hi"]:.2f}] | '
              f'CITL {m["citl"]:+.2f} [{m["citl_lo"]:+.2f}, {m["citl_hi"]:+.2f}] | '
              f'mean pred {m["mean_pred"]:.3f} vs obs {m["observed"]:.3f} | '
              f'Brier {m["brier"]:.4f} (scaled {m["scaled_brier"]:+.3f}) | '
              f'ECE {ece:.4f} | AUC {m["auc"]:.3f}')
    panels[outcome] = (y, p_raw, p_platt)

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'calibration_revision2_results.csv'),
                          index=False, encoding='utf-8-sig')
pd.concat(dec_rows, ignore_index=True).to_csv(os.path.join(HERE, 'calibration_deciles.csv'),
                                              index=False, encoding='utf-8-sig')

# Figure 5 (v2)
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
spec = [('MCI', 'raw', axes[0, 0], 'A  MCI (SVM), uncalibrated'),
        ('MCI', 'platt', axes[0, 1], 'B  MCI (SVM), after Platt scaling'),
        ('Dementia', 'raw', axes[1, 0], 'C  Dementia (RF), uncalibrated'),
        ('Dementia', 'platt', axes[1, 1], 'D  Dementia (RF), after Platt scaling')]

for oc, kind, ax, title in spec:
    y, p_raw, p_platt = panels[oc]
    p = p_raw if kind == 'raw' else p_platt
    m = cal_metrics(y, p)
    d, ece = deciles(y, p)
    hi = max(d.mean_pred.max(), d.obs_hi.max()) * 1.12
    grid = np.linspace(p.min(), p.max(), 100)
    c, lo, up = belt(y, p, grid)

    ax.plot([0, hi], [0, hi], ls='--', color='grey', lw=1.0, label='Perfect calibration')
    ax.fill_between(grid, lo, up, color='#1f77b4', alpha=0.16, lw=0, label='95% calibration belt')
    ax.plot(grid, c, color='#1f77b4', lw=1.7, label='Smoothed calibration curve')
    ax.errorbar(d.mean_pred, d.observed,
                yerr=[np.clip(d.observed - d.obs_lo, 0, None),
                      np.clip(d.obs_hi - d.observed, 0, None)],
                fmt='o', ms=4, color='#c0392b', ecolor='#c0392b', elinewidth=0.9,
                capsize=2, lw=0, label='Deciles of predicted risk (95% CI)')
    ax.axhline(y.mean(), color='grey', lw=0.8, ls=':')

    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed event rate')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.03, 0.97,
            f'Slope {m["slope"]:.2f} ({m["slope_lo"]:.2f}–{m["slope_hi"]:.2f})\n'
            f'CITL {m["citl"]:+.2f} ({m["citl_lo"]:+.2f}–{m["citl_hi"]:+.2f})\n'
            f'Brier {m["brier"]:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=7.6,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', lw=0.6))

axes[0, 0].legend(loc='lower right', frameon=False, fontsize=7.2)
fig.tight_layout()
for ext in ['png', 'pdf']:
    q = os.path.join(FIGDIR, f'Figure5_Calibration_v2.{ext}')
    fig.savefig(q, dpi=300, bbox_inches='tight')
    print('\nwrote', q)

from PIL import Image
im = Image.open(os.path.join(FIGDIR, 'Figure5_Calibration_v2.png'))
print('png size:', im.size, 'aspect', im.size[1] / im.size[0])
