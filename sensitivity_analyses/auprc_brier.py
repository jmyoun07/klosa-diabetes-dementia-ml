# -*- coding: utf-8 -*-
"""
Area under the precision-recall curve (AUPRC) and Brier score for the eight tuned
models, complementing AUC-ROC under severe class imbalance where AUC-ROC is dominated
by true negatives. AUPRC measures discrimination against the positive class and the
Brier score combines discrimination and calibration.

All eight tuned models are reproduced (hyperparameters read from
results/ML_results_all8_tuned.csv, the same stratified 70:30 split with seed 42) and
checked against the published test AUC and Brier score. On the held-out test set with
500 bootstrap resamples the script computes:

  * AUPRC (average precision) with a 95% bootstrap interval
  * the no-skill AUPRC baseline (= outcome prevalence in the test set)
  * the ratio AUPRC / baseline (fold improvement over chance)
  * the Brier score with a 95% bootstrap interval
  * the reference Brier score of a no-skill model that predicts the prevalence
    for everyone, i.e. p(1-p)
  * the scaled Brier score (Brier skill score) = 1 - BS / BS_ref, positive only when
    the model beats that no-skill model

For the two final models the Brier score is additionally recomputed after logistic
(Platt) recalibration of cross-validated out-of-fold predictions, separating the
miscalibration induced by inverse class weighting from the discriminative signal.

Outputs -> auprc_brier_results.csv, figures/Figure8_PRcurve.{png,pdf}
"""
import os, ast, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results', 'ML_results_all8_tuned.csv'))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))
OUT = os.path.join(HERE, 'auprc_brier_results.csv')
SEED, B = 42, 500

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})

LABEL = {'LASSO_LR_tuned': 'L1 Penalized Logistic Regression', 'SVC_tuned': 'Support Vector Machine',
         'RF_tuned': 'Random Forest', 'GBM_tuned': 'Gradient Boosting Machine',
         'AdaBoost_tuned': 'Adaptive Boosting', 'LightGBM_tuned': 'Light Gradient-Boosting Machine',
         'XGBoost_tuned': 'Extreme Gradient Boosting', 'CatBoost_tuned': 'Categorical Boosting'}

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
DATA = {'MCI': (np.asarray(pre['X_mci'], float), np.asarray(pre['y_mci']).astype(int)),
        'Dementia': (np.asarray(pre['X_dem'], float), np.asarray(pre['y_dem']).astype(int))}

res = pd.read_csv(RESULTS)
res['key'] = res['model'].str.replace('_tuned', '', regex=False)


def build(mt, p, y_ref=None):
    if mt == 'LASSO_LR':
        return Pipeline([('sc', StandardScaler()),
                         ('lr', LogisticRegression(C=p['C'], penalty='l1', solver='saga',
                                                   max_iter=5000, class_weight='balanced',
                                                   random_state=SEED))])
    if mt == 'SVC':
        return Pipeline([('sc', StandardScaler()),
                         ('svc', SVC(C=p['C'], gamma=p['gamma'], kernel=p['kernel'],
                                     probability=True, class_weight='balanced',
                                     random_state=SEED))])
    if mt == 'AdaBoost':
        kw = {'algorithm': 'SAMME'} if 'algorithm' in AdaBoostClassifier().get_params() else {}
        return AdaBoostClassifier(n_estimators=p['n_est'], learning_rate=p['lr'],
                                  random_state=SEED, **kw)
    if mt == 'CatBoost':
        return CatBoostClassifier(iterations=p['iterations'], depth=p['depth'],
                                  learning_rate=p['lr'], l2_leaf_reg=p['l2'],
                                  auto_class_weights='Balanced', random_seed=SEED, verbose=0)
    if mt == 'RF':
        return RandomForestClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                                      min_samples_split=p['min_split'],
                                      min_samples_leaf=p['min_leaf'], class_weight='balanced',
                                      random_state=SEED, n_jobs=-1)
    if mt == 'LightGBM':
        return LGBMClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                              learning_rate=p['lr'], num_leaves=p['num_leaves'],
                              min_child_samples=p['min_child'], class_weight='balanced',
                              random_state=SEED, verbose=-1)
    if mt == 'XGBoost':
        pos = int(y_ref.sum()); neg = len(y_ref) - pos
        return XGBClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                             learning_rate=p['lr'], subsample=p['subsample'],
                             colsample_bytree=p['colsample'],
                             scale_pos_weight=float(neg / pos), random_state=SEED,
                             eval_metric='auc', verbosity=0)
    if mt == 'GBM':
        return GradientBoostingClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                                          learning_rate=p['lr'], min_samples_split=p['min_split'],
                                          min_samples_leaf=p['min_leaf'], subsample=p['subsample'],
                                          random_state=SEED)
    raise ValueError(mt)


def fit(mt, est, X, y):
    if mt in ('GBM', 'AdaBoost'):
        est.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    else:
        est.fit(X, y)
    return est


def boot(y, p, fn, seed=SEED):
    rng = np.random.RandomState(seed); v = []
    for _ in range(B):
        idx = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[idx])) < 2:
            continue
        v.append(fn(y[idx], p[idx]))
    v = np.array(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rows, curves = [], {}
for outcome, (X, y) in DATA.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    prev = yte.mean()
    bs_ref = prev * (1 - prev)                    # Brier of a model predicting the prevalence
    print(f'\n{"="*78}\n{outcome}: test n={len(yte)}, events={yte.sum()}, prevalence={prev:.4f}')
    print(f'  no-skill AUPRC baseline = {prev:.4f} | no-skill Brier = {bs_ref:.4f}\n{"="*78}')
    sub = res[res['outcome'] == outcome]
    for _, r in sub.iterrows():
        mt = r['key']
        p = ast.literal_eval(r['best_params'])
        m = fit(mt, build(mt, p, ytr), Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]

        auc = roc_auc_score(yte, prob)
        bs = brier_score_loss(yte, prob)
        ap = average_precision_score(yte, prob)
        ap_m, ap_lo, ap_hi = boot(yte, prob, average_precision_score)
        bs_m, bs_lo, bs_hi = boot(yte, prob, brier_score_loss)
        scaled = 1 - bs / bs_ref

        ok_auc = abs(auc - r['test_auc']) < 5e-3
        ok_bs = abs(bs - r['brier_score']) < 5e-3
        flag = 'OK ' if (ok_auc and ok_bs) else 'DIFF'
        print(f'  [{flag}] {LABEL[r["model"]]:33s} AUC {auc:.3f} (pub {r["test_auc"]:.3f}) | '
              f'AUPRC {ap:.3f} [{ap_lo:.3f}-{ap_hi:.3f}] ({ap/prev:.1f}x baseline) | '
              f'BS {bs:.3f} [{bs_lo:.3f}-{bs_hi:.3f}] | scaled BS {scaled:+.3f}')

        rows.append(dict(outcome=outcome, model=LABEL[r['model']], test_auc=auc,
                         published_test_auc=r['test_auc'], reproduced=ok_auc and ok_bs,
                         auprc=ap, auprc_boot_median=ap_m, auprc_lo=ap_lo, auprc_hi=ap_hi,
                         auprc_baseline=prev, auprc_over_baseline=ap / prev,
                         brier=bs, brier_lo=bs_lo, brier_hi=bs_hi,
                         brier_noskill=bs_ref, scaled_brier=scaled))
        if (outcome == 'MCI' and mt == 'SVC') or (outcome == 'Dementia' and mt == 'RF'):
            pr, rc, _ = precision_recall_curve(yte, prob)
            curves[outcome] = (rc, pr, ap, prev)

    # Recalibrated Brier for the final model (Platt on out-of-fold predictions)
    mt = 'SVC' if outcome == 'MCI' else 'RF'
    p = ast.literal_eval(sub[sub['key'] == mt].iloc[0]['best_params'])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build(mt, p, y), X, y, cv=skf, method='predict_proba')[:, 1]
    bs_raw = brier_score_loss(y, oof)
    lr = LogisticRegression(max_iter=1000)
    eps = 1e-6
    logit = np.log(np.clip(oof, eps, 1 - eps) / (1 - np.clip(oof, eps, 1 - eps))).reshape(-1, 1)
    recal = cross_val_predict(lr, logit, y, cv=skf, method='predict_proba')[:, 1]
    bs_recal = brier_score_loss(y, recal)
    p_full = y.mean(); ref_full = p_full * (1 - p_full)
    print(f'  -> final {mt}: out-of-fold Brier {bs_raw:.4f} (scaled {1-bs_raw/ref_full:+.3f}) '
          f'-> after logistic recalibration {bs_recal:.4f} (scaled {1-bs_recal/ref_full:+.3f}); '
          f'no-skill Brier {ref_full:.4f}')
    rows.append(dict(outcome=outcome, model=f'FINAL {mt} (out-of-fold)', test_auc=np.nan,
                     published_test_auc=np.nan, reproduced=np.nan,
                     auprc=average_precision_score(y, oof), auprc_boot_median=np.nan,
                     auprc_lo=np.nan, auprc_hi=np.nan, auprc_baseline=p_full,
                     auprc_over_baseline=average_precision_score(y, oof) / p_full,
                     brier=bs_raw, brier_lo=np.nan, brier_hi=np.nan, brier_noskill=ref_full,
                     scaled_brier=1 - bs_raw / ref_full))
    rows.append(dict(outcome=outcome, model=f'FINAL {mt} (out-of-fold, recalibrated)',
                     test_auc=np.nan, published_test_auc=np.nan, reproduced=np.nan,
                     auprc=average_precision_score(y, recal), auprc_boot_median=np.nan,
                     auprc_lo=np.nan, auprc_hi=np.nan, auprc_baseline=p_full,
                     auprc_over_baseline=average_precision_score(y, recal) / p_full,
                     brier=bs_recal, brier_lo=np.nan, brier_hi=np.nan, brier_noskill=ref_full,
                     scaled_brier=1 - bs_recal / ref_full))

pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
print('\nSaved ->', OUT)

# Figure 8
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
for ax, (oc, title) in zip(axes, [('MCI', 'A  Mild cognitive impairment (SVM)'),
                                  ('Dementia', 'B  Dementia (random forest)')]):
    rc, pr, ap, base = curves[oc]
    ax.step(rc, pr, where='post', color='#1f77b4', lw=1.8,
            label=f'Model (AUPRC = {ap:.3f})')
    ax.axhline(base, color='grey', ls='--', lw=1.2,
               label=f'No-skill baseline ({base:.3f})')
    ax.set_xlabel('Recall (sensitivity)')
    ax.set_ylabel('Precision (positive predictive value)')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.45, pr[:-1].max() * 1.08))
    ax.legend(loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
for ext in ['png', 'pdf']:
    q = os.path.join(FIGDIR, f'Figure8_PRcurve.{ext}')
    fig.savefig(q, dpi=300, bbox_inches='tight')
    print('wrote', q)
