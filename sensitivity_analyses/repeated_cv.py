# -*- coding: utf-8 -*-
"""
Resampling-based performance of the eight tuned algorithms via repeated stratified
5-fold cross-validation (20 repeats = 100 folds, i.e. 20 independent partitions of the
data). Within each repeat the out-of-fold predictions are pooled and a single AUC, AUPRC
and Brier score are computed, giving 20 estimates per algorithm from which the mean,
standard deviation and 2.5th-97.5th percentile interval are derived.

To quantify the stability of the algorithm ranking, the algorithms are ranked within every
repeat, and the frequency of rank 1, the mean rank, and the range of ranks are reported for
each. Hyperparameters are the published tuned values (results/ML_results_all8_tuned.csv) and
are held fixed, so this measures the stability of the comparison, not of the tuning.

Outputs -> repeated_cv_all8_results.csv, repeated_cv_all8_ranks.csv (+ console log)
"""
import os, ast, pickle, time, warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results', 'ML_results_all8_tuned.csv'))
SEED, N_SPLITS, N_REPEATS = 42, 5, 20
N_JOBS = int(os.environ.get('N_JOBS', 14))

LABEL = {'LASSO_LR': 'L1 Penalized Logistic Regression', 'SVC': 'Support Vector Machine',
         'RF': 'Random Forest', 'GBM': 'Gradient Boosting Machine',
         'AdaBoost': 'Adaptive Boosting', 'LightGBM': 'Light Gradient-Boosting Machine',
         'XGBoost': 'Extreme Gradient Boosting', 'CatBoost': 'Categorical Boosting'}

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
DATA = {'MCI': (np.asarray(pre['X_mci'], float), np.asarray(pre['y_mci']).astype(int)),
        'Dementia': (np.asarray(pre['X_dem'], float), np.asarray(pre['y_dem']).astype(int))}

res = pd.read_csv(RESULTS)
res['key'] = res['model'].str.replace('_tuned', '', regex=False)


def build(mt, p, y_ref):
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
                                  auto_class_weights='Balanced', random_seed=SEED,
                                  verbose=0, thread_count=1)
    if mt == 'RF':
        return RandomForestClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                                      min_samples_split=p['min_split'],
                                      min_samples_leaf=p['min_leaf'], class_weight='balanced',
                                      random_state=SEED, n_jobs=1)
    if mt == 'LightGBM':
        return LGBMClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                              learning_rate=p['lr'], num_leaves=p['num_leaves'],
                              min_child_samples=p['min_child'], class_weight='balanced',
                              random_state=SEED, verbose=-1, n_jobs=1)
    if mt == 'XGBoost':
        pos = int(y_ref.sum()); neg = len(y_ref) - pos
        return XGBClassifier(n_estimators=p['n_est'], max_depth=p['max_depth'],
                             learning_rate=p['lr'], subsample=p['subsample'],
                             colsample_bytree=p['colsample'],
                             scale_pos_weight=float(neg / pos), random_state=SEED,
                             eval_metric='auc', verbosity=0, n_jobs=1)
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


def one_repeat(outcome, mt, p, X, y, rep, splits):
    """Pool out-of-fold predictions across the 5 folds of one repeat."""
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        m = fit(mt, build(mt, p, y[tr]), X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return (outcome, mt, rep,
            roc_auc_score(y, oof), average_precision_score(y, oof), brier_score_loss(y, oof))


jobs, meta = [], []
for outcome, (X, y) in DATA.items():
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    all_splits = list(rskf.split(X, y))
    per_repeat = [all_splits[i * N_SPLITS:(i + 1) * N_SPLITS] for i in range(N_REPEATS)]
    sub = res[res['outcome'] == outcome]
    for _, r in sub.iterrows():
        mt, p = r['key'], ast.literal_eval(r['best_params'])
        for rep in range(N_REPEATS):
            jobs.append(delayed(one_repeat)(outcome, mt, p, X, y, rep, per_repeat[rep]))

print(f'running {len(jobs)} (algorithm x repeat) tasks on {N_JOBS} workers ...', flush=True)
t0 = time.time()
out = Parallel(n_jobs=N_JOBS, backend='loky', verbose=1)(jobs)
print(f'done in {time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(out, columns=['outcome', 'algo', 'repeat', 'auc', 'auprc', 'brier'])
df.to_csv(os.path.join(HERE, 'repeated_cv_all8_ranks.csv'), index=False, encoding='utf-8-sig')

rows = []
for outcome in DATA:
    d = df[df.outcome == outcome]
    wide = d.pivot(index='repeat', columns='algo', values='auc')
    ranks = wide.rank(axis=1, ascending=False)
    print(f'\n{"="*100}\n{outcome}: repeated stratified {N_SPLITS}-fold CV, {N_REPEATS} repeats '
          f'(pooled out-of-fold AUC per repeat)\n{"="*100}')
    print(f'  {"algorithm":34s} {"AUC mean (SD)":>18s} {"95% interval":>18s} '
          f'{"AUPRC":>8s} {"Brier":>7s} {"rank":>10s} {"rank 1":>7s}')
    order = wide.mean().sort_values(ascending=False).index
    for a in order:
        auc = wide[a]
        ap = d[d.algo == a]['auprc'].mean()
        bs = d[d.algo == a]['brier'].mean()
        rk = ranks[a]
        print(f'  {LABEL[a]:34s} {auc.mean():.3f} ({auc.std(ddof=1):.3f})     '
              f'{np.percentile(auc, 2.5):.3f}-{np.percentile(auc, 97.5):.3f}   '
              f'{ap:.3f}  {bs:.3f}   {rk.mean():4.1f} ({int(rk.min())}-{int(rk.max())})  '
              f'{(rk == 1).mean()*100:5.0f}%')
        rows.append(dict(outcome=outcome, algorithm=LABEL[a], auc_mean=auc.mean(),
                         auc_sd=auc.std(ddof=1), auc_lo=np.percentile(auc, 2.5),
                         auc_hi=np.percentile(auc, 97.5), auprc_mean=ap, brier_mean=bs,
                         mean_rank=rk.mean(), rank_min=int(rk.min()), rank_max=int(rk.max()),
                         pct_repeats_rank1=(rk == 1).mean() * 100))
    top2 = list(order[:2])
    diff = wide[top2[0]] - wide[top2[1]]
    t = stats.ttest_rel(wide[top2[0]], wide[top2[1]])
    print(f'\n  paired comparison of the two leading algorithms ({LABEL[top2[0]]} vs {LABEL[top2[1]]}):')
    print(f'    mean AUC difference {diff.mean():+.4f} (95% interval '
          f'{np.percentile(diff, 2.5):+.4f} to {np.percentile(diff, 97.5):+.4f}), paired p = {t.pvalue:.3f}')
    print(f'    the leading algorithm ranked first in only {(ranks[top2[0]] == 1).mean()*100:.0f}% of repeats')

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'repeated_cv_all8_results.csv'),
                          index=False, encoding='utf-8-sig')
print('\nSaved -> repeated_cv_all8_results.csv, repeated_cv_all8_ranks.csv')
