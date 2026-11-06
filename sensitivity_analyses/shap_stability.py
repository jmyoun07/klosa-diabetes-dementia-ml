# -*- coding: utf-8 -*-
"""
Stability of SHAP-based feature importance across cross-validation folds for the two
final models, separating consistently important predictors from artefacts of a
particular data split.

* Repeated stratified 5-fold cross-validation with 4 repeats = 20 folds, i.e. four
  entirely different random partitions of the data.
* In every fold the final model is refitted on the training part with its published
  hyperparameters, and SHAP values are computed on the held-out part only:
    - dementia  : TreeExplainer on the random forest, all held-out rows
    - MCI       : KernelExplainer on the SVM, 100 background training rows and a
                  random sample of 100 held-out rows (matching the main analysis)
* Per fold, the importance of a feature is its mean absolute SHAP value, giving a
  20 x 22 matrix per outcome, from which each feature's mean, standard deviation,
  95% interval across folds, mean rank and its SD, and the proportion of folds in
  which the feature reached the top 10 and top 5 are reported.
* Fold-to-fold agreement is summarised by the mean pairwise Spearman and Kendall
  correlation between the 20 importance vectors.
* The ranking from the single seed-42 split used in the manuscript is compared with
  the across-fold ranking.

Outputs -> shap_stability_{mci,dementia}.csv, shap_stability_summary.csv,
           figures/FigureS1_SHAP_stability.{png,pdf}
"""
import os, pickle, time, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))
SEED, N_REPEATS, N_SPLITS = 42, 4, 5
BG_N, SVM_ROWS, NSAMPLES = 100, 100, 150

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.titlesize': 10.5,
                     'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})

PRETTY = {
    'age': 'Age', 'sex': 'Sex', 'marital': 'Marital status', 'edu': 'Educational attainment',
    'living': 'Living arrangement', 'rural': 'Region (rural)', 'income': 'Household income',
    'adl_score': 'ADL score', 'iadl_score': 'IADL score', 'grip_decreased': 'Grip strength decreased',
    'drinking': 'Current drinking', 'smoking': 'Current smoking', 'exercise': 'Regular exercise',
    'bmi_cat': 'BMI category', 'fall_exp': 'Fall experience',
    'work_participation': 'Work participation', 'cesd_total': 'Depressive symptoms (CES-D 10)',
    'hearing': 'Hearing difficulty', 'subjective_health': 'Subjective health',
    'stroke': 'Stroke treatment', 'heart_disease': 'Heart disease treatment',
    'hypertension': 'Hypertension treatment',
}

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
FEAT = list(pre['feature_cols'])
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


def shap_abs_mean(outcome, Xtr, ytr, Xte, rng):
    """mean |SHAP| per feature on the held-out rows of one fold."""
    if outcome == 'Dementia':
        m = dem_model().fit(Xtr, ytr)
        sv = shap.TreeExplainer(m).shap_values(Xte)
        sv = np.asarray(sv)
        if sv.ndim == 3:                      # (n, p, classes) or (classes, n, p)
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
        return np.abs(sv).mean(axis=0)
    m = mci_model().fit(Xtr, ytr)
    bg = Xtr[rng.choice(len(Xtr), min(BG_N, len(Xtr)), replace=False)]
    rows = Xte[rng.choice(len(Xte), min(SVM_ROWS, len(Xte)), replace=False)]
    ex = shap.KernelExplainer(lambda d: m.predict_proba(d)[:, 1], bg)
    sv = np.asarray(ex.shap_values(rows, nsamples=NSAMPLES, silent=True))
    return np.abs(sv).mean(axis=0)


def single_split_ranking(outcome, X, y):
    """Ranking from the seed-42 70:30 split used in the manuscript."""
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    rng = np.random.RandomState(SEED)
    imp = shap_abs_mean(outcome, Xtr, ytr, Xte, rng)
    return imp


summary, tables = [], {}
for outcome, (X, y) in DATA.items():
    print(f'\n{"="*84}\n{outcome}: n={len(y)}, events={y.sum()}\n{"="*84}', flush=True)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    M = []
    for k, (tr, te) in enumerate(rskf.split(X, y), 1):
        t0 = time.time()
        rng = np.random.RandomState(SEED + k)
        M.append(shap_abs_mean(outcome, X[tr], y[tr], X[te], rng))
        print(f'  fold {k:2d}/{N_SPLITS*N_REPEATS} done ({time.time()-t0:.0f}s)', flush=True)
    M = np.array(M)                                     # folds x features

    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)   # 1 = most important
    top10 = (ranks <= 10).mean(axis=0)
    top5 = (ranks <= 5).mean(axis=0)

    df = pd.DataFrame({
        'feature': [PRETTY[f] for f in FEAT],
        'mean_abs_shap': M.mean(0), 'sd': M.std(0, ddof=1),
        'ci_lo': np.percentile(M, 2.5, axis=0), 'ci_hi': np.percentile(M, 97.5, axis=0),
        'cv_pct': 100 * M.std(0, ddof=1) / M.mean(0),
        'mean_rank': ranks.mean(0), 'rank_sd': ranks.std(0, ddof=1),
        'rank_min': ranks.min(0).astype(int), 'rank_max': ranks.max(0).astype(int),
        'pct_folds_top10': 100 * top10, 'pct_folds_top5': 100 * top5,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    # single-split (seed 42) ranking used in the manuscript
    imp42 = single_split_ranking(outcome, X, y)
    r42 = stats.rankdata(-imp42)
    pretty_feat = [PRETTY[f] for f in FEAT]
    rank42_by_name = {pretty_feat[i]: int(r42[i]) for i in range(len(FEAT))}
    df['rank_single_split_seed42'] = [rank42_by_name[name] for name in df.feature]

    # fold-to-fold agreement
    sp = [stats.spearmanr(M[i], M[j]).statistic for i, j in itertools.combinations(range(len(M)), 2)]
    kt = [stats.kendalltau(M[i], M[j]).statistic for i, j in itertools.combinations(range(len(M)), 2)]
    top10_overall = set(df.feature.head(10))
    top10_single = set(np.array([PRETTY[f] for f in FEAT])[np.argsort(-imp42)[:10]])
    overlap = len(top10_overall & top10_single)

    print(f'\n  mean pairwise Spearman {np.mean(sp):.3f}, Kendall {np.mean(kt):.3f}')
    print(f'  top-10 overlap between the across-fold ranking and the single seed-42 split: {overlap}/10')
    print(f'\n  {"feature":32s} {"mean|SHAP|":>10s} {"SD":>8s} {"95% CI":>18s} {"rank (SD)":>12s} '
          f'{"range":>8s} {"%top10":>7s}')
    for _, r in df.head(10).iterrows():
        print(f'  {r.feature:32s} {r.mean_abs_shap:10.4f} {r.sd:8.4f} '
              f'[{r.ci_lo:.4f}, {r.ci_hi:.4f}] {r.mean_rank:6.1f} ({r.rank_sd:.1f}) '
              f'{int(r.rank_min):3d}-{int(r.rank_max):<3d} {r.pct_folds_top10:6.0f}%')

    df.to_csv(os.path.join(HERE, f'shap_stability_{outcome.lower()}.csv'),
              index=False, encoding='utf-8-sig')
    tables[outcome] = df
    top1_idx = pretty_feat.index(df.feature.iloc[0])
    summary.append(dict(outcome=outcome, n_folds=len(M),
                        mean_pairwise_spearman=np.mean(sp), mean_pairwise_kendall=np.mean(kt),
                        top10_overlap_with_single_split=overlap,
                        top1_feature=df.feature.iloc[0],
                        top1_pct_folds_rank1=100 * (ranks[:, top1_idx] == 1).mean()))

pd.DataFrame(summary).to_csv(os.path.join(HERE, 'shap_stability_summary.csv'),
                             index=False, encoding='utf-8-sig')

# Supplementary Figure S1
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
for ax, (oc, title) in zip(axes, [('MCI', 'A  Mild cognitive impairment (SVM)'),
                                  ('Dementia', 'B  Dementia (random forest)')]):
    d = tables[oc].head(10).iloc[::-1]
    ypos = np.arange(len(d))
    ax.barh(ypos, d.mean_abs_shap, color='#1f77b4', alpha=0.85, height=0.62)
    ax.errorbar(d.mean_abs_shap, ypos,
                xerr=[np.clip(d.mean_abs_shap - d.ci_lo, 0, None),
                      np.clip(d.ci_hi - d.mean_abs_shap, 0, None)],
                fmt='none', ecolor='#c0392b', elinewidth=1.0, capsize=2.5)
    ax.set_yticks(ypos); ax.set_yticklabels(d.feature)
    ax.set_xlabel('Mean |SHAP| across 20 cross-validation folds')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(ax.get_xlim()[1] * 0.985, i, f'{r.pct_folds_top10:.0f}%',
                va='center', ha='right', fontsize=7.5, color='0.35')
fig.tight_layout()
for ext in ['png', 'pdf']:
    q = os.path.join(FIGDIR, f'FigureS1_SHAP_stability.{ext}')
    fig.savefig(q, dpi=300, bbox_inches='tight')
    print('\nwrote', q)
from PIL import Image
print('png size:', Image.open(os.path.join(FIGDIR, 'FigureS1_SHAP_stability.png')).size)
