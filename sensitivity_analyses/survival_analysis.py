# -*- coding: utf-8 -*-
"""
Time-to-event (survival) sensitivity analysis for incident MCI / dementia. The
primary analysis models these as a binary outcome over a fixed maximum six-year
horizon (wave 7 baseline, 2018 -> wave 10, 2024); because the number of completed
follow-up assessments varies across participants (waves 8/9/10 = 2/4/6 years), a
survival-based approach that accounts for varying time-at-risk is applied here.

  (1) Reconstructs the two analytic cohorts as klosa_preprocessing.py does
      (age >= 45, treated diabetes, baseline non-dementia; MCI cohort additionally
      baseline-normal), keeping pid so that time-to-event can be derived.
  (2) Derives time to first qualifying physician diagnosis, right-censoring at the
      last wave at which the cognitive-status item (Cadd_01) was observed.
  (3) Quantifies person-years and time-at-risk heterogeneity.
  (4) Fits a Random Survival Forest and a Cox proportional-hazards model on the same
      22 predictors, with 5-fold cross-validated Harrell C-index and integrated
      Brier score, and compares the RSF risk ranking with the binary final models.

Outputs -> survival_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))          # .../MCI_data
OUT = os.path.join(HERE, 'survival_results.csv')

WAVE_YEARS = {'w08': 2.0, 'w09': 4.0, 'w10': 6.0}


# 1. Reconstruct cohorts with pid (mirrors klosa_preprocessing.py STEP 1-3)
def rd(fname, cols):
    df, _ = pyreadstat.read_sav(os.path.join(ROOT, fname), usecols=cols)
    return df


df7 = rd('w07_20260227.sav', ['pid', 'w07A002_age', 'w07C014', 'w07Cadd_01',
                              'w07gender1', 'w07F001'])
fu = {w: rd(f'{w}_20260227.sav', ['pid', f'{w}Cadd_01']) for w in ['w08', 'w09']}
fu['w10'] = rd('w10_20260317.sav', ['pid', 'w10Cadd_01'])

c = df7[df7['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]
for w in ['w08', 'w09', 'w10']:
    c = c.merge(fu[w].rename(columns={f'{w}Cadd_01': f'cad_{w}'}), on='pid', how='left')

mci_pids = c[c['w07Cadd_01'] == 5]['pid'].tolist()
dem_pids = c[c['w07Cadd_01'].isin([3, 5])]['pid'].tolist()
print(f'Cohort reconstruction: MCI n={len(mci_pids)}   dementia n={len(dem_pids)}')

# align with the preprocessed feature matrices used in the manuscript
with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
X_mci = np.asarray(pre['X_mci'], float)
X_dem = np.asarray(pre['X_dem'], float)
y_mci = np.asarray(pre['y_mci']).astype(int)
y_dem = np.asarray(pre['y_dem']).astype(int)
FEAT = list(pre['feature_cols'])
assert X_mci.shape[0] == len(mci_pids), (X_mci.shape, len(mci_pids))
assert X_dem.shape[0] == len(dem_pids), (X_dem.shape, len(dem_pids))


# 2. Derive time-to-event with right censoring at last observed wave
def surv_times(sub, target):
    """target: 3 = MCI, 1 = dementia.  Returns (event, time, n_obs_waves)."""
    ev, tm, nobs = [], [], []
    for _, r in sub.iterrows():
        t_last, event, t_event, n = 0.0, 0, None, 0
        for w in ['w08', 'w09', 'w10']:
            v = r[f'cad_{w}']
            if pd.isna(v):
                continue
            n += 1
            t_last = WAVE_YEARS[w]
            if v == target and event == 0:
                event, t_event = 1, WAVE_YEARS[w]
                break
        ev.append(event)
        tm.append(t_event if event else t_last)
        nobs.append(n)
    return np.array(ev), np.array(tm, float), np.array(nobs)


c_mci = c[c['pid'].isin(mci_pids)].set_index('pid').loc[mci_pids].reset_index()
c_dem = c[c['pid'].isin(dem_pids)].set_index('pid').loc[dem_pids].reset_index()

e_mci, t_mci, n_mci = surv_times(c_mci, 3)
e_dem, t_dem, n_dem = surv_times(c_dem, 1)

print(f'\nEvent counts vs binary outcome:  MCI {e_mci.sum()} (binary {y_mci.sum()}) | '
      f'dementia {e_dem.sum()} (binary {y_dem.sum()})')


def fu_summary(label, ev, tm, nobs):
    zero = (tm == 0)
    print(f'\n--- {label} ---')
    print(f'  events {ev.sum()}  censored {(~ev.astype(bool)).sum()}')
    print(f'  zero time-at-risk (never re-assessed): {zero.sum()} ({zero.mean()*100:.1f}%)')
    obs = tm[~zero]
    print(f'  among those with >0 follow-up: median {np.median(obs):.1f} y, '
          f'mean {obs.mean():.2f} y, range {obs.min():.0f}-{obs.max():.0f}')
    for yv in [2.0, 4.0, 6.0]:
        print(f'    last observed at {yv:.0f} y: {(tm == yv).sum():4d} '
              f'({(tm == yv).mean()*100:4.1f}%)')
    py = tm.sum()
    print(f'  total person-years {py:.0f}; incidence {ev.sum()/py*1000:.1f} per 1000 py')
    return dict(zero_n=int(zero.sum()), zero_pct=float(zero.mean()*100),
                median_fu=float(np.median(obs)), mean_fu=float(obs.mean()),
                person_years=float(py), rate_per_1000py=float(ev.sum()/py*1000))


s_mci = fu_summary('MCI cohort', e_mci, t_mci, n_mci)
s_dem = fu_summary('Dementia cohort', e_dem, t_dem, n_dem)


# 3. Survival models (RSF + Cox) with 5-fold CV C-index and integrated Brier
def eval_survival(label, X, ev, tm, seed=42):
    keep = tm > 0                       # zero time-at-risk contributes no information
    Xs, evs, tms = X[keep], ev[keep].astype(bool), tm[keep]
    print(f'\n===== {label}: survival models (n={keep.sum()}, events={evs.sum()}) =====')
    y = Surv.from_arrays(event=evs, time=tms)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    res = {}
    for name in ['RSF', 'Cox']:
        c_folds, ibs_folds = [], []
        for tr, te in skf.split(Xs, evs):
            sc = StandardScaler().fit(Xs[tr])
            xtr, xte = sc.transform(Xs[tr]), sc.transform(Xs[te])
            if name == 'RSF':
                m = RandomSurvivalForest(n_estimators=300, min_samples_split=10,
                                         min_samples_leaf=15, max_features='sqrt',
                                         random_state=seed, n_jobs=-1)
            else:
                m = CoxPHSurvivalAnalysis(alpha=0.1)
            m.fit(xtr, y[tr])
            risk = m.predict(xte)
            ci = concordance_index_censored(evs[te], tms[te], risk)[0]
            c_folds.append(ci)

            # integrated Brier score over the observable time range of this fold
            try:
                lo = max(tms[tr].min(), tms[te].min()) + 1e-6
                hi = min(tms[tr].max(), tms[te].max()) - 1e-6
                grid = np.linspace(lo, hi, 20)
                surv_fn = m.predict_survival_function(xte)
                preds = np.row_stack([[fn(t) for t in grid] for fn in surv_fn])
                ibs_folds.append(integrated_brier_score(y[tr], y[te], preds, grid))
            except Exception:
                pass

        c_folds = np.array(c_folds)
        print(f'  {name:4s} C-index  mean {c_folds.mean():.3f}  SD {c_folds.std():.3f}  '
              f'folds {np.round(c_folds, 3)}')
        if ibs_folds:
            print(f'       integrated Brier score  mean {np.mean(ibs_folds):.3f}')
        res[name] = dict(cindex_mean=float(c_folds.mean()), cindex_sd=float(c_folds.std()),
                         ibs=float(np.mean(ibs_folds)) if ibs_folds else np.nan)

    # bootstrap CI for RSF C-index on a held-out 30% split
    xtr, xte, ytr_e, yte_e, ytr_t, yte_t = train_test_split(
        Xs, evs, tms, test_size=0.3, stratify=evs, random_state=seed)
    sc = StandardScaler().fit(xtr)
    m = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15,
                             max_features='sqrt', random_state=seed, n_jobs=-1)
    m.fit(sc.transform(xtr), Surv.from_arrays(event=ytr_e, time=ytr_t))
    risk = m.predict(sc.transform(xte))
    rng = np.random.RandomState(seed); boots = []
    for _ in range(500):
        idx = rng.randint(0, len(risk), len(risk))
        if yte_e[idx].sum() < 2:
            continue
        boots.append(concordance_index_censored(yte_e[idx], yte_t[idx], risk[idx])[0])
    boots = np.array(boots)
    print(f'  RSF held-out C-index {concordance_index_censored(yte_e, yte_t, risk)[0]:.3f} '
          f'| bootstrap median {np.median(boots):.3f} '
          f'[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]')
    res['RSF']['boot_med'] = float(np.median(boots))
    res['RSF']['boot_lo'] = float(np.percentile(boots, 2.5))
    res['RSF']['boot_hi'] = float(np.percentile(boots, 97.5))

    # RSF permutation variable importance (top 8)
    sc_all = StandardScaler().fit(Xs)
    m_all = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15,
                                 max_features='sqrt', random_state=seed, n_jobs=-1)
    Xa = sc_all.transform(Xs)
    m_all.fit(Xa, y)
    base = concordance_index_censored(evs, tms, m_all.predict(Xa))[0]
    imps = []
    rng = np.random.RandomState(seed)
    for j in range(Xa.shape[1]):
        drops = []
        for _ in range(5):
            Xp = Xa.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - concordance_index_censored(evs, tms, m_all.predict(Xp))[0])
        imps.append(np.mean(drops))
    order = np.argsort(imps)[::-1][:8]
    print('  RSF permutation importance (top 8):')
    for j in order:
        print(f'    {FEAT[j]:28s} {imps[j]:+.4f}')
    res['top_predictors'] = [FEAT[j] for j in order]
    res['risk_all'] = m_all.predict(Xa)
    res['keep'] = keep
    return res


r_mci = eval_survival('MCI', X_mci, e_mci, t_mci)
r_dem = eval_survival('Dementia', X_dem, e_dem, t_dem)


# 4. Agreement between RSF risk ranking and the binary final models
def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', probability=True,
                                 class_weight='balanced', random_state=42))])


def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=42, n_jobs=-1)


def agreement(label, X, y, make, r):
    m = make(); m.fit(X, y)
    p = m.predict_proba(X)[:, 1][r['keep']]
    rho = stats.spearmanr(p, r['risk_all']).statistic
    print(f'\n{label}: Spearman rho between binary-model predicted probability and '
          f'RSF risk score = {rho:.3f}')
    return float(rho)


rho_mci = agreement('MCI', X_mci, y_mci, mci_model, r_mci)
rho_dem = agreement('Dementia', X_dem, y_dem, dem_model, r_dem)

rows = []
for lab, s, r, rho, ev in [('MCI', s_mci, r_mci, rho_mci, e_mci),
                           ('Dementia', s_dem, r_dem, rho_dem, e_dem)]:
    rows.append(dict(outcome=lab, events=int(ev.sum()), **s,
                     rsf_cindex=r['RSF']['cindex_mean'], rsf_cindex_sd=r['RSF']['cindex_sd'],
                     rsf_ibs=r['RSF']['ibs'],
                     rsf_boot_med=r['RSF']['boot_med'], rsf_boot_lo=r['RSF']['boot_lo'],
                     rsf_boot_hi=r['RSF']['boot_hi'],
                     cox_cindex=r['Cox']['cindex_mean'], cox_cindex_sd=r['Cox']['cindex_sd'],
                     cox_ibs=r['Cox']['ibs'],
                     spearman_vs_binary=rho,
                     top_predictors='; '.join(r['top_predictors'])))
pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
print('\nSaved ->', OUT)
