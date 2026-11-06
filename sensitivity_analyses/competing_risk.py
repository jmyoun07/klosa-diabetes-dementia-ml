# -*- coding: utf-8 -*-
"""
Competing-risk sensitivity analysis for the dementia cohort. In an elderly diabetic
cohort death is a major competing event; if those who die are coded as non-events, a
model may learn predictors of survival rather than of cognitive decline.

The raw KLoSA release used here (w07-w10) has no respondent-level vital-status
variable. Death is recorded only in the separate KLoSA deceased-respondent (exit)
survey, which is not part of this dataset. Death is therefore indistinguishable from
other non-response, and 34 participants re-entered after missing a wave, so absence is
not equivalent to death. The observable competing event is non-ascertainment = death
or loss to follow-up.

Analyses performed on the dementia cohort (n = 1,255; 50 events):
  (A) Descriptive accounting of the competing event.
  (B) Aalen-Johansen cumulative incidence (competing-risk aware) vs naive 1-KM.
  (C) Fine-Gray subdistribution Cox vs cause-specific Cox; comparison of the
      predictor rankings.
  (D) Composite outcome ("dementia-free survival") random survival forest vs
      dementia-only random survival forest; comparison of permutation importance.
  (E) An attrition model (random forest predicting non-ascertainment); its
      permutation importance is compared with that of the final dementia model.
  (F) Restriction to participants whose 6-year dementia status is fully ascertained
      (event observed, or seen at the final wave).

Outputs -> competing_risk_results.csv (+ console log)
"""
import os, pickle, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from scipy import stats

from lifelines import CoxPHFitter, AalenJohansenFitter, KaplanMeierFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(HERE, 'competing_risk_results.csv')
WY = {'w08': 2.0, 'w09': 4.0, 'w10': 6.0}
SEED = 42


def rd(f, cols):
    df, _ = pyreadstat.read_sav(os.path.join(ROOT, f), usecols=cols)
    return df


# cohort
df7 = rd('w07_20260227.sav', ['pid', 'w07A002_age', 'w07C014', 'w07Cadd_01', 'w07gender1', 'w07F001'])
c = df7[df7['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]
for w, f in [('w08', 'w08_20260227.sav'), ('w09', 'w09_20260227.sav'), ('w10', 'w10_20260317.sav')]:
    t = rd(f, ['pid', f'{w}Cadd_01']).rename(columns={f'{w}Cadd_01': f'cad_{w}'})
    c = c.merge(t, on='pid', how='left')

dem_pids = c[c['w07Cadd_01'].isin([3, 5])]['pid'].tolist()
c = c[c['pid'].isin(dem_pids)].set_index('pid').loc[dem_pids].reset_index()

with open(os.path.join(HERE, '_recheck_preprocessed.pkl'), 'rb') as f:
    pre = pickle.load(f)
X = np.asarray(pre['X_dem'], float)
y_bin = np.asarray(pre['y_dem']).astype(int)
FEAT = list(pre['feature_cols'])
assert X.shape[0] == len(c) == 1255

# (A) competing-event coding
status, time, last_obs, seen_final = [], [], [], []
for _, r in c.iterrows():
    ev, t_ev, lo = 0, None, 0.0
    for w in ['w08', 'w09', 'w10']:
        v = r[f'cad_{w}']
        if pd.isna(v):
            continue
        lo = WY[w]
        if v == 1 and ev == 0:
            ev, t_ev = 1, WY[w]
            break
    fin = pd.notna(r['cad_w10'])
    last_obs.append(lo); seen_final.append(fin)
    if ev:
        status.append(1); time.append(t_ev)
    elif fin:
        status.append(0); time.append(6.0)          # administratively censored
    else:
        status.append(2); time.append(min(lo + 2.0, 6.0))   # non-ascertainment

status = np.array(status); time = np.array(time, float)
last_obs = np.array(last_obs); seen_final = np.array(seen_final)

print('=' * 74)
print('(A) Competing-event accounting, dementia cohort (n=%d)' % len(c))
print('=' * 74)
print(f'  incident dementia (event of interest) : {(status==1).sum():4d} ({(status==1).mean()*100:.1f}%)')
print(f'  non-ascertainment (death or dropout)  : {(status==2).sum():4d} ({(status==2).mean()*100:.1f}%)')
print(f'    of which never reassessed at all    : {((status==2)&(last_obs==0)).sum():4d}')
print(f'    lost after wave 8 (2 y)             : {((status==2)&(last_obs==2)).sum():4d}')
print(f'    lost after wave 9 (4 y)             : {((status==2)&(last_obs==4)).sum():4d}')
print(f'  administratively censored at 6 y      : {(status==0).sum():4d} ({(status==0).mean()*100:.1f}%)')
print(f'  agreement with binary outcome coding  : {int(((status==1).astype(int)==y_bin).all())} (1 = identical)')

# (B) cumulative incidence
print('\n' + '=' * 74)
print('(B) 6-year cumulative incidence of dementia')
print('=' * 74)
aj = AalenJohansenFitter(calculate_variance=False, seed=SEED)
aj.fit(time, status, event_of_interest=1)
cif = float(aj.cumulative_density_.iloc[-1, 0])
km = KaplanMeierFitter().fit(time, (status == 1).astype(int))
naive = float(1 - km.survival_function_.iloc[-1, 0])
crude = float((status == 1).mean())
print(f'  Aalen-Johansen CIF (competing risk aware) : {cif:.4f}')
print(f'  naive 1 - Kaplan-Meier (censors competing) : {naive:.4f}')
print(f'  crude proportion (binary outcome used)     : {crude:.4f}')
print(f'  naive estimator over-states risk by         : {(naive-cif)/cif*100:.1f}% (relative)')

# (C) Fine-Gray vs cause-specific
print('\n' + '=' * 74)
print('(C) Fine-Gray subdistribution vs cause-specific Cox')
print('=' * 74)
Xs = StandardScaler().fit_transform(X)
dfX = pd.DataFrame(Xs, columns=FEAT)

# cause-specific: competing event -> censored at its time; drop zero time-at-risk
cs = dfX.copy(); cs['T'] = time; cs['E'] = (status == 1).astype(int)
cs = cs[cs['T'] > 0]
cph_cs = CoxPHFitter(penalizer=0.1).fit(cs, 'T', 'E')

# Fine-Gray: subjects with the competing event stay in the risk set to the horizon.
# Censoring here is administrative only (at 6 y), so the IPCW weights are 1 and the
# subdistribution risk set is obtained by censoring competing events at 6 y.
fg = dfX.copy(); fg['T'] = np.where(status == 1, time, 6.0); fg['E'] = (status == 1).astype(int)
cph_fg = CoxPHFitter(penalizer=0.1).fit(fg, 'T', 'E')

comp = pd.DataFrame({'cause_specific': cph_cs.params_, 'fine_gray': cph_fg.params_})
comp['abs_cs'] = comp['cause_specific'].abs()
comp['abs_fg'] = comp['fine_gray'].abs()
rho_coef = stats.spearmanr(comp['abs_cs'], comp['abs_fg']).statistic
pear = stats.pearsonr(comp['cause_specific'], comp['fine_gray'])[0]
same_sign = (np.sign(comp['cause_specific']) == np.sign(comp['fine_gray'])).mean()
print(f'  Spearman rho of |coefficient| rankings      : {rho_coef:.3f}')
print(f'  Pearson r of coefficients                   : {pear:.3f}')
print(f'  proportion of predictors with same sign     : {same_sign*100:.1f}%')
print('\n  top 8 by |cause-specific coefficient|:')
for k, r in comp.sort_values('abs_cs', ascending=False).head(8).iterrows():
    print(f'    {k:22s} cause-specific {r.cause_specific:+.3f}   Fine-Gray {r.fine_gray:+.3f}')

# (D) composite outcome RSF
print('\n' + '=' * 74)
print('(D) Composite outcome (dementia-free survival) vs dementia-only RSF')
print('=' * 74)


def rsf_importance(Xa, ev, tm, label):
    keep = tm > 0
    Xk, evk, tmk = Xa[keep], ev[keep].astype(bool), tm[keep]
    yv = Surv.from_arrays(event=evk, time=tmk)
    m = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15,
                             max_features='sqrt', random_state=SEED, n_jobs=-1)
    m.fit(Xk, yv)
    base = concordance_index_censored(evk, tmk, m.predict(Xk))[0]
    rng = np.random.RandomState(SEED); imp = []
    for j in range(Xk.shape[1]):
        d = []
        for _ in range(5):
            Xp = Xk.copy(); rng.shuffle(Xp[:, j])
            d.append(base - concordance_index_censored(evk, tmk, m.predict(Xp))[0])
        imp.append(np.mean(d))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cs_ = []
    for tr, te in skf.split(Xk, evk):
        mm = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15,
                                  max_features='sqrt', random_state=SEED, n_jobs=-1)
        mm.fit(Xk[tr], yv[tr])
        cs_.append(concordance_index_censored(evk[te], tmk[te], mm.predict(Xk[te]))[0])
    print(f'  {label:34s} n={keep.sum():4d} events={int(evk.sum()):3d}  '
          f'CV C-index {np.mean(cs_):.3f} (SD {np.std(cs_):.3f})')
    return np.array(imp), float(np.mean(cs_))


ev_dem = (status == 1).astype(int)
t_dem = np.where(status == 1, time, np.where(status == 2, time, 6.0))
t_dem = np.where(status == 2, last_obs, t_dem)          # cause-specific: censor at last obs
imp_dem, c_dem = rsf_importance(X, ev_dem, t_dem, 'dementia only (cause-specific)')

ev_comp = ((status == 1) | (status == 2)).astype(int)   # dementia OR non-ascertainment
t_comp = np.where(status == 0, 6.0, time)
imp_comp, c_comp = rsf_importance(X, ev_comp, t_comp, 'composite (dementia-free survival)')

rho_dc = stats.spearmanr(imp_dem, imp_comp).statistic
print(f'\n  Spearman rho of permutation importance (dementia vs composite): {rho_dc:.3f}')
top_dem = [FEAT[j] for j in np.argsort(imp_dem)[::-1][:5]]
top_comp = [FEAT[j] for j in np.argsort(imp_comp)[::-1][:5]]
print(f'  top-5 dementia  : {top_dem}')
print(f'  top-5 composite : {top_comp}')

# (E) explicit attrition model
print('\n' + '=' * 74)
print('(E) Attrition model (predicting non-ascertainment) vs final dementia model')
print('=' * 74)


def rf():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)


y_att = (status == 2).astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
p_att = cross_val_predict(rf(), X, y_att, cv=skf, method='predict_proba')[:, 1]
p_dem = cross_val_predict(rf(), X, y_bin, cv=skf, method='predict_proba')[:, 1]
print(f'  attrition model  cross-validated AUC : {roc_auc_score(y_att, p_att):.3f}  (events {y_att.sum()})')
print(f'  dementia model   cross-validated AUC : {roc_auc_score(y_bin, p_dem):.3f}  (events {y_bin.sum()})')
rho_risk = stats.spearmanr(p_att, p_dem).statistic
print(f'  Spearman rho between the two predicted risks : {rho_risk:.3f}')

m_att = rf().fit(X, y_att)
m_dem = rf().fit(X, y_bin)
pi_att = permutation_importance(m_att, X, y_att, n_repeats=10, random_state=SEED,
                                scoring='roc_auc', n_jobs=-1).importances_mean
pi_dem = permutation_importance(m_dem, X, y_bin, n_repeats=10, random_state=SEED,
                                scoring='roc_auc', n_jobs=-1).importances_mean
rho_imp = stats.spearmanr(pi_att, pi_dem).statistic
print(f'  Spearman rho between the two importance profiles : {rho_imp:.3f}')
print(f'  top-5 attrition : {[FEAT[j] for j in np.argsort(pi_att)[::-1][:5]]}')
print(f'  top-5 dementia  : {[FEAT[j] for j in np.argsort(pi_dem)[::-1][:5]]}')

# (F) fully ascertained subset
print('\n' + '=' * 74)
print('(F) Restriction to participants with fully ascertained 6-year status')
print('=' * 74)
full = (status == 1) | seen_final
Xf, yf = X[full], y_bin[full]
print(f'  n = {full.sum()}  events = {int(yf.sum())} ({yf.mean()*100:.1f}%)')
xtr, xte, ytr, yte = train_test_split(Xf, yf, test_size=0.3, stratify=yf, random_state=SEED)
mf = rf().fit(xtr, ytr)
pte = mf.predict_proba(xte)[:, 1]
auc_f = roc_auc_score(yte, pte)
rng = np.random.RandomState(SEED); b = []
for _ in range(500):
    idx = resample(np.arange(len(yte)), random_state=rng)
    if len(np.unique(yte[idx])) < 2:
        continue
    b.append(roc_auc_score(yte[idx], pte[idx]))
b = np.array(b)
print(f'  held-out AUC {auc_f:.3f} | bootstrap median {np.median(b):.3f} '
      f'[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]')
mfa = rf().fit(Xf, yf)
pi_f = permutation_importance(mfa, Xf, yf, n_repeats=10, random_state=SEED,
                              scoring='roc_auc', n_jobs=-1).importances_mean
rho_f = stats.spearmanr(pi_f, pi_dem).statistic
print(f'  Spearman rho of importance vs full-cohort dementia model : {rho_f:.3f}')
print(f'  top-5 : {[FEAT[j] for j in np.argsort(pi_f)[::-1][:5]]}')

# save
rows = [dict(
    n=len(c), events=int((status == 1).sum()),
    competing_n=int((status == 2).sum()), competing_pct=float((status == 2).mean() * 100),
    censored_n=int((status == 0).sum()),
    aalen_johansen_cif6=cif, naive_1_minus_km=naive, crude_proportion=crude,
    fg_vs_cs_spearman_absbeta=rho_coef, fg_vs_cs_pearson_beta=pear, fg_vs_cs_same_sign=same_sign,
    rsf_cindex_dementia=c_dem, rsf_cindex_composite=c_comp,
    rsf_importance_spearman_dem_vs_composite=rho_dc,
    attrition_auc=roc_auc_score(y_att, p_att), dementia_auc=roc_auc_score(y_bin, p_dem),
    risk_spearman_attrition_vs_dementia=rho_risk,
    importance_spearman_attrition_vs_dementia=rho_imp,
    fully_ascertained_n=int(full.sum()), fully_ascertained_events=int(yf.sum()),
    fully_ascertained_auc=auc_f, fully_ascertained_boot_lo=float(np.percentile(b, 2.5)),
    fully_ascertained_boot_hi=float(np.percentile(b, 97.5)),
    importance_spearman_fullascertained_vs_all=rho_f,
    top5_dementia='; '.join([FEAT[j] for j in np.argsort(pi_dem)[::-1][:5]]),
    top5_attrition='; '.join([FEAT[j] for j in np.argsort(pi_att)[::-1][:5]]),
    top5_composite='; '.join(top_comp),
)]
pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
print('\nSaved ->', OUT)
