# -*- coding: utf-8 -*-
"""
Missing-data pattern and sensitivity analyses.
 (1) Compare participants with vs without the (mainly) missing predictor (grip
     change) on age, sex, education, income, and outcome.
 (2) Sensitivity of the two final models to the missing-data strategy:
     primary median/mode imputation vs multiple imputation (MICE, m=10) vs
     complete-case vs grip-excluded.  Reports test AUC and 500x bootstrap CI
     for the seed=42 70:30 split.
"""
import os, pickle, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(BASE, 'final_model', 'output')
pre = pickle.load(open(os.path.join(OUT, 'preimp_data.pkl'), 'rb'))
prim = pickle.load(open(os.path.join(BASE, 'preprocessed_data.pkl'), 'rb'))
FC = pre['feature_cols']
ORD_LIMS = {'income': (1, 5), 'bmi_cat': (1, 4), 'grip_decreased': (0, 1)}

def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433, kernel='sigmoid',
                                 probability=True, class_weight='balanced', random_state=42))])
def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)

def boot(y, p, n=500, seed=42):
    rng = np.random.RandomState(seed); a = []
    for _ in range(n):
        idx = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[idx])) < 2: continue
        a.append(roc_auc_score(y[idx], p[idx]))
    a = np.array(a); return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

def evaluate(X, y, make):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    m = make(); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p); bm, lo, hi = boot(yte, p)
    return auc, bm, lo, hi, len(y), int(y.sum())

def median_impute(df):
    return df.fillna(df.median())

# (1) Missingness pattern (grip)
print("="*70); print("(1) Missingness pattern: grip-strength change (the main missing predictor)"); print("="*70)
for lab, key, ykey in [('MCI', 'X_mci_raw', 'y_mci'), ('Dementia', 'X_dem_raw', 'y_dem')]:
    X = pre[key].reset_index(drop=True); y = pre[ykey]
    miss = X['grip_decreased'].isna().values
    print(f"\n[{lab}] grip missing: {miss.sum()}/{len(X)} ({miss.mean()*100:.1f}%)")
    def c(v):
        a, b = X.loc[~miss, v].dropna(), X.loc[miss, v].dropna()
        p = stats.mannwhitneyu(a, b).pvalue
        print(f"    {v:18s} observed {a.mean():.2f}  vs missing {b.mean():.2f}   p={p:.3f}")
    c('age'); c('edu'); c('income')
    fa, fb = X.loc[~miss, 'sex'].mean(), X.loc[miss, 'sex'].mean()
    print(f"    {'female %':18s} observed {fa*100:.1f}  vs missing {fb*100:.1f}")
    oa, ob = y[~miss].mean(), y[miss].mean()
    tab = np.array([[y[~miss].sum(), (~y[~miss].astype(bool)).sum()],
                    [y[miss].sum(), (~y[miss].astype(bool)).sum()]])
    pp = stats.fisher_exact(tab)[1]
    print(f"    {'outcome rate':18s} observed {oa*100:.2f}%  vs missing {ob*100:.2f}%   p={pp:.3f}")

# (2) Sensitivity analyses
print("\n" + "="*70); print("(2) Sensitivity of final models to missing-data strategy"); print("="*70)
rows = []
for lab, make, ykey in [('MCI (SVM)', mci_model, 'y_mci'), ('Dementia (RF)', dem_model, 'y_dem')]:
    Xraw = pre['X_mci_raw' if 'MCI' in lab else 'X_dem_raw'].reset_index(drop=True)
    y = pre[ykey]
    Xp = (prim['X_mci'] if 'MCI' in lab else prim['X_dem'])
    Xp = np.asarray(Xp, float)
    print(f"\n--- {lab} ---")

    # Primary (median/mode imputation, as reported)
    a, bm, lo, hi, n, ev = evaluate(Xp, y, make)
    print(f"  Primary (median/mode)   AUC={a:.3f}  boot={bm:.3f} [{lo:.3f},{hi:.3f}]  N={n}, events={ev}")
    rows.append((lab, 'Primary (median/mode imputation)', n, ev, a, bm, lo, hi))

    # Multiple imputation (MICE, m=10)
    aucs, bms = [], []
    for s in range(10):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=s)
        Xi = pd.DataFrame(imp.fit_transform(Xraw), columns=FC)
        for col, (loB, hiB) in ORD_LIMS.items():
            Xi[col] = Xi[col].round().clip(loB, hiB)
        a, bm, _, _, _, _ = evaluate(Xi.values, y, make); aucs.append(a); bms.append(bm)
    print(f"  Multiple imputation(m=10) AUC={np.mean(aucs):.3f} (SD {np.std(aucs):.3f})  boot={np.mean(bms):.3f}")
    rows.append((lab, 'Multiple imputation (MICE, m=10)', n, ev, np.mean(aucs), np.mean(bms), np.nan, np.nan))

    # Complete-case (drop rows with any missing predictor)
    cc = Xraw.dropna(); ycc = y[cc.index.values]
    a, bm, lo, hi, ncc, evcc = evaluate(cc.values, ycc, make)
    print(f"  Complete-case            AUC={a:.3f}  boot={bm:.3f} [{lo:.3f},{hi:.3f}]  N={ncc}, events={evcc}")
    rows.append((lab, 'Complete-case', ncc, evcc, a, bm, lo, hi))

    # Grip-excluded (drop the 33%-missing predictor, median-impute the rest)
    Xge = median_impute(Xraw.drop(columns=['grip_decreased']))
    a, bm, lo, hi, n2, ev2 = evaluate(Xge.values, y, make)
    print(f"  Grip-excluded (21 vars)  AUC={a:.3f}  boot={bm:.3f} [{lo:.3f},{hi:.3f}]  N={n2}, events={ev2}")
    rows.append((lab, 'Grip-strength change excluded', n2, ev2, a, bm, lo, hi))

pd.DataFrame(rows, columns=['model','strategy','N','events','test_auc','boot_median','boot_lo','boot_hi']
             ).to_csv(os.path.join(OUT, 'missingness_sensitivity_results.csv'), index=False, encoding='utf-8-sig')
print("\nSaved -> missingness_sensitivity_results.csv")
