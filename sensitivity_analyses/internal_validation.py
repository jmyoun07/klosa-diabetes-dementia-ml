"""
Internal-validation robustness analyses.
Guards against the single random 70:30 split driving the results, and reports
optimism-corrected performance via bootstrap internal validation
(Steyerberg & Harrell, ref [35]).

Reproduces the two FINAL models exactly, then adds:
 (A) Reproduction of the single-split (seed=42) test AUC + 500x bootstrap CI.
 (B) Repeated stratified 70:30 splits (200 seeds) -> distribution of test AUC,
     showing the reported split is representative, not a lucky partition.
 (C) Bootstrap optimism-corrected AUC on the FULL dataset (Harrell/Steyerberg),
     B=500, using all data (no dependence on one split).
"""
import pickle, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.utils import resample

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
with open(os.path.join(BASE, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)
X_mci, y_mci = np.asarray(data['X_mci'], float), np.asarray(data['y_mci']).astype(int)
X_dem, y_dem = np.asarray(data['X_dem'], float), np.asarray(data['y_dem']).astype(int)

def mci_model():
    return Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433,
                                 kernel='sigmoid', probability=True,
                                 class_weight='balanced', random_state=42))])
def dem_model():
    return RandomForestClassifier(n_estimators=110, max_depth=4, min_samples_split=6,
                                  min_samples_leaf=5, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
def score(model, X, y):
    """AUC via decision_function (SVC) or predict_proba; monotonic => identical AUC."""
    if hasattr(model, 'decision_function'):
        try:
            return roc_auc_score(y, model.decision_function(X))
        except Exception:
            pass
    return roc_auc_score(y, model.predict_proba(X)[:, 1])

def boot_ci(y, p, n=500, seed=42):
    rng = np.random.RandomState(seed); a=[]
    for _ in range(n):
        idx = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[idx])) < 2: continue
        a.append(roc_auc_score(y[idx], p[idx]))
    a = np.array(a)
    return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

def run(label, X, y, make):
    print('\n' + '='*64); print(f'[{label}]  N={len(y)}  events={int(y.sum())}  rate={y.mean()*100:.2f}%'); print('='*64)

    # (A) Reproduce single split seed=42 (manuscript)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    m = make(); m.fit(Xtr, ytr)
    p_te = m.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(yte, p_te)
    bm, blo, bhi = boot_ci(yte, p_te, 500, 42)
    print(f'(A) Single split (seed 42): test AUC={test_auc:.4f} | '
          f'boot median={bm:.4f} [{blo:.4f}, {bhi:.4f}]  (test n={len(yte)}, events={int(yte.sum())})')

    # (B) Repeated stratified 70:30 splits, fixed tuned hyperparameters
    seeds = range(1000, 1200)  # 200 independent splits
    aucs = []
    for s in seeds:
        xtr, xte, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        mm = make(); mm.fit(xtr, ya)
        aucs.append(score(mm, xte, yb))
    aucs = np.array(aucs)
    print(f'(B) Repeated 70:30 splits (n=200): test AUC '
          f'mean={aucs.mean():.4f}  median={np.median(aucs):.4f}  SD={aucs.std():.4f}  '
          f'2.5-97.5%=[{np.percentile(aucs,2.5):.4f}, {np.percentile(aucs,97.5):.4f}]  '
          f'min={aucs.min():.4f}  max={aucs.max():.4f}')

    # (C) Bootstrap optimism-corrected AUC on FULL dataset (Harrell/Steyerberg)
    full = make(); full.fit(X, y)
    apparent = score(full, X, y)
    B = 500; rng = np.random.RandomState(42); opt = []
    for _ in range(B):
        idx = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[idx])) < 2: continue
        mb = make(); mb.fit(X[idx], y[idx])
        auc_boot = score(mb, X[idx], y[idx])     # performance on bootstrap sample
        auc_orig = score(mb, X, y)               # performance on original sample
        opt.append(auc_boot - auc_orig)
    optimism = float(np.mean(opt))
    corrected = apparent - optimism
    print(f'(C) Optimism-corrected (B=500, full data): apparent AUC={apparent:.4f} | '
          f'optimism={optimism:.4f} | corrected AUC={corrected:.4f}')

    return dict(outcome=label, test_auc_seed42=test_auc, boot_med=bm, boot_lo=blo, boot_hi=bhi,
                rep_mean=aucs.mean(), rep_median=float(np.median(aucs)), rep_sd=aucs.std(),
                rep_lo=np.percentile(aucs,2.5), rep_hi=np.percentile(aucs,97.5),
                rep_min=aucs.min(), rep_max=aucs.max(),
                apparent=apparent, optimism=optimism, corrected=corrected)

rows = [run('MCI (SVM)', X_mci, y_mci, mci_model),
        run('Dementia (RF)', X_dem, y_dem, dem_model)]
out = os.path.join(BASE, 'final_model', 'output', 'internal_validation_results.csv')
pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
print('\nSaved ->', out)
