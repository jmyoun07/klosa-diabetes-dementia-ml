"""
Cutoff-strategy and decision-curve analysis.
 - Operating points at clinically relevant sensitivity levels (0.80, 0.90, 0.95)
   vs the Youden index, with specificity, PPV, NPV and false-positive burden.
 - Decision curve analysis (net benefit) over the screening-relevant threshold range.
 - Practical implication of low PPV (false positives per true case detected).

Computed on 5-fold cross-validated out-of-fold predictions (33 MCI / 50 dementia
events pooled), consistent with the calibration analysis. For decision curve
analysis the predictions are first logistically recalibrated, so that threshold
probabilities correspond to risk (the dementia model is otherwise miscalibrated).
"""
import pickle, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT  = os.path.join(BASE, 'final_model', 'output')
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

def cv_oof(make, X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = make(); m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    return oof

def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    se = tp / (tp + fn) if tp + fn else 0; sp = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0; npv = tn / (tn + fn) if tn + fn else 0
    flagged = tp + fp
    return dict(thr=thr, se=se, sp=sp, ppv=ppv, npv=npv, tp=tp, fp=fp, fn=fn, tn=tn,
                flagged=flagged, pct_flagged=flagged / len(y),
                fp_per_tp=(fp / tp if tp else np.nan), nns=(flagged / tp if tp else np.nan))

def thr_for_sensitivity(y, p, target):
    """Highest threshold achieving sensitivity >= target (most specific such cutoff)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return thr[ok[0]] if len(ok) else thr[-1]

def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]

def recalibrate(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    import statsmodels.api as sm
    m = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    a, b = m.params
    return 1 / (1 + np.exp(-(a + b * lp)))

def net_benefit(y, p, pt):
    pred = (p >= pt).astype(int); n = len(y)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return tp / n - fp / n * (pt / (1 - pt))

rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (label, X, y, make) in zip(axes, [
        ('MCI (SVM)', X_mci, y_mci, mci_model),
        ('Dementia (RF)', X_dem, y_dem, dem_model)]):
    print('\n' + '=' * 70); print(f'[{label}]  prevalence={y.mean():.4f}'); print('=' * 70)
    p = cv_oof(make, X, y)

    pts = [('Youden', youden_thr(y, p))]
    for s in (0.80, 0.90, 0.95):
        pts.append((f'Se>={s:.2f}', thr_for_sensitivity(y, p, s)))
    print(f'  {"strategy":10s} {"thr":>7s} {"Se":>6s} {"Sp":>6s} {"PPV":>6s} {"NPV":>6s} '
          f'{"flagged":>8s} {"%flag":>6s} {"FP/TP":>7s} {"NNS":>6s}')
    for name, thr in pts:
        m = metrics_at(y, p, thr)
        print(f'  {name:10s} {thr:7.3f} {m["se"]:6.3f} {m["sp"]:6.3f} {m["ppv"]:6.3f} '
              f'{m["npv"]:6.3f} {m["flagged"]:8d} {m["pct_flagged"]*100:5.1f}% '
              f'{m["fp_per_tp"]:7.1f} {m["nns"]:6.1f}')
        rows.append(dict(model=label, strategy=name, **m))

    # Decision curve analysis (recalibrated probabilities)
    p_rc = recalibrate(y, p); prev = y.mean()
    grid = np.linspace(0.005, 0.20, 80)
    nb_model = [net_benefit(y, p_rc, t) for t in grid]
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in grid]
    ax.plot(grid, nb_model, '-', color='steelblue', lw=2, label='Model')
    ax.plot(grid, nb_all, '--', color='gray', lw=1, label='Screen all')
    ax.axhline(0, color='black', lw=1, label='Screen none')
    ax.set_xlim(0, 0.20); ax.set_ylim(min(-0.01, prev - 0.05), prev + 0.005)
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
    ax.set_title(label); ax.legend(fontsize=8, loc='upper right')
    print(f'  DCA net benefit (recalibrated): pt=0.03 -> model={net_benefit(y,p_rc,0.03):.4f}, '
          f'all={prev-(1-prev)*(0.03/0.97):.4f}; pt=0.05 -> model={net_benefit(y,p_rc,0.05):.4f}, '
          f'all={prev-(1-prev)*(0.05/0.95):.4f}')

plt.tight_layout()
figpath = os.path.join(BASE, 'final_model', 'figures', 'Figure_DCA.png')
fig.savefig(figpath, dpi=200, bbox_inches='tight'); fig.savefig(figpath.replace('.png', '.pdf'), bbox_inches='tight')
print('\nSaved figure ->', figpath)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'threshold_dca_results.csv'), index=False, encoding='utf-8-sig')
print('Saved ->', os.path.join(OUT, 'threshold_dca_results.csv'))
