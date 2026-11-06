"""
Generate final model figures and SHAP importance.
- MCI: SVC_tuned
- Dementia: RF_tuned
- 8 models (LR excluded)
- Figure 1: MCI - (a) 8-model AUC comparison, (b) SHAP importance
- Figure 2: Dementia - (a) 8-model AUC comparison, (b) SHAP importance
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.utils import resample

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
FIG_DIR = os.path.join(BASE, "final_model", "figures")
COEF_DIR = os.path.join(BASE, "final_model", "coefficients")
RESULTS_DIR = os.path.join(BASE, "final_model", "results")
for d in [FIG_DIR, COEF_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Load data
print("데이터 로딩...")
with open(os.path.join(BASE, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_mci, y_mci = data['X_mci'], data['y_mci']
X_dem, y_dem = data['X_dem'], data['y_dem']
FEATURE_COLS = list(data['feature_cols'])

Xtr_mci, Xte_mci, ytr_mci, yte_mci = train_test_split(
    X_mci.values, y_mci.values, test_size=0.3, stratify=y_mci.values, random_state=42)
Xtr_dem, Xte_dem, ytr_dem, yte_dem = train_test_split(
    X_dem.values, y_dem.values, test_size=0.3, stratify=y_dem.values, random_state=42)

# Build final models
print("SVC_tuned (MCI) 모델 구축...")
svc_mci = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=0.015935025356523597, gamma=0.31319323011909433, kernel='sigmoid',
                probability=True, class_weight='balanced', random_state=42))
])
svc_mci.fit(Xtr_mci, ytr_mci)

print("RF_tuned (Dementia) 모델 구축...")
rf_dem = RandomForestClassifier(
    n_estimators=110, max_depth=4, min_samples_split=6, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf_dem.fit(Xtr_dem, ytr_dem)

# SHAP computation
print("SHAP 계산 (MCI SVC) - KernelExplainer (100 samples)...")
# Use background summary for KernelExplainer
scaler_mci = svc_mci.named_steps['scaler']
Xtr_mci_scaled = scaler_mci.transform(Xtr_mci)
Xte_mci_scaled = scaler_mci.transform(Xte_mci)

# Sample background for speed
bg_idx = np.random.RandomState(42).choice(len(Xtr_mci_scaled), size=100, replace=False)
background = Xtr_mci_scaled[bg_idx]

svc_model = svc_mci.named_steps['svc']
explainer_mci = shap.KernelExplainer(svc_model.predict_proba, background)
# Use subset of test for speed
test_sample_idx = np.random.RandomState(42).choice(len(Xte_mci_scaled), size=min(100, len(Xte_mci_scaled)), replace=False)
shap_vals_mci = explainer_mci.shap_values(Xte_mci_scaled[test_sample_idx])
if isinstance(shap_vals_mci, list):
    shap_vals_mci = shap_vals_mci[1]  # positive class

print("SHAP 계산 (Dementia RF) - TreeExplainer...")
explainer_dem = shap.TreeExplainer(rf_dem)
shap_vals_dem = explainer_dem.shap_values(Xte_dem)
if isinstance(shap_vals_dem, list):
    shap_vals_dem = shap_vals_dem[1]
elif shap_vals_dem.ndim == 3:
    shap_vals_dem = shap_vals_dem[:, :, 1]

# Ensure 2D
if shap_vals_mci.ndim == 3:
    shap_vals_mci = shap_vals_mci[:, :, 1]
print(f"  MCI SHAP shape: {shap_vals_mci.shape}")
print(f"  Dem SHAP shape: {shap_vals_dem.shape}")

# Mean absolute SHAP
mci_shap_mean = np.abs(shap_vals_mci).mean(axis=0)
dem_shap_mean = np.abs(shap_vals_dem).mean(axis=0)

df_shap_mci = pd.DataFrame({
    'feature': FEATURE_COLS, 'mean_abs_shap': mci_shap_mean
}).sort_values('mean_abs_shap', ascending=False)

df_shap_dem = pd.DataFrame({
    'feature': FEATURE_COLS, 'mean_abs_shap': dem_shap_mean
}).sort_values('mean_abs_shap', ascending=False)

# Save
df_shap_mci.to_csv(os.path.join(COEF_DIR, 'MCI_SVC_tuned_SHAP.csv'), index=False, encoding='utf-8-sig')
df_shap_dem.to_csv(os.path.join(COEF_DIR, 'Dementia_RF_tuned_SHAP.csv'), index=False, encoding='utf-8-sig')

print("\nMCI SVC_tuned SHAP Top 10:")
for _, row in df_shap_mci.head(10).iterrows():
    print(f"  {row['feature']:25s} {row['mean_abs_shap']:.4f}")

print("\nDementia RF_tuned SHAP Top 10:")
for _, row in df_shap_dem.head(10).iterrows():
    print(f"  {row['feature']:25s} {row['mean_abs_shap']:.4f}")

# Check performance metrics
def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]

def bootstrap_evaluate(y_true, y_prob, n_boot=500):
    boot_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)), random_state=rng)
        yb, pb = y_true[idx], y_prob[idx]
        if len(np.unique(yb)) < 2: continue
        boot_aucs.append(roc_auc_score(yb, pb))
    ba = np.array(boot_aucs)
    return np.median(ba), np.percentile(ba, 2.5), np.percentile(ba, 97.5)

prob_mci = svc_mci.predict_proba(Xte_mci)[:, 1]
auc_mci = roc_auc_score(yte_mci, prob_mci)
thr_mci, se_mci, sp_mci = find_optimal_threshold(yte_mci, prob_mci)
boot_mci = bootstrap_evaluate(yte_mci, prob_mci)
brier_mci = brier_score_loss(yte_mci, prob_mci)
pred_mci = (prob_mci >= thr_mci).astype(int)
tp_m=((pred_mci==1)&(yte_mci==1)).sum(); fp_m=((pred_mci==1)&(yte_mci==0)).sum()
fn_m=((pred_mci==0)&(yte_mci==1)).sum(); tn_m=((pred_mci==0)&(yte_mci==0)).sum()
ppv_mci=tp_m/(tp_m+fp_m) if (tp_m+fp_m)>0 else 0
npv_mci=tn_m/(tn_m+fn_m) if (tn_m+fn_m)>0 else 0

prob_dem = rf_dem.predict_proba(Xte_dem)[:, 1]
auc_dem = roc_auc_score(yte_dem, prob_dem)
thr_dem, se_dem, sp_dem = find_optimal_threshold(yte_dem, prob_dem)
boot_dem = bootstrap_evaluate(yte_dem, prob_dem)
brier_dem = brier_score_loss(yte_dem, prob_dem)
pred_dem = (prob_dem >= thr_dem).astype(int)
tp_d=((pred_dem==1)&(yte_dem==1)).sum(); fp_d=((pred_dem==1)&(yte_dem==0)).sum()
fn_d=((pred_dem==0)&(yte_dem==1)).sum(); tn_d=((pred_dem==0)&(yte_dem==0)).sum()
ppv_dem=tp_d/(tp_d+fp_d) if (tp_d+fp_d)>0 else 0
npv_dem=tn_d/(tn_d+fn_d) if (tn_d+fn_d)>0 else 0

print(f"\n===== 최종 모델 성능 =====")
print(f"MCI SVC_tuned:  AUC={auc_mci:.4f}, Boot={boot_mci[0]:.4f} [{boot_mci[1]:.4f}, {boot_mci[2]:.4f}]")
print(f"  Se={se_mci:.3f}, Sp={sp_mci:.3f}, PPV={ppv_mci:.3f}, NPV={npv_mci:.3f}, Brier={brier_mci:.3f}")
print(f"Dem RF_tuned:   AUC={auc_dem:.4f}, Boot={boot_dem[0]:.4f} [{boot_dem[1]:.4f}, {boot_dem[2]:.4f}]")
print(f"  Se={se_dem:.3f}, Sp={sp_dem:.3f}, PPV={ppv_dem:.3f}, NPV={npv_dem:.3f}, Brier={brier_dem:.3f}")

# Final CSV
final_results = pd.DataFrame([
    {'model':'SVC_tuned_FINAL','outcome':'MCI','test_auc':auc_mci,
     'boot_median':boot_mci[0],'boot_CI95_low':boot_mci[1],'boot_CI95_high':boot_mci[2],
     'sensitivity':se_mci,'specificity':sp_mci,'ppv':ppv_mci,'npv':npv_mci,'brier':brier_mci},
    {'model':'RF_tuned_FINAL','outcome':'Dementia','test_auc':auc_dem,
     'boot_median':boot_dem[0],'boot_CI95_low':boot_dem[1],'boot_CI95_high':boot_dem[2],
     'sensitivity':se_dem,'specificity':sp_dem,'ppv':ppv_dem,'npv':npv_dem,'brier':brier_dem},
])
final_results.to_csv(os.path.join(RESULTS_DIR, 'ML_results_final.csv'), index=False, encoding='utf-8-sig')

# Copy all9 results (without LR)
df_all = pd.read_csv(os.path.join(BASE, 'ML_results_all9_tuned.csv'))
df_all = df_all[df_all['model'] != 'LR_tuned']
df_all.to_csv(os.path.join(RESULTS_DIR, 'ML_results_all8_tuned.csv'), index=False, encoding='utf-8-sig')

# Figure 1: MCI
print("\nFigure 1 생성...")
from matplotlib.patches import Patch

mci_models = [
    ('Support Vector Machine', boot_mci[0], boot_mci[1], boot_mci[2], True),
    ('Random Forest',        0.826, 0.673, 0.932, False),
    ('XGBoost',              0.769, 0.578, 0.916, False),
    ('CatBoost',             0.769, 0.570, 0.910, False),
    ('AdaBoost',             0.723, 0.531, 0.877, False),
    ('Gradient Boosting',    0.655, 0.446, 0.854, False),
    ('LightGBM',             0.648, 0.469, 0.808, False),
    ('LASSO LR',             0.624, 0.417, 0.800, False),
]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})

names = [m[0] for m in mci_models]
aucs = [m[1] for m in mci_models]
ci_lows = [m[2] for m in mci_models]
ci_highs = [m[3] for m in mci_models]
is_finals = [m[4] for m in mci_models]

err_low = [a - l for a, l in zip(aucs, ci_lows)]
err_high = [h - a for a, h in zip(aucs, ci_highs)]
colors = ['#c0392b' if f else '#3498db' for f in is_finals]

ax1.barh(range(len(names)), aucs, color=colors, edgecolor='white', height=0.65)
ax1.errorbar(aucs, range(len(names)), xerr=[err_low, err_high],
             fmt='none', ecolor='gray', capsize=2, linewidth=0.8)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlabel('Bootstrap AUC (95% CI)', fontsize=10)
ax1.set_title('(a) MCI: Optimized Model Comparison', fontsize=11, fontweight='bold')
ax1.set_xlim(0.35, 1.05)
ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax1.invert_yaxis()
for i, a in enumerate(aucs):
    ax1.text(a + 0.008, i, f'{a:.3f}', va='center', fontsize=8)

legend_elements = [Patch(facecolor='#c0392b', label='Final model'),
                   Patch(facecolor='#3498db', label='Other models')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

# Panel (b): SHAP
top_n = 15
shap_top = df_shap_mci.head(top_n)
feat_names = shap_top['feature'].tolist()[::-1]
shap_values = shap_top['mean_abs_shap'].tolist()[::-1]

ax2.barh(range(len(feat_names)), shap_values, color='#c0392b', edgecolor='white', height=0.6)
ax2.set_yticks(range(len(feat_names)))
ax2.set_yticklabels(feat_names, fontsize=9)
ax2.set_xlabel('Mean |SHAP value|', fontsize=10)
ax2.set_title('(b) MCI: SVM Feature Importance (SHAP)', fontsize=11, fontweight='bold')
for i, v in enumerate(shap_values):
    ax2.text(v + 0.0003, i, f'{v:.4f}', va='center', fontsize=7)

plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(FIG_DIR, 'Figure1_MCI.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'Figure1_MCI.pdf'), bbox_inches='tight')
plt.close()
print("Figure 1 (MCI) 저장 완료")

# Figure 2: Dementia
print("Figure 2 생성...")

dem_models = [
    ('Random Forest',        boot_dem[0], boot_dem[1], boot_dem[2], True),
    ('CatBoost',             0.754, 0.639, 0.850, False),
    ('XGBoost',              0.723, 0.615, 0.817, False),
    ('Gradient Boosting',    0.718, 0.598, 0.825, False),
    ('AdaBoost',             0.718, 0.636, 0.790, False),
    ('LightGBM',             0.694, 0.602, 0.783, False),
    ('LASSO LR',             0.690, 0.552, 0.800, False),
    ('Support Vector Machine', 0.635, 0.478, 0.773, False),
]

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})

names2 = [m[0] for m in dem_models]
aucs2 = [m[1] for m in dem_models]
ci_lows2 = [m[2] for m in dem_models]
ci_highs2 = [m[3] for m in dem_models]
is_finals2 = [m[4] for m in dem_models]

err_low2 = [a - l for a, l in zip(aucs2, ci_lows2)]
err_high2 = [h - a for a, h in zip(aucs2, ci_highs2)]
colors2 = ['#c0392b' if f else '#3498db' for f in is_finals2]

ax3.barh(range(len(names2)), aucs2, color=colors2, edgecolor='white', height=0.65)
ax3.errorbar(aucs2, range(len(names2)), xerr=[err_low2, err_high2],
             fmt='none', ecolor='gray', capsize=2, linewidth=0.8)
ax3.set_yticks(range(len(names2)))
ax3.set_yticklabels(names2, fontsize=9)
ax3.set_xlabel('Bootstrap AUC (95% CI)', fontsize=10)
ax3.set_title('(a) Dementia: Optimized Model Comparison', fontsize=11, fontweight='bold')
ax3.set_xlim(0.35, 0.95)
ax3.axvline(x=0.7, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax3.invert_yaxis()
for i, a in enumerate(aucs2):
    ax3.text(a + 0.005, i, f'{a:.3f}', va='center', fontsize=8)
ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)

# Panel (b): SHAP
shap_top2 = df_shap_dem.head(top_n)
feat_names2 = shap_top2['feature'].tolist()[::-1]
shap_values2 = shap_top2['mean_abs_shap'].tolist()[::-1]

ax4.barh(range(len(feat_names2)), shap_values2, color='#c0392b', edgecolor='white', height=0.6)
ax4.set_yticks(range(len(feat_names2)))
ax4.set_yticklabels(feat_names2, fontsize=9)
ax4.set_xlabel('Mean |SHAP value|', fontsize=10)
ax4.set_title('(b) Dementia: RF Feature Importance (SHAP)', fontsize=11, fontweight='bold')
for i, v in enumerate(shap_values2):
    ax4.text(v + 0.0003, i, f'{v:.4f}', va='center', fontsize=7)

plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(FIG_DIR, 'Figure2_Dementia.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'Figure2_Dementia.pdf'), bbox_inches='tight')
plt.close()
print("Figure 2 (Dementia) 저장 완료")

print("\n===== 모든 작업 완료 =====")
