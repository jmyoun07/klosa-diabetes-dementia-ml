# -*- coding: utf-8 -*-
"""
Generate paper tables and figures.
- Table 1: Baseline characteristics (MCI + Dementia, Yes/No/P-value)
- Table 3: Model performance comparison (8 models, both outcomes)
- Figure 1: MCI - (a) ROC curves, (b) SHAP beeswarm
- Figure 2: Dementia - (a) ROC curves, (b) SHAP beeswarm
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

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
FIG_DIR = os.path.join(BASE, "final_model", "figures")
RESULTS_DIR = os.path.join(BASE, "final_model", "results")
COEF_DIR = os.path.join(BASE, "final_model", "coefficients")
for d in [FIG_DIR, RESULTS_DIR, COEF_DIR]:
    os.makedirs(d, exist_ok=True)

# Load data
print("=" * 60)
print("데이터 로딩...")
with open(os.path.join(BASE, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_mci, y_mci = data['X_mci'], data['y_mci']
X_dem, y_dem = data['X_dem'], data['y_dem']
FEATURE_COLS = list(data['feature_cols'])

# Train/test split (same as original)
Xtr_mci, Xte_mci, ytr_mci, yte_mci = train_test_split(
    X_mci.values, y_mci.values, test_size=0.3, stratify=y_mci.values, random_state=42)
Xtr_dem, Xte_dem, ytr_dem, yte_dem = train_test_split(
    X_dem.values, y_dem.values, test_size=0.3, stratify=y_dem.values, random_state=42)

# TABLE 1: Baseline Characteristics
print("\n" + "=" * 60)
print("TABLE 1: Baseline Characteristics 생성...")

# Variable display names mapping
VAR_LABELS = {
    'age': ('Age, years', 'continuous'),
    'sex': ('Sex, female', 'binary'),
    'marital': ('Marital status', 'categorical', {1: 'Married', 2: 'Widowed', 3: 'Single/Divorced'}),
    'edu': ('Educational attainment', 'categorical', {1: 'Elementary or below', 2: 'Middle school', 3: 'High school', 4: 'College or above'}),
    'living': ('Living arrangements', 'categorical', {1: 'Alone', 2: 'With partner', 3: 'With 2 or more'}),
    'rural': ('Region, rural', 'binary'),
    'income': ('Household income quintile', 'categorical', {1: 'Q1 (lowest)', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5 (highest)'}),
    'adl_score': ('Activities of daily living score', 'continuous'),
    'iadl_score': ('Instrumental activities of daily living score', 'continuous'),
    'grip_decreased': ('Grip strength decreased', 'binary'),
    'drinking': ('Current drinking', 'binary'),
    'smoking': ('Current smoking', 'binary'),
    'exercise': ('Regular exercise', 'binary'),
    'bmi_cat': ('Body mass index category', 'categorical', {1: 'Underweight (<18.5)', 2: 'Normal (18.5-24.9)', 3: 'Overweight/Obese (25-29.9)', 4: 'Obese (>=30)'}),
    'fall_exp': ('Fall experience', 'binary'),
    'work_participation': ('Current work participation', 'binary'),
    'cesd_total': ('CES-D 10 total score', 'continuous'),
    'hearing': ('Hearing difficulty', 'categorical', {1: 'Very good', 2: 'Good', 3: 'Fair', 4: 'Poor', 5: 'Very poor'}),
    'subjective_health': ('Subjective health status', 'categorical', {1: 'Very good', 2: 'Good', 3: 'Fair', 4: 'Poor', 5: 'Very poor'}),
    'stroke': ('Stroke treatment', 'binary'),
    'heart_disease': ('Heart disease treatment', 'binary'),
    'hypertension': ('Hypertension treatment', 'binary'),
}

def compute_table1(X, y, outcome_name):
    """Compute Table 1 for one outcome."""
    df = X.copy()
    df['outcome'] = y.values

    rows = []

    total_n = len(df)
    yes_n = (df['outcome'] == 1).sum()
    no_n = (df['outcome'] == 0).sum()

    # Header row
    rows.append({
        'Variable': f'Total N',
        'Total': f'{total_n}',
        f'{outcome_name} Yes (n={yes_n})': '',
        f'{outcome_name} No (n={no_n})': '',
        'P-value': ''
    })

    for var in FEATURE_COLS:
        label, vtype = VAR_LABELS[var][0], VAR_LABELS[var][1]
        yes_data = df[df['outcome'] == 1][var]
        no_data = df[df['outcome'] == 0][var]
        all_data = df[var]

        if vtype == 'continuous':
            # Mean (SD)
            total_str = f'{all_data.mean():.1f} ({all_data.std():.1f})'
            yes_str = f'{yes_data.mean():.1f} ({yes_data.std():.1f})'
            no_str = f'{no_data.mean():.1f} ({no_data.std():.1f})'
            # t-test or Mann-Whitney
            if len(yes_data) >= 2 and len(no_data) >= 2:
                stat, pval = stats.mannwhitneyu(yes_data, no_data, alternative='two-sided')
            else:
                pval = np.nan
            pstr = format_pval(pval)
            rows.append({
                'Variable': label,
                'Total': total_str,
                f'{outcome_name} Yes (n={yes_n})': yes_str,
                f'{outcome_name} No (n={no_n})': no_str,
                'P-value': pstr
            })

        elif vtype == 'binary':
            n_yes_total = (all_data == 1).sum()
            pct_total = n_yes_total / total_n * 100
            n_yes_event = (yes_data == 1).sum()
            pct_yes = n_yes_event / yes_n * 100 if yes_n > 0 else 0
            n_yes_noevent = (no_data == 1).sum()
            pct_no = n_yes_noevent / no_n * 100 if no_n > 0 else 0

            # Chi-square or Fisher's exact
            table = pd.crosstab(df['outcome'], df[var])
            if table.shape == (2, 2):
                if table.min().min() < 5:
                    _, pval = stats.fisher_exact(table)
                else:
                    _, pval, _, _ = stats.chi2_contingency(table)
            else:
                pval = np.nan
            pstr = format_pval(pval)
            rows.append({
                'Variable': label,
                'Total': f'{n_yes_total} ({pct_total:.1f})',
                f'{outcome_name} Yes (n={yes_n})': f'{n_yes_event} ({pct_yes:.1f})',
                f'{outcome_name} No (n={no_n})': f'{n_yes_noevent} ({pct_no:.1f})',
                'P-value': pstr
            })

        elif vtype == 'categorical':
            cats = VAR_LABELS[var][2]
            # Overall chi-square
            table = pd.crosstab(df['outcome'], df[var])
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                _, pval, _, _ = stats.chi2_contingency(table)
            else:
                pval = np.nan
            pstr = format_pval(pval)

            # Variable header
            rows.append({
                'Variable': label,
                'Total': '',
                f'{outcome_name} Yes (n={yes_n})': '',
                f'{outcome_name} No (n={no_n})': '',
                'P-value': pstr
            })

            # Category rows
            for val, cat_label in cats.items():
                n_total = (all_data == val).sum()
                pct_total = n_total / total_n * 100
                n_event = (yes_data == val).sum()
                pct_event = n_event / yes_n * 100 if yes_n > 0 else 0
                n_noevent = (no_data == val).sum()
                pct_noevent = n_noevent / no_n * 100 if no_n > 0 else 0
                rows.append({
                    'Variable': f'  {cat_label}',
                    'Total': f'{n_total} ({pct_total:.1f})',
                    f'{outcome_name} Yes (n={yes_n})': f'{n_event} ({pct_event:.1f})',
                    f'{outcome_name} No (n={no_n})': f'{n_noevent} ({pct_noevent:.1f})',
                    'P-value': ''
                })

    return pd.DataFrame(rows)

def format_pval(pval):
    if pd.isna(pval):
        return ''
    elif pval < 0.001:
        return '<0.001'
    elif pval < 0.01:
        return f'{pval:.3f}'
    elif pval < 0.05:
        return f'{pval:.3f}'
    else:
        return f'{pval:.3f}'

# Generate Table 1 for both outcomes
table1_mci = compute_table1(X_mci, y_mci, 'MCI')
table1_dem = compute_table1(X_dem, y_dem, 'Dementia')

# Save as CSV
table1_mci.to_csv(os.path.join(RESULTS_DIR, 'Table1_MCI_baseline.csv'), index=False, encoding='utf-8-sig')
table1_dem.to_csv(os.path.join(RESULTS_DIR, 'Table1_Dementia_baseline.csv'), index=False, encoding='utf-8-sig')
print("  Table 1 (MCI) 저장 완료")
print("  Table 1 (Dementia) 저장 완료")

# TABLE 3: Model Performance Comparison
print("\n" + "=" * 60)
print("TABLE 3: Model Performance Comparison 생성...")

results_df = pd.read_csv(os.path.join(RESULTS_DIR, 'ML_results_all8_tuned.csv'))

# Model display name mapping
MODEL_NAMES = {
    'LASSO_LR_tuned': 'L1 Penalized Logistic Regression',
    'LightGBM_tuned': 'Light Gradient-Boosting Machine',
    'XGBoost_tuned': 'Extreme Gradient Boosting',
    'CatBoost_tuned': 'Categorical Boosting',
    'RF_tuned': 'Random Forest',
    'GBM_tuned': 'Gradient Boosting Machine',
    'AdaBoost_tuned': 'Adaptive Boosting',
    'SVC_tuned': 'Support Vector Machine',
}

# Model order (by MCI performance)
MODEL_ORDER_MCI = ['SVC_tuned', 'RF_tuned', 'XGBoost_tuned', 'CatBoost_tuned',
                    'AdaBoost_tuned', 'GBM_tuned', 'LightGBM_tuned', 'LASSO_LR_tuned']
MODEL_ORDER_DEM = ['RF_tuned', 'CatBoost_tuned', 'XGBoost_tuned', 'AdaBoost_tuned',
                    'GBM_tuned', 'LightGBM_tuned', 'LASSO_LR_tuned', 'SVC_tuned']

def build_table3(results_df, outcome, model_order, best_model):
    rows = []
    df = results_df[results_df['outcome'] == outcome]
    for m in model_order:
        row = df[df['model'] == m].iloc[0]
        is_best = (m == best_model)
        model_label = MODEL_NAMES.get(m, m)
        if is_best:
            model_label += ' *'

        auc_str = f"{row['boot_median']:.3f} ({row['boot_CI95_low']:.3f}-{row['boot_CI95_high']:.3f})"
        rows.append({
            'Model': model_label,
            'AUC (95% CI)': auc_str,
            'Test AUC': f"{row['test_auc']:.3f}",
            'Sensitivity': f"{row['opt_sensitivity']:.3f}",
            'Specificity': f"{row['opt_specificity']:.3f}",
            'PPV': f"{row['ppv']:.3f}",
            'NPV': f"{row['npv']:.3f}",
            'Brier Score': f"{row['brier_score']:.3f}",
        })
    return pd.DataFrame(rows)

table3_mci = build_table3(results_df, 'MCI', MODEL_ORDER_MCI, 'SVC_tuned')
table3_dem = build_table3(results_df, 'Dementia', MODEL_ORDER_DEM, 'RF_tuned')

# Combined Table 3
table3_combined = pd.concat([
    pd.DataFrame([{'Model': '--- Mild Cognitive Impairment ---', 'AUC (95% CI)': '', 'Test AUC': '',
                    'Sensitivity': '', 'Specificity': '', 'PPV': '', 'NPV': '', 'Brier Score': ''}]),
    table3_mci,
    pd.DataFrame([{'Model': '--- Dementia ---', 'AUC (95% CI)': '', 'Test AUC': '',
                    'Sensitivity': '', 'Specificity': '', 'PPV': '', 'NPV': '', 'Brier Score': ''}]),
    table3_dem,
], ignore_index=True)

table3_combined.to_csv(os.path.join(RESULTS_DIR, 'Table3_model_performance.csv'), index=False, encoding='utf-8-sig')
print("  Table 3 저장 완료")

# BUILD ALL 8 MODELS for ROC curves
print("\n" + "=" * 60)
print("8개 모델 구축 (ROC 곡선용)...")

# Best hyperparameters from Optuna (from ML_results_all8_tuned.csv)
import ast

def get_best_params(results_df, model_name, outcome):
    row = results_df[(results_df['model'] == model_name) & (results_df['outcome'] == outcome)].iloc[0]
    return ast.literal_eval(row['best_params'])

def build_model(model_name, params, X_train, y_train):
    """Build and fit a model given name and params."""
    if model_name == 'LASSO_LR_tuned':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=params['C'], penalty='l1', solver='saga',
                                       class_weight='balanced', max_iter=5000, random_state=42))
        ])
    elif model_name == 'SVC_tuned':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=params['C'], gamma=params['gamma'], kernel=params['kernel'],
                        probability=True, class_weight='balanced', random_state=42))
        ])
    elif model_name == 'RF_tuned':
        model = RandomForestClassifier(
            n_estimators=params['n_est'], max_depth=params['max_depth'],
            min_samples_split=params['min_split'], min_samples_leaf=params['min_leaf'],
            class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost_tuned':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        model = XGBClassifier(
            n_estimators=params['n_est'], max_depth=params['max_depth'],
            learning_rate=params['lr'], subsample=params['subsample'],
            colsample_bytree=params['colsample'], scale_pos_weight=n_neg/n_pos,
            use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == 'LightGBM_tuned':
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        model = LGBMClassifier(
            n_estimators=params['n_est'], max_depth=params['max_depth'],
            learning_rate=params['lr'], num_leaves=params['num_leaves'],
            min_child_samples=params['min_child'], scale_pos_weight=n_neg/n_pos,
            random_state=42, verbose=-1)
    elif model_name == 'CatBoost_tuned':
        model = CatBoostClassifier(
            iterations=params['iterations'], depth=params['depth'],
            learning_rate=params['lr'], l2_leaf_reg=params['l2'],
            auto_class_weights='Balanced', random_seed=42, verbose=0)
    elif model_name == 'GBM_tuned':
        model = GradientBoostingClassifier(
            n_estimators=params['n_est'], max_depth=params['max_depth'],
            learning_rate=params['lr'], min_samples_split=params['min_split'],
            min_samples_leaf=params['min_leaf'], subsample=params['subsample'],
            random_state=42)
    elif model_name == 'AdaBoost_tuned':
        model = AdaBoostClassifier(
            n_estimators=params['n_est'], learning_rate=params['lr'],
            random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model

# Build all 8 models for MCI
print("  MCI 모델 구축...")
mci_models = {}
for m in MODEL_ORDER_MCI:
    params = get_best_params(results_df, m, 'MCI')
    mci_models[m] = build_model(m, params, Xtr_mci, ytr_mci)
    print(f"    {m} 완료")

# Build all 8 models for Dementia
print("  Dementia 모델 구축...")
dem_models = {}
for m in MODEL_ORDER_DEM:
    params = get_best_params(results_df, m, 'Dementia')
    dem_models[m] = build_model(m, params, Xtr_dem, ytr_dem)
    print(f"    {m} 완료")

# FIGURE 1: MCI - (a) ROC curves, (b) SHAP beeswarm
print("\n" + "=" * 60)
print("FIGURE 1: MCI 생성...")

# Color palette for 8 models
COLORS = {
    'SVC_tuned': '#e41a1c',      # Red (best)
    'RF_tuned': '#377eb8',       # Blue
    'XGBoost_tuned': '#4daf4a',  # Green
    'CatBoost_tuned': '#984ea3', # Purple
    'AdaBoost_tuned': '#ff7f00', # Orange
    'GBM_tuned': '#a65628',      # Brown
    'LightGBM_tuned': '#f781bf', # Pink
    'LASSO_LR_tuned': '#999999', # Gray
}

SHORT_NAMES = {
    'SVC_tuned': 'SVM',
    'RF_tuned': 'RF',
    'XGBoost_tuned': 'XGBoost',
    'CatBoost_tuned': 'CatBoost',
    'AdaBoost_tuned': 'AdaBoost',
    'GBM_tuned': 'GBM',
    'LightGBM_tuned': 'LightGBM',
    'LASSO_LR_tuned': 'LASSO LR',
}

def plot_roc_curves(ax, models_dict, X_test, y_test, model_order, title, best_model):
    """Plot ROC curves for all models."""
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)

    for m in model_order:
        model = models_dict[m]
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)

        lw = 2.5 if m == best_model else 1.2
        alpha = 1.0 if m == best_model else 0.7
        label = f"{SHORT_NAMES[m]} (AUC = {auc_val:.3f})"

        ax.plot(fpr, tpr, color=COLORS[m], linewidth=lw, alpha=alpha, label=label)

    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=10)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.2)

# SHAP computation for MCI (SVM - KernelExplainer)
print("  MCI SHAP 계산 (KernelExplainer)...")
svc_mci = mci_models['SVC_tuned']
scaler_mci = svc_mci.named_steps['scaler']
Xtr_mci_sc = scaler_mci.transform(Xtr_mci)
Xte_mci_sc = scaler_mci.transform(Xte_mci)

bg_idx = np.random.RandomState(42).choice(len(Xtr_mci_sc), size=100, replace=False)
background_mci = Xtr_mci_sc[bg_idx]

svc_model_mci = svc_mci.named_steps['clf']
explainer_mci = shap.KernelExplainer(svc_model_mci.predict_proba, background_mci)
test_idx_mci = np.random.RandomState(42).choice(len(Xte_mci_sc), size=min(100, len(Xte_mci_sc)), replace=False)
shap_vals_mci = explainer_mci.shap_values(Xte_mci_sc[test_idx_mci])
if isinstance(shap_vals_mci, list):
    shap_vals_mci = shap_vals_mci[1]
if shap_vals_mci.ndim == 3:
    shap_vals_mci = shap_vals_mci[:, :, 1]

Xte_mci_display = pd.DataFrame(Xte_mci_sc[test_idx_mci], columns=FEATURE_COLS)

print("  MCI Figure 1 생성...")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel (a): ROC curves
plot_roc_curves(ax1, mci_models, Xte_mci, yte_mci, MODEL_ORDER_MCI,
                '(a) Mild Cognitive Impairment: ROC Curves', 'SVC_tuned')

# Panel (b): SHAP beeswarm
ax2_pos = ax2.get_position()
ax2.remove()
ax2_new = fig1.add_axes([0.55, 0.12, 0.42, 0.78])

# Create SHAP beeswarm using summary_plot
plt.sca(ax2_new)
shap.summary_plot(shap_vals_mci, Xte_mci_display, feature_names=FEATURE_COLS,
                  show=False, max_display=15, plot_size=None)
ax2_new.set_title('(b) MCI: SVM SHAP Summary', fontsize=12, fontweight='bold')
ax2_new.set_xlabel('SHAP value (impact on model output)', fontsize=10)

plt.savefig(os.path.join(FIG_DIR, 'Figure1_MCI.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'Figure1_MCI.pdf'), bbox_inches='tight')
plt.close()
print("  Figure 1 (MCI) 저장 완료")

# FIGURE 2: Dementia - (a) ROC curves, (b) SHAP beeswarm
print("\n" + "=" * 60)
print("FIGURE 2: Dementia 생성...")

# SHAP computation for Dementia (RF - TreeExplainer)
print("  Dementia SHAP 계산 (TreeExplainer)...")
rf_dem = dem_models['RF_tuned']
explainer_dem = shap.TreeExplainer(rf_dem)
shap_vals_dem = explainer_dem.shap_values(Xte_dem)
if isinstance(shap_vals_dem, list):
    shap_vals_dem = shap_vals_dem[1]
elif shap_vals_dem.ndim == 3:
    shap_vals_dem = shap_vals_dem[:, :, 1]

Xte_dem_display = pd.DataFrame(Xte_dem, columns=FEATURE_COLS)

print("  Dementia Figure 2 생성...")
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Panel (a): ROC curves
plot_roc_curves(ax3, dem_models, Xte_dem, yte_dem, MODEL_ORDER_DEM,
                '(a) Dementia: ROC Curves', 'RF_tuned')

# Panel (b): SHAP beeswarm
ax4_pos = ax4.get_position()
ax4.remove()
ax4_new = fig2.add_axes([0.55, 0.12, 0.42, 0.78])

plt.sca(ax4_new)
shap.summary_plot(shap_vals_dem, Xte_dem_display, feature_names=FEATURE_COLS,
                  show=False, max_display=15, plot_size=None)
ax4_new.set_title('(b) Dementia: RF SHAP Summary', fontsize=12, fontweight='bold')
ax4_new.set_xlabel('SHAP value (impact on model output)', fontsize=10)

plt.savefig(os.path.join(FIG_DIR, 'Figure2_Dementia.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'Figure2_Dementia.pdf'), bbox_inches='tight')
plt.close()
print("  Figure 2 (Dementia) 저장 완료")

# Update SHAP CSVs
print("\n" + "=" * 60)
print("SHAP CSV 업데이트...")

mci_shap_mean = np.abs(shap_vals_mci).mean(axis=0)
dem_shap_mean = np.abs(shap_vals_dem).mean(axis=0)

df_shap_mci = pd.DataFrame({
    'feature': FEATURE_COLS, 'mean_abs_shap': mci_shap_mean
}).sort_values('mean_abs_shap', ascending=False)

df_shap_dem = pd.DataFrame({
    'feature': FEATURE_COLS, 'mean_abs_shap': dem_shap_mean
}).sort_values('mean_abs_shap', ascending=False)

df_shap_mci.to_csv(os.path.join(COEF_DIR, 'MCI_SVC_tuned_SHAP.csv'), index=False, encoding='utf-8-sig')
df_shap_dem.to_csv(os.path.join(COEF_DIR, 'Dementia_RF_tuned_SHAP.csv'), index=False, encoding='utf-8-sig')

print("\nMCI SVM SHAP Top 10:")
for _, row in df_shap_mci.head(10).iterrows():
    print(f"  {row['feature']:25s} {row['mean_abs_shap']:.4f}")

print("\nDementia RF SHAP Top 10:")
for _, row in df_shap_dem.head(10).iterrows():
    print(f"  {row['feature']:25s} {row['mean_abs_shap']:.4f}")

# Update README files
print("\n" + "=" * 60)
print("README 업데이트...")

with open(os.path.join(FIG_DIR, 'README.md'), 'w', encoding='utf-8') as f:
    f.write("""# Figures

## 파일 목록
- `Figure1_MCI.png/pdf` - (a) MCI 8개 모델 ROC 곡선, (b) SVM SHAP beeswarm summary plot
- `Figure2_Dementia.png/pdf` - (a) Dementia 8개 모델 ROC 곡선, (b) RF SHAP beeswarm summary plot
""")

with open(os.path.join(RESULTS_DIR, 'README.md'), 'w', encoding='utf-8') as f:
    f.write("""# Results

## 파일 목록
- `ML_results_all8_tuned.csv` - 8개 모델 Optuna 최적화 결과 (MCI + Dementia)
- `ML_results_final.csv` - 최종 선정 모델 (MCI: SVM, Dementia: RF) 상세 성능
- `Table1_MCI_baseline.csv` - Table 1: MCI 기저 특성 (Yes/No/P-value)
- `Table1_Dementia_baseline.csv` - Table 1: Dementia 기저 특성 (Yes/No/P-value)
- `Table3_model_performance.csv` - Table 3: 8개 모델 성능 비교

## 컬럼 설명 (ML_results)
- `boot_median`: 500회 Bootstrap 중앙값 AUC
- `boot_CI95_low/high`: 95% 신뢰구간
- `test_auc`: 테스트셋 단일 AUC
- `opt_sensitivity/specificity`: Youden's J 최적 임계값에서의 Se/Sp
- `ppv/npv`: 양성예측도/음성예측도
- `brier_score`: Brier Score
""")

print("\n" + "=" * 60)
print("모든 작업 완료!")
print("=" * 60)
