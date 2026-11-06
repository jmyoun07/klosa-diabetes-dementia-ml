"""
KLoSA diabetes cohort ML prediction models for MCI and dementia (SMOTE version).
Applies the methodology of Oh et al. (2024), JMIR Medical Informatics.
Class imbalance is handled with SMOTE instead of class_weight.
SMOTE is applied only within each CV fold / to the training set to prevent data leakage.
"""

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Korean font configuration
font_candidates = [f for f in fm.findSystemFonts()
                   if any(k in f.lower() for k in ['malgun', 'nanumgothic', 'nanum'])]
if font_candidates:
    fm.fontManager.addfont(font_candidates[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_candidates[0]).get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

# Step 1: Load preprocessed data
print("=" * 60)
print("KLoSA 당뇨병 코호트 ML 모델 학습 (SMOTE)")
print("=" * 60)

with open(os.path.join(BASE_DIR, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_mci = data['X_mci']
y_mci = data['y_mci']
X_dem = data['X_dem']
y_dem = data['y_dem']
FEATURE_COLS = data['feature_cols']

print(f"\n[데이터 로딩 완료]")
print(f"  MCI 모델 - X: {X_mci.shape}, 발생률: {y_mci.mean():.3f}")
print(f"  치매 모델 - X: {X_dem.shape}, 발생률: {y_dem.mean():.3f}")


# Step 2: Model definitions (no class_weight)
def get_models():
    """Class imbalance is handled by SMOTE, so class_weight/auto_class_weights are omitted."""
    models = {
        'LR': LogisticRegression(
            max_iter=1000, random_state=42),
        'LightGBM': LGBMClassifier(
            random_state=42, verbose=-1, n_estimators=100),
        'XGBoost': XGBClassifier(
            random_state=42, eval_metric='auc', verbosity=0,
            n_estimators=100, use_label_encoder=False),
        'CatBoost': CatBoostClassifier(
            random_seed=42, verbose=0, iterations=100),
        'RF': RandomForestClassifier(
            random_state=42, n_estimators=100),
        'GBM': GradientBoostingClassifier(
            random_state=42, n_estimators=100),
        'AdaBoost': AdaBoostClassifier(
            random_state=42, n_estimators=100),
        'SVC': SVC(
            probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    return models


# Step 3: SMOTE helper
def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE and return (X_resampled, y_resampled).
    Reduce k_neighbors when the minority class has fewer than 6 samples.
    """
    n_minority = int(y.sum())
    k = min(5, n_minority - 1)
    if k < 1:
        # Too few minority samples for SMOTE; return the original data
        return X, y
    smote = SMOTE(k_neighbors=k, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


# Step 4: Evaluation (SMOTE built in)
def evaluate_model(model, X_train, y_train, X_test, y_test,
                   n_folds=5, n_bootstrap=100, random_state=42):
    """
    CV: apply SMOTE to each fold's training split, fit, then evaluate on validation.
    Test: apply SMOTE to the full training set, fit, then evaluate on the test set.
    Bootstrap: resample the test-set predictions (no SMOTE needed).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        if len(np.unique(yval)) < 2:
            continue

        # SMOTE: applied only to the in-fold training split
        Xtr_sm, ytr_sm = apply_smote(Xtr, ytr, random_state=random_state + fold)

        model.fit(Xtr_sm, ytr_sm)
        prob = model.predict_proba(Xval)[:, 1]
        cv_aucs.append(roc_auc_score(yval, prob))

    # SMOTE: applied to the full training set before the final fit
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, random_state=random_state)
    print(f"[SMOTE] {y_train.sum()}→{y_train_sm.sum()} pos / "
          f"{(y_train==0).sum()}→{(y_train_sm==0).sum()} neg", end=' | ')

    model.fit(X_train_sm, y_train_sm)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) >= 2 else np.nan

    # Bootstrap on the test set (no SMOTE; keep the real distribution)
    boot_aucs = []
    rng = np.random.RandomState(random_state)
    for _ in range(n_bootstrap):
        idx = resample(np.arange(len(y_test)), random_state=rng)
        yb, pb = y_test[idx], test_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))

    return cv_aucs, boot_aucs, test_auc


def summarize_aucs(cv_aucs, boot_aucs, test_auc):
    cv_arr   = np.array(cv_aucs)
    boot_arr = np.array(boot_aucs)
    return {
        'cv_mean':       np.mean(cv_arr),
        'cv_std':        np.std(cv_arr),
        'cv_median':     np.median(cv_arr),
        'boot_median':   np.median(boot_arr),
        'boot_IQR_low':  np.percentile(boot_arr, 25),
        'boot_IQR_high': np.percentile(boot_arr, 75),
        'boot_CI95_low': np.percentile(boot_arr, 2.5),
        'boot_CI95_high':np.percentile(boot_arr, 97.5),
        'test_auc':      test_auc,
    }


# Step 5: Run experiments
def run_experiment(X, y, label, random_state=42):
    print(f"\n{'='*60}")
    print(f"[{label}] 모델 학습 시작 (SMOTE)")
    print(f"  N={len(y)}, 발생={y.sum()}, 발생률={y.mean():.4f}")
    print(f"  클래스 비율 = {(y==0).sum()}:{y.sum()} = {(y==0).sum()//max(y.sum(),1)}:1")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state)
    print(f"  Train: {len(y_train)} (발생={y_train.sum()}), "
          f"Test: {len(y_test)} (발생={y_test.sum()})")

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  [{name}] 학습 중... ", end='', flush=True)
        try:
            cv_aucs, boot_aucs, test_auc = evaluate_model(
                model, X_train.copy(), y_train.copy(),
                X_test.copy(), y_test.copy())
            stats = summarize_aucs(cv_aucs, boot_aucs, test_auc)
            stats['model']   = name
            stats['outcome'] = label
            results.append(stats)
            trained_models[name] = (model, X_train, y_train, X_test, y_test)
            print(f"CV AUC={stats['cv_mean']:.4f}, "
                  f"Boot={stats['boot_median']:.4f} "
                  f"95%CI[{stats['boot_CI95_low']:.4f}, {stats['boot_CI95_high']:.4f}]")
        except Exception as e:
            print(f"오류: {e}")
            results.append({'model': name, 'outcome': label, 'cv_mean': np.nan})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('boot_median', ascending=False).reset_index(drop=True)

    print(f"\n[{label}] 결과 요약:")
    cols = ['model','cv_mean','cv_std','boot_median',
            'boot_IQR_low','boot_IQR_high','boot_CI95_low','boot_CI95_high','test_auc']
    print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))

    return results_df, trained_models, (X_train, y_train, X_test, y_test)


# Run experiments
results_mci, models_mci, splits_mci = run_experiment(X_mci.values, y_mci.values, 'MCI')
results_dem, models_dem, splits_dem = run_experiment(X_dem.values, y_dem.values, 'Dementia')


# Step 6: Save results
all_results = pd.concat([results_mci, results_dem], ignore_index=True)
out_path = os.path.join(BASE_DIR, 'ML_results_DM_SMOTE.csv')
all_results.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n[결과 저장] {out_path}")


# Step 7: SHAP analysis
def run_shap(model_name, models_dict, feature_names, outcome_label):
    print(f"\n[SHAP] {outcome_label} - {model_name}")
    model, Xtr, ytr, Xte, _ = models_dict[model_name]

    # Refit after applying SMOTE
    Xtr_sm, ytr_sm = apply_smote(Xtr, ytr, random_state=42)
    model.fit(Xtr_sm, ytr_sm)
    feature_names = list(feature_names)

    if model_name in ['RF', 'GBM', 'AdaBoost', 'LightGBM', 'XGBoost', 'CatBoost']:
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values
    else:
        explainer  = shap.KernelExplainer(
            model.predict_proba, shap.sample(Xtr_sm, 100))
        shap_values = explainer.shap_values(Xte[:50])
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1])
        else:
            shap_vals = np.array(shap_values)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"  상위 10개 중요 변수:")
    print(shap_df.head(10).to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n  = min(20, len(feature_names))
    top_df = shap_df.head(top_n)
    ax.barh(range(top_n), top_df['mean_abs_shap'].values[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df['feature'].values[::-1], fontsize=10)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'SHAP Feature Importance\n{outcome_label} - {model_name} (SMOTE)')
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, f'shap_smote_{outcome_label.lower()}_{model_name.lower()}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  SHAP plot 저장: {plot_path}")

    # Summary plot
    X_for_summary = Xte[:shap_vals.shape[0]]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_for_summary,
                      feature_names=feature_names, show=False, max_display=20)
    plt.title(f'SHAP Summary Plot\n{outcome_label} - {model_name} (SMOTE)')
    plt.tight_layout()
    plot_path2 = os.path.join(BASE_DIR, f'shap_smote_{outcome_label.lower()}_{model_name.lower()}_summary.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SHAP summary plot 저장: {plot_path2}")

    shap_df.to_csv(
        os.path.join(BASE_DIR, f'shap_smote_{outcome_label.lower()}_{model_name.lower()}_values.csv'),
        index=False, encoding='utf-8-sig')

    return shap_df


Xtr_mci, ytr_mci, Xte_mci, yte_mci = splits_mci
Xtr_dem, ytr_dem, Xte_dem, yte_dem = splits_dem

# Reference models from the paper
shap_mci = run_shap('RF',      models_mci, FEATURE_COLS, 'MCI')
shap_dem = run_shap('XGBoost', models_dem, FEATURE_COLS, 'Dementia')

# Best model per outcome in this run; add SHAP for it when it differs from the reference above
best_mci = results_mci.iloc[0]['model']
best_dem = results_dem.iloc[0]['model']
if best_mci != 'RF':
    print(f"\n[SHAP] MCI 최고 모델({best_mci}) 추가 분석")
    run_shap(best_mci, models_mci, FEATURE_COLS, 'MCI_best')
if best_dem != 'XGBoost':
    print(f"\n[SHAP] 치매 최고 모델({best_dem}) 추가 분석")
    run_shap(best_dem, models_dem, FEATURE_COLS, 'Dementia_best')


# Step 8: Final summary
print("\n" + "=" * 60)
print("최종 결과 요약 (SMOTE)")
print("=" * 60)

for outcome, df in [('MCI', results_mci), ('치매', results_dem)]:
    print(f"\n[{outcome}] Bootstrap AUC (median, 95%CI)")
    cols = ['model', 'boot_median', 'boot_IQR_low', 'boot_IQR_high',
            'boot_CI95_low', 'boot_CI95_high', 'test_auc']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

print(f"\n저장 파일:")
print(f"  - ML_results_DM_SMOTE.csv")
print(f"  - shap_smote_*.png / *_values.csv")
print("\n완료!")
