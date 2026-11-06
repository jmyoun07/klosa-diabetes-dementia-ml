"""
Optuna tuning for all 9 models (MCI and dementia outcomes).
Models: LR, LASSO_LR, LightGBM, XGBoost, CatBoost, RF, GBM, AdaBoost, SVC.
150 trials each with class_weight='balanced' and 500 bootstrap evaluations.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.utils import resample
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os, sys, time

BASE_DIR = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
LOG_FILE = os.path.join(BASE_DIR, "ml_all_models_tuning_log.txt")

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)

# Load data
print("=" * 70)
print("전체 9개 모델 Optuna Tuning (150 trials, class_weight='balanced')")
print("=" * 70)

with open(os.path.join(BASE_DIR, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_mci, y_mci = data['X_mci'], data['y_mci']
X_dem, y_dem = data['X_dem'], data['y_dem']
FEATURE_COLS = list(data['feature_cols'])

print(f"MCI: X={X_mci.shape}, events={y_mci.sum()}")
print(f"Dementia: X={X_dem.shape}, events={y_dem.sum()}")

Xtr_mci, Xte_mci, ytr_mci, yte_mci = train_test_split(
    X_mci.values, y_mci.values, test_size=0.3, stratify=y_mci.values, random_state=42)
Xtr_dem, Xte_dem, ytr_dem, yte_dem = train_test_split(
    X_dem.values, y_dem.values, test_size=0.3, stratify=y_dem.values, random_state=42)

print(f"\nMCI  Train: {len(ytr_mci)} (event={ytr_mci.sum()}), Test: {len(yte_mci)} (event={yte_mci.sum()})")
print(f"Dem  Train: {len(ytr_dem)} (event={ytr_dem.sum()}), Test: {len(yte_dem)} (event={yte_dem.sum()})")

# Helper functions
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
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))
    ba = np.array(boot_aucs)
    return np.median(ba), np.percentile(ba, 2.5), np.percentile(ba, 97.5)

# Optuna objective for all 9 models
def optuna_tune(model_type, X_train, y_train, n_trials=150):
    def objective(trial):
        if model_type == 'LR':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                'max_iter': 5000,
                'class_weight': 'balanced',
                'random_state': 42,
            }
            clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**params))])

        elif model_type == 'LASSO_LR':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': 'l1',
                'solver': 'saga',
                'max_iter': 5000,
                'class_weight': 'balanced',
                'random_state': 42,
            }
            clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**params))])

        elif model_type == 'SVC':
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'gamma': trial.suggest_float('gamma', 0.0001, 1, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42,
            }
            clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**params))])

        elif model_type == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_est', 50, 500),
                'learning_rate': trial.suggest_float('lr', 0.01, 2.0, log=True),
                'algorithm': 'SAMME',
                'random_state': 42,
            }
            clf = AdaBoostClassifier(**params)

        elif model_type == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
                'auto_class_weights': 'Balanced',
                'random_seed': 42, 'verbose': 0,
            }
            clf = CatBoostClassifier(**params)

        elif model_type == 'RF':
            params = {
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_leaf', 1, 10),
                'class_weight': 'balanced',
                'random_state': 42, 'n_jobs': -1,
            }
            clf = RandomForestClassifier(**params)

        elif model_type == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child', 5, 50),
                'class_weight': 'balanced',
                'random_state': 42, 'verbose': -1,
            }
            clf = LGBMClassifier(**params)

        elif model_type == 'XGBoost':
            pos = int(y_train.sum())
            neg = len(y_train) - pos
            spw = float(neg / pos) if pos > 0 else 1.0
            params = {
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
                'scale_pos_weight': spw,
                'random_state': 42, 'eval_metric': 'auc', 'verbosity': 0,
            }
            clf = XGBClassifier(**params)

        elif model_type == 'GBM':
            params = {
                'n_estimators': trial.suggest_int('n_est', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42,
            }
            clf = GradientBoostingClassifier(**params)
        else:
            return 0.5

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]
            if len(np.unique(yval)) < 2:
                continue
            try:
                if model_type == 'GBM':
                    sw = compute_sample_weight('balanced', ytr)
                    clf_copy = clf.__class__(**clf.get_params())
                    clf_copy.fit(Xtr, ytr, sample_weight=sw)
                elif model_type == 'AdaBoost':
                    sw = compute_sample_weight('balanced', ytr)
                    clf_copy = clf.__class__(**clf.get_params())
                    clf_copy.fit(Xtr, ytr, sample_weight=sw)
                elif model_type in ['LR', 'LASSO_LR', 'SVC']:
                    from sklearn.base import clone
                    clf_copy = clone(clf)
                    clf_copy.fit(Xtr, ytr)
                else:
                    clf_copy = clf.__class__(**clf.get_params())
                    clf_copy.fit(Xtr, ytr)
                prob = clf_copy.predict_proba(Xval)[:, 1]
                aucs.append(roc_auc_score(yval, prob))
            except Exception:
                aucs.append(0.5)
        return np.mean(aucs) if aucs else 0.5

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{model_type}] Best CV AUC={study.best_value:.4f}  params={study.best_params}")
    return study.best_params, study.best_value

# Build tuned model from best params
def build_tuned_model(model_type, best_params, y_ref=None):
    if model_type == 'LR':
        return Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(
            C=best_params.get('C', 1.0),
            solver=best_params.get('solver', 'lbfgs'),
            max_iter=5000, class_weight='balanced', random_state=42))])

    elif model_type == 'LASSO_LR':
        return Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(
            C=best_params.get('C', 1.0),
            penalty='l1', solver='saga',
            max_iter=5000, class_weight='balanced', random_state=42))])

    elif model_type == 'SVC':
        return Pipeline([('scaler', StandardScaler()), ('svc', SVC(
            C=best_params.get('C', 1.0),
            gamma=best_params.get('gamma', 0.01),
            kernel=best_params.get('kernel', 'rbf'),
            probability=True, class_weight='balanced', random_state=42))])

    elif model_type == 'AdaBoost':
        return AdaBoostClassifier(
            n_estimators=best_params.get('n_est', 100),
            learning_rate=best_params.get('lr', 1.0),
            algorithm='SAMME', random_state=42)

    elif model_type == 'CatBoost':
        return CatBoostClassifier(
            iterations=best_params.get('iterations', 200),
            depth=best_params.get('depth', 6),
            learning_rate=best_params.get('lr', 0.1),
            l2_leaf_reg=best_params.get('l2', 3),
            auto_class_weights='Balanced',
            random_seed=42, verbose=0)

    elif model_type == 'RF':
        return RandomForestClassifier(
            n_estimators=best_params.get('n_est', 200),
            max_depth=best_params.get('max_depth', 10),
            min_samples_split=best_params.get('min_split', 5),
            min_samples_leaf=best_params.get('min_leaf', 2),
            class_weight='balanced', random_state=42, n_jobs=-1)

    elif model_type == 'LightGBM':
        return LGBMClassifier(
            n_estimators=best_params.get('n_est', 200),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('lr', 0.1),
            num_leaves=best_params.get('num_leaves', 31),
            min_child_samples=best_params.get('min_child', 20),
            class_weight='balanced', random_state=42, verbose=-1)

    elif model_type == 'XGBoost':
        pos = int(y_ref.sum()) if y_ref is not None else 1
        neg = (len(y_ref) - pos) if y_ref is not None else 1
        spw = float(neg / pos) if pos > 0 else 1.0
        return XGBClassifier(
            n_estimators=best_params.get('n_est', 200),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('lr', 0.1),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample', 0.8),
            scale_pos_weight=spw,
            random_state=42, eval_metric='auc', verbosity=0)

    elif model_type == 'GBM':
        return GradientBoostingClassifier(
            n_estimators=best_params.get('n_est', 200),
            max_depth=best_params.get('max_depth', 5),
            learning_rate=best_params.get('lr', 0.1),
            min_samples_split=best_params.get('min_split', 5),
            min_samples_leaf=best_params.get('min_leaf', 2),
            subsample=best_params.get('subsample', 0.8),
            random_state=42)

    raise ValueError(f"Unknown: {model_type}")

# Run all 9 models
ALL_MODELS = ['LR', 'LASSO_LR', 'CatBoost', 'RF', 'LightGBM', 'XGBoost', 'GBM', 'AdaBoost', 'SVC']
all_results = []

for outcome_label, Xtr, ytr, Xte, yte in [
    ('MCI', Xtr_mci, ytr_mci, Xte_mci, yte_mci),
    ('Dementia', Xtr_dem, ytr_dem, Xte_dem, yte_dem),
]:
    print(f"\n{'='*70}")
    print(f"[{outcome_label}] 전체 9개 모델 Optuna Tuning (150 trials)")
    print(f"{'='*70}")

    for mname in ALL_MODELS:
        t0 = time.time()
        try:
            bp, bv = optuna_tune(mname, Xtr, ytr, n_trials=150)

            # Build and train
            tuned_clf = build_tuned_model(mname, bp, ytr)
            if mname == 'GBM':
                sw = compute_sample_weight('balanced', ytr)
                tuned_clf.fit(Xtr, ytr, sample_weight=sw)
            elif mname == 'AdaBoost':
                sw = compute_sample_weight('balanced', ytr)
                tuned_clf.fit(Xtr, ytr, sample_weight=sw)
            else:
                tuned_clf.fit(Xtr, ytr)

            # Test evaluation
            test_prob = tuned_clf.predict_proba(Xte)[:, 1]
            test_auc = roc_auc_score(yte, test_prob)
            opt_thr, opt_se, opt_sp = find_optimal_threshold(yte, test_prob)

            # PPV, NPV, Brier
            pred = (test_prob >= opt_thr).astype(int)
            tp = ((pred == 1) & (yte == 1)).sum()
            fp = ((pred == 1) & (yte == 0)).sum()
            fn = ((pred == 0) & (yte == 1)).sum()
            tn = ((pred == 0) & (yte == 0)).sum()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            brier = brier_score_loss(yte, test_prob)

            # Bootstrap
            boot_med, boot_low, boot_high = bootstrap_evaluate(yte, test_prob)

            elapsed = time.time() - t0
            print(f"  -> Test AUC={test_auc:.4f}, Boot={boot_med:.4f} [{boot_low:.4f}, {boot_high:.4f}]")
            print(f"     Se={opt_se:.3f}, Sp={opt_sp:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}, Brier={brier:.3f} ({elapsed:.0f}s)")

            all_results.append({
                'model': f'{mname}_tuned',
                'outcome': outcome_label,
                'phase': 'optuna_all9_22var',
                'cv_mean': bv,
                'boot_median': boot_med,
                'boot_CI95_low': boot_low,
                'boot_CI95_high': boot_high,
                'test_auc': test_auc,
                'opt_threshold': opt_thr,
                'opt_sensitivity': opt_se,
                'opt_specificity': opt_sp,
                'ppv': ppv,
                'npv': npv,
                'brier_score': brier,
                'best_params': str(bp),
            })

        except Exception as e:
            print(f"  [{mname}] ERROR: {e}")
            import traceback
            traceback.print_exc()

# Save results
df_results = pd.DataFrame(all_results)
out_csv = os.path.join(BASE_DIR, 'ML_results_all9_tuned.csv')
df_results.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {out_csv}")

# Summary
print("\n" + "=" * 70)
print("최종 요약 (Boot AUC 내림차순)")
print("=" * 70)
for outcome in ['MCI', 'Dementia']:
    sub = df_results[df_results['outcome'] == outcome].sort_values('boot_median', ascending=False)
    print(f"\n[{outcome}]")
    print(f"  {'Model':20s}  {'Boot AUC':>10s}  {'95% CI':>20s}  {'Se':>6s}  {'Sp':>6s}  {'PPV':>6s}  {'NPV':>6s}  {'Brier':>6s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for _, row in sub.iterrows():
        ci = f"[{row['boot_CI95_low']:.3f}, {row['boot_CI95_high']:.3f}]"
        print(f"  {row['model']:20s}  {row['boot_median']:10.4f}  {ci:>20s}  {row['opt_sensitivity']:6.3f}  {row['opt_specificity']:6.3f}  {row['ppv']:6.3f}  {row['npv']:6.3f}  {row['brier_score']:6.3f}")

print("\n완료!")
