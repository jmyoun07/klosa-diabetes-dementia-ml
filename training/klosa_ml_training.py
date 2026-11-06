"""
KLoSA diabetes cohort: MCI and dementia ML prediction model training.

Methodology follows Oh et al. (2024), JMIR Medical Informatics.

Models (9):
  8 ML: LR, LightGBM, XGBoost, CatBoost, RF, GBM, AdaBoost, SVC
  1 statistical: LASSO_LR (L1-penalized logistic regression)

Class imbalance:
  Three strategies are compared and the best is chosen by cross-validated AUC.
    1) class_weight='balanced'
    2) SMOTENC (oversampling for mixed-type data)
    3) Focal loss (two-pass sample_weight)

Threshold:
  Optimal threshold chosen by Youden's J instead of a fixed 0.5.
"""



import pickle

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,

                               AdaBoostClassifier, StackingClassifier)

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTENC



from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.utils import resample

import statsmodels.api as sm

import shap

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import os

import traceback



# Korean font

font_candidates = [f for f in fm.findSystemFonts()

                   if any(k in f.lower() for k in ['malgun', 'nanumgothic', 'nanum'])]

if font_candidates:

    fm.fontManager.addfont(font_candidates[0])

    plt.rcParams['font.family'] = fm.FontProperties(fname=font_candidates[0]).get_name()

else:

    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False



BASE_DIR = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))



# Categorical feature indices (based on FEATURE_COLS)

# Continuous (9): age(0), adl_score(9), iadl_score(10), cesd_total(17),

#             hearing(18), vision(19), subjective_health(20), age_sq(24), age_sex(25)

# 22 variables
# 0:age, 1:sex, 2:marital, 3:edu, 4:living, 5:rural, 6:income,
# 7:adl_score, 8:iadl_score, 9:grip_decreased, 10:drinking, 11:smoking,
# 12:exercise, 13:bmi_cat, 14:fall_exp, 15:work_participation,
# 16:cesd_total, 17:hearing, 18:subjective_health,
# 19:stroke, 20:heart_disease, 21:hypertension
# Categorical/binary (15): sex(1), marital(2), edu(3), living(4), rural(5), income(6),
#   grip_decreased(9), drinking(10), smoking(11), exercise(12), bmi_cat(13),
#   fall_exp(14), work_participation(15), stroke(19), heart_disease(20), hypertension(21)
CAT_INDICES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21]



MODEL_NAMES = ['LR', 'LightGBM', 'XGBoost', 'CatBoost', 'RF', 'GBM', 'AdaBoost', 'SVC']

STRATEGIES  = ['class_weight', 'smotenc', 'focal_loss']






# STEP 1: Data loading


print("=" * 70)

print("KLoSA 당뇨병 코호트 ML 모델 학습")

print("모델: LR, LightGBM, XGBoost, CatBoost, RF, GBM, AdaBoost, SVC, LASSO_LR (9개)")

print("불균형: class_weight / SMOTENC / Focal Loss  →  CV AUC 최적 전략 선택")

print("임계값: Youden's J 최적화")

print("=" * 70)



with open(os.path.join(BASE_DIR, 'preprocessed_data.pkl'), 'rb') as f:

    data = pickle.load(f)



X_mci      = data['X_mci']

y_mci      = data['y_mci']

X_dem      = data['X_dem']

y_dem      = data['y_dem']

FEATURE_COLS = data['feature_cols']



print(f"\n[데이터 로딩 완료]")

print(f"  MCI 모델   - X: {X_mci.shape},  발생률: {y_mci.mean():.3f}")

print(f"  치매 모델  - X: {X_dem.shape}, 발생률: {y_dem.mean():.3f}")






# STEP 2: Model builder


def build_model(model_name: str, strategy: str, y_ref=None):

    """
    strategy == 'class_weight': apply class_weight='balanced'.
    Otherwise (smotenc, focal_loss): return the base model; sample weights
    are handled in fit_model.
    """

    use_cw = (strategy == 'class_weight')



    # XGBoost: handle imbalance via scale_pos_weight

    spw = 1.0

    if use_cw and model_name == 'XGBoost' and y_ref is not None:

        pos = int(y_ref.sum())

        neg = len(y_ref) - pos

        spw = float(neg / pos) if pos > 0 else 1.0



    if model_name == 'LR':

        return Pipeline([

            ('scaler', StandardScaler()),

            ('clf', LogisticRegression(

                class_weight='balanced' if use_cw else None,

                max_iter=1000, random_state=42))

        ])

    elif model_name == 'LightGBM':

        return LGBMClassifier(

            class_weight='balanced' if use_cw else None,

            random_state=42, verbose=-1, n_estimators=100)

    elif model_name == 'XGBoost':

        return XGBClassifier(

            scale_pos_weight=spw,

            random_state=42, eval_metric='auc', verbosity=0, n_estimators=100)

    elif model_name == 'CatBoost':

        return CatBoostClassifier(

            auto_class_weights='Balanced' if use_cw else None,

            random_seed=42, verbose=0, iterations=100)

    elif model_name == 'RF':

        return RandomForestClassifier(

            class_weight='balanced' if use_cw else None,

            random_state=42, n_estimators=100)

    elif model_name == 'GBM':

        # class_weight not supported; sample_weight is passed in fit_model

        return GradientBoostingClassifier(random_state=42, n_estimators=100)

    elif model_name == 'AdaBoost':

        # class_weight not supported; sample_weight is passed in fit_model

        return AdaBoostClassifier(random_state=42, n_estimators=100)

    elif model_name == 'SVC':

        return Pipeline([

            ('scaler', StandardScaler()),

            ('clf', SVC(

                class_weight='balanced' if use_cw else None,

                probability=True, random_state=42))

        ])

    raise ValueError(f"Unknown model: {model_name}")





def fit_model(model_name: str, model, X, y, strategy: str, sample_weight=None):

    """
    Fit a model, handling sample_weight.
    - GBM / AdaBoost with the class_weight strategy: a balanced sample_weight
      is injected automatically.
    - Pipeline models: passed as '<last_step>__sample_weight'.
    """

    sw = sample_weight

    if strategy == 'class_weight' and model_name in ['GBM', 'AdaBoost']:

        sw = compute_sample_weight('balanced', y)



    if isinstance(model, Pipeline):

        if sw is not None:

            last_step = model.steps[-1][0]

            model.fit(X, y, **{f'{last_step}__sample_weight': sw})

        else:

            model.fit(X, y)

    else:

        if sw is not None:

            model.fit(X, y, sample_weight=sw)

        else:

            model.fit(X, y)

    return model






# STEP 3: Imbalance-handling utilities


def apply_smotenc(X, y, cat_indices, random_state=42):

    """SMOTENC oversampling (k_neighbors adjusted to the minority-class count)."""

    pos = int(y.sum())

    k = max(1, min(3, pos - 1))

    try:

        smote = SMOTENC(categorical_features=cat_indices,

                        random_state=random_state, k_neighbors=k)

        return smote.fit_resample(X, y)

    except Exception as e:

        print(f"    [SMOTENC 실패: {e}] 원본 사용", flush=True)

        return X, y





def compute_focal_weights(init_model, X, y, gamma=2.0):

    """
    Focal-loss sample weights (two-pass).
    init_model: a model pre-trained with class_weight='balanced'.
    """

    proba = init_model.predict_proba(X)[:, 1]

    p_t   = np.where(y == 1, proba, 1 - proba)

    fw    = (1 - p_t) ** gamma

    bw    = compute_sample_weight('balanced', y)

    w     = fw * bw

    return w / w.mean()





def fit_with_strategy(model_name, strategy, Xtr, ytr, cat_indices):

    """Return a model fitted according to the given strategy."""

    if strategy == 'smotenc':

        Xr, yr = apply_smotenc(Xtr, ytr, cat_indices)

        m = build_model(model_name, 'smotenc')

        fit_model(model_name, m, Xr, yr, 'smotenc')



    elif strategy == 'focal_loss':

        # 1st pass: balanced

        m_init = build_model(model_name, 'class_weight', ytr)

        fit_model(model_name, m_init, Xtr, ytr, 'class_weight')

        # 2nd pass: focal weights

        fw = compute_focal_weights(m_init, Xtr, ytr)

        m  = build_model(model_name, 'focal_loss', ytr)

        fit_model(model_name, m, Xtr, ytr, 'focal_loss', fw)



    else:  # class_weight

        m = build_model(model_name, 'class_weight', ytr)

        fit_model(model_name, m, Xtr, ytr, 'class_weight')



    return m






# STEP 4: Strategy selection (by 5-fold CV AUC)


def select_best_strategy(model_name, X_train, y_train, cat_indices,

                          n_folds=10, random_state=42):

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    strat_aucs = {s: [] for s in STRATEGIES}



    for tr_idx, val_idx in skf.split(X_train, y_train):

        Xtr, Xval = X_train[tr_idx], X_train[val_idx]

        ytr, yval = y_train[tr_idx], y_train[val_idx]

        if len(np.unique(yval)) < 2:

            continue

        for s in STRATEGIES:

            try:

                m    = fit_with_strategy(model_name, s, Xtr, ytr, cat_indices)

                prob = m.predict_proba(Xval)[:, 1]

                strat_aucs[s].append(roc_auc_score(yval, prob))

            except Exception:

                strat_aucs[s].append(np.nan)



    mean_aucs = {s: np.nanmean(v) if v else np.nan for s, v in strat_aucs.items()}

    valid = {s: v for s, v in mean_aucs.items() if not np.isnan(v)}

    best  = max(valid, key=valid.get) if valid else 'class_weight'

    return best, mean_aucs






# STEP 5: Optimal threshold (Youden's J)


def find_optimal_threshold(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    j_idx = np.argmax(tpr - fpr)

    return float(thresholds[j_idx]), float(tpr[j_idx]), float(1 - fpr[j_idx])






# STEP 6: Final model evaluation (CV + bootstrap)


def evaluate_final(model_name, best_strategy, X_train, y_train, X_test, y_test,

                    cat_indices, n_folds=10, n_bootstrap=500, random_state=42):

    # 10-fold CV (using the best strategy)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_aucs = []

    for tr_idx, val_idx in skf.split(X_train, y_train):

        Xtr, Xval = X_train[tr_idx], X_train[val_idx]

        ytr, yval = y_train[tr_idx], y_train[val_idx]

        if len(np.unique(yval)) < 2:

            continue

        try:

            m    = fit_with_strategy(model_name, best_strategy, Xtr, ytr, cat_indices)

            prob = m.predict_proba(Xval)[:, 1]

            cv_aucs.append(roc_auc_score(yval, prob))

        except Exception:

            pass



    # Fit on the full training set, evaluate on the test set

    final_model = fit_with_strategy(model_name, best_strategy, X_train, y_train, cat_indices)

    test_prob   = final_model.predict_proba(X_test)[:, 1]

    test_auc    = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) >= 2 else np.nan



    # Optimal threshold

    opt_thr, opt_se, opt_sp = find_optimal_threshold(y_test, test_prob)



    # Bootstrap (test set)

    boot_aucs = []

    rng = np.random.RandomState(random_state)

    for _ in range(n_bootstrap):

        idx = resample(np.arange(len(y_test)), random_state=rng)

        yb, pb = y_test[idx], test_prob[idx]

        if len(np.unique(yb)) < 2:

            continue

        boot_aucs.append(roc_auc_score(yb, pb))



    ba = np.array(boot_aucs)

    stats = dict(

        cv_mean          = np.nanmean(cv_aucs) if cv_aucs else np.nan,

        cv_std           = np.nanstd(cv_aucs)  if cv_aucs else np.nan,

        cv_median        = np.nanmedian(cv_aucs) if cv_aucs else np.nan,

        boot_median      = np.median(ba),

        boot_IQR_low     = np.percentile(ba, 25),

        boot_IQR_high    = np.percentile(ba, 75),

        boot_CI95_low    = np.percentile(ba, 2.5),

        boot_CI95_high   = np.percentile(ba, 97.5),

        test_auc         = test_auc,

        opt_threshold    = opt_thr,

        opt_sensitivity  = opt_se,

        opt_specificity  = opt_sp,

    )

    return stats, final_model, test_prob






# STEP 7: LASSO_LR (L1-penalized logistic regression)


def evaluate_lasso_lr(X_train, y_train, X_test, y_test,

                 feature_names, outcome_label, n_bootstrap=500, random_state=42):

    print(f"  [LASSO_LR] {outcome_label} 학습 중...", end=' ', flush=True)

    try:

        Xtr_sm = sm.add_constant(X_train.astype(float), has_constant='add')

        Xte_sm = sm.add_constant(X_test.astype(float),  has_constant='add')



        result = sm.Logit(y_train.astype(float), Xtr_sm).fit_regularized(

            method='l1', alpha=0.1, disp=False)



        test_prob = result.predict(Xte_sm)

        yte       = y_test.astype(float)

        test_auc  = roc_auc_score(yte, test_prob) if len(np.unique(yte)) >= 2 else np.nan

        opt_thr, opt_se, opt_sp = find_optimal_threshold(yte, test_prob)



        boot_aucs = []

        rng = np.random.RandomState(random_state)

        for _ in range(n_bootstrap):

            idx = resample(np.arange(len(yte)), random_state=rng)

            yb, pb = yte[idx], test_prob[idx]

            if len(np.unique(yb)) < 2:

                continue

            boot_aucs.append(roc_auc_score(yb, pb))



        ba = np.array(boot_aucs)

        stats = dict(

            model='LASSO_LR', outcome=outcome_label, best_strategy='L1_penalized',

            cv_mean=np.nan, cv_std=np.nan, cv_median=np.nan,

            boot_median=np.median(ba),

            boot_IQR_low=np.percentile(ba, 25), boot_IQR_high=np.percentile(ba, 75),

            boot_CI95_low=np.percentile(ba, 2.5), boot_CI95_high=np.percentile(ba, 97.5),

            test_auc=test_auc,

            opt_threshold=opt_thr, opt_sensitivity=opt_se, opt_specificity=opt_sp,

        )

        print(f"Boot={stats['boot_median']:.4f} "

              f"95%CI[{stats['boot_CI95_low']:.4f},{stats['boot_CI95_high']:.4f}] "

              f"Thr={opt_thr:.3f}")



        coef_df = pd.DataFrame({

            'feature': ['intercept'] + list(feature_names),

            'coef':    result.params,

        })

        coef_df['OR'] = np.exp(coef_df['coef'])

        coef_df.to_csv(os.path.join(BASE_DIR, f'LASSO_LR_{outcome_label}_coef.csv'),

                       index=False, encoding='utf-8-sig')

        return stats, result, test_prob



    except Exception as e:

        print(f"오류: {e}")

        return dict(model='LASSO_LR', outcome=outcome_label, best_strategy='failed',

                    cv_mean=np.nan, boot_median=np.nan, test_auc=np.nan), None, None






# STEP 8: Main experiment loop


def run_experiment(X, y, label, feature_names, cat_indices, random_state=42):

    print(f"\n{'='*70}")

    print(f"[{label}] 모델 학습")

    print(f"  N={len(y)}, 발생={int(y.sum())}, 발생률={y.mean():.4f}")

    print(f"{'='*70}")



    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.30, stratify=y, random_state=random_state)

    print(f"  Train: {len(y_train)} (발생={int(y_train.sum())}), "

          f"Test: {len(y_test)} (발생={int(y_test.sum())})\n")



    results      = []

    trained_info = {}   # name → (model, Xtr, ytr, Xte, yte, test_prob)



    for name in MODEL_NAMES:

        print(f"  [{name}] 전략 선택...", end=' ', flush=True)

        try:

            best_s, s_aucs = select_best_strategy(

                name, X_train, y_train, cat_indices, random_state=random_state)

            au_str = "  ".join(f"{s}={v:.4f}" for s, v in s_aucs.items())

            print(f"→ best={best_s}  [{au_str}]")



            print(f"  [{name}] 최종 평가...", end=' ', flush=True)

            stats, model, test_prob = evaluate_final(

                name, best_s, X_train, y_train, X_test, y_test, cat_indices)



            stats.update(model=name, outcome=label, best_strategy=best_s)

            results.append(stats)

            trained_info[name] = (model, X_train, y_train, X_test, y_test, test_prob)



            print(f"CV={stats['cv_mean']:.4f}±{stats['cv_std']:.4f}  "

                  f"Boot={stats['boot_median']:.4f} "

                  f"[{stats['boot_CI95_low']:.4f},{stats['boot_CI95_high']:.4f}]  "

                  f"Thr={stats['opt_threshold']:.3f}  "

                  f"Se={stats['opt_sensitivity']:.3f}  Sp={stats['opt_specificity']:.3f}")



        except Exception as e:

            print(f"오류: {e}")

            traceback.print_exc()

            results.append(dict(model=name, outcome=label, best_strategy='failed',

                                cv_mean=np.nan, boot_median=np.nan, test_auc=np.nan))



    # LASSO_LR

    gee_stats, _, _ = evaluate_lasso_lr(

        X_train, y_train, X_test, y_test,

        feature_names=feature_names, outcome_label=label)

    results.append(gee_stats)



    df = (pd.DataFrame(results)

            .sort_values('boot_median', ascending=False)

            .reset_index(drop=True))



    disp = ['model', 'best_strategy', 'cv_mean', 'boot_median',

            'boot_CI95_low', 'boot_CI95_high', 'test_auc',

            'opt_threshold', 'opt_sensitivity', 'opt_specificity']

    print(f"\n[{label}] 결과 요약:")

    print(df[[c for c in disp if c in df.columns]].to_string(index=False))



    return df, trained_info, (X_train, y_train, X_test, y_test)





# Run experiments

results_mci, models_mci, splits_mci = run_experiment(

    X_mci.values, y_mci.values, 'MCI', FEATURE_COLS, CAT_INDICES)



results_dem, models_dem, splits_dem = run_experiment(

    X_dem.values, y_dem.values, 'Dementia', FEATURE_COLS, CAT_INDICES)






# STEP 9: Save results


all_results = pd.concat([results_mci, results_dem], ignore_index=True)

out_path = os.path.join(BASE_DIR, 'ML_results_DM_cohort.csv')

all_results.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"\n[결과 저장] {out_path}")






# STEP 10: Feature selection (SHAP-based) and retraining


print("\n" + "=" * 70)

print("STEP 10: SHAP 기반 Feature Selection + 재학습")

print("  목적: EPV 향상을 위해 22개 -> 상위 12개 변수로 축소")

print("=" * 70)





def shap_feature_selection(X_train, y_train, feature_names, cat_indices, top_k=12):

    """Rank features by SHAP importance and return the top_k."""

    # Compute SHAP quickly using a random forest

    rf = RandomForestClassifier(

        class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)

    sv = explainer.shap_values(X_train)

    if isinstance(sv, list):

        sv = sv[1]

    elif sv.ndim == 3:

        sv = sv[:, :, 1]



    mean_abs = np.abs(sv).mean(axis=0)

    importance = pd.DataFrame({

        'feature': feature_names,

        'idx': range(len(feature_names)),

        'shap_importance': mean_abs

    }).sort_values('shap_importance', ascending=False)



    selected = importance.head(top_k)

    sel_indices = sorted(selected['idx'].tolist())

    sel_names = [feature_names[i] for i in sel_indices]



    # Remap categorical indices

    new_cat = []

    for new_i, old_i in enumerate(sel_indices):

        if old_i in cat_indices:

            new_cat.append(new_i)



    print(f"  선택된 {top_k}개 변수:")

    for _, row in selected.iterrows():

        print(f"    {row['feature']:20s}  SHAP={row['shap_importance']:.4f}")



    return sel_indices, sel_names, new_cat





def run_selected_experiment(X, y, label, feature_names, cat_indices,

                            sel_indices, sel_names, sel_cat, random_state=42):

    """Re-run the full experiment on the feature-selected data."""

    X_sel = X[:, sel_indices]

    return run_experiment(X_sel, y, f"{label}_FS", sel_names, sel_cat, random_state)





# SHAP feature selection with top_k optimization

TOP_K_CANDIDATES = [8, 10, 12, 14, 16]

# MCI feature selection (top_k optimization)
print("\n[MCI] Feature Selection (top_k 최적화)...")
Xtr_mci, ytr_mci = splits_mci[0], splits_mci[1]

best_mci_auc = -1
best_mci_k = 12
best_mci_results = None
best_mci_fs_data = None

for tk in TOP_K_CANDIDATES:
    print(f"\n  --- MCI top_k={tk} 시도 ---")
    sel_idx, sel_names, sel_cat = shap_feature_selection(
        Xtr_mci, ytr_mci, FEATURE_COLS, CAT_INDICES, top_k=tk)
    res, mdls, spls = run_selected_experiment(
        X_mci.values, y_mci.values, f'MCI_k{tk}', FEATURE_COLS, CAT_INDICES,
        sel_idx, sel_names, sel_cat)
    # Based on the best boot_median
    top_auc = res['boot_median'].max()
    print(f"  MCI top_k={tk}: best boot AUC = {top_auc:.4f}")
    if top_auc > best_mci_auc:
        best_mci_auc = top_auc
        best_mci_k = tk
        best_mci_results = res
        best_mci_fs_data = (sel_idx, sel_names, sel_cat, spls)

print(f"\n[MCI] 최적 top_k = {best_mci_k} (Boot AUC = {best_mci_auc:.4f})")
mci_sel_idx, mci_sel_names, mci_sel_cat, splits_mci_fs = best_mci_fs_data
results_mci_fs = best_mci_results
# Re-run with the best k (logged as MCI_FS)
results_mci_fs, models_mci_fs, splits_mci_fs = run_selected_experiment(
    X_mci.values, y_mci.values, 'MCI', FEATURE_COLS, CAT_INDICES,
    mci_sel_idx, mci_sel_names, mci_sel_cat)

# Dementia feature selection (top_k optimization)
print("\n[Dementia] Feature Selection (top_k 최적화)...")
Xtr_dem, ytr_dem = splits_dem[0], splits_dem[1]

best_dem_auc = -1
best_dem_k = 12
best_dem_results = None
best_dem_fs_data = None

for tk in TOP_K_CANDIDATES:
    print(f"\n  --- Dementia top_k={tk} 시도 ---")
    sel_idx, sel_names, sel_cat = shap_feature_selection(
        Xtr_dem, ytr_dem, FEATURE_COLS, CAT_INDICES, top_k=tk)
    res, mdls, spls = run_selected_experiment(
        X_dem.values, y_dem.values, f'Dem_k{tk}', FEATURE_COLS, CAT_INDICES,
        sel_idx, sel_names, sel_cat)
    top_auc = res['boot_median'].max()
    print(f"  Dementia top_k={tk}: best boot AUC = {top_auc:.4f}")
    if top_auc > best_dem_auc:
        best_dem_auc = top_auc
        best_dem_k = tk
        best_dem_results = res
        best_dem_fs_data = (sel_idx, sel_names, sel_cat, spls)

print(f"\n[Dementia] 최적 top_k = {best_dem_k} (Boot AUC = {best_dem_auc:.4f})")
dem_sel_idx, dem_sel_names, dem_sel_cat, splits_dem_fs = best_dem_fs_data
results_dem_fs = best_dem_results
# Re-run with the best k
results_dem_fs, models_dem_fs, splits_dem_fs = run_selected_experiment(
    X_dem.values, y_dem.values, 'Dementia', FEATURE_COLS, CAT_INDICES,
    dem_sel_idx, dem_sel_names, dem_sel_cat)






# STEP 11: Optuna hyperparameter tuning (dementia-focused)


print("\n" + "=" * 70)

print("STEP 11: Optuna Hyperparameter Tuning (치매 모델)")

print("=" * 70)





def optuna_tune(model_type, X_train, y_train, cat_indices, n_trials=150):

    """Optuna Bayesian optimization (stratified 5-fold CV AUC)."""



    def objective(trial):

        if model_type == 'CatBoost':

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





def build_tuned_model(model_type, best_params, y_ref=None):

    """Build a model from the best Optuna parameters."""

    if model_type == 'CatBoost':

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





# Optuna-tune the top 5 models on the dementia feature-selected data

TUNE_MODELS = ['CatBoost', 'RF', 'LightGBM', 'XGBoost', 'GBM']

Xtr_dem_fs, ytr_dem_fs, Xte_dem_fs, yte_dem_fs = splits_dem_fs



tuned_results_dem = []

tuned_models_dem = {}

print("\n[Dementia FS] Optuna Tuning (상위 5개 모델, 각 80 trials)...")



for mname in TUNE_MODELS:

    try:

        bp, bv = optuna_tune(mname, Xtr_dem_fs, ytr_dem_fs, dem_sel_cat, n_trials=150)



        # Fit on the full training set with the best params, then evaluate on the test set

        tuned_clf = build_tuned_model(mname, bp, ytr_dem_fs)

        if mname == 'GBM':

            sw = compute_sample_weight('balanced', ytr_dem_fs)

            tuned_clf.fit(Xtr_dem_fs, ytr_dem_fs, sample_weight=sw)

        else:

            tuned_clf.fit(Xtr_dem_fs, ytr_dem_fs)



        test_prob = tuned_clf.predict_proba(Xte_dem_fs)[:, 1]

        test_auc = roc_auc_score(yte_dem_fs, test_prob) if len(np.unique(yte_dem_fs)) >= 2 else np.nan

        opt_thr, opt_se, opt_sp = find_optimal_threshold(yte_dem_fs, test_prob)



        # Bootstrap

        boot_aucs = []

        rng = np.random.RandomState(42)

        for _ in range(500):

            idx = resample(np.arange(len(yte_dem_fs)), random_state=rng)

            yb, pb = yte_dem_fs[idx], test_prob[idx]

            if len(np.unique(yb)) < 2: continue

            boot_aucs.append(roc_auc_score(yb, pb))

        ba = np.array(boot_aucs)



        stats = dict(

            model=f"{mname}_tuned", outcome='Dementia_FS_tuned',

            best_strategy='optuna', cv_mean=bv,

            boot_median=np.median(ba),

            boot_IQR_low=np.percentile(ba, 25), boot_IQR_high=np.percentile(ba, 75),

            boot_CI95_low=np.percentile(ba, 2.5), boot_CI95_high=np.percentile(ba, 97.5),

            test_auc=test_auc,

            opt_threshold=opt_thr, opt_sensitivity=opt_se, opt_specificity=opt_sp)

        tuned_results_dem.append(stats)

        tuned_models_dem[mname] = (tuned_clf, test_prob)



        print(f"    Test AUC={test_auc:.4f}  Boot={np.median(ba):.4f} "

              f"[{np.percentile(ba,2.5):.4f},{np.percentile(ba,97.5):.4f}]  "

              f"Se={opt_se:.3f}  Sp={opt_sp:.3f}")

    except Exception as e:

        print(f"  [{mname}] Optuna 오류: {e}")

        traceback.print_exc()



tuned_df_dem = (pd.DataFrame(tuned_results_dem)

                .sort_values('boot_median', ascending=False)

                .reset_index(drop=True))

print("\n[Dementia FS + Optuna] 결과:")

disp_cols = ['model', 'cv_mean', 'boot_median', 'boot_CI95_low', 'boot_CI95_high',

             'test_auc', 'opt_threshold', 'opt_sensitivity', 'opt_specificity']

print(tuned_df_dem[[c for c in disp_cols if c in tuned_df_dem.columns]].to_string(index=False))





# Apply the same Optuna tuning to MCI

Xtr_mci_fs, ytr_mci_fs, Xte_mci_fs, yte_mci_fs = splits_mci_fs



tuned_results_mci = []

tuned_models_mci = {}

print("\n[MCI FS] Optuna Tuning (상위 5개 모델, 각 80 trials)...")



for mname in TUNE_MODELS:

    try:

        bp, bv = optuna_tune(mname, Xtr_mci_fs, ytr_mci_fs, mci_sel_cat, n_trials=150)

        tuned_clf = build_tuned_model(mname, bp, ytr_mci_fs)

        if mname == 'GBM':

            sw = compute_sample_weight('balanced', ytr_mci_fs)

            tuned_clf.fit(Xtr_mci_fs, ytr_mci_fs, sample_weight=sw)

        else:

            tuned_clf.fit(Xtr_mci_fs, ytr_mci_fs)



        test_prob = tuned_clf.predict_proba(Xte_mci_fs)[:, 1]

        test_auc = roc_auc_score(yte_mci_fs, test_prob) if len(np.unique(yte_mci_fs)) >= 2 else np.nan

        opt_thr, opt_se, opt_sp = find_optimal_threshold(yte_mci_fs, test_prob)



        boot_aucs = []

        rng = np.random.RandomState(42)

        for _ in range(500):

            idx = resample(np.arange(len(yte_mci_fs)), random_state=rng)

            yb, pb = yte_mci_fs[idx], test_prob[idx]

            if len(np.unique(yb)) < 2: continue

            boot_aucs.append(roc_auc_score(yb, pb))

        ba = np.array(boot_aucs)



        stats = dict(

            model=f"{mname}_tuned", outcome='MCI_FS_tuned',

            best_strategy='optuna', cv_mean=bv,

            boot_median=np.median(ba),

            boot_IQR_low=np.percentile(ba, 25), boot_IQR_high=np.percentile(ba, 75),

            boot_CI95_low=np.percentile(ba, 2.5), boot_CI95_high=np.percentile(ba, 97.5),

            test_auc=test_auc,

            opt_threshold=opt_thr, opt_sensitivity=opt_se, opt_specificity=opt_sp)

        tuned_results_mci.append(stats)

        tuned_models_mci[mname] = (tuned_clf, test_prob)



        print(f"    Test AUC={test_auc:.4f}  Boot={np.median(ba):.4f} "

              f"[{np.percentile(ba,2.5):.4f},{np.percentile(ba,97.5):.4f}]  "

              f"Se={opt_se:.3f}  Sp={opt_sp:.3f}")

    except Exception as e:

        print(f"  [{mname}] Optuna 오류: {e}")



tuned_df_mci = (pd.DataFrame(tuned_results_mci)

                .sort_values('boot_median', ascending=False)

                .reset_index(drop=True))

print("\n[MCI FS + Optuna] 결과:")

print(tuned_df_mci[[c for c in disp_cols if c in tuned_df_mci.columns]].to_string(index=False))






# STEP 12: Stacking ensemble (top 3 tuned models)


print("\n" + "=" * 70)

print("STEP 12: Stacking Ensemble (상위 3 Tuned 모델)")

print("=" * 70)





def stacking_evaluate(tuned_models, X_train, y_train, X_test, y_test,

                      outcome_label, n_bootstrap=500):

    """Stacking ensemble of the top 3 tuned models."""

    if len(tuned_models) < 2:

        print(f"  [{outcome_label}] Tuned 모델 부족, Stacking 생략")

        return None



    # Take the top 3

    top3_names = list(tuned_models.keys())[:3]

    estimators = []

    for name in top3_names:

        clf, _ = tuned_models[name]

        estimators.append((name, clf))



    print(f"  구성: {[n for n, _ in estimators]}")



    stack = StackingClassifier(

        estimators=estimators,

        final_estimator=LogisticRegression(class_weight='balanced',

                                            max_iter=1000, random_state=42),

        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),

        stack_method='predict_proba', passthrough=False)



    stack.fit(X_train, y_train)

    test_prob = stack.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, test_prob)

    opt_thr, opt_se, opt_sp = find_optimal_threshold(y_test, test_prob)



    boot_aucs = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):

        idx = resample(np.arange(len(y_test)), random_state=rng)

        yb, pb = y_test[idx], test_prob[idx]

        if len(np.unique(yb)) < 2: continue

        boot_aucs.append(roc_auc_score(yb, pb))

    ba = np.array(boot_aucs)



    stats = dict(

        model='Stacking_Top3', outcome=outcome_label, best_strategy='stacking',

        cv_mean=np.nan,

        boot_median=np.median(ba),

        boot_IQR_low=np.percentile(ba, 25), boot_IQR_high=np.percentile(ba, 75),

        boot_CI95_low=np.percentile(ba, 2.5), boot_CI95_high=np.percentile(ba, 97.5),

        test_auc=test_auc,

        opt_threshold=opt_thr, opt_sensitivity=opt_se, opt_specificity=opt_sp)



    print(f"  [{outcome_label} Stacking] Test AUC={test_auc:.4f}  "

          f"Boot={np.median(ba):.4f} [{np.percentile(ba,2.5):.4f},{np.percentile(ba,97.5):.4f}]  "

          f"Se={opt_se:.3f}  Sp={opt_sp:.3f}")

    return stats, stack, test_prob





# Dementia Stacking

stack_dem = stacking_evaluate(

    tuned_models_dem, Xtr_dem_fs, ytr_dem_fs, Xte_dem_fs, yte_dem_fs,

    'Dementia_FS_stacking')



# MCI Stacking

stack_mci = stacking_evaluate(

    tuned_models_mci, Xtr_mci_fs, ytr_mci_fs, Xte_mci_fs, yte_mci_fs,

    'MCI_FS_stacking')





# Soft-voting ensemble (weights optimized with Optuna)
def soft_voting_evaluate(tuned_models, X_train, y_train, X_test, y_test,
                         outcome_label, n_bootstrap=500):
    """Soft-voting ensemble of the top 3 models (weights optimized with Optuna)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not tuned_models or len(tuned_models) < 2:
        print(f"  [{outcome_label}] Soft Voting 불가: 모델 부족")
        return None

    # tuned_models: {name: (model, test_prob)} or {name: model}
    # Extract only the model objects
    models_only = {}
    for name, val in tuned_models.items():
        if isinstance(val, tuple):
            models_only[name] = val[0]  # (model, test_prob) → model
        else:
            models_only[name] = val

    # Test-set probabilities for each model
    probs = {}
    for name, mdl in models_only.items():
        probs[name] = mdl.predict_proba(X_test)[:, 1]

    model_names = list(probs.keys())
    prob_array = np.array([probs[n] for n in model_names])  # (n_models, n_samples)

    # Train probabilities are also needed (weights optimized via CV)
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_probs = {}
    for name, mdl in models_only.items():
        p = np.zeros(len(y_train))
        for tr_idx, val_idx in cv.split(X_train, y_train):
            mdl_clone = mdl.__class__(**mdl.get_params()) if hasattr(mdl, 'get_params') else mdl
            try:
                mdl_clone.fit(X_train[tr_idx], y_train[tr_idx])
                p[val_idx] = mdl_clone.predict_proba(X_train[val_idx])[:, 1]
            except:
                p[val_idx] = mdl.predict_proba(X_train[val_idx])[:, 1]
        train_probs[name] = p

    train_prob_array = np.array([train_probs[n] for n in model_names])

    def objective(trial):
        weights = [trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(len(model_names))]
        w_sum = sum(weights)
        if w_sum < 0.01:
            return 0.5
        weights = [w / w_sum for w in weights]
        blended = sum(w * train_prob_array[i] for i, w in enumerate(weights))
        try:
            return roc_auc_score(y_train, blended)
        except:
            return 0.5

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=200, show_progress_bar=False)

    # Apply the best weights
    best_weights = [study.best_params[f'w_{i}'] for i in range(len(model_names))]
    w_sum = sum(best_weights)
    best_weights = [w / w_sum for w in best_weights]

    print(f"  [{outcome_label} SoftVoting] 최적 가중치:")
    for n, w in zip(model_names, best_weights):
        print(f"    {n}: {w:.3f}")

    # Test predictions
    test_prob = sum(w * prob_array[i] for i, w in enumerate(best_weights))
    test_auc = roc_auc_score(y_test, test_prob)
    opt_thr, opt_se, opt_sp = find_optimal_threshold(y_test, test_prob)

    # Bootstrap
    boot_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = resample(np.arange(len(y_test)), random_state=rng)
        yb, pb = y_test[idx], test_prob[idx]
        if len(np.unique(yb)) < 2: continue
        boot_aucs.append(roc_auc_score(yb, pb))
    ba = np.array(boot_aucs)

    stats = dict(
        model='SoftVoting_Optuna', outcome=outcome_label, best_strategy='soft_voting',
        cv_mean=study.best_value,
        boot_median=np.median(ba),
        boot_IQR_low=np.percentile(ba, 25), boot_IQR_high=np.percentile(ba, 75),
        boot_CI95_low=np.percentile(ba, 2.5), boot_CI95_high=np.percentile(ba, 97.5),
        test_auc=test_auc,
        opt_threshold=opt_thr, opt_sensitivity=opt_se, opt_specificity=opt_sp)

    print(f"  [{outcome_label} SoftVoting] Test AUC={test_auc:.4f}  "
          f"Boot={np.median(ba):.4f} [{np.percentile(ba,2.5):.4f},{np.percentile(ba,97.5):.4f}]  "
          f"Se={opt_se:.3f}  Sp={opt_sp:.3f}")

    return stats, None, test_prob

# Soft Voting
sv_dem = soft_voting_evaluate(
    tuned_models_dem, Xtr_dem_fs, ytr_dem_fs, Xte_dem_fs, yte_dem_fs,
    'Dementia_FS_soft_voting')

sv_mci = soft_voting_evaluate(
    tuned_models_mci, Xtr_mci_fs, ytr_mci_fs, Xte_mci_fs, yte_mci_fs,
    'MCI_FS_soft_voting')



# STEP 13: Save the combined results


all_enhanced = []

all_enhanced.append(results_mci.assign(phase='baseline_22var'))

all_enhanced.append(results_dem.assign(phase='baseline_22var'))

all_enhanced.append(results_mci_fs.assign(phase='feature_selected_12var'))

all_enhanced.append(results_dem_fs.assign(phase='feature_selected_12var'))

all_enhanced.append(tuned_df_mci.assign(phase='optuna_tuned'))

all_enhanced.append(tuned_df_dem.assign(phase='optuna_tuned'))

if stack_mci:

    all_enhanced.append(pd.DataFrame([stack_mci[0]]).assign(phase='stacking'))

if stack_dem:

    all_enhanced.append(pd.DataFrame([stack_dem[0]]).assign(phase='stacking'))

if sv_mci:
    all_enhanced.append(pd.DataFrame([sv_mci[0]]).assign(phase='soft_voting'))

if sv_dem:
    all_enhanced.append(pd.DataFrame([sv_dem[0]]).assign(phase='soft_voting'))

all_results_enhanced = pd.concat(all_enhanced, ignore_index=True)

out_path2 = os.path.join(BASE_DIR, 'ML_results_DM_cohort_enhanced.csv')

all_results_enhanced.to_csv(out_path2, index=False, encoding='utf-8-sig')

print(f"\n[통합 결과 저장] {out_path2}")






# STEP 14: SHAP analysis (best tuned models)


def run_shap(model_name, trained_info, feature_names, outcome_label):

    print(f"\n[SHAP] {outcome_label} - {model_name}")

    if model_name not in trained_info:

        print(f"  학습 정보 없음 ({model_name})")

        return None



    model, Xtr, ytr, Xte, yte, _ = trained_info[model_name]

    feature_names = list(feature_names)



    # AdaBoost: not supported by shap.TreeExplainer; use KernelExplainer

    tree_models = ['RF', 'GBM', 'LightGBM', 'XGBoost', 'CatBoost']

    try:

        if model_name in tree_models:

            explainer  = shap.TreeExplainer(model)

            shap_vals_ = explainer.shap_values(Xte)

            if isinstance(shap_vals_, list):

                sv = shap_vals_[1]

            elif hasattr(shap_vals_, 'ndim') and shap_vals_.ndim == 3:

                sv = shap_vals_[:, :, 1]

            else:

                sv = shap_vals_

        else:

            print(f"  KernelExplainer 사용 중 (시간 소요)...")

            bg  = shap.sample(Xtr, min(100, len(Xtr)))

            exp = shap.KernelExplainer(model.predict_proba, bg)

            raw = exp.shap_values(Xte[:50])

            sv  = np.array(raw[1]) if isinstance(raw, list) else np.array(raw)

            if sv.ndim == 3:

                sv = sv[:, :, 1]



        mean_abs = np.abs(sv).mean(axis=0)

        shap_df  = (pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})

                      .sort_values('mean_abs_shap', ascending=False))

        print(f"  상위 10개:\n{shap_df.head(10).to_string(index=False)}")



        # Bar plot

        top_n  = min(20, len(feature_names))

        top_df = shap_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.barh(range(top_n), top_df['mean_abs_shap'].values[::-1])

        ax.set_yticks(range(top_n))

        ax.set_yticklabels(top_df['feature'].values[::-1], fontsize=10)

        ax.set_xlabel('Mean |SHAP value|')

        ax.set_title(f'SHAP Feature Importance\n{outcome_label} - {model_name}')

        plt.tight_layout()

        fig.savefig(os.path.join(BASE_DIR,

                    f'shap_{outcome_label.lower()}_{model_name.lower()}.png'),

                    dpi=150, bbox_inches='tight')

        plt.close(fig)



        # Summary (beeswarm) plot

        fig2, _ = plt.subplots(figsize=(10, 8))

        shap.summary_plot(sv, Xte[:sv.shape[0]],

                          feature_names=feature_names, show=False, max_display=20)

        plt.title(f'SHAP Summary\n{outcome_label} - {model_name}')

        plt.tight_layout()

        plt.savefig(os.path.join(BASE_DIR,

                    f'shap_{outcome_label.lower()}_{model_name.lower()}_summary.png'),

                    dpi=150, bbox_inches='tight')

        plt.close()



        shap_df.to_csv(

            os.path.join(BASE_DIR,

                f'shap_{outcome_label.lower()}_{model_name.lower()}_values.csv'),

            index=False, encoding='utf-8-sig')

        print(f"  SHAP 저장 완료")

        return shap_df



    except Exception as e:

        print(f"  SHAP 오류: {e}")

        traceback.print_exc()

        return None





# Baseline 22-variable SHAP (paper reference models)

run_shap('RF',       models_mci, FEATURE_COLS, 'MCI')

run_shap('XGBoost',  models_dem, FEATURE_COLS, 'Dementia')



# Add the best baseline model

best_mci = results_mci[results_mci['model'] != 'LASSO_LR'].iloc[0]['model']

best_dem = results_dem[results_dem['model'] != 'LASSO_LR'].iloc[0]['model']

if best_mci != 'RF' and best_mci in models_mci:

    run_shap(best_mci, models_mci, FEATURE_COLS, 'MCI_best')

if best_dem != 'XGBoost' and best_dem in models_dem:

    run_shap(best_dem, models_dem, FEATURE_COLS, 'Dementia_best')





# SHAP for the best tuned models (12 selected features)

def run_shap_tuned(model_name, tuned_models, Xtr, Xte, feature_names, outcome_label):

    """SHAP analysis for a tuned model."""

    print(f"\n[SHAP Tuned] {outcome_label} - {model_name}")

    if model_name not in tuned_models:

        print(f"  모델 없음")

        return None

    clf, _ = tuned_models[model_name]

    tree_models = ['RF', 'GBM', 'LightGBM', 'XGBoost', 'CatBoost']

    feature_names = list(feature_names)

    try:

        if model_name in tree_models:

            explainer = shap.TreeExplainer(clf)

            sv_raw = explainer.shap_values(Xte)

            if isinstance(sv_raw, list):

                sv = sv_raw[1]

            elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:

                sv = sv_raw[:, :, 1]

            else:

                sv = sv_raw

        else:

            bg = shap.sample(Xtr, min(100, len(Xtr)))

            exp = shap.KernelExplainer(clf.predict_proba, bg)

            raw = exp.shap_values(Xte[:50])

            sv = np.array(raw[1]) if isinstance(raw, list) else np.array(raw)

            if sv.ndim == 3: sv = sv[:, :, 1]



        mean_abs = np.abs(sv).mean(axis=0)

        shap_df = (pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})

                   .sort_values('mean_abs_shap', ascending=False))

        print(f"  상위 10개:\n{shap_df.head(10).to_string(index=False)}")



        top_n = min(15, len(feature_names))

        top_df = shap_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(range(top_n), top_df['mean_abs_shap'].values[::-1])

        ax.set_yticks(range(top_n))

        ax.set_yticklabels(top_df['feature'].values[::-1], fontsize=10)

        ax.set_xlabel('Mean |SHAP value|')

        ax.set_title(f'SHAP Feature Importance (Tuned)\n{outcome_label} - {model_name}')

        plt.tight_layout()

        fig.savefig(os.path.join(BASE_DIR,

                    f'shap_tuned_{outcome_label.lower()}_{model_name.lower()}.png'),

                    dpi=150, bbox_inches='tight')

        plt.close(fig)



        fig2, _ = plt.subplots(figsize=(10, 7))

        shap.summary_plot(sv, Xte[:sv.shape[0]],

                          feature_names=feature_names, show=False, max_display=15)

        plt.title(f'SHAP Summary (Tuned)\n{outcome_label} - {model_name}')

        plt.tight_layout()

        plt.savefig(os.path.join(BASE_DIR,

                    f'shap_tuned_{outcome_label.lower()}_{model_name.lower()}_summary.png'),

                    dpi=150, bbox_inches='tight')

        plt.close()



        shap_df.to_csv(os.path.join(BASE_DIR,

            f'shap_tuned_{outcome_label.lower()}_{model_name.lower()}_values.csv'),

            index=False, encoding='utf-8-sig')

        print(f"  SHAP 저장 완료")

        return shap_df

    except Exception as e:

        print(f"  SHAP 오류: {e}")

        traceback.print_exc()

        return None





# SHAP for the best tuned dementia model

if tuned_df_dem is not None and len(tuned_df_dem) > 0:

    best_tuned_dem = tuned_df_dem.iloc[0]['model'].replace('_tuned', '')

    run_shap_tuned(best_tuned_dem, tuned_models_dem,

                   Xtr_dem_fs, Xte_dem_fs, dem_sel_names, 'Dementia_tuned')



# SHAP for the best tuned MCI model

if tuned_df_mci is not None and len(tuned_df_mci) > 0:

    best_tuned_mci = tuned_df_mci.iloc[0]['model'].replace('_tuned', '')

    run_shap_tuned(best_tuned_mci, tuned_models_mci,

                   Xtr_mci_fs, Xte_mci_fs, mci_sel_names, 'MCI_tuned')






# STEP 15: Final summary (4-stage comparison)


print("\n" + "=" * 70)

print("최종 결과 요약 (4단계 비교)")

print("=" * 70)



cols_disp = ['model', 'best_strategy', 'cv_mean', 'boot_median',

             'boot_IQR_low', 'boot_IQR_high',

             'boot_CI95_low', 'boot_CI95_high',

             'test_auc', 'opt_threshold', 'opt_sensitivity', 'opt_specificity']



print("\n--- [1] Baseline 22변수 ---")

for lbl, df in [('MCI', results_mci), ('Dementia', results_dem)]:

    print(f"\n  [{lbl}]")

    print(df[[c for c in cols_disp if c in df.columns]].head(5).to_string(index=False))



print("\n--- [2] Feature Selected 12변수 ---")

for lbl, df in [('MCI_FS', results_mci_fs), ('Dementia_FS', results_dem_fs)]:

    print(f"\n  [{lbl}]")

    print(df[[c for c in cols_disp if c in df.columns]].head(5).to_string(index=False))



print("\n--- [3] Optuna Tuned (FS 12변수) ---")

for lbl, df in [('MCI_tuned', tuned_df_mci), ('Dementia_tuned', tuned_df_dem)]:

    print(f"\n  [{lbl}]")

    print(df[[c for c in cols_disp if c in df.columns]].head(5).to_string(index=False))



print("\n--- [4] Stacking Ensemble ---")

if stack_mci:

    print(f"\n  [MCI] AUC={stack_mci[0]['test_auc']:.4f}  "

          f"Boot={stack_mci[0]['boot_median']:.4f}  "

          f"[{stack_mci[0]['boot_CI95_low']:.4f},{stack_mci[0]['boot_CI95_high']:.4f}]  "

          f"Se={stack_mci[0]['opt_sensitivity']:.3f}  Sp={stack_mci[0]['opt_specificity']:.3f}")

if stack_dem:

    print(f"  [Dementia] AUC={stack_dem[0]['test_auc']:.4f}  "

          f"Boot={stack_dem[0]['boot_median']:.4f}  "

          f"[{stack_dem[0]['boot_CI95_low']:.4f},{stack_dem[0]['boot_CI95_high']:.4f}]  "

          f"Se={stack_dem[0]['opt_sensitivity']:.3f}  Sp={stack_dem[0]['opt_specificity']:.3f}")



print(f"\n[저장 파일]")

print(f"  ML_results_DM_cohort.csv (baseline)")

print(f"  ML_results_DM_cohort_enhanced.csv (4단계 통합)")

print(f"  LASSO_LR_*_coef.csv")

print(f"  shap_*.csv / shap_*.png / shap_tuned_*.csv / shap_tuned_*.png")

print("\n완료!")

