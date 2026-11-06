# -*- coding: utf-8 -*-
"""
Generate Table 1 and Table 2 (baseline characteristics).
Binary variables display both Yes and No rows; Sex displays Male and Female rows.
Table layout follows the reference paper format.
"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
RESULTS_DIR = os.path.join(BASE, "final_model", "results")

print("데이터 로딩...")
with open(os.path.join(BASE, 'preprocessed_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_mci, y_mci = data['X_mci'], data['y_mci']
X_dem, y_dem = data['X_dem'], data['y_dem']
FEATURE_COLS = list(data['feature_cols'])

def format_pval(pval):
    if pd.isna(pval): return ''
    if pval < 0.001: return '<.001'
    return f'{pval:.3f}'

def compute_table(X, y, outcome_label):
    df = X.copy()
    df['outcome'] = y.values

    yes = df[df['outcome'] == 1]
    no  = df[df['outcome'] == 0]
    total_n = len(df)
    yes_n   = len(yes)
    no_n    = len(no)

    col_yes   = f'Yes (n={yes_n})'
    col_no    = f'No (n={no_n})'
    col_total = f'Total (n={total_n})'

    rows = []

    def add_row(var, label, total_str, yes_str, no_str, pval_str, indent=False):
        rows.append({
            'Variable': ('  ' + label) if indent else label,
            col_total:   total_str,
            col_yes:     yes_str,
            col_no:      no_str,
            'P value':   pval_str,
        })

    def chi2_or_fisher(df, var):
        tbl = pd.crosstab(df['outcome'], df[var])
        if tbl.shape == (2,2) and tbl.min().min() < 5:
            _, pval = stats.fisher_exact(tbl)
        elif tbl.shape[0] >= 2 and tbl.shape[1] >= 2:
            _, pval, _, _ = stats.chi2_contingency(tbl)
        else:
            pval = np.nan
        return pval

    # 1. Age (continuous)
    v = 'age'
    pval = stats.mannwhitneyu(yes[v], no[v], alternative='two-sided').pvalue
    add_row(v, 'Age, years',
            f'{df[v].mean():.1f} ± {df[v].std():.1f}',
            f'{yes[v].mean():.1f} ± {yes[v].std():.1f}',
            f'{no[v].mean():.1f} ± {no[v].std():.1f}',
            format_pval(pval))

    # 2. Sex
    pval = chi2_or_fisher(df, 'sex')
    add_row('sex', 'Sex', '', '', '', format_pval(pval))
    for val, lbl in [(0, 'Male'), (1, 'Female')]:
        n_t = (df['sex']==val).sum(); p_t = n_t/total_n*100
        n_y = (yes['sex']==val).sum(); p_y = n_y/yes_n*100 if yes_n else 0
        n_n = (no['sex']==val).sum();  p_n = n_n/no_n*100  if no_n  else 0
        add_row('sex', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 3. Marital status
    pval = chi2_or_fisher(df, 'marital')
    add_row('marital', 'Marital status', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Married'), (2,'Widowed'), (3,'Single/Divorced')]:
        n_t=(df['marital']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['marital']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['marital']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('marital', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 4. Education
    pval = chi2_or_fisher(df, 'edu')
    add_row('edu', 'Educational attainment', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Elementary or below'), (2,'Middle school'), (3,'High school'), (4,'College or above')]:
        n_t=(df['edu']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['edu']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['edu']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('edu', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 5. Living arrangements
    pval = chi2_or_fisher(df, 'living')
    add_row('living', 'Living arrangements', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Alone'), (2,'With partner'), (3,'With 2 or more')]:
        n_t=(df['living']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['living']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['living']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('living', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 6. Region
    pval = chi2_or_fisher(df, 'rural')
    add_row('rural', 'Region', '', '', '', format_pval(pval))
    for val, lbl in [(0,'Urban'), (1,'Rural')]:
        n_t=(df['rural']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['rural']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['rural']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('rural', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 7. Household income
    pval = chi2_or_fisher(df, 'income')
    add_row('income', 'Household income quintile', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Q1 (lowest)'), (2,'Q2'), (3,'Q3'), (4,'Q4'), (5,'Q5 (highest)')]:
        n_t=(df['income']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['income']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['income']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('income', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 8. ADL score (continuous)
    v = 'adl_score'
    pval = stats.mannwhitneyu(yes[v], no[v], alternative='two-sided').pvalue
    add_row(v, 'ADL score',
            f'{df[v].mean():.2f} ± {df[v].std():.2f}',
            f'{yes[v].mean():.2f} ± {yes[v].std():.2f}',
            f'{no[v].mean():.2f} ± {no[v].std():.2f}',
            format_pval(pval))

    # 9. IADL score (continuous)
    v = 'iadl_score'
    pval = stats.mannwhitneyu(yes[v], no[v], alternative='two-sided').pvalue
    add_row(v, 'IADL score',
            f'{df[v].mean():.2f} ± {df[v].std():.2f}',
            f'{yes[v].mean():.2f} ± {yes[v].std():.2f}',
            f'{no[v].mean():.2f} ± {no[v].std():.2f}',
            format_pval(pval))

    # 10. Grip strength decreased (binary)
    pval = chi2_or_fisher(df, 'grip_decreased')
    add_row('grip_decreased', 'Grip strength decreased', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['grip_decreased']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['grip_decreased']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['grip_decreased']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('grip_decreased', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 11. Drinking
    pval = chi2_or_fisher(df, 'drinking')
    add_row('drinking', 'Current drinking', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['drinking']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['drinking']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['drinking']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('drinking', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 12. Smoking
    pval = chi2_or_fisher(df, 'smoking')
    add_row('smoking', 'Current smoking', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['smoking']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['smoking']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['smoking']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('smoking', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 13. Exercise
    pval = chi2_or_fisher(df, 'exercise')
    add_row('exercise', 'Regular exercise', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['exercise']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['exercise']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['exercise']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('exercise', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 14. BMI category
    pval = chi2_or_fisher(df, 'bmi_cat')
    add_row('bmi_cat', 'BMI category', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Underweight (<18.5)'), (2,'Normal (18.5–24.9)'),
                     (3,'Overweight (25.0–29.9)'), (4,'Obese (≥30.0)')]:
        n_t=(df['bmi_cat']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['bmi_cat']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['bmi_cat']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('bmi_cat', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 15. Fall experience
    pval = chi2_or_fisher(df, 'fall_exp')
    add_row('fall_exp', 'Fall experience', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['fall_exp']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['fall_exp']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['fall_exp']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('fall_exp', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 16. Work participation
    pval = chi2_or_fisher(df, 'work_participation')
    add_row('work_participation', 'Work participation', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['work_participation']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['work_participation']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['work_participation']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('work_participation', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 17. CES-D 10 score (continuous)
    v = 'cesd_total'
    pval = stats.mannwhitneyu(yes[v], no[v], alternative='two-sided').pvalue
    add_row(v, 'CES-D 10 score',
            f'{df[v].mean():.1f} ± {df[v].std():.1f}',
            f'{yes[v].mean():.1f} ± {yes[v].std():.1f}',
            f'{no[v].mean():.1f} ± {no[v].std():.1f}',
            format_pval(pval))

    # 18. Hearing
    pval = chi2_or_fisher(df, 'hearing')
    add_row('hearing', 'Hearing difficulty', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Very good'), (2,'Good'), (3,'Fair'), (4,'Poor'), (5,'Very poor')]:
        n_t=(df['hearing']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['hearing']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['hearing']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('hearing', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 19. Subjective health
    pval = chi2_or_fisher(df, 'subjective_health')
    add_row('subjective_health', 'Subjective health status', '', '', '', format_pval(pval))
    for val, lbl in [(1,'Very good'), (2,'Good'), (3,'Fair'), (4,'Poor'), (5,'Very poor')]:
        n_t=(df['subjective_health']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['subjective_health']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['subjective_health']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('subjective_health', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 20. Stroke
    pval = chi2_or_fisher(df, 'stroke')
    add_row('stroke', 'Stroke treatment', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['stroke']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['stroke']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['stroke']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('stroke', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 21. Heart disease
    pval = chi2_or_fisher(df, 'heart_disease')
    add_row('heart_disease', 'Heart disease treatment', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['heart_disease']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['heart_disease']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['heart_disease']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('heart_disease', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    # 22. Hypertension
    pval = chi2_or_fisher(df, 'hypertension')
    add_row('hypertension', 'Hypertension treatment', '', '', '', format_pval(pval))
    for val, lbl in [(0,'No'), (1,'Yes')]:
        n_t=(df['hypertension']==val).sum(); p_t=n_t/total_n*100
        n_y=(yes['hypertension']==val).sum(); p_y=n_y/yes_n*100 if yes_n else 0
        n_n=(no['hypertension']==val).sum();  p_n=n_n/no_n*100  if no_n  else 0
        add_row('hypertension', lbl, f'{n_t} ({p_t:.1f})', f'{n_y} ({p_y:.1f})', f'{n_n} ({p_n:.1f})', '', indent=True)

    return pd.DataFrame(rows)

print("Table 1 (MCI) 생성...")
t1 = compute_table(X_mci, y_mci, 'MCI onset')
t1.to_csv(os.path.join(RESULTS_DIR, 'Table1_MCI_baseline.csv'), index=False, encoding='utf-8-sig')
print("  저장 완료: Table1_MCI_baseline.csv")

print("Table 2 (Dementia) 생성...")
t2 = compute_table(X_dem, y_dem, 'Dementia onset')
t2.to_csv(os.path.join(RESULTS_DIR, 'Table2_Dementia_baseline.csv'), index=False, encoding='utf-8-sig')
print("  저장 완료: Table2_Dementia_baseline.csv")

# Preview
print("\n--- Table 1 MCI 앞부분 미리보기 ---")
print(t1.head(20).to_string(index=False))
print("\n완료!")
