# -*- coding: utf-8 -*-
"""
Survey-weighted descriptive sensitivity analysis of the KLoSA diabetic sample.

KLoSA is nationally representative, but the primary analysis does not incorporate survey
weights, clustering, or stratification. The seventh wave provides an individual
cross-sectional weight (w07wgt_c) and a longitudinal weight (w07wgt_p), and the
survey-district identifier (ed) serves as the primary sampling unit / cluster. For the two
cohorts this computes:

  * the weighted vs unweighted six-year cumulative incidence of MCI and dementia, with
    linearization (Taylor) standard errors that account for clustering by survey district;
  * weighted vs unweighted means / proportions of the key baseline characteristics
    (age, sex, education, income, depressive symptoms, hearing difficulty).

The predictive models are individual-level and are not re-fitted: discrimination and
calibration are properties of individual risk estimates, not of weighted population totals.
This analysis addresses the descriptive representativeness of the sample.

Outputs -> survey_weighted_results.csv (+ console log)
"""
import os, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.weightstats import DescrStatsW
except Exception:
    DescrStatsW = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def rd(f, cols):
    df, _ = pyreadstat.read_sav(os.path.join(ROOT, f), usecols=cols)
    return df


# cohort + weights + outcomes
base = rd('w07_20260227.sav',
          ['pid', 'w07A002_age', 'w07C014', 'w07Cadd_01', 'w07gender1', 'w07F001',
           'w07wgt_c', 'w07wgt_p', 'ed', 'w07edu', 'w07E147', 'w07C118'])
c = base[base['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]

for w, f in [('w08', 'w08_20260227.sav'), ('w09', 'w09_20260227.sav'), ('w10', 'w10_20260317.sav')]:
    t = rd(f, ['pid', f'{w}Cadd_01']).rename(columns={f'{w}Cadd_01': f'cad_{w}'})
    c = c.merge(t, on='pid', how='left')


def incident(row, target, baseline_ok):
    if row['w07Cadd_01'] not in baseline_ok:
        return np.nan
    for w in ['w08', 'w09', 'w10']:
        v = row[f'cad_{w}']
        if pd.notna(v) and v == target:
            return 1
    return 0


c['mci'] = c.apply(lambda r: incident(r, 3, [5]), axis=1)          # baseline normal only
c['dem'] = c.apply(lambda r: incident(r, 1, [3, 5]), axis=1)       # baseline non-dementia

W = 'w07wgt_c'
c = c[c[W].notna()].copy()
c['female'] = (c['w07gender1'] == 5).astype(float)
c['age'] = c['w07A002_age'].astype(float)

print('=' * 88)
print(f'Survey-weighted descriptive sensitivity analysis (cross-sectional weight {W}; '
      f'{c["ed"].nunique()} survey districts)')
print('=' * 88)
print(f'  analytic cohort with weight: {len(c)}  (dementia cohort); '
      f'MCI cohort (baseline normal): {c["mci"].notna().sum()}')


def wtd_prop(mask_df, col, w):
    """weighted proportion of col==1 among non-missing, with cluster-robust (Taylor) SE."""
    sub = mask_df[mask_df[col].notna()]
    y = sub[col].values.astype(float)
    wt = sub[w].values.astype(float)
    p = np.sum(wt * y) / np.sum(wt)
    # linearization SE with clustering by survey district (ed)
    g = sub['ed'].values
    tot = np.sum(wt)
    # z_i = w_i (y_i - p); cluster totals
    z = wt * (y - p)
    df = pd.DataFrame({'g': g, 'z': z})
    cl = df.groupby('g')['z'].sum().values
    m = len(cl)
    var = (m / (m - 1)) * np.sum(cl ** 2) / (tot ** 2)
    se = np.sqrt(max(var, 0))
    return p, se, len(sub), int(np.nansum(y))


rows = []
for lab, col in [('MCI', 'mci'), ('Dementia', 'dem')]:
    sub = c[c[col].notna()]
    unw = sub[col].mean()
    wp, wse, n, ev = wtd_prop(c, col, W)
    print(f'\n  {lab} six-year cumulative incidence:')
    print(f'    unweighted : {unw*100:5.2f}%   (n={n}, events={ev})')
    print(f'    weighted   : {wp*100:5.2f}%   (95% CI {max(0,wp-1.96*wse)*100:.2f}-{(wp+1.96*wse)*100:.2f}%, '
          f'cluster-robust)')
    print(f'    difference : {(wp-unw)*100:+.2f} percentage points')
    rows.append(dict(quantity=f'{lab} 6-year incidence (%)', unweighted=unw * 100,
                     weighted=wp * 100, weighted_lo=max(0, wp - 1.96 * wse) * 100,
                     weighted_hi=(wp + 1.96 * wse) * 100, diff_pp=(wp - unw) * 100))


def wtd_mean(df, col, w):
    sub = df[df[col].notna()]
    return np.average(sub[col], weights=sub[w]), sub[col].mean()


print('\n  Key baseline characteristics (dementia cohort), weighted vs unweighted:')
for lab, col in [('Age, years', 'age'), ('Female, %', 'female')]:
    wm, um = wtd_mean(c, col, W)
    scale = 100 if col == 'female' else 1
    print(f'    {lab:16s} unweighted {um*scale:6.2f}   weighted {wm*scale:6.2f}   '
          f'diff {(wm-um)*scale:+.2f}')
    rows.append(dict(quantity=lab, unweighted=um * scale, weighted=wm * scale,
                     weighted_lo=np.nan, weighted_hi=np.nan, diff_pp=(wm - um) * scale))

# education (w07edu) and income (w07E147) weighted distribution, if available
for lab, col in [('Education (1-4), mean', 'w07edu')]:
    if col in c.columns and c[col].notna().sum() > 0:
        cc = c[c[col].between(1, 4)]
        wm = np.average(cc[col], weights=cc[W]); um = cc[col].mean()
        print(f'    {lab:20s} unweighted {um:5.2f}   weighted {wm:5.2f}   diff {wm-um:+.2f}')
        rows.append(dict(quantity=lab, unweighted=um, weighted=wm,
                         weighted_lo=np.nan, weighted_hi=np.nan, diff_pp=wm - um))

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'survey_weighted_results.csv'), index=False,
                          encoding='utf-8-sig')
print('\nSaved -> survey_weighted_results.csv')
