# -*- coding: utf-8 -*-
"""
Follow-up completeness, attrition, and lost-to-follow-up comparison across the
variable follow-up duration. Reconstructs the diabetic cohort exactly as in
klosa_preprocessing.py and examines per-wave observation of the cognitive-status
item (Cadd_01) at waves 8-10.
"""
import os, numpy as np, pandas as pd, pyreadstat
import warnings; warnings.filterwarnings('ignore')
from scipy import stats

BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
def rd(f, cols):
    df, _ = pyreadstat.read_sav(os.path.join(BASE, f), usecols=cols); return df

df7 = rd('w07_20260227.sav', ['pid','w07A002_age','w07C014','w07Cadd_01','w07gender1','w07F001','w07edu'])
fu = {w: rd(f'{w}_20260227.sav', ['pid', f'{w}Cadd_01']) for w in ['w08','w09']}
fu['w10'] = rd('w10_20260317.sav', ['pid','w10Cadd_01'])

# Reconstruct diabetic cohort (STEP 2 of preprocessing)
c = df7[df7['w07A002_age'] >= 45].copy()
c = c[c['w07C014'].notna()]
c = c[c['w07Cadd_01'] != 1]                       # exclude baseline dementia
c = c[c['w07gender1'].notna() & c['w07F001'].notna() & c['w07Cadd_01'].notna()]
print(f"Diabetic cohort (baseline non-dementia): {len(c)}")

# merge follow-up Cadd_01
for w in ['w08','w09','w10']:
    c = c.merge(fu[w].rename(columns={f'{w}Cadd_01': f'cad_{w}'}), on='pid', how='left')

obs = {w: c[f'cad_{w}'].notna() for w in ['w08','w09','w10']}
c['n_fu'] = sum(obs[w].astype(int) for w in ['w08','w09','w10'])

def describe(sub, name):
    print(f"\n--- {name} (N={len(sub)}) ---")
    for w in ['w08','w09','w10']:
        o = sub[f'cad_{w}'].notna()
        print(f"  observed at {w}: {o.sum():4d} ({o.mean()*100:4.1f}%)")
    print(f"  complete (all 3 waves): {(sub['n_fu']==3).sum()} ({(sub['n_fu']==3).mean()*100:.1f}%)")
    for k in [2,1,0]:
        print(f"  observed at {k} wave(s): {(sub['n_fu']==k).sum()} ({(sub['n_fu']==k).mean()*100:.1f}%)")
    print(f"  observed at final wave (w10): {sub['cad_w10'].notna().sum()} ({sub['cad_w10'].notna().mean()*100:.1f}%)")
    print(f"  mean follow-up waves: {sub['n_fu'].mean():.2f}")

# MCI cohort = baseline normal (5); Dementia cohort = baseline 3 or 5
mci = c[c['w07Cadd_01'] == 5].copy()
dem = c[c['w07Cadd_01'].isin([3,5])].copy()
describe(mci, "MCI cohort (baseline normal)")
describe(dem, "Dementia cohort (baseline normal+MCI)")

# Lost-to-follow-up vs retained comparison (cohort level). Lost = observed at NO follow-up wave.
c['lost'] = (c['n_fu'] == 0)
print(f"\n=== Lost to follow-up (no follow-up wave observed): {c['lost'].sum()} / {len(c)} "
      f"({c['lost'].mean()*100:.1f}%) ===")
ret = c[~c['lost']]; lost = c[c['lost']]
def cmp_cont(v, lab):
    a, b = ret[v].dropna(), lost[v].dropna()
    p = stats.mannwhitneyu(a, b).pvalue if len(b) else np.nan
    print(f"  {lab:18s} retained {a.mean():.2f}±{a.std():.2f}  vs lost {b.mean():.2f}±{b.std():.2f}  p={p:.3f}")
def cmp_bin(v, lab, pos):
    a = (ret[v]==pos); b = (lost[v]==pos)
    tab = np.array([[a.sum(), (~a).sum()],[b.sum(), (~b).sum()]])
    p = stats.chi2_contingency(tab)[1] if b.sum()+ (~b).sum()>0 else np.nan
    print(f"  {lab:18s} retained {a.mean()*100:.1f}%  vs lost {b.mean()*100:.1f}%  p={p:.3f}")
cmp_cont('w07A002_age', 'age')
cmp_bin('w07gender1', 'female', 5)
cmp_cont('w07edu', 'education(1-4)')
cmp_bin('w07Cadd_01', 'baseline MCI', 3)

# also: of incident events, first wave of diagnosis (timing)
def first_dx(row, target):
    for w in ['w08','w09','w10']:
        v = row[f'cad_{w}']
        if pd.notna(v):
            if v == target: return w
            if target==3 and v==1: return 'dem_first'  # progressed to dementia
    return None
print("\nTiming of incident MCI (first wave reported), MCI cohort:")
mci['dxw'] = mci.apply(lambda r: first_dx(r,3), axis=1)
print("  ", dict(mci['dxw'].value_counts(dropna=True)))
print("Timing of incident dementia (first wave reported), dementia cohort:")
dem['dxw'] = dem.apply(lambda r: first_dx(r,1), axis=1)
print("  ", dict(dem['dxw'].value_counts(dropna=True)))
