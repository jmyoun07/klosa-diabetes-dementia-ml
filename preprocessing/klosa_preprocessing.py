# -*- coding: utf-8 -*-
"""
KLoSA (Korean Longitudinal Study of Ageing) - preprocessing for the diabetes
cohort MCI/dementia ML prediction models.

Data: waves 7-10 (2018-2024) SAV files, plus waves 1-6 for cumulative
comorbidity and fall history.
Reference: Oh SS et al. (2024). A Multivariable Prediction Model for Mild
Cognitive Impairment and Dementia. JMIR Med Inform 2024;12:e59396.
"""

import pandas as pd
import numpy as np
import pyreadstat
import warnings
import os

warnings.filterwarnings('ignore')

# Path configuration
DATA_DIR = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SPSS_DIR = os.path.join(DATA_DIR, "SPSS (1)")

print("=" * 60)
print("KLoSA 당뇨병 코호트 MCI·치매 ML 모델 전처리 시작")
print("=" * 60)

# STEP 1: Load SAV data
print("\n[STEP 1] 데이터 로딩...")

def load_sav(path):
    df, meta = pyreadstat.read_sav(path)
    return df, meta

# Waves 7-10 (main analysis data)
dfs = {}
for wave, fname in [
    ('w07', 'w07_20260227.sav'),
    ('w08', 'w08_20260227.sav'),
    ('w09', 'w09_20260227.sav'),
    ('w10', 'w10_20260317.sav'),
]:
    path = os.path.join(DATA_DIR, fname)
    dfs[wave], _ = load_sav(path)
    print(f"  {wave}: {dfs[wave].shape} 로딩 완료")

# Waves 1-6 (cumulative comorbidity and fall history)
dfs_hist = {}
hist_files = [
    ('w01', os.path.join(SPSS_DIR, 'w01_20260227.sav')),
    ('w02', os.path.join(SPSS_DIR, 'w02_20260227.sav')),
    ('w03', os.path.join(SPSS_DIR, 'w03_20260227.sav')),
    ('w04', os.path.join(SPSS_DIR, 'w04_20260227.sav')),
    ('w05', os.path.join(SPSS_DIR, 'w05_20260227.sav')),
    ('w06', os.path.join(SPSS_DIR, 'w06_20260227.sav')),
]
for wave, fpath in hist_files:
    dfs_hist[wave], _ = load_sav(fpath)
    print(f"  {wave}: {dfs_hist[wave].shape} 로딩 완료 (이력용)")

df6 = dfs_hist['w06']   # wave 6, for grip strength comparison
df7 = dfs['w07'].copy()

# STEP 2: Extract diabetes cohort
print("\n[STEP 2] 당뇨병 코호트 추출...")
print(f"  7차 전체 응답자: {len(df7)}")

# 1) Age 45 or older
df7_45plus = df7[df7['w07A002_age'] >= 45].copy()
print(f"  45세 이상 필터 후: {len(df7_45plus)}명")

# 2) Diabetes cohort: w07C014 not null
#    w07C014 = current diabetes treatment; includes both existing (C013 path)
#    and newly diagnosed (C011 path) patients. Using w07C013 alone would miss
#    newly diagnosed patients.
diabetic_cohort = df7_45plus[df7_45plus['w07C014'].notna()].copy()
print(f"  당뇨병 코호트 (w07C014 not null): {len(diabetic_cohort)}명")
print(f"  당뇨병 비율: {len(diabetic_cohort)/len(df7_45plus)*100:.1f}%")

# 3) Exclude baseline dementia (w07Cadd_01 == 1)
n_before = len(diabetic_cohort)
diabetic_cohort = diabetic_cohort[diabetic_cohort['w07Cadd_01'] != 1]
print(f"  기저 치매 제외 후: {len(diabetic_cohort)}명 (제외 {n_before - len(diabetic_cohort)}명)")

# 4) Exclude records missing sex or diagnosis
n_before = len(diabetic_cohort)
diabetic_cohort = diabetic_cohort[diabetic_cohort['w07gender1'].notna()]
diabetic_cohort = diabetic_cohort[diabetic_cohort['w07F001'].notna()]
diabetic_cohort = diabetic_cohort[diabetic_cohort['w07Cadd_01'].notna()]
print(f"  필수변수 결측 제외 후: {len(diabetic_cohort)}명 (제외 {n_before - len(diabetic_cohort)}명)")

# STEP 3: Build outcome variables
print("\n[STEP 3] Outcome 변수 생성...")
print("  치매/MCI: w0XCadd_01 (1=치매, 3=MCI, 5=정상)")

w7_dist = diabetic_cohort['w07Cadd_01'].value_counts().sort_index()
print(f"  7차 당뇨 코호트 Cadd_01 분포: {dict(w7_dist)}")

# Merge follow-up waves on pid
df_analysis = diabetic_cohort[['pid']].copy()
for wave in ['w08', 'w09', 'w10']:
    key_var = f'{wave}Cadd_01'
    df_wave = dfs[wave][['pid', key_var]].copy()
    df_wave.columns = ['pid', f'cadd01_{wave}']
    df_analysis = df_analysis.merge(df_wave, on='pid', how='left')
print(f"  Follow-up 병합 후: {df_analysis.shape}")

# MCI prediction model outcome
# Baseline (wave 7) normal (Cadd_01 == 5) -> incident MCI (== 3) in waves 8/9/10

mci_base_pids = diabetic_cohort[diabetic_cohort['w07Cadd_01'] == 5]['pid']
mci_base = diabetic_cohort[diabetic_cohort['pid'].isin(mci_base_pids)].copy()
mci_base = mci_base.merge(df_analysis, on='pid', how='left')

def get_mci_outcome(row):
    """Return 1 if MCI (=3) occurs in waves 8/9/10; dementia (=1) is excluded; else 0."""
    for wave in ['w08', 'w09', 'w10']:
        val = row.get(f'cadd01_{wave}', np.nan)
        if pd.notna(val):
            if val == 3:   return 1   # incident MCI
            elif val == 1: return 0   # progressed directly to dementia (excluded from MCI incidence)
    return 0

mci_base['MCI_incident'] = mci_base.apply(get_mci_outcome, axis=1)
print(f"\n  [MCI 모델] Baseline N: {len(mci_base)}, 발생: {mci_base['MCI_incident'].sum()}명")

# Dementia prediction model outcome
# Baseline (wave 7) normal + MCI (Cadd_01 in {3, 5}) -> incident dementia (== 1) in waves 8/9/10

dem_base_pids = diabetic_cohort[diabetic_cohort['w07Cadd_01'].isin([3, 5])]['pid']
dem_base = diabetic_cohort[diabetic_cohort['pid'].isin(dem_base_pids)].copy()
dem_base = dem_base.merge(df_analysis, on='pid', how='left')

def get_dementia_outcome(row):
    """Return 1 if dementia (=1) occurs in waves 8/9/10; else 0."""
    for wave in ['w08', 'w09', 'w10']:
        val = row.get(f'cadd01_{wave}', np.nan)
        if pd.notna(val) and val == 1:
            return 1
    return 0

dem_base['Dementia_incident'] = dem_base.apply(get_dementia_outcome, axis=1)
print(f"  [치매 모델] Baseline N: {len(dem_base)}, 발생: {dem_base['Dementia_incident'].sum()}명")

# STEP 4: Build cumulative comorbidity and fall history (waves 1-7)
print("\n[STEP 4] 누적 동반질환·낙상 이력 (1~7차) 생성...")

# Cumulative hypertension history
#   wave 1: w01C006 (1=yes/5=no)
#   waves 2-7: w0XC008 (1=new diagnosis/2=recurrence/3=under treatment/4=had previously); 1-4 all count as history

hyp_ever = {}  # {pid: True/False}

# Wave 1 hypertension
df1 = dfs_hist['w01'].copy()
if 'w01C006' in df1.columns:
    hyp1 = df1[df1['w01C006'] == 1]['pid'].unique()
    for pid in hyp1:
        hyp_ever[pid] = True

# Waves 2-6 hypertension (SPSS_DIR)
for wn in [2, 3, 4, 5, 6]:
    key = f'w0{wn}'
    df_w = dfs_hist[key].copy()
    hvar = f'w0{wn}C008'
    if hvar in df_w.columns:
        # 1,2,3 = current hypertension (new/recurrence/under treatment), 4 = had previously
        has_hyp = df_w[df_w[hvar].isin([1, 2, 3, 4])]['pid'].unique()
        for pid in has_hyp:
            hyp_ever[pid] = True

# Wave 7 hypertension (C008 not null = under treatment)
if 'w07C008' in df7.columns:
    has_hyp7 = df7[df7['w07C008'].notna()]['pid'].unique()
    for pid in has_hyp7:
        hyp_ever[pid] = True

print(f"  고혈압 이력 보유자: {len(hyp_ever)}명")

# Cumulative history of other chronic diseases
# cancer (C016), chronic lung disease (C023), liver disease (C028), heart disease (C033),
# cerebrovascular disease (C038), psychiatric (C043), arthritis (C048),
# plus wave 7 additions: prostate (C063), digestive (C074n), disc (C078n)

other_disease_vars = {
    'cancer':    {'w01': 'w01C016',  'wXX': 'C016'},
    'lung':      {'w01': 'w01C023',  'wXX': 'C023'},
    'liver':     {'w01': 'w01C028',  'wXX': 'C028'},
    'heart':     {'w01': 'w01C033',  'wXX': 'C033'},
    'cerebro':   {'w01': 'w01C038',  'wXX': 'C038'},
    'psych':     {'w01': 'w01C043',  'wXX': 'C043'},
    'arthritis': {'w01': 'w01C048',  'wXX': 'C048'},
}

# Other diseases: accumulate {pid: set of diseases}
other_ever = {}  # {pid: set of diseases ever had}

for dname, vinfo in other_disease_vars.items():
    # Wave 1 (existing diagnosis)
    w01_var = vinfo['w01']
    if w01_var in dfs_hist['w01'].columns:
        pids_w1 = dfs_hist['w01'][dfs_hist['w01'][w01_var] == 1]['pid'].unique()
        for pid in pids_w1:
            if pid not in other_ever: other_ever[pid] = set()
            other_ever[pid].add(dname)

    # Waves 2-6
    for wn in [2, 3, 4, 5, 6]:
        key = f'w0{wn}'
        var = f'w0{wn}{vinfo["wXX"]}'
        if var in dfs_hist[key].columns:
            pids_wn = dfs_hist[key][dfs_hist[key][var] == 1]['pid'].unique()
            for pid in pids_wn:
                if pid not in other_ever: other_ever[pid] = set()
                other_ever[pid].add(dname)

    # Wave 7
    var7 = f'w07{vinfo["wXX"]}'
    if var7 in df7.columns:
        pids_w7 = df7[df7[var7] == 1]['pid'].unique()
        for pid in pids_w7:
            if pid not in other_ever: other_ever[pid] = set()
            other_ever[pid].add(dname)

# Wave 7 additional diseases
for dname, var7 in [('prostate','w07C063'), ('digestive','w07C074n'), ('disc','w07C078n')]:
    if var7 in df7.columns:
        pids_w7 = df7[df7[var7] == 1]['pid'].unique()
        for pid in pids_w7:
            if pid not in other_ever: other_ever[pid] = set()
            other_ever[pid].add(dname)

# Cumulative comorbidity indicator: hypertension or at least one other
all_comorbid_pids = set(hyp_ever.keys()) | set(other_ever.keys())
print(f"  누적 동반질환(1종 이상) 보유자: {len(all_comorbid_pids)}명")

# Cumulative fall history
fall_ever = set()

# Wave 1: w01C056 (1=yes, 5=no)
if 'w01C056' in dfs_hist['w01'].columns:
    pids_fall1 = dfs_hist['w01'][dfs_hist['w01']['w01C056'] == 1]['pid'].unique()
    fall_ever.update(pids_fall1)

# Waves 2-6: w0XC056 (1=yes)
for wn in [2, 3, 4, 5, 6]:
    key = f'w0{wn}'
    fvar = f'w0{wn}C056'
    if fvar in dfs_hist[key].columns:
        pids_fall = dfs_hist[key][dfs_hist[key][fvar] == 1]['pid'].unique()
        fall_ever.update(pids_fall)

# Wave 7: w07C056 (1=yes)
if 'w07C056' in df7.columns:
    pids_fall7 = df7[df7['w07C056'] == 1]['pid'].unique()
    fall_ever.update(pids_fall7)

print(f"  낙상 경험(1회 이상) 이력자: {len(fall_ever)}명")

# STEP 4.5: Diabetes duration - backtrack through waves 1-7
print("\n[STEP 4.5] 당뇨 이환 기간 산출 (1~7차 역추적)...")

# Survey year per wave
WAVE_SURVEY_YEARS = {
    'w01': 2006, 'w02': 2008, 'w03': 2010,
    'w04': 2012, 'w05': 2014, 'w06': 2016,
}
BASELINE_YEAR = 2018   # wave 7 reference year

dm_first_known = {}  # {pid: int(first known year)}

# Step A: exact diagnosis year from w07C012y (newly diagnosed patients)
if 'w07C012y' in df7.columns:
    valid7 = df7[df7['w07C012y'].notna()].copy()
    valid7['_yr'] = pd.to_numeric(valid7['w07C012y'], errors='coerce')
    valid7 = valid7[(valid7['_yr'] >= 1930) & (valid7['_yr'] <= 2018)]
    for _, row in valid7.iterrows():
        dm_first_known[row['pid']] = int(row['_yr'])
print(f"  w07C012y 정확년도 확보: {len(dm_first_known)}명")

# Step B: estimate earliest wave with reported diabetes from waves 1-6
# Wave 1 (w01): diabetes diagnosis (1=yes/5=no) - try multiple candidate variable names
# Waves 2-6: diabetes status (1=new/2=recurrence/3=under treatment/4=had previously) - try multiple candidates
DM_CAND = {
    'w01': (['w01C012', 'w01C010', 'w01C014'], [1]),
    'w02': (['w02C010', 'w02C011', 'w02C012'], [1, 2, 3, 4]),
    'w03': (['w03C010', 'w03C011', 'w03C012'], [1, 2, 3, 4]),
    'w04': (['w04C010', 'w04C011', 'w04C012'], [1, 2, 3, 4]),
    'w05': (['w05C010', 'w05C011', 'w05C012'], [1, 2, 3, 4]),
    'w06': (['w06C010', 'w06C011', 'w06C012'], [1, 2, 3, 4]),
}

found_vars = {}  # log of the variable names actually used
for wave_key, wave_year in sorted(WAVE_SURVEY_YEARS.items(), key=lambda x: x[1]):
    df_w = dfs_hist[wave_key]
    candidates, pos_vals = DM_CAND[wave_key]
    dm_var = None
    for vname in candidates:
        if vname in df_w.columns:
            dm_var = vname
            break
    if dm_var is None:
        print(f"  {wave_key}: 당뇨 변수 없음 (후보: {candidates})")
        continue
    found_vars[wave_key] = dm_var
    pos_pids = df_w[df_w[dm_var].isin(pos_vals)]['pid'].unique()
    updated = 0
    for pid in pos_pids:
        curr = dm_first_known.get(pid, None)
        # If there is no earlier record than this wave's survey year, set it to that year
        # (wave year = latest lower-bound year at which diabetes was first confirmed for this wave)
        if curr is None or wave_year < curr:
            dm_first_known[pid] = wave_year
            updated += 1
    print(f"  {wave_key}({wave_year}): {dm_var} 사용, 당뇨 보유 {len(pos_pids)}명, "
          f"dm_first_known 갱신 {updated}명")

# Step B-2: cohort members not yet in dm_first_known -> assign 2018
#    w07C014 not null but not captured in waves 1-6 = newly diagnosed in wave 7
w7_new = 0
for pid in diabetic_cohort['pid'].values:
    if pid not in dm_first_known:
        dm_first_known[pid] = 2018  # first confirmed in wave 7 -> duration 0 years
        w7_new += 1
print(f"  w07(2018): 7차 신규 당뇨 {w7_new}명 → dm_first_known 배정 (유병기간=0)")

print(f"  최종 dm_first_known 확보: {len(dm_first_known)}명 / "
      f"당뇨코호트 {len(diabetic_cohort)}명")

# Step C: check diabetes-duration distribution within the cohort
cohort_pids = set(diabetic_cohort['pid'].values)
dm_dur_vals = [BASELINE_YEAR - yr for pid, yr in dm_first_known.items()
               if pid in cohort_pids and 0 <= BASELINE_YEAR - yr <= 80]
if dm_dur_vals:
    arr = np.array(dm_dur_vals)
    print(f"  이환기간 (코호트) - N={len(arr)}, "
          f"중앙값={np.median(arr):.0f}년, 평균={np.mean(arr):.1f}년, "
          f"min={arr.min():.0f}, max={arr.max():.0f}")
else:
    print("  이환기간 산출 불가 (변수 미확인)")


# STEP 5: Build predictor variables (wave 7 baseline)
print("\n[STEP 5] 예측변수 생성 중...")

def build_features(df_base, df7_full, df6_full, all_comorbid_pids, fall_ever,
                   dm_first_known):
    """Build the independent variables from the wave 7 baseline data."""
    df = df_base[['pid']].merge(df7_full, on='pid', how='left')

    # Age
    df['age'] = df['w07A002_age']

    # Sex
    # w07gender1: 1=male, 5=female -> 0=male, 1=female (binary)
    df['sex'] = (df['w07gender1'] == 5).astype(int)

    # Marital status (3 categories)
    # w07marital: 1=married, 2=separated, 3=divorced, 4=widowed, 5=never married
    # -> 1=married, 2=widowed, 3=other (separated/divorced/never married)
    def recode_marital(x):
        if pd.isna(x): return np.nan
        if x == 1:     return 1   # married
        elif x == 4:   return 2   # widowed (4=widowed, distinct from 3=divorced!)
        else:          return 3   # other (separated/divorced/never married)
    df['marital'] = df['w07marital'].apply(recode_marital)

    # Education (4 ordered categories)
    # w07edu: 1=elementary or less, 2=middle school, 3=high school, 4=college or more (used as-is)
    df['edu'] = df['w07edu'].where(df['w07edu'].isin([1,2,3,4]), other=np.nan)

    # Living arrangement (3 categories)
    # 1=living alone, 2=with spouse (2 people), 3=family of 2 or more
    def recode_living(row):
        hhsize  = row.get('w07hhsize', np.nan)
        marital = row.get('w07marital', np.nan)
        if pd.isna(hhsize): return np.nan
        if hhsize == 1:                    return 1   # living alone
        elif hhsize == 2 and marital == 1: return 2   # with spouse (2 people)
        else:                              return 3   # family of 2 or more
    df['living'] = df.apply(recode_living, axis=1)

    # Residential area
    # w07region3: 1=large city, 2=small/mid city, 3=rural -> 0=urban (large city), 1=rural (small/mid + rural)
    if 'w07region3' in df.columns:
        df['rural'] = (df['w07region3'] != 1).astype(int)
    elif 'w07region2' in df.columns:
        df['rural'] = (df['w07region2'] == 2).astype(int)
    else:
        df['rural'] = 0

    # Income (5 ordered categories)
    # w07E147: last year's total household income (unit: 10,000 KRW)
    #   -9=refused, -8=don't know -> treated as missing
    # -> 1=<1000 / 2=1000-2000 / 3=2000-3000 / 4=3000-5000 / 5=>=5000 (10,000 KRW)
    if 'w07E147' in df.columns:
        income_raw = df['w07E147'].where(df['w07E147'] > 0, other=np.nan)
        def recode_income(x):
            if pd.isna(x): return np.nan
            if x < 1000:   return 1
            elif x < 2000: return 2
            elif x < 3000: return 3
            elif x < 5000: return 4
            else:          return 5
        df['income'] = income_raw.apply(recode_income)
    else:
        df['income'] = np.nan

    # Social participation (currently working, binary)
    # w07D001: currently working (1=yes, 5=no)
    if 'w07D001' in df.columns:
        df['work_participation'] = (df['w07D001'] == 1).astype(int)
    else:
        df['work_participation'] = 0

    # ADL score (0-7, continuous)
    # C201-C207: 1 point if >=3, summed
    adl_vars = ['w07C201','w07C202','w07C203','w07C204','w07C205','w07C206','w07C207']
    for v in adl_vars:
        if v in df.columns:
            df[f'_{v}_bin'] = (df[v] >= 3).astype(float)
            df[f'_{v}_bin'] = df[f'_{v}_bin'].where(df[v].notna(), other=np.nan)
    adl_bin_cols = [f'_{v}_bin' for v in adl_vars if f'_{v}_bin' in df.columns]
    df['adl_score'] = df[adl_bin_cols].sum(axis=1)

    # IADL score (0-10, continuous)
    # C208-C217: 1 point if >=3, summed
    iadl_vars = ['w07C208','w07C209','w07C210','w07C211','w07C212',
                 'w07C213','w07C214','w07C215','w07C216','w07C217']
    for v in iadl_vars:
        if v in df.columns:
            df[f'_{v}_bin'] = (df[v] >= 3).astype(float)
            df[f'_{v}_bin'] = df[f'_{v}_bin'].where(df[v].notna(), other=np.nan)
    iadl_bin_cols = [f'_{v}_bin' for v in iadl_vars if f'_{v}_bin' in df.columns]
    df['iadl_score'] = df[iadl_bin_cols].sum(axis=1)

    # Grip strength decrease (based on dominant hand)
    # w07C501: 1=right-handed, 3=ambidextrous, 5=left-handed
    # Use the maximum of two measurements of the dominant hand
    # Compare wave 7 vs wave 6 -> decrease (1) if wave 7 < wave 6
    def get_dominant_grip(row, wave='w07'):
        """Identify the dominant hand and return its maximum grip over two measurements."""
        hand_var = f'{wave}C501'
        r1, r2 = f'{wave}C507', f'{wave}C508'  # right hand, measurements 1 and 2
        l1, l2 = f'{wave}C509', f'{wave}C510'  # left hand, measurements 1 and 2
        handedness = row.get(hand_var, np.nan)
        if handedness == 1:        # right-handed -> right hand
            vals = [row.get(r1, np.nan), row.get(r2, np.nan)]
        elif handedness == 5:      # left-handed -> left hand
            vals = [row.get(l1, np.nan), row.get(l2, np.nan)]
        else:                      # ambidextrous (3) or missing -> overall max
            vals = [row.get(r1, np.nan), row.get(r2, np.nan),
                    row.get(l1, np.nan), row.get(l2, np.nan)]
        valid = [v for v in vals if pd.notna(v) and v > 0]
        return max(valid) if valid else np.nan

    # Wave 7 dominant-hand grip
    grip_cols_7 = ['w07C501','w07C507','w07C508','w07C509','w07C510']
    if all(c in df.columns for c in grip_cols_7):
        df['grip_7'] = df.apply(lambda r: get_dominant_grip(r, 'w07'), axis=1)
    else:
        df['grip_7'] = np.nan

    # Wave 6 dominant-hand grip (use wave 6 handedness if available, otherwise wave 7)
    grip_cols_6 = ['w06C501','w06C507','w06C508','w06C509','w06C510']
    if all(c in df6_full.columns for c in grip_cols_6):
        df6_grip = df6_full[['pid'] + grip_cols_6].copy()
        df6_grip['grip_6'] = df6_grip.apply(lambda r: get_dominant_grip(r, 'w06'), axis=1)
        df = df.merge(df6_grip[['pid','grip_6']], on='pid', how='left')
    else:
        df['grip_6'] = np.nan

    df['grip_decreased'] = np.where(
        df['grip_7'].notna() & df['grip_6'].notna(),
        (df['grip_7'] < df['grip_6']).astype(int),
        np.nan
    )

    # Current drinking
    # w07C122: 1=yes, 5=no
    df['drinking'] = (df['w07C122'] == 1).astype(int) if 'w07C122' in df.columns else 0

    # Current smoking
    # w07C117: 1=yes, 5=no
    df['smoking'] = (df['w07C117'] == 1).astype(int) if 'w07C117' in df.columns else 0

    # Regular exercise
    # w07C108: 1=yes, 5=no
    df['exercise'] = (df['w07C108'] == 1).astype(int) if 'w07C108' in df.columns else 0

    # BMI (4 ordered categories)
    # w07C105: weight (kg), w07C107: height (cm)
    # -9/-8 treated as missing, implausible values removed
    # -> 1=underweight (BMI<18.5), 2=normal (18.5-25), 3=overweight (25-30), 4=obese (>=30)
    if 'w07C105' in df.columns and 'w07C107' in df.columns:
        weight = df['w07C105'].where(df['w07C105'] >= 30, other=np.nan)
        height = df['w07C107'].where(df['w07C107'] >= 100, other=np.nan)
        df['bmi'] = weight / (height / 100) ** 2
    else:
        df['bmi'] = np.nan

    def recode_bmi(bmi):
        if pd.isna(bmi): return np.nan
        elif bmi < 18.5: return 1   # underweight
        elif bmi < 25:   return 2   # normal
        elif bmi < 30:   return 3   # overweight
        else:            return 4   # obese
    df['bmi_cat'] = df['bmi'].apply(recode_bmi)

    # Fall experience (cumulative history, waves 1-7)
    df['fall_exp'] = df['pid'].isin(fall_ever).astype(int)

    # CES-D total score (10 items, continuous, 0-30)
    # w07C142-C151: 1=rarely (0), 2=sometimes (1), 3=often (2), 4=most of the time (3)
    # Reverse-coded items: C146 (getting along well), C149 (without complaint)
    cesd_items = ['w07C142','w07C143','w07C144','w07C145','w07C146',
                  'w07C147','w07C148','w07C149','w07C150','w07C151']
    cesd_reverse = ['w07C146', 'w07C149']
    for c in cesd_items:
        if c in df.columns:
            df[f'_cesd_{c}'] = df[c].apply(lambda x: x - 1 if x in [1,2,3,4] else np.nan)
    for c in cesd_reverse:
        col_name = f'_cesd_{c}'
        if col_name in df.columns:
            df[col_name] = 3 - df[col_name]
    cesd_cols = [f'_cesd_{c}' for c in cesd_items if f'_cesd_{c}' in df.columns]
    df['cesd_total'] = df[cesd_cols].sum(axis=1)

    # Hearing (continuous, 1-5)
    # w07C083: 1=very good ... 5=very poor (a major modifiable risk factor per the Lancet Commission)
    if 'w07C083' in df.columns:
        df['hearing'] = df['w07C083'].where(df['w07C083'].isin([1,2,3,4,5]), other=np.nan)
    else:
        df['hearing'] = np.nan

    # Self-rated health (continuous, 1-5)
    # w07C152: 1=very good ... 5=very poor
    if 'w07C152' in df.columns:
        df['subjective_health'] = df['w07C152'].where(df['w07C152'].isin([1,2,3,4,5]), other=np.nan)
    else:
        df['subjective_health'] = np.nan

    # Cerebrovascular disease (stroke), currently treated (binary)
    # w07C041: 1=under treatment, 5=no -> 1 if under treatment or C038 diagnosis history = 1
    if 'w07C041' in df.columns:
        df['stroke'] = (df['w07C041'] == 1).astype(int)
    elif 'w07C038' in df.columns:
        df['stroke'] = (df['w07C038'] == 1).astype(int)
    else:
        df['stroke'] = 0

    # Heart disease, currently treated (binary)
    # w07C036: 1=under treatment, 5=no
    if 'w07C036' in df.columns:
        df['heart_disease'] = (df['w07C036'] == 1).astype(int)
    elif 'w07C033' in df.columns:
        df['heart_disease'] = (df['w07C033'] == 1).astype(int)
    else:
        df['heart_disease'] = 0

    # Hypertension, currently treated (binary)
    # w07C009: 1=under treatment, 5=no
    if 'w07C009' in df.columns:
        df['hypertension'] = (df['w07C009'] == 1).astype(int)
    elif 'w07C006' in df.columns:
        df['hypertension'] = (df['w07C006'] == 1).astype(int)
    else:
        df['hypertension'] = 0

    # Cardio-cerebrovascular disease count (composite, 0-3)
    # stroke + heart_disease + hypertension; combines 3 variables into 1 to improve EPV
    df['cvd_count'] = df['stroke'].fillna(0) + df['heart_disease'].fillna(0) + df['hypertension'].fillna(0)

    # Functional impairment composite score (continuous)
    # adl_score + iadl_score; combines 2 variables into 1
    df['functional_score'] = df['adl_score'].fillna(0) + df['iadl_score'].fillna(0)

    # Established long-term diabetes (binary)
    # Confirmed in waves 1-6 = diabetes for at least 2 years
    # Confirmed only in wave 7 = newly diagnosed or not captured in earlier waves
    # Reference: Rawlings et al. (2014) Diabetes Care - long-term diabetes increases cognitive-decline risk
    BASELINE_YR = 2018
    dm_dur_map = {pid: BASELINE_YR - yr for pid, yr in dm_first_known.items()}
    df['dm_established'] = df['pid'].map(dm_dur_map).fillna(0)
    df['dm_established'] = (df['dm_established'] > 0).astype(int)  # 1=confirmed in earlier wave, 0=wave 7 only

    return df


# Build features
print("  MCI 모델 피처 생성 중...")
mci_features = build_features(mci_base, df7, df6, all_comorbid_pids, fall_ever,
                               dm_first_known)
print("  치매 모델 피처 생성 중...")
dem_features = build_features(dem_base, df7, df6, all_comorbid_pids, fall_ever,
                               dm_first_known)

# baseline_mci: baseline cognitive status (dementia model only)
# The dementia cohort includes normal (5) + MCI (3); MCI is a strong predictor of dementia conversion
dem_features['baseline_mci'] = (dem_base['w07Cadd_01'].values == 3).astype(int)
mci_features['baseline_mci'] = 0  # MCI model is entirely normal (5) -> constant (auto-dropped by SHAP)
print(f"  치매 코호트 기저 MCI: {dem_features['baseline_mci'].sum()}명 / {len(dem_features)}명")

# STEP 6: Handle missing values
print("\n[STEP 6] 결측치 처리...")

# Feature columns; SHAP selects the most informative subset, so all candidates are included here.
FEATURE_COLS = [
    # Demographic characteristics (7)
    'age',              # continuous (45-98 years)
    'sex',              # 0=male, 1=female (binary)
    'marital',          # 1=married, 2=widowed, 3=other
    'edu',              # 1=elementary or less, 2=middle, 3=high school, 4=college or more
    'living',           # 1=living alone, 2=with spouse (2 people), 3=family of 2 or more
    'rural',            # 0=urban, 1=rural
    'income',           # 1=<1000, 2=1000-2000, 3=2000-3000, 4=3000-5000, 5=>=5000 (10,000 KRW)
    # Functional status (2)
    'adl_score',        # continuous (0-7), ADL score
    'iadl_score',       # continuous (0-10), IADL score
    # Health behaviors (6)
    'grip_decreased',   # 0=maintained/increased, 1=decreased (dominant hand)
    'drinking',         # 0=no, 1=yes
    'smoking',          # 0=no, 1=yes
    'exercise',         # 0=no, 1=yes
    'bmi_cat',          # 1=underweight, 2=normal, 3=overweight, 4=obese
    'fall_exp',         # 0=no, 1=yes (cumulative waves 1-7)
    # Social participation
    'work_participation', # 0=not economically active, 1=currently working
    # Mental health
    'cesd_total',       # continuous (0-30), CES-D 10-item total score
    # Sensory function
    'hearing',          # continuous (1-5), 1=very good ... 5=very poor
    # Self-rated health
    'subjective_health',# continuous (1-5), 1=very good ... 5=very poor
    # Cardio-cerebrovascular disease (3)
    'stroke',           # 0=no, 1=yes (cumulative waves 1-7)
    'heart_disease',    # 0=no, 1=yes (cumulative waves 1-7)
    'hypertension',     # 0=no, 1=yes (cumulative waves 1-7)
]

print(f"  피처 수: {len(FEATURE_COLS)}")

def handle_missing(df, feature_cols, verbose=True):
    df = df.copy()
    total_missing = 0
    for col in feature_cols:
        if col not in df.columns:
            if verbose: print(f"  변수 없음: {col} (0으로 채움)")
            df[col] = 0
            continue
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        total_missing += null_count
        null_pct = null_count / len(df) * 100
        if verbose and null_pct > 3:
            print(f"  결측 {null_pct:.1f}%: {col} ({null_count}개) → 중앙값/최빈값 대체")
        if df[col].dtype in ['float64', 'int64']:
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(0)
    if verbose:
        print(f"  총 결측치 처리: {total_missing}개")
    return df

print("  MCI 모델 결측치 처리...")
mci_features = handle_missing(mci_features, FEATURE_COLS)
print("  치매 모델 결측치 처리...")
dem_features = handle_missing(dem_features, FEATURE_COLS)

# STEP 7: Split feature matrix and outcome
print("\n[STEP 7] Feature Matrix 구성...")

mci_features['MCI_incident'] = mci_base['MCI_incident'].values
dem_features['Dementia_incident'] = dem_base['Dementia_incident'].values

X_mci = mci_features[FEATURE_COLS].astype(float)
y_mci = mci_features['MCI_incident'].astype(int)
X_dem = dem_features[FEATURE_COLS].astype(float)
y_dem = dem_features['Dementia_incident'].astype(int)

print(f"  MCI 모델  - X: {X_mci.shape}, 발생: {y_mci.sum()}명({y_mci.mean()*100:.2f}%)")
print(f"  치매 모델 - X: {X_dem.shape}, 발생: {y_dem.sum()}명({y_dem.mean()*100:.2f}%)")

# STEP 8: Print variable codebook
print("\n[STEP 8] 변수 현황 코드북 출력...")

def print_codebook(X_mci, X_dem, feature_cols):
    """Variable codebook: number/name/description/type/raw variable/value/missing count/distribution."""
    n = len(X_mci)

    codebook = [
        ( 1, 'age',              '나이',                  '연속형',    'w07A002_age',       '연속값 (세)'),
        ( 2, 'sex',              '성별',                  '이진형',    'w07gender1',        '0=남, 1=여'),
        ( 3, 'marital',          '혼인상태',              '범주형(3)', 'w07marital',        '1=혼인중, 2=사별, 3=기타'),
        ( 4, 'edu',              '교육수준',              '범주형(4)', 'w07edu',            '1=초졸이하, 2=중졸, 3=고졸, 4=대졸이상'),
        ( 5, 'living',           '거주형태',              '범주형(3)', 'w07hhsize',         '1=독거, 2=배우자, 3=2인이상'),
        ( 6, 'rural',            '거주지역',              '이진형',    'w07region3',        '0=도시, 1=농촌'),
        ( 7, 'income',           '가구 총소득',           '범주형(5)', 'w07E147',           '1~5(5분위)'),
        ( 8, 'work_participation','현재 노동 여부',        '이진형',    'w07D001',           '0=아니오, 1=예'),
        ( 9, 'adl_score',        'ADL 점수',              '연속형',    'w07C201~C207',      '0~7점'),
        (10, 'iadl_score',       'IADL 점수',             '연속형',    'w07C208~C217',      '0~10점'),
        (11, 'grip_decreased',   '악력 감소(주손)',         '이진형',    'w07C501+C507~510/w06C501+C507~510', '0=유지, 1=감소'),
        (12, 'drinking',         '현재 음주',             '이진형',    'w07C122',           '0=아니오, 1=예'),
        (13, 'smoking',          '현재 흡연',             '이진형',    'w07C117',           '0=아니오, 1=예'),
        (14, 'exercise',         '규칙적 운동',           '이진형',    'w07C108',           '0=아니오, 1=예'),
        (15, 'bmi_cat',          'BMI 비만도',            '범주형(4)', 'w07C105+C107',      '1=저체중~4=비만'),
        (16, 'fall_exp',         '낙상경험 (1~7차)',       '이진형',    'w01~w07C056',       '0=없음, 1=있음'),
        (17, 'cesd_total',       'CES-D 우울총점',         '연속형',    'w07C142~C151',      '0~30점(10항목)'),
        (18, 'hearing',          '청력',                   '연속형',    'w07C083',           '1=좋음~5=나쁨'),
        (19, 'subjective_health','주관적 건강',             '연속형',    'w07C152',           '1=좋음~5=나쁨'),
        (20, 'stroke',           '뇌혈관질환 치료',         '이진형',    'w07C041',           '0=없음, 1=치료중'),
        (21, 'heart_disease',    '심장질환 치료',           '이진형',    'w07C036',           '0=없음, 1=치료중'),
        (22, 'hypertension',     '고혈압 치료',             '이진형',    'w07C009',           '0=없음, 1=치료중'),
    ]

    # Pre-imputation missingness info (manual)
    premiss = {
        'grip_decreased': (398, 32.0),
        'income':         (8, 0.6),
        'bmi_cat':        (5, 0.4),
    }

    SEP = " | "
    hdr = ("번호".rjust(4) + SEP + "변수명".ljust(20) + SEP + "변수내용".ljust(16) +
           SEP + "유형".ljust(10) + SEP + "원시변수".ljust(18) +
           SEP + "Value 설명".ljust(38) + SEP + "결측(처리전)".ljust(14) + SEP + "MCI 분포")
    print()
    print(hdr)
    print("-" * 160)

    for row in codebook:
        num, vname, desc, dtype, raw, val = row

        # Missingness info
        if vname in premiss:
            nc, np_ = premiss[vname]
            miss_str = str(nc) + "(" + str(np_) + "%)"
        else:
            miss_str = "0(0.0%)"

        # Distribution
        col = X_mci[vname] if vname in X_mci.columns else pd.Series(dtype=float)
        if dtype == '연속형':
            dist = "평균 " + str(round(float(col.mean()), 2))
        elif dtype == '이진형':
            pct = round(float(col.mean()) * 100, 1)
            dist = str(pct) + "% 해당"
        else:
            vc = col.value_counts().sort_index()
            dist = " | ".join([str(int(k)) + "=" + str(int(v)) for k, v in vc.items()])

        print(str(num).rjust(4) + SEP + vname.ljust(20) + SEP + desc.ljust(16) +
              SEP + dtype.ljust(10) + SEP + raw.ljust(18) +
              SEP + val.ljust(38) + SEP + miss_str.ljust(14) + SEP + dist)

    print()
    print("  [요약] 총 독립변수: " + str(len(feature_cols)) + "개")
    print("    연속형  (6개): age, adl_score, iadl_score, cesd_total, hearing, subjective_health")
    print("    범주형  (3개): marital(3단계), edu(4단계), living(3단계), income(5단계), bmi_cat(4단계)")
    print("    이진형  (9개): sex, rural, work_participation, grip_decreased,")
    print("                   drinking, smoking, exercise, fall_exp, stroke, heart_disease, hypertension")

print_codebook(X_mci, X_dem, FEATURE_COLS)

# STEP 9: Save
print("\n[STEP 9] 전처리 결과 저장...")

import pickle
save_path = os.path.join(DATA_DIR, 'preprocessed_data.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({
        'X_mci': X_mci, 'y_mci': y_mci,
        'X_dem': X_dem, 'y_dem': y_dem,
        'feature_cols': FEATURE_COLS,
    }, f)
print(f"  저장 완료: {save_path}")

print("\n" + "=" * 60)
print("전처리 완료!")
print("=" * 60)
