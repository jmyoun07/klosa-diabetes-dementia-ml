# -*- coding: utf-8 -*-
"""Competing-risk analysis with DEATH as the competing event (from the KLoSA exit
survey) and non-death dropout treated as censoring / missingness."""
import glob, os, pickle, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings("ignore")
from lifelines import AalenJohansenFitter, KaplanMeierFitter, CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SP = os.path.dirname(os.path.abspath(__file__))
# Directory holding the raw KLoSA wave files (w08_*.sav, w09_*.sav, w10_*.sav) and
# final_model/output/_recheck_preprocessed.pkl. Set KLOSA_DATA_DIR to its location;
# the KLoSA microdata are access-controlled and not distributed with this code.
BASE = os.environ.get("KLOSA_DATA_DIR", os.path.join(os.path.dirname(SP), "data"))
# Individual-level death linkage (pid, death_year, death_month, seen_w10), derived
# from the KLoSA exit survey. Not redistributable; see README.md for how to build it.
link = pd.read_csv(os.path.join(SP, "cohort_death_linkage.csv"))

# ---- outcome timing: first physician-reported dementia / MCI at wave 8, 9, 10
status = {"pid": link["pid"].values}
for wv, yrs in (("w08", 2), ("w09", 4), ("w10", 6)):
    f = glob.glob(os.path.join(BASE, "%s_*.sav" % wv))[0]
    col = [c for c in pyreadstat.read_sav(f, metadataonly=True)[1].column_names
           if c.lower() == "%scadd_01" % wv][0]
    d, _ = pyreadstat.read_sav(f, usecols=["pid", col])
    d["pid"] = d["pid"].astype("int64")
    status[wv] = link["pid"].map(d.set_index("pid")[col]).values
S = pd.DataFrame(status)

WAVE_YEARS = {"w08": 2, "w09": 4, "w10": 6}


def build(outcome_code):
    """time, event: 0 censored, 1 outcome, 2 death."""
    t, e = [], []
    for i, row in S.iterrows():
        dem_t = None
        for wv, yr in WAVE_YEARS.items():
            v = row[wv]
            if pd.notna(v) and v == outcome_code:
                dem_t = yr
                break
        dy = link["death_year"].iloc[i]
        dm = link["death_month"].iloc[i]
        death_t = None
        if pd.notna(dy) and 2018 <= dy <= 2024:
            death_t = (dy - 2018) + (dm - 7) / 12.0 if pd.notna(dm) else (dy - 2018)
            death_t = max(0.05, min(6.0, death_t))
        last_seen = 0.0
        for wv, yr in WAVE_YEARS.items():
            if pd.notna(row[wv]):
                last_seen = yr
        if dem_t is not None and (death_t is None or dem_t <= death_t):
            t.append(dem_t); e.append(1)
        elif death_t is not None:
            t.append(death_t); e.append(2)
        else:
            t.append(last_seen if last_seen > 0 else 0.05); e.append(0)
    return np.array(t, float), np.array(e, int)


print("=" * 84)
print("DEMENTIA COHORT (n = %d): death identified from the KLoSA exit survey" % len(link))
print("=" * 84)
t, e = build(1)
n = len(t)
print("  events: dementia %d | death %d | censored (non-death dropout / administrative) %d"
      % ((e == 1).sum(), (e == 2).sum(), (e == 0).sum()))

aj = AalenJohansenFitter(calculate_variance=True, seed=42)
aj.fit(t, e, event_of_interest=1)
ci_dem = float(aj.cumulative_density_.iloc[-1, 0])
aj2 = AalenJohansenFitter(calculate_variance=False, seed=42)
aj2.fit(t, e, event_of_interest=2)
ci_death = float(aj2.cumulative_density_.iloc[-1, 0])
km = KaplanMeierFitter().fit(t, (e == 1).astype(int))
km_naive = 1 - float(km.survival_function_.iloc[-1, 0])
crude = (e == 1).mean()

print("\n  Six-year cumulative incidence of dementia")
print("    Aalen-Johansen, DEATH as competing event      : %.4f" % ci_dem)
print("    naive Kaplan-Meier complement (censors death) : %.4f  (+%.1f%% relative)"
      % (km_naive, 100 * (km_naive - ci_dem) / ci_dem))
print("    crude proportion (binary outcome used)        : %.4f" % crude)
print("    Aalen-Johansen cumulative incidence of DEATH  : %.4f" % ci_death)

# ---------------- Fine-Gray vs cause-specific Cox, death as competing event
pre = pickle.load(open(os.path.join(BASE, "final_model", "output", "_recheck_preprocessed.pkl"), "rb"))
X = np.asarray(pre["X_dem"], float)
cols = list(pre["feature_cols"])
df = pd.DataFrame(X, columns=cols)
df["T"], df["E"] = t, e

cs = CoxPHFitter(penalizer=0.1).fit(
    df.assign(ev=(df["E"] == 1).astype(int)).drop(columns=["E"]), "T", "ev")
cs_coef = cs.params_

# Fine-Gray via the subdistribution risk set: those who die are kept in the risk
# set to the horizon (administrative censoring only at 6 years)
fg_df = df.copy()
fg_df.loc[fg_df["E"] == 2, "T"] = 6.0
fg = CoxPHFitter(penalizer=0.1).fit(
    fg_df.assign(ev=(fg_df["E"] == 1).astype(int)).drop(columns=["E"]), "T", "ev")
fg_coef = fg.params_

print("\n  Fine-Gray (subdistribution) vs cause-specific Cox, death as the competing event")
print("    coefficients agree in sign for %d of %d predictors"
      % (int((np.sign(fg_coef) == np.sign(cs_coef)).sum()), len(cols)))
print("    Pearson r = %.3f | Spearman of |coef| ranks = %.3f"
      % (np.corrcoef(fg_coef, cs_coef)[0, 1],
         pd.Series(fg_coef).abs().rank().corr(pd.Series(cs_coef).abs().rank(), method="spearman")))

# ---------------- IPCW for NON-DEATH dropout only, among those still alive
alive = (e != 2) | (t >= 6.0)
observed_w10 = link["seen_w10"].values.astype(bool)
died_in = np.array([pd.notna(y) and 2018 <= y <= 2024 for y in link["death_year"]])
# censoring = alive at 6 years but not assessed at wave 10
elig = ~died_in
cens = elig & ~observed_w10
print("\n  Non-death dropout (censoring), estimated among survivors only")
print("    survivors (not known to have died) : %d" % elig.sum())
print("    of whom not assessed at wave 10    : %d (%.1f%% of survivors)"
      % (cens.sum(), 100 * cens.sum() / elig.sum()))

Xe, ye = X[elig], (~cens[elig]).astype(int)      # 1 = ascertained
pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
p = cross_val_predict(pipe, Xe, ye, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      method="predict_proba")[:, 1]
print("    ascertainment among survivors is predictable from baseline: AUC %.3f"
      % roc_auc_score(ye, p))
p = np.clip(p, 0.05, 1.0)
w = ye.mean() / p
y_dem = np.asarray(pre["y_dem"]).astype(int)[elig]
ipcw_inc = float(np.sum(w * ye * y_dem) / np.sum(w * ye))
print("    six-year cumulative incidence of dementia among survivors:")
print("      unweighted, ascertained only : %.4f" % y_dem[ye == 1].mean())
print("      IPCW-weighted                : %.4f" % ipcw_inc)

pd.DataFrame([dict(
    n=n, dementia=(e == 1).sum(), deaths=(e == 2).sum(), censored=(e == 0).sum(),
    aj_dementia=ci_dem, km_naive=km_naive, crude=crude, aj_death=ci_death,
    survivors=int(elig.sum()), survivor_dropout=int(cens.sum()),
    ipcw_incidence_survivors=ipcw_inc,
    unweighted_incidence_survivors=float(y_dem[ye == 1].mean()))]).to_csv(
    os.path.join(SP, "competing_death_results.csv"), index=False)
print("\nwrote competing_death_results.csv")
