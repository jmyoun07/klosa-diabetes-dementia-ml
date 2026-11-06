# -*- coding: utf-8 -*-
"""K-MMSE variable availability audit (Supplementary Table S16).

The outcome in this study is a self-reported physician diagnosis rather than a
standardized cognitive measure. This script audits what the KLoSA release actually
contains for the Korean Mini-Mental State Examination (K-MMSE):

  * The cognitive-function and MMSE generated variables that were previously
    distributed for waves 1-9 have been withdrawn by the Korea Employment
    Information Service (KEIS) and are no longer included in the release, with no
    plan to re-issue them (per the KEIS data notice).
  * Waves 7-9 of the release used here therefore contain no cognitive test.
  * Wave 10 (2024) contains the 19 raw K-MMSE items (w10C401-w10C419), which can be
    scored to the standard 30-point total. Item coding: 5 = no sub-item correct
    (0 points), 1-4 = number of sub-items correct, and -8/-9 = refusal / don't know
    (treated as missing).

Because the K-MMSE exists only at the final wave, it cannot define an incident
outcome (there is no baseline measurement). This audit quantifies, within the
analytic cohort, how many participants have a usable wave-10 K-MMSE and the
item-level response availability that determines it.

Output -> kmmse_variable_audit.csv (+ console log)
"""
import glob, os, warnings
import numpy as np, pandas as pd, pyreadstat
warnings.filterwarnings("ignore")

# Directory holding the raw KLoSA wave files. The KLoSA microdata are
# access-controlled and are not distributed with this code; set KLOSA_DATA_DIR to
# the directory that contains w07_*.sav and w10_*.sav.
BASE_DIR = os.environ.get("KLOSA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

ITEMS = [f"w10C4{i:02d}" for i in range(1, 20)]        # 19 K-MMSE items
MAXP = {"w10C401": 3, "w10C402": 3, "w10C403": 1, "w10C404": 1, "w10C405": 4, "w10C406": 1,
        "w10C407": 3, "w10C408": 1, "w10C409": 1, "w10C410": 1, "w10C411": 1, "w10C412": 1,
        "w10C413": 1, "w10C414": 1, "w10C415": 1, "w10C416": 3, "w10C417": 1, "w10C418": 1,
        "w10C419": 1}
assert sum(MAXP.values()) == 30

# What the current KLoSA release contains for cognitive assessment, wave by wave.
COGNITIVE_WAVES = [
    ("w01", None), ("w02", None), ("w03", None), ("w04", None), ("w05", None),
    ("w06", None), ("w07", None), ("w08", None), ("w09", None),
    ("w10", "19 raw K-MMSE items (w10C401-w10C419)"),
]


def sav(wave, cols):
    f = glob.glob(os.path.join(BASE_DIR, "%s_*.sav" % wave))
    if not f:
        raise SystemExit("Missing %s_*.sav in %s. Set KLOSA_DATA_DIR to the KLoSA "
                         "data directory (the microdata are access-controlled and are "
                         "not distributed with this code)." % (wave, BASE_DIR))
    df, _ = pyreadstat.read_sav(f[0], usecols=cols)
    return df


# ---- analytic cohort: age >= 45, treated diabetes, cognitively normal at baseline
df7 = sav("w07", ["pid", "w07A002_age", "w07C014", "w07Cadd_01", "w07gender1", "w07F001"])
c = df7[df7["w07A002_age"] >= 45].copy()
c = c[c["w07C014"].notna()]                      # currently treated for diabetes
c = c[c["w07Cadd_01"].notna() & (c["w07Cadd_01"] != 1)]   # not demented at baseline
c = c[c["w07gender1"].notna() & c["w07F001"].notna()]
mci = c[c["w07Cadd_01"] == 5]["pid"]             # baseline normal -> MCI cohort
dem = c[c["w07Cadd_01"].isin([3, 5])]["pid"]     # baseline non-demented -> dementia cohort

# ---- wave-10 K-MMSE items, scored
w10 = sav("w10", ["pid"] + ITEMS)
for it in ITEMS:
    w10[it] = w10[it].where(w10[it] > 0)         # -8/-9 refusals and non-responses -> missing
score = pd.DataFrame({it: np.minimum(np.where(w10[it] == 5, 0, w10[it]), MAXP[it]) for it in ITEMS},
                     index=w10.index)
w10["mmse_total"] = score.sum(axis=1, min_count=len(ITEMS))   # NaN unless all 19 items present

# ---- provenance table
print("=" * 78)
print("K-MMSE availability by wave in the KLoSA release used here")
print("=" * 78)
prov = []
for wv, note in COGNITIVE_WAVES:
    print("  %-5s %s" % (wv, note or "no cognitive test / MMSE variable"))
    prov.append(dict(wave=wv, cognitive_content=note or "none (withdrawn by KEIS for w01-w09)"))


def audit(name, pids):
    d = w10[w10["pid"].isin(pids)]
    n = len(pids)
    rows = []
    for it in ITEMS:
        present = int(d[it].notna().sum())
        rows.append(dict(cohort=name, item=it, max_points=MAXP[it],
                         n_present=present, pct_present=round(100 * present / n, 1)))
    complete = int(d["mmse_total"].notna().sum())
    print("\n  %s (n = %d)" % (name, n))
    print("    participants with a complete 19-item K-MMSE at wave 10: %d (%.1f%%)"
          % (complete, 100 * complete / n))
    if complete:
        print("    mean total among those scored: %.1f (SD %.1f)"
              % (d["mmse_total"].mean(), d["mmse_total"].std()))
    print("    per-item availability: min %.1f%%, max %.1f%%"
          % (min(r["pct_present"] for r in rows), max(r["pct_present"] for r in rows)))
    return rows, dict(cohort=name, n=n, complete_kmmse=complete,
                      pct_complete=round(100 * complete / n, 1))


print("\n" + "=" * 78)
print("Item-level K-MMSE availability within the analytic cohort")
print("=" * 78)
item_rows, cohort_rows = [], []
for name, pids in [("MCI cohort", mci), ("Dementia cohort", dem)]:
    r, summ = audit(name, pids)
    item_rows += r
    cohort_rows.append(summ)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmmse_variable_audit.csv")
pd.concat([pd.DataFrame(prov), pd.DataFrame(cohort_rows), pd.DataFrame(item_rows)],
          axis=0, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote", os.path.basename(out))
