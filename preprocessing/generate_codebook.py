"""Generate the predictor codebook (Supplementary Table S1).

The codebook is built directly from the analytic feature set stored in
preprocessed_data.pkl, so the variable list always matches the model inputs. For
each of the 22 predictors it records the KLoSA source item(s), the coding, the
pre-imputation missingness, and the distribution in the analytic cohort.
"""
import os
import pickle
import pandas as pd

BASE_DIR = os.environ.get("KLOSA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

with open(os.path.join(BASE_DIR, "preprocessed_data.pkl"), "rb") as f:
    data = pickle.load(f)
X_mci, y_mci = data["X_mci"], data["y_mci"]
X_dem, y_dem = data["X_dem"], data["y_dem"]
FEATURE_COLS = list(data["feature_cols"])

# Predictor metadata (description, type, KLoSA source item, coding), kept in sync
# with the derivation in klosa_preprocessing.py.
META = {
    "age":               ("Age", "continuous", "w07A002_age", "years"),
    "sex":               ("Sex", "binary", "w07gender1", "0=male, 1=female"),
    "marital":           ("Marital status", "categorical(3)", "w07marital", "1=married, 2=widowed, 3=other"),
    "edu":               ("Education level", "categorical(4)", "w07edu", "1=<=elementary, 2=middle, 3=high, 4=>=college"),
    "living":            ("Living arrangement", "categorical(3)", "w07hhsize", "1=alone, 2=with spouse, 3=two or more"),
    "rural":             ("Residential area", "binary", "w07region3", "0=urban, 1=rural"),
    "income":            ("Household income", "categorical(5)", "w07E147", "1-5 (quintiles)"),
    "work_participation": ("Current employment", "binary", "w07D001", "0=no, 1=yes"),
    "adl_score":         ("ADL score", "continuous", "w07C201-C207", "0-7"),
    "iadl_score":        ("IADL score", "continuous", "w07C208-C217", "0-10"),
    "grip_decreased":    ("Grip-strength decline", "binary", "w07C501,C507-C510 vs w06", "0=maintained, 1=decreased"),
    "drinking":          ("Current drinking", "binary", "w07C122", "0=no, 1=yes"),
    "smoking":           ("Current smoking", "binary", "w07C117", "0=no, 1=yes"),
    "exercise":          ("Regular exercise", "binary", "w07C108", "0=no, 1=yes"),
    "bmi_cat":           ("BMI category", "categorical(4)", "w07C105,C107", "1=underweight ... 4=obese"),
    "fall_exp":          ("Fall experience (w01-w07)", "binary", "w01-w07C056", "0=no, 1=yes"),
    "cesd_total":        ("CES-D depression score", "continuous", "w07C142-C151", "0-30 (10 items)"),
    "hearing":           ("Hearing", "continuous", "w07C083", "1=good ... 5=poor"),
    "subjective_health": ("Self-rated health", "continuous", "w07C152", "1=good ... 5=poor"),
    "stroke":            ("Cerebrovascular disease (treated)", "binary", "w07C041", "0=no, 1=under treatment"),
    "heart_disease":     ("Heart disease (treated)", "binary", "w07C036", "0=no, 1=under treatment"),
    "hypertension":      ("Hypertension (treated)", "binary", "w07C009", "0=no, 1=under treatment"),
}

# Pre-imputation missingness (percent of the analytic cohort). Every other
# predictor was complete before imputation.
PRE_IMPUTATION_MISSING = {"grip_decreased": 32.0, "income": 0.6, "bmi_cat": 0.4}


def distribution(name, vtype):
    if name not in getattr(X_dem, "columns", []):
        return ""
    s = X_dem[name]
    if vtype.startswith("continuous"):
        return "mean %.2f (SD %.2f)" % (s.mean(), s.std())
    if vtype.startswith("binary"):
        return "%.1f%% positive" % (s.mean() * 100)
    vc = s.value_counts().sort_index()
    return "; ".join("%g=%d" % (k, v) for k, v in vc.items())


rows = []
for i, name in enumerate(FEATURE_COLS, 1):
    desc, vtype, source, coding = META.get(name, (name, "", "", ""))
    rows.append({
        "no": i,
        "variable": name,
        "description": desc,
        "type": vtype,
        "source_item": source,
        "coding": coding,
        "pre_imputation_missing_pct": PRE_IMPUTATION_MISSING.get(name, 0.0),
        "distribution": distribution(name, vtype),
    })

codebook = pd.DataFrame(rows)
print("Predictor codebook (%d variables; MCI cohort n=%d, dementia cohort n=%d)"
      % (len(FEATURE_COLS), len(X_mci), len(X_dem)))
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
print(codebook.to_string(index=False))

out = os.path.join(BASE_DIR, "variable_codebook.csv")
codebook.to_csv(out, index=False, encoding="utf-8-sig")
print("\nsaved:", out)

print("\nOutcomes:")
print("  MCI:      n=%d, events=%d (%.1f%%)" % (len(y_mci), int(y_mci.sum()), y_mci.mean() * 100))
print("  Dementia: n=%d, events=%d (%.1f%%)" % (len(y_dem), int(y_dem.sum()), y_dem.mean() * 100))
