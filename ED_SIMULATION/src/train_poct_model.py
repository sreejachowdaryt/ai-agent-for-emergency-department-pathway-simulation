# src/train_poct_model.py
"""
POCT Amenability Classifier — Random Forest
============================================

Train the POCT amenability classifier used by the hybrid ED simulation.

This script trains a Random Forest classifier on the synthetic ED case dataset
to estimate whether a patient is likely to benefit from POCT-based acceleration
during the assessment stage.

Dataset and modelling assumptions:
- Trained on the hybrid synthetic dataset derived from MIMIC-IV-ED and MIMIC-III
- Assessment duration is represented by the proxy:
      initial_assessment_time - arrival_time
- pathway_outcome is used in the updated dataset structure
- first_careunit may be null for discharged patients and is filled as UNKNOWN
- admission_type may include multiple categories sampled from the broader
  MIMIC distribution

Feature engineering:
- diagnosis code
- admission type
- careunit
- arrival hour
- arrival weekday
- assessment-duration bucket

Outputs:
- trained Random Forest model
- model metadata (encoders, feature mappings, performance metrics)
- precomputed POCT probability lookup table for O(1) simulation-time inference
- feature importance figure for interpretation

The trained model supports the ML-enhanced simulation by enabling fast,
precomputed POCT decision support during runtime.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

DATA_PATH   = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"
OUT_DIR     = "../data"
FIGURES_DIR = "../figures"
os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_OUT  = os.path.join(OUT_DIR, "poct_model.pkl")
META_OUT   = os.path.join(OUT_DIR, "poct_model_meta.pkl")
LOOKUP_OUT = os.path.join(OUT_DIR, "poct_lookup.pkl")

# ----------------------------------------------------------
# POCT AMENABILITY LABELS
# ----------------------------------------------------------

POCT_HIGH = {
    '99592', '0389', '51881', '4280',
    '42731', '41401', '5849', '53081',
}

POCT_STANDARD = {
    '4019', '25000', '5990', '486',
    '2859', '5070', '496', '2762',
}

# ----------------------------------------------------------
# LOAD AND PREPARE DATA
# ----------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Parse datetime columns
for col in ["arrival_time", "initial_assessment_time"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Triage duration proxy: initial_assessment_time - arrival_time
# This is the combined triage+doctor stage from MIMIC-IV-ED edstays
# Previously used triage_end - triage_start (old dataset structure)
df["triage_duration_h"] = (
    df["initial_assessment_time"] - df["arrival_time"]
).dt.total_seconds() / 3600.0

df["arrival_hour"]    = df["arrival_time"].dt.hour
df["arrival_weekday"] = df["arrival_time"].dt.weekday

# Assign POCT label from diagnosis code
df["diag_str"] = df["primary_diagnosis_code"].astype(str).str.strip()

def assign_poct(code):
    if code in POCT_HIGH:       return "high"
    elif code in POCT_STANDARD: return "standard"
    return "none"

df["poct_benefit"]  = df["diag_str"].apply(assign_poct)
df["poct_amenable"] = (df["poct_benefit"] != "none").astype(int)

print(f"  Total cases:       {len(df):,}")
print(f"  POCT amenable:     {df['poct_amenable'].sum():,} ({df['poct_amenable'].mean():.1%})")
print(f"  Not amenable:      {(df['poct_amenable']==0).sum():,} ({(df['poct_amenable']==0).mean():.1%})")
print()

# Column check
print("Column check:")
for col in ["arrival_time", "initial_assessment_time", "triage_duration_h",
            "pathway_outcome", "first_careunit", "admission_type",
            "primary_diagnosis_code"]:
    present = col in df.columns
    sample  = str(df[col].iloc[0])[:35] if present else "MISSING"
    print(f"  {'✓' if present else '✗'}  {col:<30} {sample}")
print()

# ----------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------

TOP_N_DIAG = 30
top_diags  = df["diag_str"].value_counts().head(TOP_N_DIAG).index.tolist()

def encode_diag(code):
    return top_diags.index(code) if code in top_diags else TOP_N_DIAG

df["diag_encoded"] = df["diag_str"].apply(encode_diag)

le_admit = LabelEncoder()
le_cu    = LabelEncoder()

df["admission_type_enc"] = le_admit.fit_transform(
    df["admission_type"].fillna("EMERGENCY")
)
# first_careunit is null for discharge patients — fill as UNKNOWN
df["careunit_enc"] = le_cu.fit_transform(
    df["first_careunit"].fillna("UNKNOWN")
)

# Triage duration bucket: 0.1h intervals, clipped to 0–2h (120 min)
# New dataset has longer assessment times (mean 32 min vs old 10 min)
df["triage_bucket"] = (
    df["triage_duration_h"].clip(0, 2.0) * 10
).round().astype(int).clip(0, 20)

FEATURES = [
    "diag_encoded",
    "admission_type_enc",
    "careunit_enc",
    "arrival_hour",
    "arrival_weekday",
    "triage_bucket",
]

X = df[FEATURES].fillna(0).values
y = df["poct_amenable"].values

# ----------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  Training set: {len(X_train):,} samples")
print(f"  Test set:     {len(X_test):,} samples")
print()

# ----------------------------------------------------------
# TRAIN RANDOM FOREST
# ----------------------------------------------------------

print("Training Random Forest classifier...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
acc_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")

print(f"  5-fold CV AUC (training set):  {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")
print(f"  5-fold CV Acc (training set):  {acc_cv.mean():.3f} ± {acc_cv.std():.3f}")

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)

print(f"  Hold-out test AUC:             {test_auc:.3f}")
print()
print("  Classification report (hold-out test set):")
print(classification_report(y_test, y_pred,
                            target_names=["Not amenable", "POCT amenable"]))

# ----------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ----------------------------------------------------------

feat_names  = ["Diagnosis Code", "Admission Type", "Care Unit",
               "Arrival Hour", "Arrival Weekday", "Assessment Duration"]
importances = rf.feature_importances_
order       = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_facecolor("#FAFAFA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#E8E8E8", linewidth=0.8, zorder=0)
colors = ["#C0392B" if importances[i] == importances.max() else "#2E86AB"
          for i in order]
bars = ax.barh([feat_names[i] for i in order],
               [importances[i] for i in order],
               color=colors, zorder=3, edgecolor="white")
for bar, imp in zip(bars, [importances[i] for i in order]):
    ax.text(imp + 0.003, bar.get_y() + bar.get_height()/2,
            f"{imp:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_title("Random Forest — Feature Importance\n(POCT Amenability Classifier)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "poct_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: ../figures/poct_feature_importance.png")

# ----------------------------------------------------------
# PRECOMPUTE LOOKUP TABLE
# ----------------------------------------------------------

print("\nPrecomputing POCT probability lookup table...")

n_diag   = TOP_N_DIAG + 1
n_admit  = len(le_admit.classes_)
n_cu     = len(le_cu.classes_)
n_hour   = 24
n_wday   = 7
n_triage = 21   # 0–20 buckets (0.1h intervals up to 2h)

total_combos = n_diag * n_admit * n_cu * n_hour * n_wday * n_triage
print(f"  Combinations: {total_combos:,}")

rows, keys = [], []
for d in range(n_diag):
    for a in range(n_admit):
        for c in range(n_cu):
            for h in range(n_hour):
                for w in range(n_wday):
                    for t in range(n_triage):
                        rows.append([d, a, c, h, w, t])
                        keys.append((d, a, c, h, w, t))

X_all  = np.array(rows)
probs  = rf.predict_proba(X_all)[:, 1]
lookup = {k: float(p) for k, p in zip(keys, probs)}

print(f"  Done. {len(lookup):,} entries computed.")
print(f"  POCT-amenable at threshold 0.55: "
      f"{sum(1 for p in lookup.values() if p >= 0.55)/len(lookup):.1%} of combos")

# ----------------------------------------------------------
# SAVE
# ----------------------------------------------------------

meta = {
    "features":      FEATURES,
    "feat_names":    feat_names,
    "top_diags":     top_diags,
    "le_admit":      le_admit,
    "le_cu":         le_cu,
    "poct_high":     POCT_HIGH,
    "poct_standard": POCT_STANDARD,
    "auc_cv_mean":   float(auc_cv.mean()),
    "auc_cv_std":    float(auc_cv.std()),
    "test_auc":      float(test_auc),
    "n_estimators":  200,
    "threshold":     0.55,
    "n_diag":        n_diag,
    "n_admit":       n_admit,
    "n_cu":          n_cu,
    "n_triage":      n_triage,
}

with open(MODEL_OUT,  "wb") as f: pickle.dump(rf,     f)
with open(META_OUT,   "wb") as f: pickle.dump(meta,   f)
with open(LOOKUP_OUT, "wb") as f: pickle.dump(lookup, f)

print(f"\n  Model  → {MODEL_OUT}")
print(f"  Meta   → {META_OUT}")
print(f"  Lookup → {LOOKUP_OUT}")
print("\n" + "="*55)
print("  Training complete. Run ed_simulation_ml.py next.")
print("="*55)