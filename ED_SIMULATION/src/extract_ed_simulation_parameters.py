# extract_ed_simulation_parameters.py
"""
Extract simulation parameters from the synthetic ED dataset.

Outputs (all saved to ../data/):
  ed_interarrival_times_hours.csv   - time between consecutive patient arrivals
  ed_triage_durations_hours.csv     - arrival → initial assessment (triage proxy)
  ed_treatment_durations_hours.csv  - initial assessment → ward transfer in
  ed_ward_stay_durations_hours.csv  - ward transfer in → ward transfer out  ← NEW
  ed_icu_los_hours.csv              - ICU LOS from icu_los_hours column      ← NEW
  ed_branch_probabilities.csv       - discharge / admission / ICU split

Why each duration is computed the way it is:
  - triage_duration:    arrival → initial_assessment is the operational delay
                        before clinical contact begins (triage proxy).
  - treatment_duration: initial_assessment → first_transfer_in is the ED
                        treatment stage before the patient moves to a ward.
  - ward_stay:          first_transfer_in → first_transfer_out is actual time
                        spent occupying a ward bed (Ward 1).
  - icu_los:            taken directly from icu_los_hours column which was
                        computed during dataset generation.

Previously treatment_duration was used as a proxy for ward stay — this was
incorrect because the gap (assessment → ward transfer in) was only minutes
long in most cases, producing unrealistically short LOS values.
"""

import os
import pandas as pd

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_PATH = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"
OUT_DIR = "../data"
os.makedirs(OUT_DIR, exist_ok=True)

INTERARRIVAL_OUT  = os.path.join(OUT_DIR, "ed_interarrival_times_hours.csv")
TRIAGE_OUT        = os.path.join(OUT_DIR, "ed_triage_durations_hours.csv")
TREATMENT_OUT     = os.path.join(OUT_DIR, "ed_treatment_durations_hours.csv")
WARD_STAY_OUT     = os.path.join(OUT_DIR, "ed_ward_stay_durations_hours.csv")
ICU_LOS_OUT       = os.path.join(OUT_DIR, "ed_icu_los_hours.csv")
BRANCH_OUT        = os.path.join(OUT_DIR, "ed_branch_probabilities.csv")

# ---------------------------------------------------------
# LOAD & PARSE
# ---------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

datetime_cols = [
    "arrival_time",
    "initial_assessment_time",
    "first_transfer_in",
    "first_transfer_out",
    "second_transfer_in",
    "second_transfer_out",
    "icu_admission_time",
    "icu_discharge_time",
    "discharge_time",
]

for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

df = df.dropna(subset=["arrival_time"]).copy()
df = df.sort_values("arrival_time").reset_index(drop=True)
print(f"Loaded {len(df)} cases.")

# ---------------------------------------------------------
# 1. INTERARRIVAL TIMES
# ---------------------------------------------------------
# Time between consecutive patient arrivals (hours).
# Drives how frequently patients enter the simulation.

df["interarrival_hours"] = df["arrival_time"].diff().dt.total_seconds() / 3600.0
interarrival = df["interarrival_hours"].dropna()
interarrival = interarrival[interarrival >= 0]

pd.DataFrame({"interarrival_hours": interarrival}).to_csv(INTERARRIVAL_OUT, index=False)
print(f"  Interarrival times:  {len(interarrival)} rows | "
      f"mean={interarrival.mean():.4f}h  median={interarrival.median():.4f}h")

# ---------------------------------------------------------
# 2. TRIAGE DURATION
# ---------------------------------------------------------
# arrival_time → initial_assessment_time
# Represents the delay before first clinical contact (triage/registration).

df["triage_duration_hours"] = (
    df["initial_assessment_time"] - df["arrival_time"]
).dt.total_seconds() / 3600.0

triage = df["triage_duration_hours"].dropna()
triage = triage[triage > 0]

# Cap at 99th percentile to remove data artefacts
upper = triage.quantile(0.99)
triage = triage[triage <= upper]

pd.DataFrame({"triage_duration_hours": triage}).to_csv(TRIAGE_OUT, index=False)
print(f"  Triage durations:    {len(triage)} rows | "
      f"mean={triage.mean():.4f}h  median={triage.median():.4f}h")

# ---------------------------------------------------------
# 3. TREATMENT DURATION (ED cubicle stage)
# ---------------------------------------------------------
# initial_assessment_time → first_transfer_in
# This is the time a patient spends with a doctor in a cubicle
# before being moved to a ward bed.

df["treatment_duration_hours"] = (
    df["first_transfer_in"] - df["initial_assessment_time"]
).dt.total_seconds() / 3600.0

treatment = df["treatment_duration_hours"].dropna()
treatment = treatment[treatment > 0]

upper = treatment.quantile(0.95)
treatment = treatment[treatment <= upper]

pd.DataFrame({"treatment_duration_hours": treatment}).to_csv(TREATMENT_OUT, index=False)
print(f"  Treatment durations: {len(treatment)} rows | "
      f"mean={treatment.mean():.4f}h  median={treatment.median():.4f}h")

# ---------------------------------------------------------
# 4. WARD STAY DURATION  ← KEY FIX
# ---------------------------------------------------------
# first_transfer_in → first_transfer_out
# This is the actual time a patient occupies a ward bed (Ward 1).
# Previously this was missing — treatment_duration was being used
# as a proxy, which was far too short.

df["ward_stay_hours"] = (
    df["first_transfer_out"] - df["first_transfer_in"]
).dt.total_seconds() / 3600.0

ward_stay = df["ward_stay_hours"].dropna()
ward_stay = ward_stay[ward_stay > 0]

# Cap at 99th percentile — some cases have very long ward stays
upper_ward = ward_stay.quantile(0.99)
ward_stay = ward_stay[ward_stay <= upper_ward]

pd.DataFrame({"ward_stay_hours": ward_stay}).to_csv(WARD_STAY_OUT, index=False)
print(f"  Ward stay durations: {len(ward_stay)} rows | "
      f"mean={ward_stay.mean():.2f}h  median={ward_stay.median():.2f}h")

# ---------------------------------------------------------
# 5. ICU LOS  ← NEW
# ---------------------------------------------------------
# Taken directly from the icu_los_hours column generated in the dataset.
# Only available for cases with an ICU admission.

if "icu_los_hours" in df.columns:
    icu_los = pd.to_numeric(df["icu_los_hours"], errors="coerce").dropna()
    icu_los = icu_los[icu_los > 0]
    upper_icu = icu_los.quantile(0.99)
    icu_los = icu_los[icu_los <= upper_icu]

    pd.DataFrame({"icu_los_hours": icu_los}).to_csv(ICU_LOS_OUT, index=False)
    print(f"  ICU LOS:             {len(icu_los)} rows | "
          f"mean={icu_los.mean():.2f}h  median={icu_los.median():.2f}h")
else:
    print("  WARNING: icu_los_hours column not found — ICU LOS file not created.")

# ---------------------------------------------------------
# 6. BRANCH PROBABILITIES
# ---------------------------------------------------------
# Derived from the corrected event log where Callout Cancelled is
# a distinct outcome, separate from Discharge.
#
# Four outcomes from Ward1 Transfer Out (matches updated DFG):
#   DISCHARGED   → patient went home                    (65.1%)
#   CANCELLED    → callout cancelled / redirected        ( 9.7%)
#   TRANSFERRED  → patient moved to Ward2 then exits    ( 9.7%)
#   ICU          → patient escalated to ICU             (15.4%)
#
# Additionally: Ward2 → ICU probability
#   Of 1,297 patients who went to Ward2, 200 were then escalated to ICU
#   p_ward2_to_icu = 200 / 1297 = 0.1542
#
# NOTE: In the simulation, DISCHARGED and CANCELLED are merged into a
# single "discharge" outcome since both result in the patient leaving
# the ED without requiring a boarding slot. The distinction is preserved
# here for completeness and potential future use.

outcomes = df["callout_outcome"].astype(str).str.strip().str.upper()
valid    = outcomes.isin(["DISCHARGED", "TRANSFERRED", "ICU", "CANCELLED"])
outcomes = outcomes[valid]

total        = len(outcomes)
p_discharged = (outcomes == "DISCHARGED").sum() / total
p_cancelled  = (outcomes == "CANCELLED").sum()  / total
p_admission  = (outcomes == "TRANSFERRED").sum() / total
p_icu        = (outcomes == "ICU").sum()          / total

# Simulation uses discharge + cancelled combined (both exit ED without boarding)
p_discharge_sim = p_discharged + p_cancelled

# Ward2 → ICU transition probability
# Patients who went Ward2 Transfer Out → ICU Admission
ward2_patients = df["second_transfer_in"].notna().sum()
ward2_to_icu   = df[
    df["second_transfer_in"].notna() & df["icu_admission_time"].notna()
].shape[0]
p_ward2_to_icu = ward2_to_icu / ward2_patients if ward2_patients > 0 else 0.0

branch_df = pd.DataFrame([{
    "p_discharge":     round(p_discharge_sim, 6),   # discharged + cancelled
    "p_admission":     round(p_admission,     6),   # transferred (Ward2, exits ED)
    "p_icu":           round(p_icu,           6),   # direct ICU from Ward1
    "p_discharged":    round(p_discharged,    6),   # home discharge only
    "p_cancelled":     round(p_cancelled,     6),   # callout cancelled only
    "p_ward2_to_icu":  round(p_ward2_to_icu,  6),   # Ward2 → ICU escalation
}])

branch_df.to_csv(BRANCH_OUT, index=False)

print(f"\n  Branch probabilities (from Ward1 Transfer Out):")
print(f"    discharged       = {p_discharged:.4f}  (home discharge)")
print(f"    cancelled        = {p_cancelled:.4f}  (callout cancelled)")
print(f"    discharge (sim)  = {p_discharge_sim:.4f}  (discharged + cancelled combined)")
print(f"    admission        = {p_admission:.4f}  (Ward2 transfer)")
print(f"    icu              = {p_icu:.4f}  (direct ICU escalation)")
print(f"    sum              = {p_discharge_sim + p_admission + p_icu:.4f}  (should be 1.0)")
print(f"\n  Ward2 → ICU probability:")
print(f"    ward2 patients   = {ward2_patients}")
print(f"    ward2 → icu      = {ward2_to_icu}")
print(f"    p_ward2_to_icu   = {p_ward2_to_icu:.4f}")

print("\nAll parameter files saved to:", OUT_DIR)