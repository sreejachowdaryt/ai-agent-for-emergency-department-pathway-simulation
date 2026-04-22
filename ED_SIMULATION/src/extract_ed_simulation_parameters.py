# src/extract_ed_simulation_parameters.py
"""
Extract simulation parameters from the NEW hybrid synthetic ED dataset.

This version is aligned with the current ed_cases.csv structure.

Dataset fields used:
- arrival_time
- initial_assessment_time
- boarding_start_time
- ed_departure_time
- first_transfer_in
- first_transfer_out
- second_transfer_in
- second_transfer_out
- ed_los_hours
- total_careunit_los_hours
- pathway_outcome

Outputs saved to ../data/:

1. ed_interarrival_times_hours.csv
   Time between consecutive arrivals.

2. ed_assessment_durations_hours.csv
   initial_assessment_time - arrival_time
   This is the ED assessment/service proxy for the combined triage+doctor stage.

3. ed_boarding_durations_hours.csv
   ed_departure_time - boarding_start_time
   Only for admitted/transferred patients.

4. ed_ed_los_hours.csv
   Total ED length of stay distribution.

5. ed_first_careunit_stay_hours.csv
   first_transfer_out - first_transfer_in

6. ed_second_careunit_stay_hours.csv
   second_transfer_out - second_transfer_in

7. ed_branch_probabilities.csv
   Outcome probabilities:
   - p_discharge
   - p_admission
   - p_transferred

8. ed_post_first_careunit_transition_probabilities.csv
   Branch after first careunit:
   - p_hospital_discharge_after_first
   - p_second_careunit_after_first

Important note:
- Queue waiting times are NOT extracted here.
  They emerge from the simulation.
- The ED assessment duration here is a synthetic proxy for the
  combined pre-decision clinical stage.
"""

import os
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
INPUT_PATH = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"
OUT_DIR = "../data"
os.makedirs(OUT_DIR, exist_ok=True)

INTERARRIVAL_OUT = os.path.join(OUT_DIR, "ed_interarrival_times_hours.csv")
ASSESSMENT_OUT = os.path.join(OUT_DIR, "ed_assessment_durations_hours.csv")
BOARDING_OUT = os.path.join(OUT_DIR, "ed_boarding_durations_hours.csv")
ED_LOS_OUT = os.path.join(OUT_DIR, "ed_ed_los_hours.csv")
FIRST_CAREUNIT_STAY_OUT = os.path.join(OUT_DIR, "ed_first_careunit_stay_hours.csv")
SECOND_CAREUNIT_STAY_OUT = os.path.join(OUT_DIR, "ed_second_careunit_stay_hours.csv")
BRANCH_OUT = os.path.join(OUT_DIR, "ed_branch_probabilities.csv")
POST_FIRST_BRANCH_OUT = os.path.join(OUT_DIR, "ed_post_first_careunit_transition_probabilities.csv")


def save_series(series: pd.Series, output_path: str, column_name: str):
    series = pd.to_numeric(series, errors="coerce").dropna()
    pd.DataFrame({column_name: series}).to_csv(output_path, index=False)


print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

dt_cols = [
    "arrival_time",
    "initial_assessment_time",
    "boarding_start_time",
    "ed_departure_time",
    "first_transfer_in",
    "first_transfer_out",
    "second_transfer_in",
    "second_transfer_out",
    "discharge_time",
]

for col in dt_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

df = df.dropna(subset=["arrival_time"]).copy()
df = df.sort_values("arrival_time").reset_index(drop=True)

print(f"Loaded {len(df):,} cases.")
print()

# -------------------------------------------------------------------
# 1. Interarrival times
# -------------------------------------------------------------------
df["interarrival_hours"] = df["arrival_time"].diff().dt.total_seconds() / 3600.0
interarrival = df["interarrival_hours"].dropna()
interarrival = interarrival[interarrival >= 0]

save_series(interarrival, INTERARRIVAL_OUT, "interarrival_hours")

print(f"Interarrival times:              {len(interarrival):,} rows | "
      f"mean={interarrival.mean():.4f}h  "
      f"median={interarrival.median():.4f}h  "
      f"(~{interarrival.mean()*60:.1f} min mean gap)")

# -------------------------------------------------------------------
# 2. Assessment duration proxy
# -------------------------------------------------------------------
df["assessment_duration_h"] = (
    df["initial_assessment_time"] - df["arrival_time"]
).dt.total_seconds() / 3600.0

assessment = df["assessment_duration_h"].dropna()
assessment = assessment[assessment > 0]
assessment = assessment[assessment <= assessment.quantile(0.99)]

save_series(assessment, ASSESSMENT_OUT, "assessment_duration_hours")

print(f"Assessment duration:             {len(assessment):,} rows | "
      f"mean={assessment.mean()*60:.1f} min  "
      f"median={assessment.median()*60:.1f} min  "
      f"p95={assessment.quantile(0.95)*60:.1f} min")

# -------------------------------------------------------------------
# 3. Boarding duration
# -------------------------------------------------------------------
df["boarding_duration_h"] = (
    df["ed_departure_time"] - df["boarding_start_time"]
).dt.total_seconds() / 3600.0

boarding = df["boarding_duration_h"].dropna()
boarding = boarding[boarding > 0]
boarding = boarding[boarding <= boarding.quantile(0.99)]

save_series(boarding, BOARDING_OUT, "boarding_duration_hours")

print(f"Boarding duration:               {len(boarding):,} rows | "
      f"mean={boarding.mean()*60:.1f} min  "
      f"median={boarding.median()*60:.1f} min  "
      f"p95={boarding.quantile(0.95)*60:.1f} min")

# -------------------------------------------------------------------
# 4. ED LOS
# -------------------------------------------------------------------
ed_los = pd.to_numeric(df["ed_los_hours"], errors="coerce").dropna()
ed_los = ed_los[ed_los > 0]
ed_los = ed_los[ed_los <= ed_los.quantile(0.99)]

save_series(ed_los, ED_LOS_OUT, "ed_los_hours")

print(f"ED length of stay:               {len(ed_los):,} rows | "
      f"mean={ed_los.mean():.2f}h  "
      f"median={ed_los.median():.2f}h  "
      f"p95={ed_los.quantile(0.95):.2f}h")

# -------------------------------------------------------------------
# 5. First careunit stay
# -------------------------------------------------------------------
df["first_careunit_stay_h"] = (
    df["first_transfer_out"] - df["first_transfer_in"]
).dt.total_seconds() / 3600.0

first_stay = df["first_careunit_stay_h"].dropna()
first_stay = first_stay[first_stay > 0]
first_stay = first_stay[first_stay <= first_stay.quantile(0.99)]

save_series(first_stay, FIRST_CAREUNIT_STAY_OUT, "first_careunit_stay_hours")

print(f"First careunit stay:             {len(first_stay):,} rows | "
      f"mean={first_stay.mean():.2f}h  "
      f"median={first_stay.median():.2f}h  "
      f"p95={first_stay.quantile(0.95):.2f}h")

# -------------------------------------------------------------------
# 6. Second careunit stay
# -------------------------------------------------------------------
df["second_careunit_stay_h"] = (
    df["second_transfer_out"] - df["second_transfer_in"]
).dt.total_seconds() / 3600.0

second_stay = df["second_careunit_stay_h"].dropna()
second_stay = second_stay[second_stay > 0]
second_stay = second_stay[second_stay <= second_stay.quantile(0.99)]

save_series(second_stay, SECOND_CAREUNIT_STAY_OUT, "second_careunit_stay_hours")

print(f"Second careunit stay:            {len(second_stay):,} rows | "
      f"mean={second_stay.mean():.2f}h  "
      f"median={second_stay.median():.2f}h  "
      f"p95={second_stay.quantile(0.95):.2f}h")

# -------------------------------------------------------------------
# 7. Branch probabilities
# -------------------------------------------------------------------
print()
outcomes = df["pathway_outcome"].astype(str).str.strip().str.upper()
valid = outcomes.isin(["DISCHARGED", "ADMITTED", "TRANSFERRED"])
outcomes = outcomes[valid]
total = len(outcomes)

p_discharge = (outcomes == "DISCHARGED").sum() / total
p_admission = (outcomes == "ADMITTED").sum() / total
p_transferred = (outcomes == "TRANSFERRED").sum() / total

print(f"Branch probabilities (from {total:,} cases):")
print(f"  discharged   = {p_discharge:.4f} ({p_discharge:.1%})")
print(f"  admitted     = {p_admission:.4f} ({p_admission:.1%})")
print(f"  transferred  = {p_transferred:.4f} ({p_transferred:.1%})")
print(f"  sum          = {p_discharge + p_admission + p_transferred:.4f}")

pd.DataFrame([{
    "p_discharge": round(p_discharge, 6),
    "p_admission": round(p_admission, 6),
    "p_transferred": round(p_transferred, 6),
}]).to_csv(BRANCH_OUT, index=False)

# -------------------------------------------------------------------
# 8. Post-first-careunit branch probabilities
# -------------------------------------------------------------------
has_first = df["first_transfer_out"].notna()
n_first = has_first.sum()
n_second = df["second_transfer_in"].notna().sum()
n_direct_discharge_after_first = n_first - n_second

p_second_after_first = n_second / n_first if n_first > 0 else 0.0
p_discharge_after_first = n_direct_discharge_after_first / n_first if n_first > 0 else 0.0

print()
print("Branch after first careunit:")
print(f"  first careunit cases:             {n_first:,}")
print(f"  direct hospital discharge:        {n_direct_discharge_after_first:,}")
print(f"  moved to second careunit:         {n_second:,}")
print(f"  p_hospital_discharge_after_first  = {p_discharge_after_first:.4f}")
print(f"  p_second_careunit_after_first     = {p_second_after_first:.4f}")

pd.DataFrame([{
    "p_hospital_discharge_after_first": round(p_discharge_after_first, 6),
    "p_second_careunit_after_first": round(p_second_after_first, 6),
}]).to_csv(POST_FIRST_BRANCH_OUT, index=False)

print(f"\nAll parameter files saved to: {OUT_DIR}")
print()
print("Files produced:")
for f, desc in [
    ("ed_interarrival_times_hours.csv", "Arrival process"),
    ("ed_assessment_durations_hours.csv", "ED assessment service proxy"),
    ("ed_boarding_durations_hours.csv", "Boarding service time"),
    ("ed_ed_los_hours.csv", "Total ED LOS"),
    ("ed_first_careunit_stay_hours.csv", "First careunit LOS"),
    ("ed_second_careunit_stay_hours.csv", "Second careunit LOS"),
    ("ed_branch_probabilities.csv", "ED pathway branch probabilities"),
    ("ed_post_first_careunit_transition_probabilities.csv", "Branch after first careunit"),
]:
    full = os.path.join(OUT_DIR, f)
    status = "✓" if os.path.exists(full) else "✗ MISSING"
    print(f"  {status}  {f:<50} {desc}")