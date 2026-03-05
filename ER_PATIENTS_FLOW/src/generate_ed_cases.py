# src/generate_ed_cases.py
"""
Generate a synthetic ED dataset (patients.csv + ed_cases.csv).

Key feature:
- Admissions per patient are sampled from an empirical probability distribution
  derived from MIMIC-III (output of derive_admissions_pmf_from_mimic.py).

Algorithm (high-level):
1) Generate NUM_PATIENTS synthetic patients (demographics only).
2) Load MIMIC-derived admissions-per-patient PMF (num_admissions -> probability).
3) For each patient, sample N admissions ~ Categorical(PMF), using buckets 0..4 and 5+ tail.
4) For each admission, generate a coherent timestamp chain:
   arrival -> assessment -> transfers -> (optional ICU) -> discharge -> callout
5) Enforce sequential admissions (no overlaps) by placing admission k+1 after discharge k
   using a sampled inter-admission gap (mean/std in hours).
"""

import os
import random
import pandas as pd
from datetime import timedelta

from utils import (
    random_arrival_time,
    random_time_after,
    derive_age_group,
    load_gap_series, # loads MIMIC gap distributions
    sample_empirical, # resample from emprical series
    format_datetime_cols, # consistent MIMIC-like datetime format
)


random.seed(42) # Reproducibility

# =========================================================
# CONFIG
# =========================================================
NUM_PATIENTS = 2000

OUT_DIR = os.path.join("Synthetic_dataset", "data")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENTS_PATH = os.path.join(OUT_DIR, "patients.csv")
ED_CASES_PATH = os.path.join(OUT_DIR, "ed_cases.csv")

# MIMIC-derived PMF file (created by src/derive_admissions_pmf_from_mimic.py)
MIMIC_PMF_PATH = os.path.join(OUT_DIR, "mimic_admissions_count_probabilities.csv")

# MIMIC-derived Inter-admission time gaps: Days (created by src/compute_time_gaps_from_mimic.py)
MIMIC_GAPS_PATH = os.path.join(OUT_DIR, "mimic_interadmission_gaps_days.csv")

# MIMIC-derived branch probabilities (created by src/extract_activity_gaps_from_mimic.py)
BRANCH_PROB_PATH = os.path.join(OUT_DIR, "mimic_branch_probabilities.csv")

# Activity gap distributions produced by extract_activity_gaps_from_mimic.py
GAP_ARRIVAL_TO_FIRST_IN_H  = os.path.join(OUT_DIR, "mimic_gap_arrival_to_first_transfer_in_hours.csv")
GAP_TRANSFER_STAY_H        = os.path.join(OUT_DIR, "mimic_gap_transfer_stay_hours.csv")
GAP_BETWEEN_TRANSFERS_H    = os.path.join(OUT_DIR, "mimic_gap_between_transfer1_out_and_transfer2_in_hours.csv")
GAP_ICU_LOS_H              = os.path.join(OUT_DIR, "mimic_gap_icu_los_hours.csv")
GAP_LAST_TO_DISCHARGE_H    = os.path.join(OUT_DIR, "mimic_gap_last_activity_to_discharge_hours.csv")
GAP_DISCHARGE_TO_CALLOUT_M = os.path.join(OUT_DIR, "mimic_gap_discharge_to_callout_create_minutes.csv")
GAP_CALLOUT_ACK_M          = os.path.join(OUT_DIR, "mimic_gap_callout_create_to_ack_minutes.csv")

# How far back first admissions can start
LOOKBACK_DAYS = 365 * 3
sim_end = pd.Timestamp.now()
sim_start = sim_end - pd.Timedelta(days=LOOKBACK_DAYS)

# Value pools
GENDERS = ["M", "F"]
ADMISSION_TYPES = ["EMERGENCY", "URGENT", "ELECTIVE"]
CAREUNITS = ["WARD", "MICU", "SICU"]
DIAGNOSES = ["I21", "J18", "S06", "N39"]
DISCHARGE_LOCATIONS = ["HOME", "REHAB", "ICU", "DEATH"]
CALLOUT_OUTCOMES = ["DISCHARGED", "TRANSFERRED", "ICU", "CANCELLED"]

# Date-time formatting (MIMIC-like)
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATETIME_COLS = [
    "arrival_time",
    "initial_assessment_time",
    "first_transfer_in",
    "first_transfer_out",
    "second_transfer_in",
    "second_transfer_out",
    "icu_admission_time",
    "icu_discharge_time",
    "discharge_time",
    "callout_time",
    "callout_ack_time",
]

# Admissions-per-patient PMF handling
TAIL_BUCKET_START = 5         # collapse all counts >= 5 into a "5+" bucket
MAX_TAIL_ADMISSIONS = 10      # when "5+" is sampled, expand to a concrete 5..MAX_TAIL_ADMISSIONS


# =========================================================
# PMF loading + sampling (from MIMIC)
# =========================================================
def load_mimic_admissions_pmf(path: str):
    """
    Load admissions-count probabilities derived from MIMIC.
    Expected CSV columns: num_admissions, probability

    Returns:
      options: [0,1,2,3,4,5]  where 5 means "5+"
      weights: corresponding probabilities (normalised)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find PMF file at: {path}\n"
            f"Run: python src/derive_admissions_pmf_from_mimic.py first."
        )

    pmf = pd.read_csv(path)
    if "num_admissions" not in pmf.columns or "probability" not in pmf.columns:
        raise ValueError(
            f"PMF file must contain columns: num_admissions, probability. Found: {list(pmf.columns)}"
        )

    pmf["num_admissions"] = pmf["num_admissions"].astype(int)
    pmf["probability"] = pmf["probability"].astype(float)

    probs = dict(zip(pmf["num_admissions"], pmf["probability"]))

    # buckets 0..4
    options = list(range(0, TAIL_BUCKET_START))  # [0,1,2,3,4]
    weights = [float(probs.get(k, 0.0)) for k in options]

    # tail bucket >=5
    tail_prob = float(pmf.loc[pmf["num_admissions"] >= TAIL_BUCKET_START, "probability"].sum())
    options.append(TAIL_BUCKET_START)  # represent 5+
    weights.append(tail_prob)

    # normalise (safety)
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"PMF probabilities sum to 0. Check file: {path}")
    weights = [w / total for w in weights]

    return options, weights


def sample_num_admissions(options, weights) -> int:
    """
    Sample admissions per patient using MIMIC-derived PMF.
    If the tail bucket (5+) is selected, expand to a concrete value in [5..MAX_TAIL_ADMISSIONS].
    """
    n = random.choices(options, weights=weights, k=1)[0]
    if n == TAIL_BUCKET_START:
        return random.randint(TAIL_BUCKET_START, MAX_TAIL_ADMISSIONS)
    return n


# ==================================================
# Inter-admission Time Gaps + sampling (from MIMIC)
# ==================================================
def load_mimic_gap_days(path: str) -> pd.Series:
    """
    Load MIMIC-derived inter-admission gaps in days.
    Returns a Series of positive gap values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find MIMIC gaps file at: {path}\n"
            f"Run: python src/compute_time_gaps_from_mimic.py first."
        )

    g = pd.read_csv(path)
    if "gap_days" not in g.columns:
        raise ValueError(f"{path} must contain a 'gap_days' column")

    gaps = pd.to_numeric(g["gap_days"], errors="coerce").dropna()
    gaps = gaps[gaps > 0]

    if gaps.empty:
        raise ValueError("MIMIC gaps file has no positive gap_days values.")
    return gaps


def sample_gap_days_empirical(gaps: pd.Series, min_days: float = 1.0) -> float:
    """
    Empirically resample a gap from MIMIC gap distribution.
    min_days prevents unrealistically tiny gaps (optional safeguard).
    """
    while True:
        val = float(gaps.sample(1).iloc[0])
        if val >= min_days:
            return val


# =========================================================
# Load branch probabilities (from MIMIC)
# =========================================================
def load_branch_probabilities(path: str) -> tuple[float, float]:
    """
    Load branching probabilities created by extract_activity_gaps_from_mimic.py
    Expected columns: p_second_transfer, p_icu
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find branch probability file at: {path}\n"
            f"Run: python src/extract_activity_gaps_from_mimic.py first."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    for col in ["p_second_transfer", "p_icu"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain '{col}' column")

    p_second = float(df.loc[0, "p_second_transfer"])
    p_icu = float(df.loc[0, "p_icu"])

    # safety clamp
    p_second = min(max(p_second, 0.0), 1.0)
    p_icu = min(max(p_icu, 0.0), 1.0)
    return p_second, p_icu


# =========================================================
# Case generation helpers Defines the sequence of events
# arrival → initial_assessment → transfers → (optional ICU) → discharge → callout → ack
# =========================================================
def generate_one_case(
    patient_id: int,
    case_id: int,
    arrival_time: pd.Timestamp,
    p_second_transfer: float,
    p_icu: float,
    s_arrival_to_first_in_h: pd.Series,
    s_transfer_stay_h: pd.Series,
    s_between_transfers_h: pd.Series,
    s_icu_los_h: pd.Series,
    s_last_to_discharge_h: pd.Series,
    s_discharge_to_callout_m: pd.Series,
    s_callout_ack_m: pd.Series,
) -> dict:
    """
    Generate one admission case.
    Option A: assessment delay remains a small random delay (5–60 mins).
    After that, activity times are sampled from MIMIC-derived empirical gap distributions.
    """

    # Option A: small operational delay for assessment (keeps your earlier approach)
    initial_assessment_time = random_time_after(arrival_time, 5, 60)

    # Gap: arrival -> first transfer in (hours from MIMIC)
    gap1_h = sample_empirical(s_arrival_to_first_in_h, min_val=0.01)
    first_transfer_in = initial_assessment_time + timedelta(hours=gap1_h)

    # Gap: transfer stay (hours from MIMIC)
    stay1_h = sample_empirical(s_transfer_stay_h, min_val=0.01)
    first_transfer_out = first_transfer_in + timedelta(hours=stay1_h)

    # Branch: second transfer?
    has_second_transfer = (random.random() < p_second_transfer)

    second_transfer_in = second_transfer_out = None
    second_careunit = None

    last_transfer_out = first_transfer_out

    if has_second_transfer:
        gap_between_h = sample_empirical(s_between_transfers_h, min_val=0.01)
        second_transfer_in = first_transfer_out + timedelta(hours=gap_between_h)

        stay2_h = sample_empirical(s_transfer_stay_h, min_val=0.01)
        second_transfer_out = second_transfer_in + timedelta(hours=stay2_h)

        second_careunit = random.choice(CAREUNITS)
        last_transfer_out = second_transfer_out

    # Branch: ICU?
    has_icu = (random.random() < p_icu)

    icu_admission_time = icu_discharge_time = icu_los_hours = None
    last_activity_time = last_transfer_out

    if has_icu:
        # small operational delay to ICU start (you can later learn this too)
        icu_admission_time = random_time_after(last_transfer_out, 10, 60)

        los_h = sample_empirical(s_icu_los_h, min_val=0.01)
        icu_discharge_time = icu_admission_time + timedelta(hours=los_h)
        icu_los_hours = round(los_h, 2)

        last_activity_time = icu_discharge_time

    # Gap: last activity -> discharge (hours from MIMIC)
    last_to_discharge_h = sample_empirical(s_last_to_discharge_h, min_val=0.01)
    discharge_time = last_activity_time + timedelta(hours=last_to_discharge_h)

    # Gap: discharge -> callout create (minutes from MIMIC)
    d2c_min = sample_empirical(s_discharge_to_callout_m, min_val=0.0)
    callout_time = discharge_time + timedelta(minutes=d2c_min)

    # Gap: callout create -> ack (minutes from MIMIC)
    c2a_min = sample_empirical(s_callout_ack_m, min_val=0.0)
    callout_ack_time = callout_time + timedelta(minutes=c2a_min)

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "arrival_time": arrival_time,
        "initial_assessment_time": initial_assessment_time,
        "admission_type": random.choice(ADMISSION_TYPES),
        "first_careunit": random.choice(CAREUNITS),
        "first_transfer_in": first_transfer_in,
        "first_transfer_out": first_transfer_out,
        "second_careunit": second_careunit,
        "second_transfer_in": second_transfer_in,
        "second_transfer_out": second_transfer_out,
        "icu_admission_time": icu_admission_time,
        "icu_discharge_time": icu_discharge_time,
        "icu_los_hours": icu_los_hours,
        "primary_diagnosis_code": random.choice(DIAGNOSES),
        "discharge_time": discharge_time,
        "discharge_location": random.choice(DISCHARGE_LOCATIONS),
        "callout_time": callout_time,
        "callout_ack_time": callout_ack_time,
        "callout_outcome": random.choice(CALLOUT_OUTCOMES),
    }


def format_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to MIMIC-like string format."""
    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(DATETIME_FMT)
    return df


# =========================================================
# Main
# =========================================================
def main():
    # ------------------------------
    # 0) Load MIMIC admissions PMF
    # ------------------------------
    options, weights = load_mimic_admissions_pmf(MIMIC_PMF_PATH)
    print("Loaded MIMIC PMF (bucketed 0..4, 5+):")
    print(list(zip(options, [round(w, 6) for w in weights])))
    
    # ---------------------------------------
    # 0) Load MIMIC inter-admission time gap
    # ---------------------------------------
    mimic_gaps_days = load_mimic_gap_days(MIMIC_GAPS_PATH)
    print(f"Loaded {len(mimic_gaps_days)} MIMIC inter-admission gaps (days).")

     # ---------------------------------------
    # 0) Load branching probabilities
    # ---------------------------------------
    p_second_transfer, p_icu = load_branch_probabilities(BRANCH_PROB_PATH)
    print(f"Loaded branching probabilities: p_second_transfer={p_second_transfer:.4f}, p_icu={p_icu:.4f}")

    # ---------------------------------------
    # 0) Load activity gap distributions
    # ---------------------------------------
    s_arrival_to_first_in_h = load_gap_series(GAP_ARRIVAL_TO_FIRST_IN_H, "gap_hours")
    s_transfer_stay_h = load_gap_series(GAP_TRANSFER_STAY_H, "gap_hours")
    s_between_transfers_h = load_gap_series(GAP_BETWEEN_TRANSFERS_H, "gap_hours")
    s_icu_los_h = load_gap_series(GAP_ICU_LOS_H, "gap_hours")
    s_last_to_discharge_h = load_gap_series(GAP_LAST_TO_DISCHARGE_H, "gap_hours")
    s_discharge_to_callout_m = load_gap_series(GAP_DISCHARGE_TO_CALLOUT_M, "gap_minutes")
    s_callout_ack_m = load_gap_series(GAP_CALLOUT_ACK_M, "gap_minutes")

    print("Loaded activity-gap distributions from MIMIC.")

    # ------------------------------
    # 1) Generate PATIENTS table
    # ------------------------------
    patients_rows = []
    for patient_id in range(1, NUM_PATIENTS + 1):
        gender = random.choice(GENDERS)
        age_years = random.randint(0, 95)
        age_group = derive_age_group(age_years)
        patients_rows.append(
            {
                "patient_id": patient_id,
                "gender": gender,
                "age_years": age_years,
                "age_group": age_group,
            }
        )

    patients_df = pd.DataFrame(patients_rows)
    patients_df.to_csv(PATIENTS_PATH, index=False)
    print(f"Generated patients -> {PATIENTS_PATH} (rows={len(patients_df)})")

    # ------------------------------
    # 2) Generate ED CASES table
    # ------------------------------
    rows = []
    case_id_counter = 1000

    for patient in patients_rows:
        patient_id = patient["patient_id"]

        # Sample number of admissions from MIMIC-derived PMF
        n_adm = sample_num_admissions(options, weights)

        # If n_adm==0, patient appears only in PATIENTS table (rare if P(0)=0)
        if n_adm == 0:
            continue

        # First admission arrival time
        arrival = random_arrival_time(lookback_days=LOOKBACK_DAYS)
        prev_discharge = None

        for adm_idx in range(n_adm):
            if adm_idx == 0:
                arrival_time = arrival
            else:
                # Ensure no overlap: next admission starts after previous discharge + gap (Inter-admission gaps between admissions for same patient)
                gap_days = sample_gap_days_empirical(mimic_gaps_days, min_days=7)
                arrival_time = prev_discharge + timedelta(days=gap_days)
            
            # If the next admission would be outside the simulation window, stop
            if pd.Timestamp(arrival_time) > sim_end:
                break

            row = generate_one_case(
                patient_id=patient_id,
                case_id=case_id_counter,
                arrival_time=arrival_time,
                p_second_transfer=p_second_transfer,
                p_icu=p_icu,
                s_arrival_to_first_in_h=s_arrival_to_first_in_h,
                s_transfer_stay_h=s_transfer_stay_h,
                s_between_transfers_h=s_between_transfers_h,
                s_icu_los_h=s_icu_los_h,
                s_last_to_discharge_h=s_last_to_discharge_h,
                s_discharge_to_callout_m=s_discharge_to_callout_m,
                s_callout_ack_m=s_callout_ack_m,
            )

            case_id_counter += 1
            rows.append(row)

            prev_discharge = row["discharge_time"]

    ed_df = pd.DataFrame(rows)

    # Sort for readability
    if not ed_df.empty:
        ed_df = ed_df.sort_values(["patient_id", "arrival_time"]).reset_index(drop=True)

    # Format datetime strings
    ed_df = format_datetimes(ed_df)

    ed_df.to_csv(ED_CASES_PATH, index=False)
    print(f"Generated ED cases -> {ED_CASES_PATH} (rows={len(ed_df)})")


if __name__ == "__main__":
    main()
