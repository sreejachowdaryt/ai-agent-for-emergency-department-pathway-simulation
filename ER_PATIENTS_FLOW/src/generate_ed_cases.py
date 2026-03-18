# src/generate_ed_cases.py
"""
Generate a synthetic ED dataset (patients.csv + ed_cases.csv).

This version keeps the existing time/pathway logic but improves the
categorical fields using empirically derived distributions from MIMIC.

What is now MIMIC-derived:
- admissions per patient
- inter-admission gaps
- second-transfer probability
- ICU probability
- activity gap distributions
- callout outcome distribution
- admission_type distribution
- first_careunit distribution
- second_careunit distribution conditioned on first_careunit
- primary_diagnosis_code distribution
- discharge_location distribution
- arrival hour-of-day and weekday patterns

What remains synthetic by design:
- exact patient-level arrival timestamps are synthetically generated
  within the simulation window, but are constrained by MIMIC-derived
  hour-of-day and weekday admission patterns
- initial_assessment_time as a short operational delay after arrival
"""

import os
import random
import pandas as pd
from datetime import timedelta

from utils import (
    random_time_after,
    derive_age_group,
    load_gap_series,
    sample_empirical,
)

from mimic_paths import get_mimic_paths


random.seed(42)  # Reproducibility

# =========================================================
# CONFIG
# =========================================================
NUM_PATIENTS = 10000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)      # .../ER_PATIENTS_FLOW
REPO_ROOT = os.path.dirname(PROJECT_ROOT)     # .../ai-agent-for-emergency-department-pathway-simulation

OUT_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENTS_PATH = os.path.join(OUT_DIR, "patients.csv")
ED_CASES_PATH = os.path.join(OUT_DIR, "ed_cases.csv")

# MIMIC-derived PMF file
MIMIC_PMF_PATH = os.path.join(OUT_DIR, "mimic_admissions_count_probabilities.csv")

# MIMIC-derived inter-admission gaps
MIMIC_GAPS_PATH = os.path.join(OUT_DIR, "mimic_interadmission_gaps_days.csv")

# MIMIC-derived branch probabilities
BRANCH_PROB_PATH = os.path.join(OUT_DIR, "mimic_branch_probabilities.csv")

# MIMIC-derived activity gap distributions
GAP_ARRIVAL_TO_FIRST_IN_H = os.path.join(OUT_DIR, "mimic_gap_arrival_to_first_transfer_in_hours.csv")
GAP_TRANSFER_STAY_H = os.path.join(OUT_DIR, "mimic_gap_transfer_stay_hours.csv")
GAP_BETWEEN_TRANSFERS_H = os.path.join(OUT_DIR, "mimic_gap_between_transfer1_out_and_transfer2_in_hours.csv")
GAP_ICU_LOS_H = os.path.join(OUT_DIR, "mimic_gap_icu_los_hours.csv")
GAP_LAST_TO_DISCHARGE_H = os.path.join(OUT_DIR, "mimic_gap_last_activity_to_discharge_hours.csv")
GAP_DISCHARGE_TO_CALLOUT_M = os.path.join(OUT_DIR, "mimic_gap_discharge_to_callout_create_minutes.csv")
GAP_CALLOUT_ACK_M = os.path.join(OUT_DIR, "mimic_gap_callout_create_to_ack_minutes.csv")

# Reference MIMIC folder lives at repo root
MIMIC_DIR = os.path.join(REPO_ROOT, "Reference_mimic_iii")

# Debug statements
print("PROJECT_ROOT =", PROJECT_ROOT)
print("REPO_ROOT =", REPO_ROOT)
print("MIMIC_DIR =", MIMIC_DIR)

# How far back first admissions can start
LOOKBACK_DAYS = 365 * 3
sim_end = pd.Timestamp.now()

# Value pools / defaults
GENDERS = ["M", "F"]
DEFAULT_DIAGNOSIS = "UNKNOWN"

# Keep only the most frequent diagnosis codes to avoid an excessively sparse distribution
TOP_N_DIAGNOSIS_CODES = 50

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
TAIL_BUCKET_START = 5
MAX_TAIL_ADMISSIONS = 10


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names for safer cross-file handling.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    return df


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def build_probability_dict(series: pd.Series, top_n: int | None = None) -> dict[str, float]:
    """
    Convert a categorical series into a probability dictionary.
    """
    s = series.astype(str).str.strip().str.upper()
    s = s[s.notna() & (s != "") & (s != "NAN")]

    if top_n is not None:
        top_values = s.value_counts().head(top_n).index
        s = s[s.isin(top_values)]

    if s.empty:
        return {}

    probs = s.value_counts(normalize=True).to_dict()
    return {str(k): float(v) for k, v in probs.items()}


def sample_from_distribution(prob_dict: dict[str, float], default_value: str) -> str:
    """
    Sample one categorical value from a probability dictionary.
    """
    if not prob_dict:
        return default_value

    values = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(values, weights=weights, k=1)[0]


def load_arrival_temporal_distributions(mimic_dir: str) -> tuple[dict[int, float], dict[int, float]]:
    """
    Learn hour-of-day and weekday probabilities from MIMIC ADMISSIONS.ADMITTIME.
    Uses emergency admissions as the ED-like cohort.

    Returns:
        hour_probs: {0..23: prob}
        weekday_probs: {0..6: prob}  # Monday=0 ... Sunday=6
    """
    paths = get_mimic_paths(mimic_dir)

    adm = pd.read_csv(
        paths["ADMISSIONS"],
        usecols=["ADMISSION_TYPE", "ADMITTIME"]
    )
    adm = normalize_cols(adm)
    adm["ADMITTIME"] = to_dt(adm["ADMITTIME"])

    adm = adm.dropna(subset=["ADMITTIME"]).copy()

    # Restrict to emergency admissions for ED-like realism
    adm = adm[
        adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().eq("EMERGENCY")
    ].copy()

    if adm.empty:
        hour_probs = {h: 1 / 24 for h in range(24)}
        weekday_probs = {d: 1 / 7 for d in range(7)}
        return hour_probs, weekday_probs

    hour_probs = (
        adm["ADMITTIME"].dt.hour.value_counts(normalize=True).sort_index().to_dict()
    )
    weekday_probs = (
        adm["ADMITTIME"].dt.weekday.value_counts(normalize=True).sort_index().to_dict()
    )

    hour_probs = {h: float(hour_probs.get(h, 0.0)) for h in range(24)}
    weekday_probs = {d: float(weekday_probs.get(d, 0.0)) for d in range(7)}

    hour_total = sum(hour_probs.values())
    weekday_total = sum(weekday_probs.values())

    if hour_total > 0:
        hour_probs = {h: v / hour_total for h, v in hour_probs.items()}
    else:
        hour_probs = {h: 1 / 24 for h in range(24)}

    if weekday_total > 0:
        weekday_probs = {d: v / weekday_total for d, v in weekday_probs.items()}
    else:
        weekday_probs = {d: 1 / 7 for d in range(7)}

    return hour_probs, weekday_probs


def sample_mimic_informed_arrival_time(
    lookback_days: int,
    hour_probs: dict[int, float],
    weekday_probs: dict[int, float],
) -> pd.Timestamp:
    """
    Generate a synthetic arrival timestamp within the lookback window, but with
    hour-of-day and weekday patterns learned from MIMIC ADMITTIME.
    """
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=lookback_days)

    target_weekday = random.choices(
        list(weekday_probs.keys()),
        weights=list(weekday_probs.values()),
        k=1
    )[0]

    target_hour = random.choices(
        list(hour_probs.keys()),
        weights=list(hour_probs.values()),
        k=1
    )[0]

    for _ in range(100):
        random_day_offset = random.randint(0, lookback_days)
        candidate_date = (start + pd.Timedelta(days=random_day_offset)).normalize()
        if candidate_date.weekday() == target_weekday:
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            candidate = candidate_date + pd.Timedelta(
                hours=target_hour,
                minutes=minute,
                seconds=second
            )
            if candidate <= now:
                return candidate

    random_day_offset = random.randint(0, lookback_days)
    candidate_date = (start + pd.Timedelta(days=random_day_offset)).normalize()
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return candidate_date + pd.Timedelta(
        hours=target_hour,
        minutes=minute,
        seconds=second
    )


def sample_mimic_informed_arrival_time_after(
    earliest_time: pd.Timestamp,
    sim_end: pd.Timestamp,
    hour_probs: dict[int, float],
    weekday_probs: dict[int, float],
    max_attempts: int = 500,
) -> pd.Timestamp:
    """
    Sample a MIMIC-informed arrival timestamp on or after earliest_time,
    while respecting learned weekday and hour-of-day distributions.
    """
    earliest_time = pd.Timestamp(earliest_time)
    sim_end = pd.Timestamp(sim_end)

    if earliest_time > sim_end:
        return earliest_time

    start_date = earliest_time.normalize()
    end_date = sim_end.normalize()

    if start_date > end_date:
        return earliest_time

    target_hour = random.choices(
        list(hour_probs.keys()),
        weights=list(hour_probs.values()),
        k=1
    )[0]

    for _ in range(max_attempts):
        target_weekday = random.choices(
            list(weekday_probs.keys()),
            weights=list(weekday_probs.values()),
            k=1
        )[0]

        total_days = (end_date - start_date).days
        if total_days < 0:
            break

        day_offset = random.randint(0, total_days)
        candidate_date = start_date + pd.Timedelta(days=day_offset)

        if candidate_date.weekday() != target_weekday:
            continue

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        candidate = candidate_date + pd.Timedelta(
            hours=target_hour,
            minutes=minute,
            seconds=second
        )

        if candidate < earliest_time:
            continue

        if candidate <= sim_end:
            return candidate

    fallback = earliest_time
    if fallback <= sim_end:
        return fallback

    return sim_end


def sample_conditional_second_careunit(
    first_careunit: str,
    second_careunit_transition_probs: dict[str, dict[str, float]],
    fallback_probs: dict[str, float],
    default_value: str = "WARD",
) -> str:
    """
    Sample second careunit conditional on the sampled first careunit.
    Falls back to overall second-careunit probabilities if the first careunit
    is unseen in the transition map.
    """
    conditional_probs = second_careunit_transition_probs.get(first_careunit, None)

    if conditional_probs:
        return sample_from_distribution(conditional_probs, default_value=default_value)

    return sample_from_distribution(fallback_probs, default_value=default_value)


# =========================================================
# PMF loading + sampling (from MIMIC)
# =========================================================
def load_mimic_admissions_pmf(path: str):
    """
    Load admissions-count probabilities derived from MIMIC.
    Expected CSV columns: num_admissions, probability

    Returns:
      options: [0,1,2,3,4,5] where 5 means "5+"
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

    options = list(range(0, TAIL_BUCKET_START))
    weights = [float(probs.get(k, 0.0)) for k in options]

    tail_prob = float(pmf.loc[pmf["num_admissions"] >= TAIL_BUCKET_START, "probability"].sum())
    options.append(TAIL_BUCKET_START)
    weights.append(tail_prob)

    total = sum(weights)
    if total <= 0:
        raise ValueError(f"PMF probabilities sum to 0. Check file: {path}")
    weights = [w / total for w in weights]

    return options, weights


def sample_num_admissions(options, weights) -> int:
    """
    Sample admissions per patient using the MIMIC-derived PMF.
    If the tail bucket (5+) is selected, expand to a concrete value.
    """
    n = random.choices(options, weights=weights, k=1)[0]
    if n == TAIL_BUCKET_START:
        return random.randint(TAIL_BUCKET_START, MAX_TAIL_ADMISSIONS)
    return n


# =========================================================
# Inter-admission gaps
# =========================================================
def load_mimic_gap_days(path: str) -> pd.Series:
    """
    Load MIMIC-derived inter-admission gaps in days.
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
    Resample an inter-admission gap from the empirical MIMIC distribution.
    """
    while True:
        val = float(gaps.sample(1).iloc[0])
        if val >= min_days:
            return val


# =========================================================
# Load branch probabilities
# =========================================================
def load_branch_probabilities(path: str) -> tuple[float, float]:
    """
    Load branch probabilities created by extract_activity_gaps_from_mimic.py

    Expected columns:
    - p_second_transfer
    - p_icu
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

    p_second = min(max(p_second, 0.0), 1.0)
    p_icu = min(max(p_icu, 0.0), 1.0)
    return p_second, p_icu


# =========================================================
# Load empirically derived categorical distributions
# =========================================================
def load_mimic_categorical_distributions(mimic_dir: str) -> dict[str, dict[str, float]]:
    """
    Build MIMIC-derived probability distributions for the categorical fields
    that were previously assigned randomly.
    """
    paths = get_mimic_paths(mimic_dir)

    # -------------------------
    # ADMISSIONS
    # -------------------------
    adm = pd.read_csv(
        paths["ADMISSIONS"],
        usecols=["HADM_ID", "ADMISSION_TYPE", "DISCHARGE_LOCATION"]
    )
    adm = normalize_cols(adm)

    adm_em = adm.loc[
        adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().eq("EMERGENCY"),
        "HADM_ID"
    ].dropna().unique()
    adm_em_set = set(adm_em)

    admission_type_probs = build_probability_dict(adm["ADMISSION_TYPE"])

    discharge_location_probs = build_probability_dict(
        adm.loc[adm["HADM_ID"].isin(adm_em_set), "DISCHARGE_LOCATION"]
    )

    # -------------------------
    # TRANSFERS
    # -------------------------
    tr = pd.read_csv(
        paths["TRANSFERS"],
        usecols=["HADM_ID", "INTIME", "CURR_CAREUNIT"]
    )
    tr = normalize_cols(tr)
    tr["INTIME"] = to_dt(tr["INTIME"])
    tr = tr.dropna(subset=["HADM_ID", "INTIME", "CURR_CAREUNIT"]).copy()

    tr = tr[tr["HADM_ID"].isin(adm_em_set)].copy()
    tr = tr.sort_values(["HADM_ID", "INTIME"])

    first_careunits = tr.groupby("HADM_ID")["CURR_CAREUNIT"].first()
    first_careunit_probs = build_probability_dict(first_careunits)

    transfer_sequences = tr.groupby("HADM_ID")["CURR_CAREUNIT"].apply(list)

    second_careunits = transfer_sequences[
        transfer_sequences.apply(lambda x: len(x) >= 2)
    ].apply(lambda x: x[1])

    second_careunit_probs = build_probability_dict(second_careunits)

    second_careunit_transition_probs = {}

    paired_sequences = transfer_sequences[
        transfer_sequences.apply(lambda x: len(x) >= 2)
    ]

    for seq in paired_sequences:
        first_cu = str(seq[0]).strip().upper()
        second_cu = str(seq[1]).strip().upper()

        if first_cu not in second_careunit_transition_probs:
            second_careunit_transition_probs[first_cu] = {}

        second_careunit_transition_probs[first_cu][second_cu] = (
            second_careunit_transition_probs[first_cu].get(second_cu, 0) + 1
        )

    for first_cu, next_counts in second_careunit_transition_probs.items():
        total = sum(next_counts.values())
        if total > 0:
            second_careunit_transition_probs[first_cu] = {
                k: float(v) / total for k, v in next_counts.items()
            }

    # -------------------------
    # DIAGNOSES
    # -------------------------
    diagnosis_probs = {}

    diag_path = paths.get("DIAGNOSES_ICD", None)

    if diag_path and os.path.exists(diag_path):
        diag = pd.read_csv(
            diag_path,
            usecols=["HADM_ID", "ICD9_CODE"]
        )
        diag = normalize_cols(diag)
        diag = diag.dropna(subset=["HADM_ID", "ICD9_CODE"]).copy()

        diag = diag[diag["HADM_ID"].isin(adm_em_set)].copy()

        diagnosis_probs = build_probability_dict(
            diag["ICD9_CODE"],
            top_n=TOP_N_DIAGNOSIS_CODES
        )

    if not diagnosis_probs:
        diagnosis_probs = {DEFAULT_DIAGNOSIS: 1.0}

    # -------------------------
    # CALLOUT
    # -------------------------
    callout = pd.read_csv(
        paths["CALLOUT"],
        usecols=["CALLOUT_OUTCOME"]
    )
    callout = normalize_cols(callout)

    co = callout["CALLOUT_OUTCOME"].astype(str).str.strip().str.upper()

    valid_non_escalated = ["DISCHARGED", "CANCELLED"]
    co = co[co.isin(valid_non_escalated)]

    if co.empty:
        callout_outcome_probs = {
            "DISCHARGED": 0.85,
            "CANCELLED": 0.15,
        }
    else:
        probs = co.value_counts(normalize=True).to_dict()
        callout_outcome_probs = {k: float(probs.get(k, 0.0)) for k in valid_non_escalated}

    return {
        "admission_type_probs": admission_type_probs,
        "first_careunit_probs": first_careunit_probs,
        "second_careunit_probs": second_careunit_probs,
        "second_careunit_transition_probs": second_careunit_transition_probs,
        "diagnosis_probs": diagnosis_probs,
        "discharge_location_probs": discharge_location_probs,
        "callout_outcome_probs": callout_outcome_probs,
    }


def sample_non_escalated_callout_outcome(callout_outcome_probs: dict[str, float]) -> str:
    """
    For cases without ICU and without second transfer, sample a callout outcome
    from the non-escalated disposition categories only.
    """
    allowed = ["DISCHARGED", "CANCELLED"]
    weights = [callout_outcome_probs.get(k, 0.0) for k in allowed]

    total = sum(weights)
    if total <= 0:
        return "DISCHARGED"

    weights = [w / total for w in weights]
    return random.choices(allowed, weights=weights, k=1)[0]


# =========================================================
# Case generation
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
    admission_type_probs: dict[str, float],
    first_careunit_probs: dict[str, float],
    second_careunit_probs: dict[str, float],
    second_careunit_transition_probs: dict[str, dict[str, float]],
    diagnosis_probs: dict[str, float],
    discharge_location_probs: dict[str, float],
    callout_outcome_probs: dict[str, float],
) -> dict:
    """
    Generate one admission case.
    """

    # -----------------------------------------------------
    # Synthetic arrival -> initial assessment buffer
    # -----------------------------------------------------
    initial_assessment_time = random_time_after(arrival_time, 5, 60)

    # -----------------------------------------------------
    # MIMIC-derived first careunit
    # -----------------------------------------------------
    first_careunit = sample_from_distribution(
        first_careunit_probs,
        default_value="WARD"
    )

    # -----------------------------------------------------
    # MIMIC-derived first transfer timing
    # -----------------------------------------------------
    gap1_h = sample_empirical(s_arrival_to_first_in_h, min_val=0.01)
    first_transfer_in = initial_assessment_time + timedelta(hours=gap1_h)

    stay1_h = sample_empirical(s_transfer_stay_h, min_val=0.01)
    first_transfer_out = first_transfer_in + timedelta(hours=stay1_h)

    # -----------------------------------------------------
    # Branch: second transfer
    # -----------------------------------------------------
    has_second_transfer = (random.random() < p_second_transfer)

    second_transfer_in = None
    second_transfer_out = None
    second_careunit = None

    last_transfer_out = first_transfer_out

    if has_second_transfer:
        gap_between_h = sample_empirical(s_between_transfers_h, min_val=0.01)
        second_transfer_in = first_transfer_out + timedelta(hours=gap_between_h)

        stay2_h = sample_empirical(s_transfer_stay_h, min_val=0.01)
        second_transfer_out = second_transfer_in + timedelta(hours=stay2_h)

        second_careunit = sample_conditional_second_careunit(
            first_careunit=first_careunit,
            second_careunit_transition_probs=second_careunit_transition_probs,
            fallback_probs=second_careunit_probs,
            default_value="WARD",
        )

        last_transfer_out = second_transfer_out

    # -----------------------------------------------------
    # Branch: ICU
    # -----------------------------------------------------
    has_icu = (random.random() < p_icu)

    icu_admission_time = None
    icu_discharge_time = None
    icu_los_hours = None
    last_activity_time = last_transfer_out

    if has_icu:
        icu_admission_time = random_time_after(last_transfer_out, 10, 60)

        los_h = sample_empirical(s_icu_los_h, min_val=0.01)
        icu_discharge_time = icu_admission_time + timedelta(hours=los_h)
        icu_los_hours = round(los_h, 2)

        last_activity_time = icu_discharge_time

    # -----------------------------------------------------
    # Last activity -> discharge
    # -----------------------------------------------------
    last_to_discharge_h = sample_empirical(s_last_to_discharge_h, min_val=0.01)
    discharge_time = last_activity_time + timedelta(hours=last_to_discharge_h)

    # -----------------------------------------------------
    # Discharge -> callout create
    # -----------------------------------------------------
    d2c_min = sample_empirical(s_discharge_to_callout_m, min_val=0.0)
    callout_time = discharge_time + timedelta(minutes=d2c_min)

    # -----------------------------------------------------
    # Callout create -> acknowledge
    # -----------------------------------------------------
    c2a_min = sample_empirical(s_callout_ack_m, min_val=0.0)
    callout_ack_time = callout_time + timedelta(minutes=c2a_min)

    # -----------------------------------------------------
    # Other MIMIC-derived categorical fields
    # -----------------------------------------------------
    admission_type = sample_from_distribution(
        admission_type_probs,
        default_value="EMERGENCY"
    )

    primary_diagnosis_code = sample_from_distribution(
        diagnosis_probs,
        default_value=DEFAULT_DIAGNOSIS
    )

    discharge_location = sample_from_distribution(
        discharge_location_probs,
        default_value="HOME"
    )

    # -----------------------------------------------------
    # Pathway-consistent callout outcome
    # -----------------------------------------------------
    if has_icu:
        callout_outcome = "ICU"
    elif has_second_transfer:
        callout_outcome = "TRANSFERRED"
    else:
        callout_outcome = sample_non_escalated_callout_outcome(callout_outcome_probs)

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "arrival_time": arrival_time,
        "initial_assessment_time": initial_assessment_time,
        "admission_type": admission_type,
        "first_careunit": first_careunit,
        "first_transfer_in": first_transfer_in,
        "first_transfer_out": first_transfer_out,
        "second_careunit": second_careunit,
        "second_transfer_in": second_transfer_in,
        "second_transfer_out": second_transfer_out,
        "icu_admission_time": icu_admission_time,
        "icu_discharge_time": icu_discharge_time,
        "icu_los_hours": icu_los_hours,
        "primary_diagnosis_code": primary_diagnosis_code,
        "discharge_time": discharge_time,
        "discharge_location": discharge_location,
        "callout_time": callout_time,
        "callout_ack_time": callout_ack_time,
        "callout_outcome": callout_outcome,
    }


def format_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns to MIMIC-like string format.
    """
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
    # 0) Load MIMIC inter-admission gap distribution
    # ---------------------------------------
    mimic_gaps_days = load_mimic_gap_days(MIMIC_GAPS_PATH)
    print(f"Loaded {len(mimic_gaps_days)} MIMIC inter-admission gaps (days).")

    # ---------------------------------------
    # 0) Load branch probabilities
    # ---------------------------------------
    p_second_transfer, p_icu = load_branch_probabilities(BRANCH_PROB_PATH)
    print(
        f"Loaded branching probabilities: "
        f"p_second_transfer={p_second_transfer:.4f}, p_icu={p_icu:.4f}"
    )

    # ----------------------------------------------------------------------
    # 0) Load ADMITTIME-derived temporal probabilities for arrival_time
    # ----------------------------------------------------------------------
    hour_probs, weekday_probs = load_arrival_temporal_distributions(MIMIC_DIR)
    print("\nLoaded MIMIC arrival temporal distributions.")
    print("hour_probs:", {k: round(v, 4) for k, v in hour_probs.items() if v > 0})
    print("weekday_probs:", {k: round(v, 4) for k, v in weekday_probs.items() if v > 0})

    # ---------------------------------------
    # 0) Load categorical distributions from MIMIC
    # ---------------------------------------
    categorical = load_mimic_categorical_distributions(MIMIC_DIR)

    admission_type_probs = categorical["admission_type_probs"]
    first_careunit_probs = categorical["first_careunit_probs"]
    second_careunit_probs = categorical["second_careunit_probs"]
    second_careunit_transition_probs = categorical["second_careunit_transition_probs"]
    diagnosis_probs = categorical["diagnosis_probs"]
    discharge_location_probs = categorical["discharge_location_probs"]
    callout_outcome_probs = categorical["callout_outcome_probs"]

    print("\nLoaded MIMIC categorical distributions:")
    print("admission_type_probs:", {k: round(v, 4) for k, v in admission_type_probs.items()})
    print("first_careunit_probs:", {k: round(v, 4) for k, v in first_careunit_probs.items()})
    print("second_careunit_probs:", {k: round(v, 4) for k, v in second_careunit_probs.items()})
    print(
        "second_careunit_transition_probs (sample):",
        {
            k: {kk: round(vv, 4) for kk, vv in list(v.items())[:5]}
            for k, v in list(second_careunit_transition_probs.items())[:5]
        }
    )
    print("diagnosis_probs (top shown):", dict(list({k: round(v, 4) for k, v in diagnosis_probs.items()}.items())[:10]))
    print("discharge_location_probs:", {k: round(v, 4) for k, v in discharge_location_probs.items()})
    print("callout_outcome_probs:", {k: round(v, 4) for k, v in callout_outcome_probs.items()})

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

    print("\nLoaded activity-gap distributions from MIMIC.")

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

        n_adm = sample_num_admissions(options, weights)

        if n_adm == 0:
            continue

        arrival = sample_mimic_informed_arrival_time(
            lookback_days=LOOKBACK_DAYS,
            hour_probs=hour_probs,
            weekday_probs=weekday_probs,
        )

        prev_discharge = None

        for adm_idx in range(n_adm):
            if adm_idx == 0:
                arrival_time = arrival
            else:
                gap_days = sample_gap_days_empirical(mimic_gaps_days, min_days=7)
                earliest_next_arrival = prev_discharge + timedelta(days=gap_days)

                arrival_time = sample_mimic_informed_arrival_time_after(
                    earliest_time=earliest_next_arrival,
                    sim_end=sim_end,
                    hour_probs=hour_probs,
                    weekday_probs=weekday_probs,
                )

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
                admission_type_probs=admission_type_probs,
                first_careunit_probs=first_careunit_probs,
                second_careunit_probs=second_careunit_probs,
                second_careunit_transition_probs=second_careunit_transition_probs,
                diagnosis_probs=diagnosis_probs,
                discharge_location_probs=discharge_location_probs,
                callout_outcome_probs=callout_outcome_probs,
            )

            case_id_counter += 1
            rows.append(row)

            prev_discharge = row["discharge_time"]

    ed_df = pd.DataFrame(rows)

    if not ed_df.empty:
        ed_df = ed_df.sort_values(["patient_id", "arrival_time"]).reset_index(drop=True)

    ed_df = format_datetimes(ed_df)

    ed_df.to_csv(ED_CASES_PATH, index=False)
    print(f"Generated ED cases -> {ED_CASES_PATH} (rows={len(ed_df)})")


if __name__ == "__main__":
    main()