# src/generate_ed_cases.py
"""
Generate a synthetic ED dataset (patients.csv + ed_cases.csv).

Hybrid design:
- ED phase calibrated from MIMIC-IV-ED edstays
- Post-ED admitted careunit pathway calibrated from MIMIC-III transfers

ED model:
Arrival -> Triage/Doctor -> Decision -> Discharge OR Boarding -> ED departure

Admitted pathway only:
Boarding -> first_careunit -> optional second_careunit

MIMIC-IV-ED derived:
- arrival hour/weekday patterns
- ED LOS
- ED disposition probabilities

MIMIC-III derived:
- admissions per patient
- inter-admission gaps
- first careunit distribution
- second careunit distribution
- second careunit conditional on first careunit
- second transfer probability
- careunit LOS
- gap between first and second transfer
- last careunit to discharge

Synthetic by design:
- initial_assessment_time
- boarding_start_time

IMPORTANT FIX IN THIS VERSION:
- ensures arrival_time <= initial_assessment_time <= ed_departure_time
- ensures for admitted cases:
  initial_assessment_time <= boarding_start_time <= ed_departure_time
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

random.seed(42)

# =========================================================
# CONFIG
# =========================================================
NUM_PATIENTS = 50000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENTS_PATH = os.path.join(OUT_DIR, "patients.csv")
ED_CASES_PATH = os.path.join(OUT_DIR, "ed_cases.csv")

MIMIC_PMF_PATH = os.path.join(OUT_DIR, "mimic_admissions_count_probabilities.csv")
MIMIC_GAPS_PATH = os.path.join(OUT_DIR, "mimic_interadmission_gaps_days.csv")
BRANCH_PROB_PATH = os.path.join(OUT_DIR, "mimic_branch_probabilities.csv")

# ED phase files from MIMIC-IV-ED
ED_ARRIVAL_HOUR_PROB_PATH = os.path.join(OUT_DIR, "mimic_ed_arrival_hour_probabilities.csv")
ED_ARRIVAL_WEEKDAY_PROB_PATH = os.path.join(OUT_DIR, "mimic_ed_arrival_weekday_probabilities.csv")
ED_DISPOSITION_PROB_PATH = os.path.join(OUT_DIR, "mimic_ed_disposition_probabilities.csv")
ED_LOS_OVERALL_PATH = os.path.join(OUT_DIR, "mimic_ed_los_hours.csv")
ED_LOS_ADMITTED_PATH = os.path.join(OUT_DIR, "mimic_ed_los_admitted_hours.csv")
ED_LOS_HOME_PATH = os.path.join(OUT_DIR, "mimic_ed_los_home_hours.csv")

# Post-ED admitted pathway files from MIMIC-III
FIRST_CAREUNIT_PROB_PATH = os.path.join(OUT_DIR, "mimic_first_careunit_probabilities.csv")
SECOND_CAREUNIT_PROB_PATH = os.path.join(OUT_DIR, "mimic_second_careunit_probabilities.csv")
SECOND_CAREUNIT_TRANSITION_PATH = os.path.join(OUT_DIR, "mimic_second_careunit_transition_probabilities.csv")
GAP_CAREUNIT_STAY_H = os.path.join(OUT_DIR, "mimic_gap_careunit_stay_hours.csv")
GAP_BETWEEN_TRANSFERS_H = os.path.join(OUT_DIR, "mimic_gap_between_transfer1_out_and_transfer2_in_hours.csv")
GAP_LAST_CAREUNIT_TO_DISCHARGE_H = os.path.join(OUT_DIR, "mimic_gap_last_careunit_to_discharge_hours.csv")

# MIMIC reference root
MIMIC_DIR = os.path.join(REPO_ROOT, "Reference_mimic_iii")

print("PROJECT_ROOT =", PROJECT_ROOT)
print("REPO_ROOT =", REPO_ROOT)
print("MIMIC_DIR =", MIMIC_DIR)

LOOKBACK_DAYS = 365 * 3
sim_end = pd.Timestamp.now()

GENDERS = ["M", "F"]
DEFAULT_DIAGNOSIS = "UNKNOWN"
TOP_N_DIAGNOSIS_CODES = 50

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATETIME_COLS = [
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

TAIL_BUCKET_START = 5
MAX_TAIL_ADMISSIONS = 10


# =========================================================
# HELPERS
# =========================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
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
    if not prob_dict:
        return default_value

    values = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(values, weights=weights, k=1)[0]


def load_probability_csv(path: str, key_col: str, prob_col: str = "probability") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if key_col not in df.columns or prob_col not in df.columns:
        raise ValueError(f"{path} must contain columns: {key_col}, {prob_col}")

    probs = dict(zip(df[key_col], df[prob_col]))
    total = sum(float(v) for v in probs.values())
    if total <= 0:
        raise ValueError(f"Probabilities in {path} sum to 0")

    return {k: float(v) / total for k, v in probs.items()}


def load_transition_probability_csv(path: str) -> dict[str, dict[str, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = {"first_careunit", "second_careunit", "probability"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {required}")

    transition_probs = {}
    for _, row in df.iterrows():
        first = str(row["first_careunit"]).strip().upper()
        second = str(row["second_careunit"]).strip().upper()
        prob = float(row["probability"])

        transition_probs.setdefault(first, {})
        transition_probs[first][second] = prob

    return transition_probs


def load_branch_probabilities(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if "p_second_transfer" not in df.columns:
        raise ValueError(f"{path} must contain 'p_second_transfer'")

    p_second = float(df.loc[0, "p_second_transfer"])
    return min(max(p_second, 0.0), 1.0)


def load_mimic_admissions_pmf(path: str):
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
    n = random.choices(options, weights=weights, k=1)[0]
    if n == TAIL_BUCKET_START:
        return random.randint(TAIL_BUCKET_START, MAX_TAIL_ADMISSIONS)
    return n


def load_mimic_gap_days(path: str) -> pd.Series:
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
    while True:
        val = float(gaps.sample(1).iloc[0])
        if val >= min_days:
            return val


def load_arrival_temporal_distributions_from_edstays() -> tuple[dict[int, float], dict[int, float]]:
    hour_probs = load_probability_csv(ED_ARRIVAL_HOUR_PROB_PATH, "hour")
    weekday_probs = load_probability_csv(ED_ARRIVAL_WEEKDAY_PROB_PATH, "weekday")
    return hour_probs, weekday_probs


def sample_mimic_informed_arrival_time(
    lookback_days: int,
    hour_probs: dict[int, float],
    weekday_probs: dict[int, float],
) -> pd.Timestamp:
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
        if candidate_date.weekday() == int(target_weekday):
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            candidate = candidate_date + pd.Timedelta(
                hours=int(target_hour),
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
        hours=int(target_hour),
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

        if candidate_date.weekday() != int(target_weekday):
            continue

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        candidate = candidate_date + pd.Timedelta(
            hours=int(target_hour),
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
    conditional_probs = second_careunit_transition_probs.get(first_careunit, None)
    if conditional_probs:
        return sample_from_distribution(conditional_probs, default_value=default_value)
    return sample_from_distribution(fallback_probs, default_value=default_value)


def load_mimic_categorical_distributions(mimic_dir: str) -> dict[str, dict[str, float]]:
    paths = get_mimic_paths(mimic_dir)

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

    diagnosis_probs = {}
    diag_path = paths.get("DIAGNOSES_ICD", None)

    if diag_path and os.path.exists(diag_path):
        diag = pd.read_csv(diag_path, usecols=["HADM_ID", "ICD9_CODE"])
        diag = normalize_cols(diag)
        diag = diag.dropna(subset=["HADM_ID", "ICD9_CODE"]).copy()
        diag = diag[diag["HADM_ID"].isin(adm_em_set)].copy()

        diagnosis_probs = build_probability_dict(
            diag["ICD9_CODE"],
            top_n=TOP_N_DIAGNOSIS_CODES
        )

    if not diagnosis_probs:
        diagnosis_probs = {DEFAULT_DIAGNOSIS: 1.0}

    return {
        "admission_type_probs": admission_type_probs,
        "diagnosis_probs": diagnosis_probs,
        "discharge_location_probs": discharge_location_probs,
    }


def load_ed_disposition_probabilities() -> dict[str, float]:
    return load_probability_csv(ED_DISPOSITION_PROB_PATH, "disposition")


def sample_pathway_outcome(disposition_probs: dict[str, float]) -> str:
    raw_disp = sample_from_distribution(disposition_probs, default_value="HOME")

    if raw_disp == "ADMITTED":
        return "ADMITTED"
    if raw_disp == "TRANSFER":
        return "TRANSFERRED"
    return "DISCHARGED"


def sample_assessment_time(arrival_time: pd.Timestamp, ed_departure_time: pd.Timestamp) -> pd.Timestamp:
    """
    Generate a synthetic assessment/doctor-completion time that is always
    between arrival and ED departure.
    """
    earliest_assessment = arrival_time + timedelta(minutes=5)

    # Ultra-short ED stay fallback
    if ed_departure_time <= earliest_assessment:
        candidate = ed_departure_time - timedelta(minutes=1)
        if candidate < arrival_time:
            return arrival_time
        return candidate

    latest_allowed = ed_departure_time - timedelta(minutes=1)

    candidate = random_time_after(arrival_time, 5, 60)

    if candidate > latest_allowed:
        return latest_allowed

    return candidate


def sample_boarding_start_time(
    initial_assessment_time: pd.Timestamp,
    ed_departure_time: pd.Timestamp
) -> pd.Timestamp:
    """
    Generate a boarding start time that always lies between assessment
    completion and ED departure.
    """
    if ed_departure_time <= initial_assessment_time:
        return initial_assessment_time

    gap_seconds = max((ed_departure_time - initial_assessment_time).total_seconds(), 60)
    frac = random.uniform(0.4, 0.85)
    candidate = initial_assessment_time + timedelta(seconds=gap_seconds * frac)

    latest_allowed = ed_departure_time - timedelta(minutes=1)

    if latest_allowed < initial_assessment_time:
        return initial_assessment_time

    if candidate > latest_allowed:
        return latest_allowed

    return candidate


# =========================================================
# CASE GENERATION
# =========================================================
def generate_one_case(
    patient_id: int,
    case_id: int,
    arrival_time: pd.Timestamp,
    p_second_transfer: float,
    s_ed_los_overall_h: pd.Series,
    s_ed_los_admitted_h: pd.Series,
    s_ed_los_home_h: pd.Series,
    s_careunit_stay_h: pd.Series,
    s_between_transfers_h: pd.Series,
    s_last_careunit_to_discharge_h: pd.Series,
    admission_type_probs: dict[str, float],
    diagnosis_probs: dict[str, float],
    discharge_location_probs: dict[str, float],
    first_careunit_probs: dict[str, float],
    second_careunit_probs: dict[str, float],
    second_careunit_transition_probs: dict[str, dict[str, float]],
    disposition_probs: dict[str, float],
) -> dict:
    """
    Generate one ED case using hybrid ED + transfer logic.
    """

    admission_type = sample_from_distribution(admission_type_probs, default_value="EMERGENCY")
    primary_diagnosis_code = sample_from_distribution(diagnosis_probs, default_value=DEFAULT_DIAGNOSIS)
    discharge_location = sample_from_distribution(discharge_location_probs, default_value="HOME")

    pathway_outcome = sample_pathway_outcome(disposition_probs)

    boarding_start_time = None
    ed_departure_time = None
    initial_assessment_time = None

    first_careunit = None
    first_transfer_in = None
    first_transfer_out = None

    second_careunit = None
    second_transfer_in = None
    second_transfer_out = None

    total_careunit_los_hours = None
    ed_los_hours = None
    discharge_time = None

    # ---------------------------------------------
    # Branch A: direct discharge from ED
    # ---------------------------------------------
    if pathway_outcome == "DISCHARGED":
        base_ed_los = (
            sample_empirical(s_ed_los_home_h, min_val=0.10)
            if len(s_ed_los_home_h) > 0
            else sample_empirical(s_ed_los_overall_h, min_val=0.10)
        )

        ed_departure_time = arrival_time + timedelta(hours=base_ed_los)
        ed_los_hours = round(base_ed_los, 2)

        initial_assessment_time = sample_assessment_time(arrival_time, ed_departure_time)
        discharge_time = ed_departure_time

        return {
            "patient_id": patient_id,
            "case_id": case_id,
            "arrival_time": arrival_time,
            "initial_assessment_time": initial_assessment_time,
            "boarding_start_time": boarding_start_time,
            "ed_departure_time": ed_departure_time,
            "ed_los_hours": ed_los_hours,
            "admission_type": admission_type,
            "first_careunit": first_careunit,
            "first_transfer_in": first_transfer_in,
            "first_transfer_out": first_transfer_out,
            "second_careunit": second_careunit,
            "second_transfer_in": second_transfer_in,
            "second_transfer_out": second_transfer_out,
            "total_careunit_los_hours": total_careunit_los_hours,
            "primary_diagnosis_code": primary_diagnosis_code,
            "discharge_time": discharge_time,
            "discharge_location": discharge_location,
            "pathway_outcome": pathway_outcome,
        }

    # ---------------------------------------------
    # Branch B: admitted / transferred after ED
    # ---------------------------------------------
    base_ed_los = (
        sample_empirical(s_ed_los_admitted_h, min_val=0.10)
        if len(s_ed_los_admitted_h) > 0
        else sample_empirical(s_ed_los_overall_h, min_val=0.10)
    )

    ed_departure_time = arrival_time + timedelta(hours=base_ed_los)
    ed_los_hours = round(base_ed_los, 2)

    initial_assessment_time = sample_assessment_time(arrival_time, ed_departure_time)
    boarding_start_time = sample_boarding_start_time(initial_assessment_time, ed_departure_time)

    first_careunit = sample_from_distribution(first_careunit_probs, default_value="WARD")

    # Patient leaves ED into first careunit at ed_departure_time
    first_transfer_in = ed_departure_time

    stay1_h = sample_empirical(s_careunit_stay_h, min_val=0.01)
    first_transfer_out = first_transfer_in + timedelta(hours=stay1_h)

    total_los = stay1_h
    has_second_transfer = (random.random() < p_second_transfer)

    if has_second_transfer:
        second_careunit = sample_conditional_second_careunit(
            first_careunit=first_careunit,
            second_careunit_transition_probs=second_careunit_transition_probs,
            fallback_probs=second_careunit_probs,
            default_value="WARD",
        )

        gap_between_h = sample_empirical(s_between_transfers_h, min_val=0.01)
        second_transfer_in = first_transfer_out + timedelta(hours=gap_between_h)

        stay2_h = sample_empirical(s_careunit_stay_h, min_val=0.01)
        second_transfer_out = second_transfer_in + timedelta(hours=stay2_h)

        total_los += stay2_h
        last_activity_time = second_transfer_out
        pathway_outcome = "TRANSFERRED"
    else:
        last_activity_time = first_transfer_out
        pathway_outcome = "ADMITTED"

    total_careunit_los_hours = round(total_los, 2)

    last_to_discharge_h = sample_empirical(s_last_careunit_to_discharge_h, min_val=0.01)
    discharge_time = last_activity_time + timedelta(hours=last_to_discharge_h)

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "arrival_time": arrival_time,
        "initial_assessment_time": initial_assessment_time,
        "boarding_start_time": boarding_start_time,
        "ed_departure_time": ed_departure_time,
        "ed_los_hours": ed_los_hours,
        "admission_type": admission_type,
        "first_careunit": first_careunit,
        "first_transfer_in": first_transfer_in,
        "first_transfer_out": first_transfer_out,
        "second_careunit": second_careunit,
        "second_transfer_in": second_transfer_in,
        "second_transfer_out": second_transfer_out,
        "total_careunit_los_hours": total_careunit_los_hours,
        "primary_diagnosis_code": primary_diagnosis_code,
        "discharge_time": discharge_time,
        "discharge_location": discharge_location,
        "pathway_outcome": pathway_outcome,
    }


def format_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(DATETIME_FMT)
    return df


# =========================================================
# MAIN
# =========================================================
def main():
    options, weights = load_mimic_admissions_pmf(MIMIC_PMF_PATH)
    print("Loaded MIMIC PMF (bucketed 0..4, 5+):")
    print(list(zip(options, [round(w, 6) for w in weights])))

    mimic_gaps_days = load_mimic_gap_days(MIMIC_GAPS_PATH)
    print(f"Loaded {len(mimic_gaps_days)} MIMIC inter-admission gaps (days).")

    p_second_transfer = load_branch_probabilities(BRANCH_PROB_PATH)
    print(f"Loaded p_second_transfer={p_second_transfer:.4f}")

    hour_probs, weekday_probs = load_arrival_temporal_distributions_from_edstays()
    print("Loaded ED arrival temporal distributions from MIMIC-IV-ED edstays.")

    disposition_probs = load_ed_disposition_probabilities()
    print("Loaded ED disposition probabilities from MIMIC-IV-ED edstays.")

    categorical = load_mimic_categorical_distributions(MIMIC_DIR)
    admission_type_probs = categorical["admission_type_probs"]
    diagnosis_probs = categorical["diagnosis_probs"]
    discharge_location_probs = categorical["discharge_location_probs"]

    first_careunit_probs = load_probability_csv(FIRST_CAREUNIT_PROB_PATH, "careunit")
    second_careunit_probs = load_probability_csv(SECOND_CAREUNIT_PROB_PATH, "careunit")
    second_careunit_transition_probs = load_transition_probability_csv(SECOND_CAREUNIT_TRANSITION_PATH)

    s_ed_los_overall_h = load_gap_series(ED_LOS_OVERALL_PATH, "ed_los_hours")
    s_ed_los_admitted_h = load_gap_series(ED_LOS_ADMITTED_PATH, "ed_los_hours")
    s_ed_los_home_h = load_gap_series(ED_LOS_HOME_PATH, "ed_los_hours")

    s_careunit_stay_h = load_gap_series(GAP_CAREUNIT_STAY_H, "gap_hours")
    s_between_transfers_h = load_gap_series(GAP_BETWEEN_TRANSFERS_H, "gap_hours")
    s_last_careunit_to_discharge_h = load_gap_series(GAP_LAST_CAREUNIT_TO_DISCHARGE_H, "gap_hours")

    print("\nLoaded all hybrid MIMIC-derived distributions.")

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

        first_arrival = sample_mimic_informed_arrival_time(
            lookback_days=LOOKBACK_DAYS,
            hour_probs=hour_probs,
            weekday_probs=weekday_probs,
        )

        prev_discharge = None

        for adm_idx in range(n_adm):
            if adm_idx == 0:
                arrival_time = first_arrival
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
                s_ed_los_overall_h=s_ed_los_overall_h,
                s_ed_los_admitted_h=s_ed_los_admitted_h,
                s_ed_los_home_h=s_ed_los_home_h,
                s_careunit_stay_h=s_careunit_stay_h,
                s_between_transfers_h=s_between_transfers_h,
                s_last_careunit_to_discharge_h=s_last_careunit_to_discharge_h,
                admission_type_probs=admission_type_probs,
                diagnosis_probs=diagnosis_probs,
                discharge_location_probs=discharge_location_probs,
                first_careunit_probs=first_careunit_probs,
                second_careunit_probs=second_careunit_probs,
                second_careunit_transition_probs=second_careunit_transition_probs,
                disposition_probs=disposition_probs,
            )

            rows.append(row)
            case_id_counter += 1
            prev_discharge = pd.Timestamp(row["discharge_time"])

    ed_df = pd.DataFrame(rows)

    if not ed_df.empty:
        ed_df = ed_df.sort_values(["patient_id", "arrival_time"]).reset_index(drop=True)

        # Final safety checks
        dt_arrival = pd.to_datetime(ed_df["arrival_time"], errors="coerce")
        dt_assess = pd.to_datetime(ed_df["initial_assessment_time"], errors="coerce")
        dt_depart = pd.to_datetime(ed_df["ed_departure_time"], errors="coerce")
        dt_board = pd.to_datetime(ed_df["boarding_start_time"], errors="coerce")

        invalid_assessment = ((dt_assess < dt_arrival) | (dt_assess > dt_depart)).sum()
        invalid_boarding = (
            dt_board.notna() & ((dt_board < dt_assess) | (dt_board > dt_depart))
        ).sum()

        print(f"Validation: invalid assessment rows = {int(invalid_assessment)}")
        print(f"Validation: invalid boarding rows = {int(invalid_boarding)}")

    ed_df = format_datetimes(ed_df)
    ed_df.to_csv(ED_CASES_PATH, index=False)

    print(f"Generated ED cases -> {ED_CASES_PATH} (rows={len(ed_df)})")


if __name__ == "__main__":
    main()