# src/ed_simulation_ml.py
"""
Hybrid ML + Rule-Based Agent ED Simulation — Deliverable 3
===========================================================

This script extends the baseline ED simulation with two coordinated interventions:

1. Machine learning-based POCT streaming at the assessment stage
2. Rule-based boarding prioritisation and escalation for non-discharge patients

Dataset basis:
- Arrival schedule loaded directly from the synthetic ed_cases.csv file
  to preserve realistic temporal ordering and repeated patient visits
- Patients are simulated independently once they enter the system

HYBRID AGENT MECHANISM:

ML intervention at assessment:
- A pre-trained Random Forest model estimates the probability that a
  patient is suitable for POCT-based acceleration
- Prediction uses sampled clinical and pathway-related features:
  diagnosis code, admission type, careunit, arrival hour, weekday,
  and assessment-duration bucket
- If the POCT probability exceeds the configured threshold, the patient
  receives a reduced assessment time based on severity
- Low- and medium-severity POCT patients with high confidence and
  discharge outcome are flagged as fast-tracked

Rule-based intervention at boarding:
- Applies severity-based boarding priority using SimPy PriorityResource
- Applies severity-dependent boarding service times to model escalation
  in bed allocation and inpatient handover

Performance metrics:
- assessment waiting time
- boarding waiting time
- boarding waiting time by severity
- total ED length of stay (LOS)
- POCT usage and fast-track counts
- NHS 4-hour target compliance

This model evaluates whether combining upstream diagnostic acceleration
with downstream boarding prioritisation improves ED flow more effectively
than the baseline and rule-based-only simulation variants.
"""

import os
import pickle
import random
import simpy
import pandas as pd
import numpy as np

from resources_ml import EDResourcesML
from patient import Patient
from ai_agent import (
    get_boarding_priority,
    get_boarding_service_mean_hours,
)

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

NUM_REPLICATIONS   = 10
SIM_DURATION_HOURS = 365 * 24
WARMUP_HOURS       = 7 * 24
BASE_SEED          = 42

# kept only as reference
ARRIVAL_RATE_PER_HOUR = 150 / 24

ASSESSMENT_SERVICE_H = 32.5 / 60
BOARDING_MEAN_H      = 147 / 60

# POCT-adjusted assessment service times (hours)
POCT_ASSESSMENT_TIME = {
    "critical": 16.0 / 60,
    "high":     20.0 / 60,
    "medium":   23.0 / 60,
    "low":      16.0 / 60,
}

POCT_THRESHOLD      = 0.55
FASTTRACK_THRESHOLD = 0.75

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BRANCH_PATH      = "../data/ed_branch_probabilities.csv"
LOOKUP_PATH      = "../data/poct_lookup.pkl"
META_PATH        = "../data/poct_model_meta.pkl"
ED_CASES_PATH    = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"

OUT_DIR          = "../data"
PATIENT_LOG_OUT  = os.path.join(OUT_DIR, "simulation_ml_patient_log.csv")
SUMMARY_OUT      = os.path.join(OUT_DIR, "simulation_ml_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD PARAMETERS
# ----------------------------------------------------------

def load_branch_probabilities(path: str) -> dict:
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        "discharge":   float(row["p_discharge"]),
        "admission":   float(row["p_admission"]),
        "transferred": float(row["p_transferred"]),
    }


def load_arrival_schedule(path: str) -> pd.DataFrame:
    """
    Load dataset-driven arrivals from the synthetic ED cases file.
    Required columns:
      - patient_id
      - case_id
      - arrival_time
    """
    df = pd.read_csv(path, usecols=["patient_id", "case_id", "arrival_time"])
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    df = df.dropna(subset=["patient_id", "case_id", "arrival_time"]).copy()
    df = df.sort_values(["arrival_time", "case_id"]).reset_index(drop=True)
    return df


def build_sim_arrival_schedule(df: pd.DataFrame, sim_duration_hours: float) -> pd.DataFrame:
    """
    Convert dataset timestamps to simulation hours while preserving ordering
    and repeated patient admissions. The full dataset time window is
    linearly mapped into the simulation horizon.
    """
    df = df.copy()

    start_time = df["arrival_time"].min()
    df["arrival_offset_h"] = (
        df["arrival_time"] - start_time
    ).dt.total_seconds() / 3600.0

    max_offset = df["arrival_offset_h"].max()

    if max_offset <= 0:
        df["sim_arrival_h"] = 0.0
    else:
        scale = sim_duration_hours / max_offset
        df["sim_arrival_h"] = df["arrival_offset_h"] * scale

    return df[["patient_id", "case_id", "sim_arrival_h"]].copy()


print("Loading simulation parameters (hybrid ML + rule-based agent)...")
branch_probs = load_branch_probabilities(BRANCH_PATH)

if not os.path.exists(LOOKUP_PATH):
    raise FileNotFoundError(
        f"Lookup table not found: {LOOKUP_PATH}\nRun train_poct_model.py first."
    )

with open(LOOKUP_PATH, "rb") as f:
    POCT_LOOKUP = pickle.load(f)

with open(META_PATH, "rb") as f:
    poct_meta = pickle.load(f)

top_diags   = poct_meta["top_diags"]
le_admit    = poct_meta["le_admit"]
le_cu       = poct_meta["le_cu"]
TOP_N_DIAGS = len(top_diags)
n_triage    = poct_meta.get("n_triage", 21)

DIAG_TO_IDX  = {str(d): i for i, d in enumerate(top_diags)}
ADMIT_TO_IDX = {str(c): i for i, c in enumerate(le_admit.classes_)}
CU_TO_IDX    = {str(c): i for i, c in enumerate(le_cu.classes_)}

print(f"  Assessment mean:        {ASSESSMENT_SERVICE_H*60:.1f} min")
print(f"  Baseline boarding mean: {BOARDING_MEAN_H*60:.1f} min")
print(f"  Branch probs:           discharge={branch_probs['discharge']:.4f}  "
      f"admission={branch_probs['admission']:.4f}  "
      f"transferred={branch_probs['transferred']:.4f}")
print(f"  POCT lookup loaded:     {len(POCT_LOOKUP):,} entries")
print(f"  CV AUC:                 {poct_meta['auc_cv_mean']:.3f} ± {poct_meta['auc_cv_std']:.3f}")
print(f"  Test AUC:               {poct_meta.get('test_auc', 'N/A')}")
print(f"  POCT threshold:         {POCT_THRESHOLD}")
print(f"  Fast-track threshold:   {FASTTRACK_THRESHOLD}")
print()

# ----------------------------------------------------------
# DATASET DISTRIBUTIONS (for feature sampling)
# ----------------------------------------------------------

_df         = pd.read_csv(ED_CASES_PATH)
DIAG_CODES  = _df["primary_diagnosis_code"].astype(str).tolist()
CAREUNITS   = _df["first_careunit"].fillna("UNKNOWN").astype(str).tolist()
ADMIT_TYPES = _df["admission_type"].fillna("EMERGENCY").astype(str).tolist()
del _df

# ----------------------------------------------------------
# SEVERITY / OUTCOME
# ----------------------------------------------------------

SEVERITY_WEIGHTS = {
    "low":      0.40,
    "medium":   0.35,
    "high":     0.18,
    "critical": 0.07,
}

def assign_severity(rng):
    return rng.choices(
        list(SEVERITY_WEIGHTS.keys()),
        weights=list(SEVERITY_WEIGHTS.values()),
        k=1
    )[0]

def sample_outcome(rng):
    r = rng.random()
    if r < branch_probs["discharge"]:
        return "discharge"
    elif r < branch_probs["discharge"] + branch_probs["admission"]:
        return "admission"
    return "transferred"

# ----------------------------------------------------------
# POCT LOOKUP (O(1))
# ----------------------------------------------------------

def get_poct_prob(diag_code, admit_type, careunit,
                  arrival_hour, arrival_weekday, assess_duration_h):
    d = DIAG_TO_IDX.get(str(diag_code), TOP_N_DIAGS)
    a = ADMIT_TO_IDX.get(str(admit_type), 0)
    c = CU_TO_IDX.get(str(careunit), 0)
    h = int(arrival_hour) % 24
    w = int(arrival_weekday) % 7
    t = min(int(round(float(assess_duration_h) * 10)), n_triage - 1)
    return POCT_LOOKUP.get((d, a, c, h, w, t), 0.5)

# ----------------------------------------------------------
# PATIENT FLOW
# ----------------------------------------------------------

def patient_flow(env, patient, resources, rng,
                 completed_list, warmup, poct_stats):
    patient.arrival_time = env.now

    # Sample clinical features
    idx        = rng.randint(0, len(DIAG_CODES) - 1)
    diag_code  = DIAG_CODES[idx]
    careunit   = CAREUNITS[idx]
    admit_type = ADMIT_TYPES[idx]

    arrival_hour = int(env.now) % 24
    arrival_wday = int(env.now / 24) % 7

    # ── STAGE 1: ASSESSMENT with POCT prediction ──────────────────────
    poct_prob = get_poct_prob(
        diag_code, admit_type, careunit,
        arrival_hour, arrival_wday, ASSESSMENT_SERVICE_H
    )
    is_poct = poct_prob >= POCT_THRESHOLD

    if is_poct:
        poct_stats["poct_applied"] += 1
        assess_time = POCT_ASSESSMENT_TIME[patient.severity]
    else:
        assess_time = rng.expovariate(1.0 / ASSESSMENT_SERVICE_H)

    with resources.assessment_bay.request() as assess_req:
        yield assess_req
        patient.assessment_start = env.now
        yield env.timeout(assess_time)
        patient.assessment_end = env.now

    # ── OUTCOME DECISION ─────────────────────────────────────────────
    outcome = sample_outcome(rng)

    # Fast-track flag (logical marker only; no separate resource pathway)
    is_fasttrack = (
        is_poct and
        poct_prob >= FASTTRACK_THRESHOLD and
        patient.severity in ("low", "medium") and
        outcome == "discharge"
    )
    if is_fasttrack:
        poct_stats["fasttrack_applied"] += 1

    patient.outcome = outcome

    # ── STAGE 2: BOARDING (non-discharge, priority + escalation) ─────
    if outcome in ("admission", "transferred"):
        priority = get_boarding_priority(patient.severity)
        boarding_mean_h = get_boarding_service_mean_hours(patient.severity)

        with resources.boarding_slot.request(priority=priority) as board_req:
            yield board_req
            patient.boarding_start = env.now
            yield env.timeout(rng.expovariate(1.0 / boarding_mean_h))
            patient.boarding_end = env.now

    patient.departure_time = env.now

    if patient.arrival_time >= warmup:
        completed_list.append(patient)

# ----------------------------------------------------------
# ARRIVAL GENERATOR — DATASET-DRIVEN
# ----------------------------------------------------------

def generate_arrivals_from_schedule(env, resources, rng, completed_list,
                                    warmup, arrival_schedule, poct_stats):
    for row in arrival_schedule.itertuples(index=False):
        delay = float(row.sim_arrival_h) - env.now
        if delay > 0:
            yield env.timeout(delay)

        patient = Patient(case_id=int(row.case_id), severity=assign_severity(rng))
        patient.patient_id = int(row.patient_id)

        env.process(patient_flow(
            env, patient, resources, rng,
            completed_list, warmup, poct_stats
        ))

# ----------------------------------------------------------
# SINGLE REPLICATION
# ----------------------------------------------------------

def run_replication(rep_id, arrival_schedule):
    seed       = BASE_SEED + rep_id
    rng        = random.Random(seed)
    env        = simpy.Environment()
    resources  = EDResourcesML(env)
    completed  = []
    poct_stats = {"poct_applied": 0, "fasttrack_applied": 0}

    env.process(generate_arrivals_from_schedule(
        env, resources, rng, completed,
        WARMUP_HOURS, arrival_schedule, poct_stats
    ))
    env.run(until=SIM_DURATION_HOURS)
    return completed, poct_stats

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

def _valid(v):  return [x for x in v if x is not None]
def smean(v):   return float(np.mean(_valid(v)))          if _valid(v) else 0.0
def smax(v):    return float(np.max(_valid(v)))           if _valid(v) else 0.0
def spct(v, p): return float(np.percentile(_valid(v), p)) if _valid(v) else 0.0


def compute_metrics(patients, poct_stats):
    n = len(patients)
    if n == 0:
        return {}

    outcomes       = [p.outcome for p in patients]
    assess_waits   = [p.assessment_wait for p in patients]
    boarding_waits = [
        p.boarding_wait for p in patients
        if p.outcome in ("admission", "transferred")
    ]
    total_los      = [p.total_los for p in patients]

    base = {
        "n_patients":            n,
        "n_discharge":           outcomes.count("discharge"),
        "n_admission":           outcomes.count("admission"),
        "n_transferred":         outcomes.count("transferred"),
        "n_poct_applied":        poct_stats["poct_applied"],
        "n_fasttrack":           poct_stats["fasttrack_applied"],

        "assessment_wait_mean":  smean(assess_waits),
        "assessment_wait_p95":   spct(assess_waits, 95),
        "assessment_wait_max":   smax(assess_waits),

        "boarding_wait_mean":    smean(boarding_waits),
        "boarding_wait_p95":     spct(boarding_waits, 95),
        "boarding_wait_max":     smax(boarding_waits),

        "total_los_mean":        smean(total_los),
        "total_los_p95":         spct(total_los, 95),
        "total_los_max":         smax(total_los),
    }

    for sev in ["critical", "high", "medium", "low"]:
        waits = [
            p.boarding_wait for p in patients
            if p.outcome in ("admission", "transferred")
            and p.severity == sev
            and p.boarding_wait is not None
        ]
        base[f"boarding_wait_{sev}_mean"] = float(np.mean(waits)) if waits else 0.0

    return base

# ----------------------------------------------------------
# PRINT
# ----------------------------------------------------------

def print_summary(m, label=""):
    n = max(m.get("n_patients", 1), 1)
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Patients:              {m.get('n_patients',0):>6}")
    print(f"  Discharge:             {m.get('n_discharge',0):>6}  ({m.get('n_discharge',0)/n*100:.1f}%)")
    print(f"  Admission:             {m.get('n_admission',0):>6}  ({m.get('n_admission',0)/n*100:.1f}%)")
    print(f"  Transferred:           {m.get('n_transferred',0):>6}  ({m.get('n_transferred',0)/n*100:.1f}%)")
    print(f"  POCT applied:          {m.get('n_poct_applied',0):>6}  ({m.get('n_poct_applied',0)/n*100:.1f}%)")
    print(f"  Fast-tracked:          {m.get('n_fasttrack',0):>6}  ({m.get('n_fasttrack',0)/n*100:.1f}%)")
    print(f"  ---")
    print(f"  Assessment wait mean:  {m.get('assessment_wait_mean',0)*60:>7.2f} min")
    print(f"  Assessment wait p95:   {m.get('assessment_wait_p95',0)*60:>7.2f} min")
    print(f"  Boarding wait mean:    {m.get('boarding_wait_mean',0)*60:>7.2f} min")
    print(f"  Boarding wait p95:     {m.get('boarding_wait_p95',0)*60:>7.2f} min")
    print(f"  Total LOS mean:        {m.get('total_los_mean',0)*60:>7.2f} min")
    print(f"  Total LOS p95:         {m.get('total_los_p95',0)*60:>7.2f} min")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run_simulation():
    raw_arrivals = load_arrival_schedule(ED_CASES_PATH)
    arrival_schedule = build_sim_arrival_schedule(raw_arrivals, SIM_DURATION_HOURS)

    print(f"\n{'='*58}")
    print("  Hybrid ML + Rule-Based Agent Simulation")
    print(f"  Arrival source: dataset-driven schedule (ed_cases.csv)")
    print(f"  Visits per replication: {len(arrival_schedule):,}")
    print(f"  Unique patients:        {arrival_schedule['patient_id'].nunique():,}")
    print(f"  Replications:          {NUM_REPLICATIONS}")
    print(f"  Horizon:               {SIM_DURATION_HOURS//24} days")
    print(f"  Warm-up:               {WARMUP_HOURS//24} days")
    print(f"  Agent:                 ML POCT + priority/escalation boarding")
    print(f"  Inference:             Precomputed lookup table (O(1))")
    print(f"{'='*58}\n")

    all_metrics  = []
    all_patients = []

    for rep in range(1, NUM_REPLICATIONS + 1):
        print(f"Running replication {rep}/{NUM_REPLICATIONS}...", end=" ", flush=True)
        patients, poct_stats = run_replication(rep, arrival_schedule)
        print(f"done — {len(patients):,} patients  "
              f"(POCT: {poct_stats['poct_applied']:,}, "
              f"fast-track: {poct_stats['fasttrack_applied']:,})")

        m = compute_metrics(patients, poct_stats)
        m["replication"] = rep
        all_metrics.append(m)

        for p in patients:
            row = p.to_dict()
            row["patient_id"] = getattr(p, "patient_id", None)
            row["replication"] = rep
            all_patients.append(row)

        print_summary(m, label=f"Replication {rep}")

    mdf  = pd.DataFrame(all_metrics)
    cols = [c for c in mdf.columns if c != "replication"]
    mu   = mdf[cols].mean()
    sd   = mdf[cols].std()

    print(f"\n{'='*58}")
    print(f"  ML AGENT SUMMARY  (mean ± std, {NUM_REPLICATIONS} replications)")
    print(f"{'='*58}")
    for col in cols:
        if any(x in col for x in ["wait", "los"]):
            print(f"  {col:<40} {mu[col]*60:>8.2f} min  ±  {sd[col]*60:.2f} min")
        else:
            print(f"  {col:<40} {mu[col]:>8.1f}      ±  {sd[col]:.1f}")

    four_hour = [
        1 if p.get("total_los", 999) <= 4.0 else 0
        for p in all_patients if p.get("total_los") is not None
    ]
    if four_hour:
        pct = sum(four_hour) / len(four_hour) * 100
        print(f"\n  NHS 4-hour compliance:  {pct:.1f}%  (target ≥95%)")

    pd.DataFrame(all_patients).to_csv(PATIENT_LOG_OUT, index=False)
    mdf.to_csv(SUMMARY_OUT, index=False)
    print(f"\n  Patient log → {PATIENT_LOG_OUT}  ({len(all_patients):,} rows)")
    print(f"  Summary     → {SUMMARY_OUT}")
    return mdf

if __name__ == "__main__":
    run_simulation()