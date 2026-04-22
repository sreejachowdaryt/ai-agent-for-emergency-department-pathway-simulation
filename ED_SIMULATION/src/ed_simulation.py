# src/ed_simulation.py
"""
Baseline ED Simulation — Deliverable 2
=======================================

This script implements the baseline ED model used for comparison against
the AI-enhanced simulation variants.

DATASET: Hybrid MIMIC-IV-ED + MIMIC-III synthetic dataset.
- ED phase (arrival, assessment, boarding) derived from MIMIC-IV-ED edstays.
- Post-ED inpatient phase (careunit stays) derived from MIMIC-III transfers.
- Simulation models ED-only pathway; patients exit at boarding handover.

PATIENT PATHWAY:
  Arrival
    → wait for assessment bay
    → assessment (triage + doctor combined, 32.5 min mean)
    → OUTCOME DECISION
        discharge   (61.2%) → patient exits immediately
        admission   (34.6%) → waits for boarding slot → exits ED
        transferred  (4.2%) → waits for boarding slot → exits ED

ARRIVALS:
  Arrival schedule is read directly from the synthetic ed_cases.csv file.
  This preserves repeated patient admissions and their temporal ordering.
  Visits are still processed independently once they enter the simulation.

RESOURCES (Little's Law, target ρ = 0.80–0.90):
  assessment_bay:  5
  boarding_slot:   7

PERFORMANCE METRICS:
  assessment_wait  — arrival → assessment bay available
  boarding_wait    — assessment end → boarding slot available (non-discharge)
  total_los        — arrival → ED departure
  NHS 4-hour compliance — proportion with total_los ≤ 240 min
"""

import os
import random
import simpy
import pandas as pd
import numpy as np

from resources import EDResources
from patient import Patient

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

NUM_REPLICATIONS   = 10
SIM_DURATION_HOURS = 365 * 24
WARMUP_HOURS       = 7  * 24
BASE_SEED          = 42

ARRIVAL_RATE_PER_HOUR = 150 / 24   # kept for reference only

# Combined assessment stage (triage + doctor proxy)
# From dataset: initial_assessment_time - arrival_time, mean = 32.5 min
ASSESSMENT_SERVICE_H = 32.5 / 60

# Boarding: from dataset (ed_departure_time - boarding_start_time)
# Mean = 147 min, median = 120 min
BOARDING_MEAN_H = 147 / 60

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BRANCH_PATH      = "../data/ed_branch_probabilities.csv"
ED_CASES_PATH    = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"
OUT_DIR          = "../data"
PATIENT_LOG_OUT  = os.path.join(OUT_DIR, "simulation_patient_log.csv")
SUMMARY_OUT      = os.path.join(OUT_DIR, "simulation_summary.csv")
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
    Convert real timestamps to simulation hours while preserving ordering
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


print("Loading simulation parameters...")
branch_probs = load_branch_probabilities(BRANCH_PATH)

print(f"  Assessment mean:   {ASSESSMENT_SERVICE_H*60:.1f} min")
print(f"  Boarding mean:     {BOARDING_MEAN_H*60:.1f} min")
print(f"  Branch probs:      discharge={branch_probs['discharge']:.4f}  "
      f"admission={branch_probs['admission']:.4f}  "
      f"transferred={branch_probs['transferred']:.4f}")

# ----------------------------------------------------------
# SEVERITY ASSIGNMENT
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
        weights=list(SEVERITY_WEIGHTS.values()), k=1
    )[0]

# ----------------------------------------------------------
# OUTCOME SAMPLING
# ----------------------------------------------------------

def sample_outcome(rng):
    """
    Sample outcome using MIMIC-IV-ED derived disposition probabilities.
    discharge    61.0% — discharged home directly from ED
    admission    34.6% — admitted to ward (boards in ED first)
    transferred   4.4% — transferred to another unit (boards in ED first)

    No severity modifier — flat dataset-derived rates.
    Severity is used only by the AI agent for boarding priority.
    """
    r = rng.random()
    if r < branch_probs["discharge"]:
        return "discharge"
    elif r < branch_probs["discharge"] + branch_probs["admission"]:
        return "admission"
    return "transferred"

# ----------------------------------------------------------
# PATIENT FLOW
# ----------------------------------------------------------

def patient_flow(env, patient, resources, rng, completed_list, warmup):
    """
    ED-only patient pathway:

    STAGE 1 — Assessment (assessment_bay required)
      Patient waits for an assessment bay, then receives combined
      triage + doctor assessment (32.5 min mean, exponential).
      Released when assessment complete.

    STAGE 2 — Boarding (boarding_slot required, non-discharge only)
      Patient waits in ED for bed confirmation.
      Exponential service time, mean 147 min.
      Patient exits ED when boarding complete.

    Discharge patients exit immediately after assessment.
    """

    patient.arrival_time = env.now

    # ── STAGE 1: ASSESSMENT ───────────────────────────────────────────
    with resources.assessment_bay.request() as assess_req:
        yield assess_req
        patient.assessment_start = env.now
        yield env.timeout(rng.expovariate(1.0 / ASSESSMENT_SERVICE_H))
        patient.assessment_end = env.now

    # ── OUTCOME DECISION ─────────────────────────────────────────────
    outcome = sample_outcome(rng)
    patient.outcome = outcome

    # ── STAGE 2: BOARDING (non-discharge only) ────────────────────────
    if outcome in ("admission", "transferred"):
        with resources.boarding_slot.request() as board_req:
            yield board_req
            patient.boarding_start = env.now
            yield env.timeout(rng.expovariate(1.0 / BOARDING_MEAN_H))
            patient.boarding_end = env.now

    # ── DEPARTURE ────────────────────────────────────────────────────
    patient.departure_time = env.now

    if patient.arrival_time >= warmup:
        completed_list.append(patient)

# ----------------------------------------------------------
# ARRIVAL GENERATOR — DATASET-DRIVEN
# ----------------------------------------------------------

def generate_arrivals_from_schedule(env, resources, rng, completed_list,
                                    warmup, arrival_schedule):
    """
    Generate arrivals directly from the synthetic dataset schedule.
    Repeated patient_id values naturally preserve multi-admission structure
    in the arrival stream.
    """
    for row in arrival_schedule.itertuples(index=False):
        delay = float(row.sim_arrival_h) - env.now
        if delay > 0:
            yield env.timeout(delay)

        patient = Patient(case_id=int(row.case_id), severity=assign_severity(rng))

        # attach patient_id dynamically so it can be written to logs
        patient.patient_id = int(row.patient_id)

        env.process(patient_flow(env, patient, resources, rng, completed_list, warmup))

# ----------------------------------------------------------
# SINGLE REPLICATION
# ----------------------------------------------------------

def run_replication(rep_id, arrival_schedule):
    seed      = BASE_SEED + rep_id
    rng       = random.Random(seed)
    env       = simpy.Environment()
    resources = EDResources(env)
    completed = []

    env.process(
        generate_arrivals_from_schedule(
            env, resources, rng, completed, WARMUP_HOURS, arrival_schedule
        )
    )
    env.run(until=SIM_DURATION_HOURS)
    return completed

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

def _valid(v):  return [x for x in v if x is not None]
def smean(v):   return float(np.mean(_valid(v)))          if _valid(v) else 0.0
def smax(v):    return float(np.max(_valid(v)))           if _valid(v) else 0.0
def spct(v, p): return float(np.percentile(_valid(v), p)) if _valid(v) else 0.0


def compute_metrics(patients):
    n = len(patients)
    if n == 0:
        return {}
    outcomes       = [p.outcome         for p in patients]
    assess_waits   = [p.assessment_wait for p in patients]
    boarding_waits = [p.boarding_wait   for p in patients
                      if p.outcome in ("admission", "transferred")]
    total_los      = [p.total_los       for p in patients]

    return {
        "n_patients":            n,
        "n_discharge":           outcomes.count("discharge"),
        "n_admission":           outcomes.count("admission"),
        "n_transferred":         outcomes.count("transferred"),

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

# ----------------------------------------------------------
# PRINT
# ----------------------------------------------------------

def print_summary(m, label=""):
    n = max(m.get("n_patients", 1), 1)
    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"{'='*56}")
    print(f"  Patients:              {m.get('n_patients',0):>6}")
    print(f"  Discharge:             {m.get('n_discharge',0):>6}  ({m.get('n_discharge',0)/n*100:.1f}%)")
    print(f"  Admission:             {m.get('n_admission',0):>6}  ({m.get('n_admission',0)/n*100:.1f}%)")
    print(f"  Transferred:           {m.get('n_transferred',0):>6}  ({m.get('n_transferred',0)/n*100:.1f}%)")
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

    print(f"\n{'='*56}")
    print("  Baseline ED Simulation")
    print(f"  Arrival source: dataset-driven schedule (ed_cases.csv)")
    print(f"  Visits per replication: {len(arrival_schedule):,}")
    print(f"  Unique patients:        {arrival_schedule['patient_id'].nunique():,}")
    print(f"  Replications:          {NUM_REPLICATIONS}")
    print(f"  Horizon:               {SIM_DURATION_HOURS//24} days")
    print(f"  Warm-up:               {WARMUP_HOURS//24} days")
    print(f"{'='*56}\n")

    all_metrics  = []
    all_patients = []

    for rep in range(1, NUM_REPLICATIONS + 1):
        print(f"Running replication {rep}/{NUM_REPLICATIONS}...", end=" ", flush=True)
        patients = run_replication(rep, arrival_schedule)
        print(f"done — {len(patients):,} patients.")
        m = compute_metrics(patients)
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

    print(f"\n{'='*56}")
    print("  SUMMARY ACROSS ALL REPLICATIONS  (mean ± std)")
    print(f"{'='*56}")
    for col in cols:
        if any(x in col for x in ["wait", "los"]):
            print(f"  {col:<30} {mu[col]*60:>8.2f} min  ±  {sd[col]*60:.2f} min")
        else:
            print(f"  {col:<30} {mu[col]:>8.1f}      ±  {sd[col]:.1f}")

    four_hour = [1 if p.get("total_los", 999) <= 4.0 else 0
                 for p in all_patients if p.get("total_los") is not None]
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