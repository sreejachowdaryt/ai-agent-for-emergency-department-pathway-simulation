# ed_simulation_ai.py
"""
Rule-Based Priority + Escalation Agent ED Simulation — Deliverable 3
====================================================================
AI Agent for Emergency Department Pathway Simulation
BSc Computer Science with AI, University of Leeds

This version matches the dataset-driven baseline simulation, but adds
a rule-based boarding intervention.

DATASET-DRIVEN ARRIVALS:
  Arrival schedule is read directly from the synthetic ed_cases.csv file.
  This preserves repeated patient admissions and temporal ordering.
  Visits are still processed independently once they enter the simulation.

AI AGENT MECHANISM:
  When a non-discharge patient requests a boarding slot, the agent:
    1. assigns a severity-based priority integer
    2. assigns a severity-based boarding mean time

  Priority:
    critical → 1
    high     → 2
    medium   → 3
    low      → 4

  Boarding mean time:
    critical → 90 min
    high     → 110 min
    medium   → 147 min
    low      → 147 min

This can be interpreted as a boarding escalation rule:
higher-acuity patients trigger faster inpatient bed allocation /
handover processes while also receiving queue priority.
"""

import os
import random
import simpy
import pandas as pd
import numpy as np

from resources_ai import EDResourcesAI
from patient import Patient
from ai_agent import (
    get_boarding_priority,
    get_boarding_service_mean_hours,
    describe_agent,
)

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

NUM_REPLICATIONS   = 10
SIM_DURATION_HOURS = 365 * 24
WARMUP_HOURS       = 7 * 24
BASE_SEED          = 42

# kept only as baseline reference in print/docstring
ARRIVAL_RATE_PER_HOUR = 150 / 24

# Combined assessment stage (triage + doctor proxy)
ASSESSMENT_SERVICE_H = 32.5 / 60

# Baseline boarding mean (reference)
BOARDING_MEAN_H = 147 / 60

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BRANCH_PATH      = "../data/ed_branch_probabilities.csv"
ED_CASES_PATH    = "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv"
OUT_DIR          = "../data"
PATIENT_LOG_OUT  = os.path.join(OUT_DIR, "simulation_ai_patient_log.csv")
SUMMARY_OUT      = os.path.join(OUT_DIR, "simulation_ai_summary.csv")
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


print("Loading simulation parameters (rule-based AI agent)...")
branch_probs = load_branch_probabilities(BRANCH_PATH)

print(f"  Assessment mean:       {ASSESSMENT_SERVICE_H*60:.1f} min")
print(f"  Baseline boarding mean:{BOARDING_MEAN_H*60:.1f} min")
print(f"  Branch probs:          discharge={branch_probs['discharge']:.4f}  "
      f"admission={branch_probs['admission']:.4f}  "
      f"transferred={branch_probs['transferred']:.4f}")
print()
print(describe_agent())
print()

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
# PATIENT FLOW — AI AGENT VERSION
# ----------------------------------------------------------

def patient_flow(env, patient, resources, rng, completed_list, warmup):
    """
    Baseline flow with rule-based intervention at boarding:
      1. severity-based priority
      2. severity-based boarding escalation (faster boarding for
         critical/high cases)
    """
    patient.arrival_time = env.now

    # ── STAGE 1: ASSESSMENT (unchanged) ──────────────────────────────
    with resources.assessment_bay.request() as assess_req:
        yield assess_req
        patient.assessment_start = env.now
        yield env.timeout(rng.expovariate(1.0 / ASSESSMENT_SERVICE_H))
        patient.assessment_end = env.now

    # ── OUTCOME DECISION (unchanged) ─────────────────────────────────
    outcome = sample_outcome(rng)
    patient.outcome = outcome

    # ── STAGE 2: BOARDING — AI AGENT ACTIVE HERE ─────────────────────
    if outcome in ("admission", "transferred"):
        priority = get_boarding_priority(patient.severity)
        boarding_mean_h = get_boarding_service_mean_hours(patient.severity)

        with resources.boarding_slot.request(priority=priority) as board_req:
            yield board_req
            patient.boarding_start = env.now
            yield env.timeout(rng.expovariate(1.0 / boarding_mean_h))
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
        patient.patient_id = int(row.patient_id)

        env.process(patient_flow(env, patient, resources, rng, completed_list, warmup))

# ----------------------------------------------------------
# SINGLE REPLICATION
# ----------------------------------------------------------

def run_replication(rep_id, arrival_schedule):
    seed      = BASE_SEED + rep_id
    rng       = random.Random(seed)
    env       = simpy.Environment()
    resources = EDResourcesAI(env)
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

    outcomes       = [p.outcome for p in patients]
    assess_waits   = [p.assessment_wait for p in patients]
    boarding_waits = [
        p.boarding_wait for p in patients
        if p.outcome in ("admission", "transferred")
    ]
    total_los      = [p.total_los for p in patients]

    boarding_by_sev = {}
    for sev in ["critical", "high", "medium", "low"]:
        waits = [
            p.boarding_wait for p in patients
            if p.outcome in ("admission", "transferred")
            and p.severity == sev
            and p.boarding_wait is not None
        ]
        boarding_by_sev[f"boarding_wait_{sev}_mean"] = (
            float(np.mean(waits)) if waits else 0.0
        )

    base = {
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
    base.update(boarding_by_sev)
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
    print(f"  ---")
    print(f"  Assessment wait mean:  {m.get('assessment_wait_mean',0)*60:>7.2f} min")
    print(f"  Assessment wait p95:   {m.get('assessment_wait_p95',0)*60:>7.2f} min")
    print(f"  Boarding wait mean:    {m.get('boarding_wait_mean',0)*60:>7.2f} min")
    print(f"  Boarding wait p95:     {m.get('boarding_wait_p95',0)*60:>7.2f} min")
    print(f"  Total LOS mean:        {m.get('total_los_mean',0)*60:>7.2f} min")
    print(f"  Total LOS p95:         {m.get('total_los_p95',0)*60:>7.2f} min")
    print(f"  ---  Boarding wait by severity  ---")
    for sev in ["critical", "high", "medium", "low"]:
        print(f"  {sev:<10} boarding wait: "
              f"{m.get(f'boarding_wait_{sev}_mean', 0)*60:>7.2f} min")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run_simulation():
    raw_arrivals = load_arrival_schedule(ED_CASES_PATH)
    arrival_schedule = build_sim_arrival_schedule(raw_arrivals, SIM_DURATION_HOURS)

    print(f"\n{'='*58}")
    print("  Rule-Based Priority + Escalation Agent Simulation")
    print(f"  Arrival source: dataset-driven schedule (ed_cases.csv)")
    print(f"  Visits per replication: {len(arrival_schedule):,}")
    print(f"  Unique patients:        {arrival_schedule['patient_id'].nunique():,}")
    print(f"  Replications:          {NUM_REPLICATIONS}")
    print(f"  Horizon:               {SIM_DURATION_HOURS//24} days")
    print(f"  Warm-up:               {WARMUP_HOURS//24} days")
    print(f"  Agent:                 Severity-weighted boarding priority + escalation")
    print(f"{'='*58}\n")

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

    print(f"\n{'='*58}")
    print("  AI AGENT SUMMARY  (mean ± std, 10 replications)")
    print(f"{'='*58}")
    for col in cols:
        if any(x in col for x in ["wait", "los"]):
            print(f"  {col:<38} {mu[col]*60:>8.2f} min  ±  {sd[col]*60:.2f} min")
        else:
            print(f"  {col:<38} {mu[col]:>8.1f}      ±  {sd[col]:.1f}")

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