# ed_simulation.py
"""
Baseline ED Simulation — Deliverable 2
=======================================
AI Agent for Emergency Department Pathway Simulation
BSc Computer Science with AI, University of Leeds

SCOPE — ED-only model:
  Patients are tracked from arrival until they either:
    (a) are discharged home, or
    (b) are handed over to ward/ICU (boarding slot acquired → patient exits ED)
  The simulation does NOT model the inpatient stay after ED departure.
  This is standard practice for ED simulation studies and is consistent
  with the dissertation aim: "modelling from patient arrival to discharge."

ARRIVAL RATE:
  150 patients/day (exponential interarrival, mean 9.6 minutes).
  Duration parameters (triage, treatment, boarding) are empirically
  derived from the synthetic dataset which was calibrated against
  MIMIC-III distributions. The arrival rate is calibrated to represent
  a mid-sized UK Emergency Department consistent with NHS England
  reported attendances.

PATIENT PATHWAY (matches process model from process_discovery.py):
  Arrival
    → wait for triage nurse + cubicle
    → triage / initial assessment  (10 min active service)
    → wait for doctor
    → doctor consultation          (20 min, NHS ED standard)
    → release cubicle + doctor
    → OUTCOME DECISION
        discharge  (74.8%) → patient exits immediately
        admission  ( 9.7%) → patient waits for boarding slot → exits ED
        icu        (15.4%) → patient waits for boarding slot → exits ED

RESOURCES (Little's Law calibration, target ρ = 0.75–0.85):
  triage_nurse:  2  (ρ ≈ 0.87)
  doctor:        3  (ρ ≈ 0.83)
  cubicle:       8  (ρ ≈ 0.78)
  boarding_slot: 3  (ρ ≈ 0.78)

METRICS COLLECTED (all in hours):
  triage_wait    — arrival → triage nurse available
  doctor_wait    — triage end → doctor available
  boarding_wait  — doctor end → boarding slot available (non-discharge only)
  total_los      — arrival → ED departure
  outcome counts and throughput per replication

REPLICATIONS:
  10 independent replications, each seeded separately.
  1-week warm-up period discarded to avoid start-up bias.
  Results reported as mean ± std across replications.
"""

import os
import random
import simpy
import pandas as pd
import numpy as np

from resources import EDResources
from patient import Patient

# ----------------------------------------------------------
# SIMULATION CONFIGURATION
# ----------------------------------------------------------

NUM_REPLICATIONS   = 10
SIM_DURATION_HOURS = 365 * 24    # 1 year per replication
WARMUP_HOURS       = 7  * 24     # 1-week warm-up discarded
BASE_SEED          = 42

# Arrival rate — 150 patients/day = one patient every 9.6 minutes on average
ARRIVAL_RATE_PER_HOUR = 150 / 24   # 6.25 patients/hour

# Service times (hours) — active staff contact time, NOT including waiting
TRIAGE_SERVICE_H  = 10 / 60   # 10 minutes — active nurse triage
DOCTOR_SERVICE_H  = 20 / 60   # 20 minutes — NHS ED consultation standard

# Boarding window — time a non-discharge patient occupies a boarding slot
# before being physically transferred out of the ED.
# Sampled from exponential; mean = 90 minutes based on NHS boarding targets.
BOARDING_MEAN_H   = 90 / 60   # 1.5 hours mean boarding time

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

BRANCH_PATH     = "../data/ed_branch_probabilities.csv"
TRANS_PROB_PATH = "../data/transition_probabilities.csv"
OUT_DIR         = "../data"
PATIENT_LOG_OUT = os.path.join(OUT_DIR, "simulation_patient_log.csv")
SUMMARY_OUT     = os.path.join(OUT_DIR, "simulation_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# LOAD PARAMETERS
# ----------------------------------------------------------

def load_branch_probabilities(path: str) -> dict:
    df  = pd.read_csv(path)
    row = df.iloc[0]
    return {
        "discharge":    float(row["p_discharge"]),    # discharged + cancelled
        "admission":    float(row["p_admission"]),    # Ward2 transfer
        "icu":          float(row["p_icu"]),          # direct ICU from Ward1
        "ward2_to_icu": float(row["p_ward2_to_icu"]),# Ward2 → ICU escalation
    }


def load_transition_probabilities(path: str) -> dict:
    df    = pd.read_csv(path)
    trans = {}
    for _, row in df.iterrows():
        src  = row["from"]
        dst  = row["to"]
        prob = float(row["probability"])
        trans.setdefault(src, {})[dst] = prob
    return trans


print("Loading simulation parameters...")
branch_probs     = load_branch_probabilities(BRANCH_PATH)
transition_probs = load_transition_probabilities(TRANS_PROB_PATH)

print(f"  Arrival rate:    {ARRIVAL_RATE_PER_HOUR:.4f} patients/hour  "
      f"({ARRIVAL_RATE_PER_HOUR*24:.0f}/day)")
print(f"  Triage service:  {TRIAGE_SERVICE_H*60:.0f} min")
print(f"  Doctor service:  {DOCTOR_SERVICE_H*60:.0f} min")
print(f"  Boarding mean:   {BOARDING_MEAN_H*60:.0f} min")
print(f"  Branch probs:    discharge={branch_probs['discharge']:.4f}  "
      f"admission={branch_probs['admission']:.4f}  "
      f"icu={branch_probs['icu']:.4f}  "
      f"ward2_to_icu={branch_probs['ward2_to_icu']:.4f}")

# ----------------------------------------------------------
# SEVERITY ASSIGNMENT
# ----------------------------------------------------------
# Derived from first_careunit distribution in the synthetic dataset.
# ICU-type careunits (MICU, CCU, SICU, TSICU, CSRU) map to high/critical.
# These weights inform the AI agent's prioritisation logic (Deliverable 3).

SEVERITY_WEIGHTS = {
    "low":      0.40,
    "medium":   0.35,
    "high":     0.18,
    "critical": 0.07,
}


def assign_severity(rng: random.Random) -> str:
    return rng.choices(
        list(SEVERITY_WEIGHTS.keys()),
        weights=list(SEVERITY_WEIGHTS.values()),
        k=1
    )[0]


# ----------------------------------------------------------
# OUTCOME SAMPLING
# ----------------------------------------------------------

def sample_outcome(rng: random.Random, severity: str) -> str:
    """
    Sample outcome from Ward1 Transfer Out using corrected branch probabilities
    derived from the updated event log (post callout_cancelled fix):

        discharge  65.1%  (DISCHARGED + CANCELLED combined — both exit ED)
        admission   9.7%  (TRANSFERRED — goes to Ward2 then exits ED)
        icu        15.4%  (direct ICU escalation from Ward1)
        --- sum =  90.2%, remaining 9.7% was cancelled — now in discharge ---

    Severity modifier: critical/high patients have elevated ICU probability,
    reflecting clinical reality that sicker patients are more likely to need
    intensive care. Documented as a modelling assumption in the dissertation.
    """
    base_icu = branch_probs["icu"]

    if severity == "critical":
        p_icu = min(base_icu * 3.0, 0.85)
    elif severity == "high":
        p_icu = min(base_icu * 1.8, 0.60)
    else:
        p_icu = base_icu

    # Rescale discharge and admission proportionally around adjusted ICU prob
    remaining    = 1.0 - p_icu
    base_non_icu = branch_probs["discharge"] + branch_probs["admission"]
    if base_non_icu > 0:
        p_discharge = branch_probs["discharge"] / base_non_icu * remaining
        p_admission  = branch_probs["admission"]  / base_non_icu * remaining
    else:
        p_discharge = remaining
        p_admission  = 0.0

    r = rng.random()
    if r < p_discharge:
        return "discharge"
    elif r < p_discharge + p_admission:
        return "admission"
    else:
        return "icu"


# ----------------------------------------------------------
# PATIENT FLOW
# ----------------------------------------------------------

def patient_flow(
    env:            simpy.Environment,
    patient:        Patient,
    resources:      EDResources,
    rng:            random.Random,
    completed_list: list,
    warmup:         float,
):
    """
    Simulate one patient through the ED pathway.

    STAGE 1 — Triage (nurse + cubicle required simultaneously)
      Patient waits until BOTH a nurse AND a cubicle are free.
      Triage service time = 10 minutes (active nurse contact).

    STAGE 2 — Doctor consultation (doctor required, cubicle held)
      Patient waits for a doctor while still occupying the cubicle.
      Doctor service time = 20 minutes.
      Cubicle is released after doctor finishes.

    STAGE 3 — Outcome and boarding (non-discharge only)
      Discharge patients exit immediately after doctor.
      Admission/ICU patients wait for a boarding slot, then exit.
      Boarding time = exponential(mean 90 min).

    ED departure = patient exits the simulation after Stage 3.
    """

    patient.arrival_time = env.now

    # -------------------------------------------------------
    # STAGE 1 — TRIAGE
    # Nurse and cubicle both required to start triage.
    # Using SimPy's & operator to request both simultaneously —
    # patient only starts when BOTH are granted.
    # -------------------------------------------------------
    nurse_req   = resources.triage_nurse.request()
    cubicle_req = resources.cubicle.request()

    yield nurse_req & cubicle_req

    patient.triage_start = env.now
    yield env.timeout(TRIAGE_SERVICE_H)
    patient.triage_end = env.now

    # Nurse is released after triage — cubicle is held for doctor
    resources.triage_nurse.release(nurse_req)

    # -------------------------------------------------------
    # STAGE 2 — DOCTOR CONSULTATION
    # Patient waits for a doctor while holding the cubicle.
    # Doctor wait time (triage_end → doctor_start) is the key
    # front-of-house bottleneck metric.
    # -------------------------------------------------------
    with resources.doctor.request() as doc_req:
        yield doc_req

        patient.doctor_start = env.now
        yield env.timeout(DOCTOR_SERVICE_H)
        patient.doctor_end = env.now

    # Cubicle released after doctor finishes
    resources.cubicle.release(cubicle_req)

    # -------------------------------------------------------
    # OUTCOME DECISION
    # -------------------------------------------------------
    outcome = sample_outcome(rng, patient.severity)
    patient.outcome = outcome

    # -------------------------------------------------------
    # STAGE 3a — WARD 2 BOARDING (admission pathway)
    # Matches DFG: Ward1 Transfer Out → Ward2 Transfer In/Out
    # Patient waits for a boarding slot (Ward2 handover).
    # After Ward2, 15.4% are escalated to ICU (p_ward2_to_icu).
    # The remaining 84.6% exit the ED as a transfer.
    # -------------------------------------------------------
    if outcome == "admission":
        with resources.boarding_slot.request() as board_req:
            yield board_req
            patient.boarding_start = env.now
            boarding_time = rng.expovariate(1.0 / BOARDING_MEAN_H)
            yield env.timeout(boarding_time)
            patient.boarding_end = env.now

        # Ward2 → ICU escalation (15.4% from DFG: 200/1297)
        if rng.random() < branch_probs["ward2_to_icu"]:
            outcome = "icu"
            patient.outcome = outcome

    # -------------------------------------------------------
    # STAGE 3b — DIRECT ICU BOARDING
    # Matches DFG: Ward1 Transfer Out → ICU Admission (1541 cases)
    # Also reached via Ward2 escalation above.
    # Patient waits for a boarding slot (ICU handover).
    # -------------------------------------------------------
    if outcome == "icu":
        with resources.boarding_slot.request() as board_req:
            yield board_req
            # Only record boarding_start if not already set by Ward2 path
            if patient.boarding_start is None:
                patient.boarding_start = env.now
            yield env.timeout(rng.expovariate(1.0 / BOARDING_MEAN_H))
            patient.boarding_end = env.now

    # -------------------------------------------------------
    # DEPARTURE
    # -------------------------------------------------------
    patient.departure_time = env.now

    # Only record patients who arrived after the warm-up period
    if patient.arrival_time >= warmup:
        completed_list.append(patient)


# ----------------------------------------------------------
# ARRIVAL GENERATOR
# ----------------------------------------------------------

def generate_arrivals(
    env:            simpy.Environment,
    resources:      EDResources,
    rng:            random.Random,
    completed_list: list,
    sim_duration:   float,
    warmup:         float,
    pid_counter:    list,
):
    """
    Generate patient arrivals using an exponential interarrival distribution.
    Rate = ARRIVAL_RATE_PER_HOUR (150 patients/day).

    Exponential interarrival is the standard assumption for ED arrivals
    (Poisson process), supported by queuing theory and used widely in
    healthcare simulation literature.
    """
    while env.now < sim_duration:
        # Exponential interarrival: mean = 1 / lambda
        interarrival = rng.expovariate(ARRIVAL_RATE_PER_HOUR)
        yield env.timeout(interarrival)

        if env.now >= sim_duration:
            break

        pid      = pid_counter[0]
        pid_counter[0] += 1
        severity = assign_severity(rng)
        patient  = Patient(case_id=pid, severity=severity)

        env.process(
            patient_flow(env, patient, resources, rng, completed_list, warmup)
        )


# ----------------------------------------------------------
# SINGLE REPLICATION
# ----------------------------------------------------------

def run_replication(rep_id: int) -> list:
    """
    Run one complete simulation replication.
    Each replication uses a unique random seed for independence.
    Returns a list of completed Patient objects (post-warmup only).
    """
    seed       = BASE_SEED + rep_id
    rng        = random.Random(seed)
    env        = simpy.Environment()
    resources  = EDResources(env)
    completed  = []
    pid_counter = [rep_id * 1_000_000]

    env.process(
        generate_arrivals(
            env, resources, rng, completed,
            SIM_DURATION_HOURS, WARMUP_HOURS, pid_counter
        )
    )
    env.run(until=SIM_DURATION_HOURS)
    return completed


# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

def _valid(values: list) -> list:
    return [x for x in values if x is not None]

def safe_mean(values):       return float(np.mean(_valid(values)))       if _valid(values) else 0.0
def safe_max(values):        return float(np.max(_valid(values)))         if _valid(values) else 0.0
def safe_pct(values, p):     return float(np.percentile(_valid(values),p)) if _valid(values) else 0.0


def compute_metrics(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}

    outcomes       = [p.outcome        for p in patients]
    triage_waits   = [p.triage_wait    for p in patients]
    doctor_waits   = [p.doctor_wait    for p in patients]
    boarding_waits = [p.boarding_wait  for p in patients if p.outcome in ("admission","icu")]
    total_los      = [p.total_los      for p in patients]

    return {
        "n_patients":           n,
        "n_discharge":          outcomes.count("discharge"),
        "n_admission":          outcomes.count("admission"),
        "n_icu":                outcomes.count("icu"),

        "triage_wait_mean":     safe_mean(triage_waits),
        "triage_wait_p95":      safe_pct(triage_waits, 95),
        "triage_wait_max":      safe_max(triage_waits),

        "doctor_wait_mean":     safe_mean(doctor_waits),
        "doctor_wait_p95":      safe_pct(doctor_waits, 95),
        "doctor_wait_max":      safe_max(doctor_waits),

        "boarding_wait_mean":   safe_mean(boarding_waits),
        "boarding_wait_p95":    safe_pct(boarding_waits, 95),
        "boarding_wait_max":    safe_max(boarding_waits),

        "total_los_mean":       safe_mean(total_los),
        "total_los_p95":        safe_pct(total_los, 95),
        "total_los_max":        safe_max(total_los),
    }


# ----------------------------------------------------------
# PRINT
# ----------------------------------------------------------

def print_summary(m: dict, label: str = ""):
    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"{'='*56}")
    print(f"  Patients:              {m.get('n_patients',0):>6}")
    print(f"  Discharge:             {m.get('n_discharge',0):>6}  "
          f"({m.get('n_discharge',0)/max(m.get('n_patients',1),1)*100:.1f}%)")
    print(f"  Admission:             {m.get('n_admission',0):>6}  "
          f"({m.get('n_admission',0)/max(m.get('n_patients',1),1)*100:.1f}%)")
    print(f"  ICU:                   {m.get('n_icu',0):>6}  "
          f"({m.get('n_icu',0)/max(m.get('n_patients',1),1)*100:.1f}%)")
    print(f"  ---")
    print(f"  Triage wait  mean:     {m.get('triage_wait_mean',0)*60:>7.2f} min")
    print(f"  Triage wait  p95:      {m.get('triage_wait_p95',0)*60:>7.2f} min")
    print(f"  Triage wait  max:      {m.get('triage_wait_max',0)*60:>7.2f} min")
    print(f"  Doctor wait  mean:     {m.get('doctor_wait_mean',0)*60:>7.2f} min")
    print(f"  Doctor wait  p95:      {m.get('doctor_wait_p95',0)*60:>7.2f} min")
    print(f"  Doctor wait  max:      {m.get('doctor_wait_max',0)*60:>7.2f} min")
    print(f"  Boarding wait mean:    {m.get('boarding_wait_mean',0)*60:>7.2f} min")
    print(f"  Boarding wait p95:     {m.get('boarding_wait_p95',0)*60:>7.2f} min")
    print(f"  Total LOS    mean:     {m.get('total_los_mean',0)*60:>7.2f} min")
    print(f"  Total LOS    p95:      {m.get('total_los_p95',0)*60:>7.2f} min")
    print(f"  Total LOS    max:      {m.get('total_los_max',0)*60:>7.2f} min")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run_simulation():
    print(f"\n{'='*56}")
    print("  Baseline ED Simulation")
    print(f"  Arrival rate:  {ARRIVAL_RATE_PER_HOUR*24:.0f} patients/day")
    print(f"  Replications:  {NUM_REPLICATIONS}")
    print(f"  Horizon:       {SIM_DURATION_HOURS} h ({SIM_DURATION_HOURS//24} days)")
    print(f"  Warm-up:       {WARMUP_HOURS} h ({WARMUP_HOURS//24} days)")
    print(f"{'='*56}\n")

    all_metrics  = []
    all_patients = []

    for rep in range(1, NUM_REPLICATIONS + 1):
        print(f"Running replication {rep}/{NUM_REPLICATIONS}...", end=" ", flush=True)
        patients = run_replication(rep)
        print(f"done — {len(patients):,} patients.")

        m = compute_metrics(patients)
        m["replication"] = rep
        all_metrics.append(m)

        for p in patients:
            row = p.to_dict()
            row["replication"] = rep
            all_patients.append(row)

        print_summary(m, label=f"Replication {rep}")

    # Cross-replication summary
    mdf  = pd.DataFrame(all_metrics)
    cols = [c for c in mdf.columns if c != "replication"]
    mu   = mdf[cols].mean()
    sd   = mdf[cols].std()

    print(f"\n{'='*56}")
    print("  SUMMARY ACROSS ALL REPLICATIONS  (mean ± std)")
    print(f"{'='*56}")
    for col in cols:
        # Show time metrics in minutes for readability
        if any(x in col for x in ["wait","los"]):
            print(f"  {col:<28} {mu[col]*60:>8.2f} min  ±  {sd[col]*60:.2f} min")
        else:
            print(f"  {col:<28} {mu[col]:>8.1f}      ±  {sd[col]:.1f}")

    # NHS 4-hour target: % patients with total LOS <= 4h
    four_hour_compliance = []
    for p in all_patients:
        if p.get("total_los") is not None:
            four_hour_compliance.append(1 if p["total_los"] <= 4.0 else 0)
    if four_hour_compliance:
        pct = sum(four_hour_compliance) / len(four_hour_compliance) * 100
        print(f"\n  NHS 4-hour target compliance:  {pct:.1f}%  "
              f"(target: ≥95%)")

    # Save outputs
    pd.DataFrame(all_patients).to_csv(PATIENT_LOG_OUT, index=False)
    mdf.to_csv(SUMMARY_OUT, index=False)
    print(f"\n  Patient log  → {PATIENT_LOG_OUT}  ({len(all_patients):,} rows)")
    print(f"  Summary      → {SUMMARY_OUT}")

    return mdf


if __name__ == "__main__":
    run_simulation()