# src/resources.py
"""
ED Resources — calibrated for a mid-sized UK Emergency Department
with an effective workload of approximately 150–160 patients/day
after scaling the synthetic dataset to a 365-day simulation horizon.

Resource capacities are calibrated using empirical arrival rates
derived from the dataset, guided by Little's Law (L = λW),
targeting utilisation (ρ) of approximately 0.80–0.90 per resource pool.

Assumptions:
    Effective arrival rate: λ ≈ 150 patients/day ≈ 6.25 patients/hour
    Utilisation: ρ = λ / (c × μ) = λW / c

Resource Configurations:

Resource         Capacity  Service time  ρ       Rationale
-----------      --------  ------------  ------  --------------------------------
assessment_bay   5         32.5 min      0.85    Combined triage+doctor proxy
boarding_slot    7         147 min       0.85    Empirically derived from dataset

Modelling Notes:

NOTE on resource redesign:
  The hybrid dataset uses a single 'initial_assessment_time - arrival_time'
  proxy (mean 32.5 min) representing the combined triage and doctor stage.
  Separate triage nurse, doctor, and cubicle resources are therefore replaced
  by a single assessment_bay resource.

  Boarding mean = 147 min (from ed_departure_time - boarding_start_time
  in the MIMIC-IV-ED derived dataset).
  Non-discharge rate ≈ 39% of 150/day = 58.5 patients/day = 2.44/hour.
  With 7 slots: ρ = 2.44 × (147/60) / 7 ≈ 0.85, which lies within the
  target utilisation range.
  With 3 or 4 slots the system would be overloaded (ρ > 1.0).

NOTE on boarding_slot:
  Represents the ED exit-block period — the patient is medically ready
  to leave the ED but cannot do so until an inpatient bed is confirmed.
  This is the main bottleneck for non-discharge patients and the stage
  where AI-based prioritisation has the greatest effect.
"""

import simpy


class EDResources:
    def __init__(self, env: simpy.Environment):
        self.assessment_bay = simpy.Resource(env, capacity=5)
        self.boarding_slot  = simpy.Resource(env, capacity=7)

    def utilisation_snapshot(self) -> dict:
        return {
            "assessment_bay_queue":   len(self.assessment_bay.queue),
            "assessment_bay_in_use":  self.assessment_bay.count,
            "boarding_slot_queue":    len(self.boarding_slot.queue),
            "boarding_slot_in_use":   self.boarding_slot.count,
        }