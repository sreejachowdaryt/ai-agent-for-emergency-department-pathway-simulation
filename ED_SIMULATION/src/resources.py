# resources.py
"""
ED Resources — calibrated for a mid-sized UK Emergency Department
receiving 150 patients/day.

Resource capacities are derived using Little's Law (L = λW) targeting
a utilisation (ρ) of 0.75–0.85 per resource pool, which produces
realistic queue formation without permanent overload.

    λ  = 150 patients/day = 6.25 patients/hour
    ρ  = λ / (c × μ)   where c = capacity, μ = service rate

Resource         Capacity  Service time  ρ      Rationale
-----------      --------  ------------  -----  --------------------------
triage_nurse     2         10 min        0.87   Active nurse assessment time
doctor           3         20 min        0.83   NHS ED consultation standard
cubicle          8         30 min        0.78   Triage + treatment combined
boarding_slot    3         90 min        0.78   ED boarding before ward/ICU

NOTE on boarding_slot:
  In the ED-only model, once a patient is assigned a boarding slot they
  are recorded as "pending transfer" and depart the ED after a short
  boarding window (empirically sampled). The slot represents the ED's
  physical capacity to hold admitted patients while awaiting a ward/ICU
  bed confirmation. This is the primary bottleneck for non-discharge
  patients and is where the AI agent intervention has the most impact.

NOTE on triage service time vs triage_duration in data:
  The triage_duration column in ed_cases.csv (mean ~32 min) represents
  arrival -> initial_assessment, which INCLUDES queue waiting time.
  The 10-minute service time here is the active nurse contact time only.
  Waiting time then emerges naturally from queue dynamics.
"""

import simpy


class EDResources:
    def __init__(self, env: simpy.Environment):
        self.triage_nurse  = simpy.Resource(env, capacity=2)
        self.doctor        = simpy.Resource(env, capacity=3)
        self.cubicle       = simpy.Resource(env, capacity=8)
        self.boarding_slot = simpy.Resource(env, capacity=3)

    def utilisation_snapshot(self) -> dict:
        return {
            "triage_nurse_queue":    len(self.triage_nurse.queue),
            "triage_nurse_in_use":   self.triage_nurse.count,
            "doctor_queue":          len(self.doctor.queue),
            "doctor_in_use":         self.doctor.count,
            "cubicle_queue":         len(self.cubicle.queue),
            "cubicle_in_use":        self.cubicle.count,
            "boarding_slot_queue":   len(self.boarding_slot.queue),
            "boarding_slot_in_use":  self.boarding_slot.count,
        }