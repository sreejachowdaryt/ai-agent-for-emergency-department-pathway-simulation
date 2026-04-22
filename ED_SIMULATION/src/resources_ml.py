# src/resources_ml.py
"""
ED resource configuration for the hybrid ML + AI-enhanced simulation model.

This module uses the same resource structure as the rule-based AI model,
with a priority-based boarding queue.

Key difference from baseline:
- boarding_slot is implemented as a SimPy PriorityResource instead of a standard Resource

Purpose:
- Enables severity-based prioritisation of patients during boarding
- Works in combination with the ML-based POCT intervention applied at the assessment stage

All resource capacities remain unchanged:
- assessment_bay capacity = 5
- boarding_slot capacity  = 7

This ensures that any observed performance differences across:
  - baseline simulation
  - rule-based AI simulation
  - hybrid ML + AI simulation

are due to decision-making logic rather than changes in resource capacity.
"""

import simpy


class EDResourcesML:
    def __init__(self, env: simpy.Environment):
        self.assessment_bay = simpy.Resource(env, capacity=5)
        self.boarding_slot  = simpy.PriorityResource(env, capacity=7)

    def utilisation_snapshot(self) -> dict:
        return {
            "assessment_bay_queue":   len(self.assessment_bay.queue),
            "assessment_bay_in_use":  self.assessment_bay.count,
            "boarding_slot_queue":    len(self.boarding_slot.queue),
            "boarding_slot_in_use":   self.boarding_slot.count,
        }