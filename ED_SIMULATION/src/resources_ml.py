# resources_ml.py
"""
ED Resources — ML Agent version (Deliverable 3, ML Extension)

Identical to resources_ai.py — PriorityResource for boarding slot.
Separated into its own file so the three simulations are fully independent.

boarding_slot capacity = 7 (matches baseline and rule-based agent)
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