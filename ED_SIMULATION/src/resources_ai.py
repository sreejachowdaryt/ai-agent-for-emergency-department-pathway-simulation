# resources_ai.py
"""
ED Resources — AI Agent version (Deliverable 3)

Identical to resources.py with ONE change:
  boarding_slot is a SimPy PriorityResource instead of Resource.

This enables the AI agent to prioritise critical patients at boarding.
All capacities identical to baseline — any difference in results is
caused only by the priority ordering, not by additional resources.

boarding_slot capacity = 7 (matches baseline resources.py)
"""

import simpy


class EDResourcesAI:
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