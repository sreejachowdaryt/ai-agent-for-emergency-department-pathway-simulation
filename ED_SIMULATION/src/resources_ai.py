# src/resources_ai.py
"""
ED resource configuration for the AI-enhanced simulation model.

This module extends the baseline resource configuration by replacing
the boarding resource with a priority-based queue.

Key difference from baseline:
- boarding_slot is implemented as a SimPy PriorityResource instead of a standard Resource

Purpose:
- Enables the AI agent to prioritise higher-severity patients when
  allocating boarding slots

All other aspects remain unchanged:
- assessment_bay capacity = 5
- boarding_slot capacity  = 7

This ensures that any observed performance differences between the
baseline and AI simulations are due to prioritisation logic rather
than changes in resource capacity.
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