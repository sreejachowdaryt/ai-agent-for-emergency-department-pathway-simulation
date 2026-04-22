# patient.py
"""
Patient entity for the ED simulation (ED-only model).

Timestamps map to the new hybrid ED pathway:
  Arrival → Assessment (triage+doctor combined) → Boarding (if non-discharge) → Departure

The old separate triage/doctor stage timestamps are replaced by a
single assessment stage, matching the dataset's initial_assessment_time proxy.

All times are in simulation hours.
"""


class Patient:

    def __init__(self, case_id: int, patient_id: int = None, severity: str = None):
        self.case_id = case_id
        self.patient_id = patient_id
        self.severity = severity   # low / medium / high / critical
        self.outcome  = None       # discharge / admission / transferred

        # Stage timestamps (simulation hours)
        self.arrival_time       = None
        self.assessment_start   = None  # assessment bay available
        self.assessment_end     = None  # assessment complete, outcome decided
        self.boarding_start     = None  # boarding slot available (non-discharge)
        self.boarding_end       = None  # boarding complete, patient exits ED
        self.departure_time     = None

    # ------------------------------------------------------------------
    # Derived waiting time properties
    # ------------------------------------------------------------------

    @property
    def assessment_wait(self):
        if self.assessment_start is not None and self.arrival_time is not None:
            return self.assessment_start - self.arrival_time
        return None

    @property
    def boarding_wait(self):
        if self.boarding_start is not None and self.assessment_end is not None:
            return self.boarding_start - self.assessment_end
        return None

    @property
    def total_los(self):
        if self.departure_time is not None and self.arrival_time is not None:
            return self.departure_time - self.arrival_time
        return None

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "severity": self.severity,
            "outcome": self.outcome,
            "arrival_time": self.arrival_time,
            "assessment_wait": self.assessment_wait,
            "boarding_wait": self.boarding_wait,
            "total_los": self.total_los,
        }

    def __repr__(self):
        return (
            f"Patient(patient_id={self.patient_id}, case_id={self.case_id}, "
            f"severity={self.severity}, outcome={self.outcome})"
        )