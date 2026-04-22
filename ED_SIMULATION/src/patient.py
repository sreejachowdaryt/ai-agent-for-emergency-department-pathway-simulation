# src/patient.py
"""
Patient entity for the ED simulation model.

This class represents an individual patient (case) moving through the
ED-only simulation pathway:

Arrival → Assessment → Boarding (if admitted/transferred) → Departure

Model assumptions:
- The assessment stage combines triage and doctor consultation,
  consistent with the synthetic dataset proxy:
      initial_assessment_time − arrival_time
- Boarding represents the delay between clinical decision and ED exit
  for non-discharge patients

All timestamps are recorded in simulation time (hours).

The class also provides derived performance metrics:
- assessment_wait
- boarding_wait
- total length of stay (LOS)

These are used for evaluation and simulation output logging.
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