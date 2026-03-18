# patient.py
"""
Patient entity for the ED simulation (ED-only model).

Timestamps recorded here map to the ED pathway stages:
  Arrival → Triage → Doctor consultation → Boarding (if non-discharge) → Departure

All times are in simulation hours. Derived properties convert to
waiting times for use in performance metrics.
"""


class Patient:

    def __init__(self, case_id: int, severity: str = None):
        self.case_id  = case_id
        self.severity = severity       # low / medium / high / critical
        self.outcome  = None           # discharge / admission / icu

        # Stage timestamps
        self.arrival_time    = None
        self.triage_start    = None    # nurse + cubicle both available
        self.triage_end      = None
        self.doctor_start    = None
        self.doctor_end      = None
        self.boarding_start  = None    # boarding slot available (non-discharge)
        self.boarding_end    = None
        self.departure_time  = None

    # ------------------------------------------------------------------
    # Derived waiting time properties
    # ------------------------------------------------------------------

    @property
    def triage_wait(self):
        """Arrival → triage nurse available (hours)."""
        if self.triage_start is not None and self.arrival_time is not None:
            return self.triage_start - self.arrival_time
        return None

    @property
    def doctor_wait(self):
        """Triage end → doctor available (hours)."""
        if self.doctor_start is not None and self.triage_end is not None:
            return self.doctor_start - self.triage_end
        return None

    @property
    def boarding_wait(self):
        """Doctor end → boarding slot available (hours). Non-discharge only."""
        if self.boarding_start is not None and self.doctor_end is not None:
            return self.boarding_start - self.doctor_end
        return None

    @property
    def total_los(self):
        """Arrival → ED departure (hours)."""
        if self.departure_time is not None and self.arrival_time is not None:
            return self.departure_time - self.arrival_time
        return None

    def to_dict(self) -> dict:
        return {
            "case_id":        self.case_id,
            "severity":       self.severity,
            "outcome":        self.outcome,
            "arrival_time":   self.arrival_time,
            "triage_wait":    self.triage_wait,
            "doctor_wait":    self.doctor_wait,
            "boarding_wait":  self.boarding_wait,
            "total_los":      self.total_los,
        }

    def __repr__(self):
        return (f"Patient(case_id={self.case_id}, severity={self.severity}, "
                f"outcome={self.outcome})")