# src/ai_agent.py
"""
AI Agent — Rule-Based Boarding Priority + Escalation Agent
==========================================================

This module implements a rule-based operational agent that intervenes
at the boarding stage for non-discharge patients.

AGENT TYPE:
  Rule-based operational agent

WHAT THE AGENT DOES:
  The agent operates at the boarding stage for non-discharge patients.

  It applies two rule-based decisions:
  1. severity-based boarding priority
  2. severity-based boarding escalation (faster boarding for
     critical/high-acuity cases)

PRIORITY RULES:
  Severity    Priority Number
  --------    ---------------
  critical    1
  high        2
  medium      3
  low         4

BOARDING ESCALATION RULES:
  Severity    Mean boarding time
  --------    ------------------
  critical    90 min
  high        110 min
  medium      147 min
  low         147 min

Interpretation:
- Higher-acuity patients receive faster access to boarding slots
  and reduced boarding durations
- Lower-acuity patients follow standard boarding behaviour

This agent provides a transparent and interpretable baseline for
evaluating AI-driven prioritisation strategies in the ED simulation.
"""

SEVERITY_TO_PRIORITY = {
    "critical": 1,
    "high":     2,
    "medium":   3,
    "low":      4,
}

DEFAULT_PRIORITY = 3

# Boarding mean time in HOURS
SEVERITY_TO_BOARDING_MEAN_H = {
    "critical": 90 / 60,
    "high":     110 / 60,
    "medium":   147 / 60,
    "low":      147 / 60,
}

DEFAULT_BOARDING_MEAN_H = 147 / 60


def get_boarding_priority(severity: str) -> int:
    """
    Return integer priority for SimPy PriorityResource.
    Lower number = higher priority.
    """
    return SEVERITY_TO_PRIORITY.get(severity, DEFAULT_PRIORITY)


def get_boarding_service_mean_hours(severity: str) -> float:
    """
    Return severity-dependent boarding mean time in hours.
    """
    return SEVERITY_TO_BOARDING_MEAN_H.get(severity, DEFAULT_BOARDING_MEAN_H)


def describe_agent() -> str:
    lines = [
        "AI Agent: Rule-Based Boarding Priority + Escalation Agent",
        "=" * 57,
        "Decision point: boarding slot request (post-assessment)",
        "Mechanism:      SimPy PriorityResource + severity-based boarding time",
        "",
        "Priority rules:",
    ]

    for severity, priority in sorted(SEVERITY_TO_PRIORITY.items(), key=lambda x: x[1]):
        lines.append(
            f"  {severity:<10} → priority {priority}  "
            f"({'highest' if priority == 1 else 'lowest' if priority == 4 else 'intermediate'})"
        )

    lines.append("")
    lines.append("Boarding escalation rules:")
    for severity in ["critical", "high", "medium", "low"]:
        mean_min = SEVERITY_TO_BOARDING_MEAN_H[severity] * 60
        lines.append(f"  {severity:<10} → {mean_min:.0f} min mean boarding time")

    lines.append("")
    lines.append("Tie-breaking: arrival order within same severity level")
    lines.append("Unaffected:   arrivals, assessment stage, outcome assignment, capacities")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_agent())
    print()
    for sev in ["critical", "high", "medium", "low"]:
        p = get_boarding_priority(sev)
        b = get_boarding_service_mean_hours(sev) * 60
        print(f"  {sev:<10} → priority {p}, boarding mean {b:.0f} min")