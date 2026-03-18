# create_event_log.py
"""
Convert the synthetic ED dataset into an event log for process mining.

Each row in ed_cases.csv becomes a TRACE (one patient's journey).
Each activity within that trace is a timestamped EVENT.

FIX applied vs original version:
  The original script appended a "Discharge" event for every patient
  using the discharge_time column, which is populated for ALL patients
  regardless of outcome. This caused the process tree to show Discharge
  as the universal terminal activity, misrepresenting ICU and transferred
  patients.

  The fix: the terminal activity is now determined by callout_outcome:

    callout_outcome    Terminal activity      Timestamp used
    ---------------    -----------------      --------------
    DISCHARGED         Discharge              discharge_time
    CANCELLED          Callout Cancelled      discharge_time
    TRANSFERRED        Ward2 Transfer Out     second_transfer_out
                       (already in pathway — no extra event added)
    ICU                ICU Discharge          icu_discharge_time
                       (already in pathway — no extra event added)

  For TRANSFERRED and ICU patients, the correct terminal activity is
  already recorded as part of their pathway activities (Ward2 Transfer Out
  and ICU Discharge respectively). Adding a "Discharge" event on top of
  those was the source of the misrepresentation.

Output columns:
    case_id    — patient case identifier (trace ID for process mining)
    activity   — activity name
    timestamp  — datetime of the activity
    outcome    — callout_outcome (retained as case attribute for filtering)
"""

import pandas as pd


def create_event_log(input_path: str, output_path: str):

    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} cases loaded.")

    events = []

    # ------------------------------------------------------------------
    # Activity columns that are ALWAYS added if the timestamp exists.
    # These form the core pathway backbone for every patient.
    # "Discharge" is intentionally excluded here — it is handled
    # conditionally below based on callout_outcome.
    # ------------------------------------------------------------------
    pathway_activities = [
        ("Arrival",              "arrival_time"),
        ("Initial Assessment",   "initial_assessment_time"),
        ("Ward1 Transfer In",    "first_transfer_in"),
        ("Ward1 Transfer Out",   "first_transfer_out"),
        ("Ward2 Transfer In",    "second_transfer_in"),
        ("Ward2 Transfer Out",   "second_transfer_out"),
        ("ICU Admission",        "icu_admission_time"),
        ("ICU Discharge",        "icu_discharge_time"),
    ]

    for _, row in df.iterrows():

        case_id = row["case_id"]
        outcome = str(row.get("callout_outcome", "")).strip().upper()

        # --- Core pathway activities ---
        for activity, col in pathway_activities:
            if col in row and pd.notna(row[col]):
                events.append({
                    "case_id":   case_id,
                    "activity":  activity,
                    "timestamp": row[col],
                    "outcome":   outcome,
                })

        # --- Terminal activity (outcome-dependent) ---
        if outcome in ("DISCHARGED",):
            # Standard home discharge
            if pd.notna(row.get("discharge_time")):
                events.append({
                    "case_id":   case_id,
                    "activity":  "Discharge",
                    "timestamp": row["discharge_time"],
                    "outcome":   outcome,
                })

        elif outcome == "CANCELLED":
            # Callout was cancelled — patient disposition redirected
            # Uses discharge_time as the timestamp of the cancellation event
            if pd.notna(row.get("discharge_time")):
                events.append({
                    "case_id":   case_id,
                    "activity":  "Callout Cancelled",
                    "timestamp": row["discharge_time"],
                    "outcome":   outcome,
                })

        elif outcome == "TRANSFERRED":
            # Patient transferred out via Ward2 Transfer Out.
            # That activity is already added above — no extra terminal event needed.
            # The trace correctly ends at Ward2 Transfer Out.
            pass

        elif outcome == "ICU":
            # Patient escalated to ICU — trace ends at ICU Discharge.
            # That activity is already added above — no extra terminal event needed.
            pass

        else:
            # Unknown outcome — fall back to discharge_time if available
            if pd.notna(row.get("discharge_time")):
                events.append({
                    "case_id":   case_id,
                    "activity":  "Discharge",
                    "timestamp": row["discharge_time"],
                    "outcome":   outcome,
                })

    # ------------------------------------------------------------------
    # Build, sort and save the event log
    # ------------------------------------------------------------------
    event_log = pd.DataFrame(events)
    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"])
    event_log = event_log.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    event_log.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Summary for verification
    # ------------------------------------------------------------------
    print(f"\nEvent log created: {output_path}")
    print(f"  Total events:  {len(event_log):,}")
    print(f"  Total traces:  {event_log['case_id'].nunique():,}")
    print(f"\n  Terminal activity distribution:")
    terminal_events = (
        event_log
        .sort_values("timestamp")
        .groupby("case_id")
        .last()["activity"]
        .value_counts()
    )
    for activity, count in terminal_events.items():
        pct = count / event_log["case_id"].nunique() * 100
        print(f"    {activity:<25} {count:>6} ({pct:.1f}%)")

    print(f"\n  Activity event counts:")
    for activity, count in event_log["activity"].value_counts().items():
        print(f"    {activity:<25} {count:>6}")


if __name__ == "__main__":

    create_event_log(
        "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv",
        "../data/event_log.csv"
    )