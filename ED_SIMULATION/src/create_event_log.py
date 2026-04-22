# src/create_event_log.py
"""
Convert the synthetic ED dataset into an ordered event log for process mining.

Each row in ed_cases.csv becomes one TRACE (one patient case).
Each activity in that trace becomes a timestamped EVENT.

This version fixes equal-timestamp ordering issues by assigning
an explicit event_order so that process mining does not infer
spurious loops/concurrency.

Pathway represented:

DISCHARGED:
    Arrival -> Initial Assessment -> Discharge

ADMITTED:
    Arrival -> Initial Assessment -> Boarding Start -> ED Departure
    -> First Careunit In -> First Careunit Out -> Hospital Discharge

TRANSFERRED:
    Arrival -> Initial Assessment -> Boarding Start -> ED Departure
    -> First Careunit In -> First Careunit Out
    -> Second Careunit In -> Second Careunit Out -> Hospital Discharge
"""

import pandas as pd

df = pd.read_csv("../data/event_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = df.sort_values(["case_id", "timestamp", "event_order"]).reset_index(drop=True)
df["next_activity"] = df.groupby("case_id")["activity"].shift(-1)

bad = df[
    (df["activity"] == "Initial Assessment") &
    (df["next_activity"] == "Arrival")
]

print("Initial Assessment -> Arrival count:", len(bad))
print(bad[["case_id", "activity", "timestamp", "next_activity"]].head(10))

ACTIVITY_ORDER = {
    "Arrival": 1,
    "Initial Assessment": 2,
    "Boarding Start": 3,
    "Discharge": 4,
    "ED Departure": 5,
    "First Careunit In": 6,
    "First Careunit Out": 7,
    "Second Careunit In": 8,
    "Second Careunit Out": 9,
    "Hospital Discharge": 10,
}


def add_event(events, case_id, activity, timestamp, row, outcome):
    if pd.notna(timestamp):
        events.append({
            "case_id": case_id,
            "activity": activity,
            "timestamp": timestamp,
            "event_order": ACTIVITY_ORDER[activity],
            "pathway_outcome": outcome,
            "admission_type": row.get("admission_type"),
            "primary_diagnosis_code": row.get("primary_diagnosis_code"),
            "discharge_location": row.get("discharge_location"),
        })


def create_event_log(input_path: str, output_path: str):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} cases loaded.")

    dt_cols = [
        "arrival_time",
        "initial_assessment_time",
        "boarding_start_time",
        "ed_departure_time",
        "first_transfer_in",
        "first_transfer_out",
        "second_transfer_in",
        "second_transfer_out",
        "discharge_time",
    ]

    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    events = []

    for _, row in df.iterrows():
        case_id = row["case_id"]
        outcome = str(row.get("pathway_outcome", "")).strip().upper()

        add_event(events, case_id, "Arrival", row.get("arrival_time"), row, outcome)
        add_event(events, case_id, "Initial Assessment", row.get("initial_assessment_time"), row, outcome)

        if outcome == "DISCHARGED":
            add_event(events, case_id, "Discharge", row.get("ed_departure_time"), row, outcome)

        elif outcome == "ADMITTED":
            add_event(events, case_id, "Boarding Start", row.get("boarding_start_time"), row, outcome)
            add_event(events, case_id, "ED Departure", row.get("ed_departure_time"), row, outcome)
            add_event(events, case_id, "First Careunit In", row.get("first_transfer_in"), row, outcome)
            add_event(events, case_id, "First Careunit Out", row.get("first_transfer_out"), row, outcome)
            add_event(events, case_id, "Hospital Discharge", row.get("discharge_time"), row, outcome)

        elif outcome == "TRANSFERRED":
            add_event(events, case_id, "Boarding Start", row.get("boarding_start_time"), row, outcome)
            add_event(events, case_id, "ED Departure", row.get("ed_departure_time"), row, outcome)
            add_event(events, case_id, "First Careunit In", row.get("first_transfer_in"), row, outcome)
            add_event(events, case_id, "First Careunit Out", row.get("first_transfer_out"), row, outcome)
            add_event(events, case_id, "Second Careunit In", row.get("second_transfer_in"), row, outcome)
            add_event(events, case_id, "Second Careunit Out", row.get("second_transfer_out"), row, outcome)
            add_event(events, case_id, "Hospital Discharge", row.get("discharge_time"), row, outcome)

        else:
            add_event(events, case_id, "ED Departure", row.get("ed_departure_time"), row, outcome)

    event_log = pd.DataFrame(events)

    if event_log.empty:
        print("No events created.")
        return

    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"], errors="coerce")

    event_log = (
        event_log
        .sort_values(["case_id", "timestamp", "event_order"])
        .reset_index(drop=True)
    )

    event_log.to_csv(output_path, index=False)

    print(f"\nEvent log created: {output_path}")
    print(f"  Total events:  {len(event_log):,}")
    print(f"  Total traces:  {event_log['case_id'].nunique():,}")

    print("\n  Terminal activity distribution:")
    terminal = (
        event_log
        .sort_values(["case_id", "timestamp", "event_order"])
        .groupby("case_id")
        .last()["activity"]
        .value_counts()
    )

    total_traces = event_log["case_id"].nunique()
    for activity, count in terminal.items():
        pct = count / total_traces * 100
        print(f"    {activity:<30} {count:>6} ({pct:.1f}%)")

    print("\n  All activity event counts:")
    for activity, count in event_log["activity"].value_counts().items():
        print(f"    {activity:<30} {count:>6}")


if __name__ == "__main__":
    create_event_log(
        "../../ER_PATIENTS_FLOW/Synthetic_dataset/data/ed_cases.csv",
        "../data/event_log.csv"
    )