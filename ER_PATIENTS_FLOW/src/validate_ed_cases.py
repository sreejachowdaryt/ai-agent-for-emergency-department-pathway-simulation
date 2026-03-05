# Logical Checks

import os
import pandas as pd

INPUT_PATH = os.path.join("Synthetic_dataset", "data", "ed_cases.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("ed_cases.csv is empty. Generate data first.")
        return

    # Convert datetime columns
    dt_cols = [
        "arrival_time", "initial_assessment_time", "first_transfer_in", "first_transfer_out",
        "second_transfer_in", "second_transfer_out", "icu_admission_time", "icu_discharge_time",
        "discharge_time", "callout_time", "callout_ack_time"
    ]

    for c in dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    errors = []

    # Rule 1: arrival < assessment < discharge
    bad = df[~(df["arrival_time"] < df["initial_assessment_time"])]
    for idx in bad.index:
        errors.append((idx, "arrival_time must be before initial_assessment_time"))

    bad = df[~(df["initial_assessment_time"] < df["discharge_time"])]
    for idx in bad.index:
        errors.append((idx, "initial_assessment_time must be before discharge_time"))

    # Rule 2: first transfer in < out
    bad = df[~(df["first_transfer_in"] < df["first_transfer_out"])]
    for idx in bad.index:
        errors.append((idx, "first_transfer_in must be before first_transfer_out"))

    # Rule 3: second transfer (if present) in < out
    second_present = df["second_transfer_in"].notna()
    bad = df[second_present & ~(df["second_transfer_in"] < df["second_transfer_out"])]
    for idx in bad.index:
        errors.append((idx, "second_transfer_in must be before second_transfer_out when present"))

    # Rule 4: ICU (if present) admission < discharge
    icu_present = df["icu_admission_time"].notna()
    bad = df[icu_present & ~(df["icu_admission_time"] < df["icu_discharge_time"])]
    for idx in bad.index:
        errors.append((idx, "icu_admission_time must be before icu_discharge_time when present"))

    # Rule 5: no overlapping admissions per patient
    df_sorted = df.sort_values(["patient_id", "arrival_time"]).reset_index()
    for pid, g in df_sorted.groupby("patient_id"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            prev_dis = g.loc[i - 1, "discharge_time"]
            cur_arr = g.loc[i, "arrival_time"]
            if pd.notna(prev_dis) and pd.notna(cur_arr) and not (cur_arr > prev_dis):
                errors.append((int(g.loc[i, "index"]), f"overlap with previous admission for patient {pid}"))

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issues")
        for e in errors[:25]:
            print(" -", e)
    else:
        print("VALIDATION PASSED ✅ No logical errors found.")

if __name__ == "__main__":
    main()
