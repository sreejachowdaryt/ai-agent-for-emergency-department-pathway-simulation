# Computed Time gaps from MIMIC-III dataset for multiple admissions (Inter-admission Time Gaps)

import os
import pandas as pd
import numpy as np

def find_file(root_dir: str, filename: str) -> str:
    """
    Search for filename anywhere under root_dir and return the first match.
    This avoids hardcoding paths like ADMISSIONS.csv/ADMISSIONS.csv.
    """
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find {filename} under: {root_dir}")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIMIC_DIR = os.path.join(PROJECT_ROOT, "..", "Reference_mimic_iii")
admissions_path = find_file(MIMIC_DIR, "ADMISSIONS.csv")


def main():
    admissions = pd.read_csv(
        admissions_path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]
    )

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])

    admissions = admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])

    gaps_days = []

    for pid, group in admissions.groupby("SUBJECT_ID"):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            gap = group.loc[i, "ADMITTIME"] - group.loc[i - 1, "DISCHTIME"]
            gap_days = gap.total_seconds() / (3600 * 24)

            if gap_days > 0:
                gaps_days.append(gap_days)

    gaps_days = np.array(gaps_days)

    out_path = os.path.join("Synthetic_dataset", "data", "mimic_interadmission_gaps_days.csv")
    pd.DataFrame({"gap_days": gaps_days}).to_csv(out_path, index=False)
    print("Saved gaps to:", out_path)

    print("\nInter-admission gaps (days)")
    print("----------------------------------")
    print(f"Mean gap: {gaps_days.mean():.2f} days")
    print(f"Std gap: {gaps_days.std(ddof=1):.2f} days")
    print(f"Median gap: {np.median(gaps_days):.2f} days")
    print(f"95th percentile: {np.percentile(gaps_days,95):.2f} days")
    print(f"Max gap: {gaps_days.max():.2f} days")

if __name__ == "__main__":
    main()
