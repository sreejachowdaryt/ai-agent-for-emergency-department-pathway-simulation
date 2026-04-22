# src/analyse_admissions.py
"""
Analyse admissions per patient in the synthetic ED dataset and compare with MIMIC-derived distributions.

This script:
- Computes the number of admissions per patient from ed_cases.csv
- Generates distribution and probability tables (including zero-admission patients)
- Produces a plot of synthetic admissions
- Compares synthetic probabilities with MIMIC-derived probabilities using bucketed groups (0–4, 5+)
- Outputs comparison plots and CSV files for validation

Used to validate that the synthetic dataset preserves realistic admission patterns.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ER_PATIENTS_FLOW/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../ER_PATIENTS_FLOW

OUT_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
PATIENTS_PATH = os.path.join(OUT_DIR, "patients.csv")
ED_CASES_PATH = os.path.join(OUT_DIR, "ed_cases.csv")

# Path to MIMIC-derived probabilities created by derive_admissions_pmf_from_mimic.py
MIMIC_PROB_PATH = os.path.join(OUT_DIR, "mimic_admissions_count_probabilities.csv")

# Bucket config to match generator (0..4 and 5+)
TAIL_BUCKET_START = 5


def bucket_0_4_5plus(s: pd.Series) -> pd.Series:
    """Collapse counts >=5 into a single '5+' bucket represented by integer 5."""
    s = s.copy()
    s[s >= TAIL_BUCKET_START] = TAIL_BUCKET_START
    return s


def main():
    print("OUT_DIR =", OUT_DIR)
    print("PATIENTS_PATH =", PATIENTS_PATH)
    print("ED_CASES_PATH =", ED_CASES_PATH)

    patients = pd.read_csv(PATIENTS_PATH)
    cases = pd.read_csv(ED_CASES_PATH)

    # Count admissions per patient from ed_cases
    if cases.empty:
        counts = pd.Series(dtype=int)
    else:
        counts = cases.groupby("patient_id")["case_id"].nunique()

    # Merge counts onto all patients, fill missing with 0
    all_counts = patients[["patient_id"]].copy()
    all_counts["num_admissions"] = all_counts["patient_id"].map(counts).fillna(0).astype(int)

    # ----------------------------------------------------------------
    # Synthetic: distribution + probability (unbucketed, includes 0)
    # ----------------------------------------------------------------
    dist = all_counts["num_admissions"].value_counts().sort_index()
    dist_df = dist.reset_index()
    dist_df.columns = ["num_admissions", "num_patients"]
    dist_df.to_csv(os.path.join(OUT_DIR, "admissions_per_patient_distribution.csv"), index=False)

    prob = dist / dist.sum()
    prob_df = prob.reset_index()
    prob_df.columns = ["num_admissions", "probability"]
    prob_df.to_csv(os.path.join(OUT_DIR, "admissions_count_probabilities.csv"), index=False)

    # ---------------------------
    # Plot 1: synthetic histogram
    # ---------------------------
    plt.figure()
    dist.plot(kind="bar")
    plt.xlabel("Number of admissions per patient")
    plt.ylabel("Number of patients")
    plt.title("Admissions per patient distribution (synthetic)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "admissions_per_patient_histogram.png"), dpi=200)
    plt.close()

    # =========================================================
    # Plot 2 — grouped bar chart (Synthetic vs MIMIC probabilities)
    # Buckets: 0,1,2,3,4,5+ (represented as 5)
    # =========================================================

    if not os.path.exists(MIMIC_PROB_PATH):
        print(f"WARNING: {MIMIC_PROB_PATH} not found. Skipping Synthetic vs MIMIC probability comparison plot.")
        print("Saved histogram + distribution + probabilities (including 0 admissions).")
        return

    # Synthetic probabilities in the same 0..4, 5+ bucket scheme
    syn_bucketed = bucket_0_4_5plus(all_counts["num_admissions"])
    syn_prob_bucket = syn_bucketed.value_counts(normalize=True).sort_index()

    # Ensure indices 0..5 exist
    for k in range(0, 6):
        if k not in syn_prob_bucket.index:
            syn_prob_bucket.loc[k] = 0.0
    syn_prob_bucket = syn_prob_bucket.sort_index()

    # Load MIMIC probabilities and bucket them the same way
    mimic_df = pd.read_csv(MIMIC_PROB_PATH)
    if not {"num_admissions", "probability"}.issubset(set(mimic_df.columns)):
        print(f"WARNING: {MIMIC_PROB_PATH} missing required columns. Skipping comparison plot.")
        print("Saved histogram + distribution + probabilities (including 0 admissions).")
        return

    mimic_df["num_admissions"] = mimic_df["num_admissions"].astype(int)
    mimic_df["probability"] = mimic_df["probability"].astype(float)

    mimic_bucket = mimic_df.copy()
    mimic_bucket.loc[mimic_bucket["num_admissions"] >= TAIL_BUCKET_START, "num_admissions"] = TAIL_BUCKET_START
    mimic_prob_bucket = mimic_bucket.groupby("num_admissions")["probability"].sum().sort_index()

    # Ensure indices 0..5 exist
    for k in range(0, 6):
        if k not in mimic_prob_bucket.index:
            mimic_prob_bucket.loc[k] = 0.0
    mimic_prob_bucket = mimic_prob_bucket.sort_index()

    # Save comparison table
    labels = ["0", "1", "2", "3", "4", "5+"]
    comp_df = pd.DataFrame({
        "bucket": labels,
        "synthetic_probability": syn_prob_bucket.values,
        "mimic_probability": mimic_prob_bucket.values
    })
    comp_df.to_csv(os.path.join(OUT_DIR, "admissions_probability_synthetic_vs_mimic.csv"), index=False)

    # Grouped bar chart
    x = np.arange(len(labels))
    width = 0.38

    plt.figure()
    plt.bar(x - width / 2, syn_prob_bucket.values, width, label="Synthetic")
    plt.bar(x + width / 2, mimic_prob_bucket.values, width, label="MIMIC")
    plt.xticks(x, labels)
    plt.xlabel("Number of admissions per patient (bucketed)")
    plt.ylabel("Probability")
    plt.title("Admissions per patient probability: Synthetic vs MIMIC (0..4, 5+)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "admissions_probability_synthetic_vs_mimic.png"), dpi=200)
    plt.close()

    print("Saved histogram + distribution + probabilities (including 0 admissions).")
    print("Saved Synthetic vs MIMIC probability comparison plot + CSV.")


if __name__ == "__main__":
    main()