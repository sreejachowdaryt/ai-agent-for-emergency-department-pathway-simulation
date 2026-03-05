import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("Synthetic_dataset", "data")
PATIENTS_PATH = os.path.join(OUT_DIR, "patients.csv")
ED_CASES_PATH = os.path.join(OUT_DIR, "ed_cases.csv")

def main():
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

    # Distribution and probabilities (now includes 0)
    dist = all_counts["num_admissions"].value_counts().sort_index()
    dist_df = dist.reset_index()
    dist_df.columns = ["num_admissions", "num_patients"]
    dist_df.to_csv(os.path.join(OUT_DIR, "admissions_per_patient_distribution.csv"), index=False)

    prob = dist / dist.sum()
    prob_df = prob.reset_index()
    prob_df.columns = ["num_admissions", "probability"]
    prob_df.to_csv(os.path.join(OUT_DIR, "admissions_count_probabilities.csv"), index=False)

    # Plot
    plt.figure()
    dist.plot(kind="bar")
    plt.xlabel("Number of admissions per patient")
    plt.ylabel("Number of patients")
    plt.title("Admissions per patient distribution (synthetic, MIMIC-derived)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "admissions_per_patient_histogram.png"), dpi=200)
    plt.close()

    print("Saved histogram + distribution + probabilities (including 0 admissions).")

if __name__ == "__main__":
    main()
