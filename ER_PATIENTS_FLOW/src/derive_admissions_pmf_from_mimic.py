# Probablity Distribution of admisisons per patient using the MIMIC-III dataset (Patient.csv and Admission.csv)

import os
import pandas as pd

# -------------------------------------------------
# Config
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIMIC_DIR = os.path.join(PROJECT_ROOT, "..", "Reference_mimic_iii")

OUT_DIR = os.path.join("Synthetic_dataset", "data")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DIST = os.path.join(OUT_DIR, "mimic_admissions_per_patient_distribution.csv")
OUT_PROB = os.path.join(OUT_DIR, "mimic_admissions_count_probabilities.csv")


def find_file(root_dir: str, filename: str) -> str:
    """
    Search for filename anywhere under root_dir and return the first match.
    This avoids hardcoding paths like ADMISSIONS.csv/ADMISSIONS.csv.
    """
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find {filename} under: {root_dir}")


def main():
    # Auto-detect the actual file locations
    admissions_path = find_file(MIMIC_DIR, "ADMISSIONS.csv")
    patients_path = find_file(MIMIC_DIR, "PATIENTS.csv")

    print("Using ADMISSIONS file:", admissions_path)
    print("Using PATIENTS file:", patients_path)

    admissions = pd.read_csv(admissions_path, usecols=["SUBJECT_ID", "HADM_ID"])
    patients = pd.read_csv(patients_path, usecols=["SUBJECT_ID"])

    # 1) admissions per patient (unique hadm_id per subject_id)
    adm_counts = admissions.groupby("SUBJECT_ID")["HADM_ID"].nunique()

    # 2) include "0 admissions" patients if they exist
    all_patients = patients["SUBJECT_ID"].unique()
    counts_all = pd.Series(0, index=all_patients, dtype=int)
    counts_all.loc[adm_counts.index] = adm_counts.astype(int)

    # 3) distribution: number of admissions -> number of patients
    dist = counts_all.value_counts().sort_index()
    dist_df = dist.reset_index()
    dist_df.columns = ["num_admissions", "num_patients"]
    dist_df.to_csv(OUT_DIST, index=False)

    # 4) probabilities (PMF)
    prob = dist / dist.sum()
    prob_df = prob.reset_index()
    prob_df.columns = ["num_admissions", "probability"]
    prob_df.to_csv(OUT_PROB, index=False)

    # Print summary
    print("\nSaved:")
    print(" -", OUT_DIST)
    print(" -", OUT_PROB)

    print("\nTop probabilities:")
    print(prob_df.head(10).to_string(index=False))

    # Collapsed tail summary (0..4, 5+)
    p0 = float(prob.get(0, 0.0))
    p1 = float(prob.get(1, 0.0))
    p2 = float(prob.get(2, 0.0))
    p3 = float(prob.get(3, 0.0))
    p4 = float(prob.get(4, 0.0))
    p5plus = float(prob[prob.index >= 5].sum())

    print("\nCollapsed PMF (0..4, 5+):")
    print(
        f"P(0)={p0:.6f}, P(1)={p1:.6f}, P(2)={p2:.6f}, "
        f"P(3)={p3:.6f}, P(4)={p4:.6f}, P(5+)={p5plus:.6f}"
    )


if __name__ == "__main__":
    main()
