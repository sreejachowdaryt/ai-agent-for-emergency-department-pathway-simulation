# src/validate_ed_cases.py
"""
Logical + statistical validation for Synthetic_dataset/data/ed_cases.csv

NEW HYBRID DATASET VALIDATION

Logical checks:
 - arrival_time <= initial_assessment_time <= ed_departure_time
 - if boarding_start_time present:
       initial_assessment_time <= boarding_start_time <= ed_departure_time
 - if first transfer present:
       first_transfer_in == ed_departure_time
       first_transfer_in < first_transfer_out
 - if second transfer present:
       first_transfer_out <= second_transfer_in
       second_transfer_in < second_transfer_out
 - discharge_time >= ed_departure_time
 - no overlapping admissions per patient
 - outcome-consistency checks:
       DISCHARGED rows should not have careunit fields
       ADMITTED rows should have first-careunit fields and no second-careunit fields
       TRANSFERRED rows should have both first and second-careunit fields

Statistical validation:
 - pathway outcome proportions
 - second transfer rate vs target p_second_transfer
 - MIMIC vs Synthetic overlays:
      * ED LOS overall
      * ED LOS admitted
      * ED LOS discharged/home
      * careunit stay duration
      * arrival hour-of-day distribution
      * arrival weekday distribution

Saves figures to: Synthetic_dataset/figures/
"""

import os
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "figures")

INPUT_PATH = os.path.join(DATA_DIR, "ed_cases.csv")
BRANCH_PROB_PATH = os.path.join(DATA_DIR, "mimic_branch_probabilities.csv")

# MIMIC-IV-ED derived
MIMIC_ED_LOS_OVERALL = os.path.join(DATA_DIR, "mimic_ed_los_hours.csv")
MIMIC_ED_LOS_ADMITTED = os.path.join(DATA_DIR, "mimic_ed_los_admitted_hours.csv")
MIMIC_ED_LOS_HOME = os.path.join(DATA_DIR, "mimic_ed_los_home_hours.csv")
MIMIC_ED_HOUR_PROBS = os.path.join(DATA_DIR, "mimic_ed_arrival_hour_probabilities.csv")
MIMIC_ED_WEEKDAY_PROBS = os.path.join(DATA_DIR, "mimic_ed_arrival_weekday_probabilities.csv")
MIMIC_ED_DISPOSITION_PROBS = os.path.join(DATA_DIR, "mimic_ed_disposition_probabilities.csv")

# MIMIC-III derived
MIMIC_CAREUNIT_STAY = os.path.join(DATA_DIR, "mimic_gap_careunit_stay_hours.csv")


def _load_series(path: str, col: str) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    s = pd.read_csv(path)[col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _clip_to_p99(mimic: pd.Series, syn: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    mimic = pd.to_numeric(mimic, errors="coerce").dropna()
    syn = pd.to_numeric(syn, errors="coerce").dropna()

    if mimic.empty or syn.empty:
        return mimic, syn, float("nan")

    p99 = float(mimic.quantile(0.99))
    mimic_c = mimic[mimic <= p99]
    syn_c = syn[syn <= p99]
    return mimic_c, syn_c, p99


def _load_probability_csv(path: str, key_col: str, prob_col: str = "probability") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if key_col not in df.columns or prob_col not in df.columns:
        raise ValueError(f"{path} must contain columns: {key_col}, {prob_col}")

    probs = dict(zip(df[key_col], df[prob_col]))
    total = sum(float(v) for v in probs.values())
    if total <= 0:
        raise ValueError(f"Probabilities in {path} sum to 0")

    return {k: float(v) / total for k, v in probs.items()}


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_DIR =", DATA_DIR)
    print("INPUT_PATH =", INPUT_PATH)
    print("FIG_DIR =", FIG_DIR)

    if not os.path.exists(INPUT_PATH):
        print(f"Cannot find {INPUT_PATH}. Generate data first.")
        return

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("ed_cases.csv is empty. Generate data first.")
        return

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

    for c in dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    errors = []

    # =====================================================
    # LOGICAL CHECKS
    # =====================================================

    # 1) arrival <= assessment <= ed_departure
    bad = df[
        df["arrival_time"].isna() |
        df["initial_assessment_time"].isna() |
        df["ed_departure_time"].isna() |
        (df["initial_assessment_time"] < df["arrival_time"]) |
        (df["initial_assessment_time"] > df["ed_departure_time"])
    ]
    for idx in bad.index:
        errors.append((idx, "arrival_time <= initial_assessment_time <= ed_departure_time violated"))

    # 2) boarding must lie between assessment and ed departure
    boarding_present = df["boarding_start_time"].notna()
    bad = df[
        boarding_present &
        (
            (df["boarding_start_time"] < df["initial_assessment_time"]) |
            (df["boarding_start_time"] > df["ed_departure_time"])
        )
    ]
    for idx in bad.index:
        errors.append((idx, "boarding_start_time must lie between initial_assessment_time and ed_departure_time"))

    # 3) ed_departure <= discharge
    bad = df[
        df["ed_departure_time"].notna() &
        df["discharge_time"].notna() &
        (df["ed_departure_time"] > df["discharge_time"])
    ]
    for idx in bad.index:
        errors.append((idx, "ed_departure_time must be before or equal to discharge_time"))

    # 4) first transfer consistency
    first_present = df["first_transfer_in"].notna() | df["first_transfer_out"].notna() | df["first_careunit"].notna()
    bad = df[
        first_present &
        (
            df["first_transfer_in"].isna() |
            df["first_transfer_out"].isna() |
            df["first_careunit"].isna() |
            ~(df["first_transfer_in"] == df["ed_departure_time"]) |
            ~(df["first_transfer_in"] < df["first_transfer_out"])
        )
    ]
    for idx in bad.index:
        errors.append((idx, "first transfer fields inconsistent with ed_departure_time or invalid ordering"))

    # 5) second transfer consistency
    second_present = df["second_transfer_in"].notna() | df["second_transfer_out"].notna() | df["second_careunit"].notna()
    bad = df[
        second_present &
        (
            df["second_transfer_in"].isna() |
            df["second_transfer_out"].isna() |
            df["second_careunit"].isna() |
            df["first_transfer_out"].isna() |
            ~(df["first_transfer_out"] <= df["second_transfer_in"]) |
            ~(df["second_transfer_in"] < df["second_transfer_out"])
        )
    ]
    for idx in bad.index:
        errors.append((idx, "second transfer fields inconsistent or invalid ordering"))

    # 6) no overlapping admissions per patient
    if "patient_id" in df.columns:
        df_sorted = df.sort_values(["patient_id", "arrival_time"]).reset_index()
        for pid, g in df_sorted.groupby("patient_id"):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                prev_dis = g.loc[i - 1, "discharge_time"]
                cur_arr = g.loc[i, "arrival_time"]
                if pd.notna(prev_dis) and pd.notna(cur_arr) and not (cur_arr > prev_dis):
                    errors.append((int(g.loc[i, "index"]), f"overlap with previous admission for patient {pid}"))

    # 7) outcome consistency
    discharged_mask = df["pathway_outcome"] == "DISCHARGED"
    admitted_mask = df["pathway_outcome"] == "ADMITTED"
    transferred_mask = df["pathway_outcome"] == "TRANSFERRED"

    discharged_bad = df[
        discharged_mask &
        df[
            [
                "boarding_start_time",
                "first_careunit",
                "first_transfer_in",
                "first_transfer_out",
                "second_careunit",
                "second_transfer_in",
                "second_transfer_out",
                "total_careunit_los_hours",
            ]
        ].notna().any(axis=1)
    ]
    for idx in discharged_bad.index:
        errors.append((idx, "DISCHARGED row should not have boarding/careunit fields populated"))

    admitted_bad = df[
        admitted_mask &
        (
            df["boarding_start_time"].isna() |
            df["first_careunit"].isna() |
            df["first_transfer_in"].isna() |
            df["first_transfer_out"].isna() |
            df["total_careunit_los_hours"].isna() |
            df["second_careunit"].notna() |
            df["second_transfer_in"].notna() |
            df["second_transfer_out"].notna()
        )
    ]
    for idx in admitted_bad.index:
        errors.append((idx, "ADMITTED row must have first-careunit fields and no second-careunit fields"))

    transferred_bad = df[
        transferred_mask &
        (
            df["boarding_start_time"].isna() |
            df["first_careunit"].isna() |
            df["first_transfer_in"].isna() |
            df["first_transfer_out"].isna() |
            df["second_careunit"].isna() |
            df["second_transfer_in"].isna() |
            df["second_transfer_out"].isna() |
            df["total_careunit_los_hours"].isna()
        )
    ]
    for idx in transferred_bad.index:
        errors.append((idx, "TRANSFERRED row must have both first and second careunit fields"))

    if errors:
        print(f"VALIDATION FAILED (logical): {len(errors)} issues")
        for e in errors[:100]:
            print(" -", e)
        print("\nFix logical errors before relying on statistics/plots.")
        return
    else:
        print("VALIDATION PASSED: No logical errors found.")

    # =====================================================
    # STATISTICAL VALIDATION
    # =====================================================
    print("\n--- Statistical Validation ---")

    outcome_probs = df["pathway_outcome"].value_counts(normalize=True).to_dict()
    second_transfer_rate = (df["pathway_outcome"] == "TRANSFERRED").mean()

    print("Observed pathway outcome probabilities (Synthetic):")
    print(f" - DISCHARGED:   {outcome_probs.get('DISCHARGED', 0.0):.4f}")
    print(f" - ADMITTED:     {outcome_probs.get('ADMITTED', 0.0):.4f}")
    print(f" - TRANSFERRED:  {outcome_probs.get('TRANSFERRED', 0.0):.4f}")

    target_second = None
    if os.path.exists(BRANCH_PROB_PATH):
        branch = pd.read_csv(BRANCH_PROB_PATH)
        if not branch.empty and "p_second_transfer" in branch.columns:
            target_second = float(branch.loc[0, "p_second_transfer"])
            print("\nTarget probabilities (from MIMIC extraction):")
            print(f" - p_second_transfer:   {target_second:.4f}")
        else:
            print("\nWARNING: mimic_branch_probabilities.csv missing expected columns.")
    else:
        print("\nWARNING: mimic_branch_probabilities.csv not found; skipping target comparison.")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not available. Skipping plots.")
        return

    os.makedirs(FIG_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Plot 1: second transfer probability
    # -----------------------------------------------------
    if target_second is not None:
        plt.figure()
        plt.bar(["Synthetic", "Target"], [second_transfer_rate, target_second])
        plt.title("Second Transfer Probability")
        plt.ylabel("Probability")
        out = os.path.join(FIG_DIR, "second_transfer_probability.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 2: ED LOS overall
    # -----------------------------------------------------
    syn_ed_los = pd.to_numeric(df["ed_los_hours"], errors="coerce").dropna()
    if os.path.exists(MIMIC_ED_LOS_OVERALL):
        mimic_ed_los = _load_series(MIMIC_ED_LOS_OVERALL, "ed_los_hours")
        ks = ks_2samp(mimic_ed_los.dropna(), syn_ed_los.dropna())
        print(f"\nKS (ED LOS overall): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

        mimic_c, syn_c, p99 = _clip_to_p99(mimic_ed_los, syn_ed_los)
        print(f"Plot clipping (ED LOS overall): <= 99th percentile of MIMIC = {p99:.2f} hours")

        plt.figure()
        plt.hist(mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
        plt.hist(syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
        plt.title("ED Length of Stay (Overall)")
        plt.xlabel("Hours")
        plt.ylabel("Density")
        plt.legend()
        out = os.path.join(FIG_DIR, "ed_los_overall_hours.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 3: ED LOS admitted
    # -----------------------------------------------------
    syn_ed_los_adm = pd.to_numeric(
        df.loc[df["pathway_outcome"].isin(["ADMITTED", "TRANSFERRED"]), "ed_los_hours"],
        errors="coerce"
    ).dropna()

    if os.path.exists(MIMIC_ED_LOS_ADMITTED):
        mimic_ed_los_adm = _load_series(MIMIC_ED_LOS_ADMITTED, "ed_los_hours")
        if not syn_ed_los_adm.empty and not mimic_ed_los_adm.empty:
            ks = ks_2samp(mimic_ed_los_adm.dropna(), syn_ed_los_adm.dropna())
            print(f"\nKS (ED LOS admitted): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

            mimic_c, syn_c, p99 = _clip_to_p99(mimic_ed_los_adm, syn_ed_los_adm)
            print(f"Plot clipping (ED LOS admitted): <= 99th percentile of MIMIC = {p99:.2f} hours")

            plt.figure()
            plt.hist(mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
            plt.hist(syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
            plt.title("ED Length of Stay (Admitted/Transferred)")
            plt.xlabel("Hours")
            plt.ylabel("Density")
            plt.legend()
            out = os.path.join(FIG_DIR, "ed_los_admitted_hours.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 4: ED LOS discharged/home
    # -----------------------------------------------------
    syn_ed_los_home = pd.to_numeric(
        df.loc[df["pathway_outcome"] == "DISCHARGED", "ed_los_hours"],
        errors="coerce"
    ).dropna()

    if os.path.exists(MIMIC_ED_LOS_HOME):
        mimic_ed_los_home = _load_series(MIMIC_ED_LOS_HOME, "ed_los_hours")
        if not syn_ed_los_home.empty and not mimic_ed_los_home.empty:
            ks = ks_2samp(mimic_ed_los_home.dropna(), syn_ed_los_home.dropna())
            print(f"\nKS (ED LOS discharged/home): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

            mimic_c, syn_c, p99 = _clip_to_p99(mimic_ed_los_home, syn_ed_los_home)
            print(f"Plot clipping (ED LOS discharged/home): <= 99th percentile of MIMIC = {p99:.2f} hours")

            plt.figure()
            plt.hist(mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
            plt.hist(syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
            plt.title("ED Length of Stay (Discharged)")
            plt.xlabel("Hours")
            plt.ylabel("Density")
            plt.legend()
            out = os.path.join(FIG_DIR, "ed_los_discharged_hours.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 5: careunit stay duration
    # -----------------------------------------------------
    syn_stay = (
        (df["first_transfer_out"] - df["first_transfer_in"]).dt.total_seconds() / 3600
    )
    syn_stay = pd.to_numeric(syn_stay, errors="coerce").dropna()
    syn_stay = syn_stay[syn_stay > 0]

    if os.path.exists(MIMIC_CAREUNIT_STAY):
        mimic_stay = _load_series(MIMIC_CAREUNIT_STAY, "gap_hours")

        if not syn_stay.empty and not mimic_stay.empty:
            ks = ks_2samp(mimic_stay.dropna(), syn_stay.dropna())
            print(f"\nKS (careunit stay): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

            mimic_c, syn_c, p99 = _clip_to_p99(mimic_stay, syn_stay)
            print(f"Plot clipping (careunit stay): <= 99th percentile of MIMIC = {p99:.2f} hours")

            plt.figure()
            plt.hist(mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
            plt.hist(syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
            plt.title("First Careunit Stay Duration")
            plt.xlabel("Hours")
            plt.ylabel("Density")
            plt.legend()
            out = os.path.join(FIG_DIR, "careunit_stay_hours.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 6: arrival hour-of-day distribution
    # -----------------------------------------------------
    if "arrival_time" in df.columns:
        mimic_hour_probs = _load_probability_csv(MIMIC_ED_HOUR_PROBS, "hour")
        mimic_weekday_probs = _load_probability_csv(MIMIC_ED_WEEKDAY_PROBS, "weekday")

        syn_hour_probs = df["arrival_time"].dt.hour.value_counts(normalize=True).sort_index().to_dict()
        syn_hour_probs = {h: float(syn_hour_probs.get(h, 0.0)) for h in range(24)}

        plt.figure()
        plt.bar(
            [h - 0.2 for h in range(24)],
            [syn_hour_probs[h] for h in range(24)],
            width=0.4,
            label="Synthetic"
        )
        plt.bar(
            [h + 0.2 for h in range(24)],
            [mimic_hour_probs.get(h, 0.0) for h in range(24)],
            width=0.4,
            label="MIMIC"
        )
        plt.xticks(range(24))
        plt.xlabel("Hour of day")
        plt.ylabel("Probability")
        plt.title("Arrival Hour-of-Day Distribution: Synthetic vs MIMIC-IV-ED")
        plt.legend()
        out = os.path.join(FIG_DIR, "arrival_hour_distribution.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

        # -------------------------------------------------
        # Plot 7: arrival weekday distribution
        # -------------------------------------------------
        syn_weekday_probs = df["arrival_time"].dt.weekday.value_counts(normalize=True).sort_index().to_dict()
        syn_weekday_probs = {d: float(syn_weekday_probs.get(d, 0.0)) for d in range(7)}

        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        plt.figure()
        plt.bar(
            [d - 0.2 for d in range(7)],
            [syn_weekday_probs[d] for d in range(7)],
            width=0.4,
            label="Synthetic"
        )
        plt.bar(
            [d + 0.2 for d in range(7)],
            [mimic_weekday_probs.get(d, 0.0) for d in range(7)],
            width=0.4,
            label="MIMIC"
        )
        plt.xticks(range(7), weekday_labels)
        plt.xlabel("Weekday")
        plt.ylabel("Probability")
        plt.title("Arrival Weekday Distribution: Synthetic vs MIMIC-IV-ED")
        plt.legend()
        out = os.path.join(FIG_DIR, "arrival_weekday_distribution.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    # -----------------------------------------------------
    # Plot 8: pathway outcome probability chart
    # -----------------------------------------------------
    plt.figure()
    labels = ["DISCHARGED", "ADMITTED", "TRANSFERRED"]
    values = [outcome_probs.get(lbl, 0.0) for lbl in labels]
    plt.bar(labels, values)
    plt.title("Synthetic Pathway Outcome Probabilities")
    plt.ylabel("Probability")
    out = os.path.join(FIG_DIR, "pathway_outcome_probabilities.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out}")

    print("\nValidation completed")
    print(f"Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()