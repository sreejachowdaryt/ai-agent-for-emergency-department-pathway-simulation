# src/validate_ed_cases.py
# Logical + Statistical validation for Synthetic_dataset/data/ed_cases.csv
#
# Logical checks:
#  - arrival < assessment < discharge
#  - transfer in < out (and optional second transfer)
#  - ICU admission < ICU discharge (when present)
#  - no overlapping admissions per patient
#
# Statistical validation:
#  - ICU rate vs target p_icu
#  - second transfer rate vs target p_second_transfer
#  - MIMIC vs Synthetic distribution overlays:
#      * arrival -> first transfer gap (hours)
#      * transfer stay duration (hours)
#      * ICU LOS (hours)
#      * arrival hour-of-day distribution
#      * arrival weekday distribution
#
# Saves figures to: Synthetic_dataset/figures/

import os
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ER_PATIENTS_FLOW/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../ER_PATIENTS_FLOW

DATA_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "figures")

INPUT_PATH = os.path.join(DATA_DIR, "ed_cases.csv")
BRANCH_PROB_PATH = os.path.join(DATA_DIR, "mimic_branch_probabilities.csv")

MIMIC_GAP_ARRIVAL = os.path.join(DATA_DIR, "mimic_gap_arrival_to_first_transfer_in_hours.csv")
MIMIC_TRANSFER_STAY = os.path.join(DATA_DIR, "mimic_gap_transfer_stay_hours.csv")
MIMIC_ICU_LOS = os.path.join(DATA_DIR, "mimic_gap_icu_los_hours.csv")


def _load_series(path: str, col: str) -> pd.Series:
    """Load a numeric column from a CSV as a cleaned Series."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    s = pd.read_csv(path)[col]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def _clip_to_p99(mimic: pd.Series, syn: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """
    Clip both series to the 99th percentile of the MIMIC distribution
    so plots focus on the main mass (tails can be huge in MIMIC).
    Returns (mimic_clipped, syn_clipped, p99_value).
    """
    mimic = pd.to_numeric(mimic, errors="coerce").dropna()
    syn = pd.to_numeric(syn, errors="coerce").dropna()
    if mimic.empty or syn.empty:
        return mimic, syn, float("nan")

    p99 = float(mimic.quantile(0.99))
    mimic_c = mimic[mimic <= p99]
    syn_c = syn[syn <= p99]
    return mimic_c, syn_c, p99


def _load_mimic_arrival_temporal_distributions() -> tuple[dict[int, float], dict[int, float]]:
    """
    Load MIMIC-derived arrival hour and weekday distributions directly from ADMISSIONS.ADMITTIME.
    Uses EMERGENCY admissions only, consistent with generate_ed_cases.py.
    """
    mimic_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "Reference_mimic_iii")

    # local import to avoid changing other files
    from mimic_paths import get_mimic_paths

    paths = get_mimic_paths(mimic_dir)

    adm = pd.read_csv(
        paths["ADMISSIONS"],
        usecols=["ADMISSION_TYPE", "ADMITTIME"]
    )
    adm.columns = (
        adm.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"], errors="coerce")
    adm = adm.dropna(subset=["ADMITTIME"]).copy()

    adm = adm[
        adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().eq("EMERGENCY")
    ].copy()

    if adm.empty:
        hour_probs = {h: 1 / 24 for h in range(24)}
        weekday_probs = {d: 1 / 7 for d in range(7)}
        return hour_probs, weekday_probs

    hour_probs = adm["ADMITTIME"].dt.hour.value_counts(normalize=True).sort_index().to_dict()
    weekday_probs = adm["ADMITTIME"].dt.weekday.value_counts(normalize=True).sort_index().to_dict()

    hour_probs = {h: float(hour_probs.get(h, 0.0)) for h in range(24)}
    weekday_probs = {d: float(weekday_probs.get(d, 0.0)) for d in range(7)}

    hour_total = sum(hour_probs.values())
    weekday_total = sum(weekday_probs.values())

    if hour_total > 0:
        hour_probs = {h: v / hour_total for h, v in hour_probs.items()}
    else:
        hour_probs = {h: 1 / 24 for h in range(24)}

    if weekday_total > 0:
        weekday_probs = {d: v / weekday_total for d, v in weekday_probs.items()}
    else:
        weekday_probs = {d: 1 / 7 for d in range(7)}

    return hour_probs, weekday_probs


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

    # ----------------
    # LOGICAL CHECKS
    # ----------------

    bad = df[~(df["arrival_time"] < df["initial_assessment_time"])]
    for idx in bad.index:
        errors.append((idx, "arrival_time must be before initial_assessment_time"))

    bad = df[~(df["initial_assessment_time"] < df["discharge_time"])]
    for idx in bad.index:
        errors.append((idx, "initial_assessment_time must be before discharge_time"))

    bad = df[~(df["first_transfer_in"] < df["first_transfer_out"])]
    for idx in bad.index:
        errors.append((idx, "first_transfer_in must be before first_transfer_out"))

    if "second_transfer_in" in df.columns and "second_transfer_out" in df.columns:
        second_present = df["second_transfer_in"].notna()
        bad = df[second_present & ~(df["second_transfer_in"] < df["second_transfer_out"])]
        for idx in bad.index:
            errors.append((idx, "second_transfer_in must be before second_transfer_out when present"))

    if "icu_admission_time" in df.columns and "icu_discharge_time" in df.columns:
        icu_present = df["icu_admission_time"].notna()
        bad = df[icu_present & ~(df["icu_admission_time"] < df["icu_discharge_time"])]
        for idx in bad.index:
            errors.append((idx, "icu_admission_time must be before icu_discharge_time when present"))

    if "patient_id" in df.columns:
        df_sorted = df.sort_values(["patient_id", "arrival_time"]).reset_index()
        for pid, g in df_sorted.groupby("patient_id"):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                prev_dis = g.loc[i - 1, "discharge_time"]
                cur_arr = g.loc[i, "arrival_time"]
                if pd.notna(prev_dis) and pd.notna(cur_arr) and not (cur_arr > prev_dis):
                    errors.append((int(g.loc[i, "index"]), f"overlap with previous admission for patient {pid}"))

    if errors:
        print(f"VALIDATION FAILED (logical): {len(errors)} issues")
        for e in errors[:50]:
            print(" -", e)
        print("\nFix logical errors before relying on any statistics/plots.")
        return
    else:
        print("VALIDATION PASSED: No logical errors found.")

    # -----------------------------
    # STATISTICAL VALIDATION
    # -----------------------------
    print("\n--- Statistical Validation ---")

    icu_rate = df["icu_admission_time"].notna().mean() if "icu_admission_time" in df.columns else float("nan")
    second_transfer_rate = df["second_transfer_in"].notna().mean() if "second_transfer_in" in df.columns else float("nan")

    print("Observed probabilities (Synthetic):")
    print(f" - ICU admission rate:        {icu_rate:.4f}")
    print(f" - Second transfer rate:      {second_transfer_rate:.4f}")

    target_icu = None
    target_second = None

    if os.path.exists(BRANCH_PROB_PATH):
        branch = pd.read_csv(BRANCH_PROB_PATH)
        if not branch.empty and {"p_icu", "p_second_transfer"}.issubset(branch.columns):
            target_icu = float(branch.loc[0, "p_icu"])
            target_second = float(branch.loc[0, "p_second_transfer"])

            print("\nTarget probabilities (from MIMIC extraction):")
            print(f" - p_icu:               {target_icu:.4f}")
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

    # Plot 1: ICU probability bar chart
    if target_icu is not None:
        plt.figure()
        plt.bar(["Synthetic", "Target"], [icu_rate, target_icu])
        plt.title("ICU Admission Probability")
        plt.ylabel("Probability")
        out = os.path.join(FIG_DIR, "icu_probability.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    # Plot 2: Second transfer probability bar chart
    if target_second is not None:
        plt.figure()
        plt.bar(["Synthetic", "Target"], [second_transfer_rate, target_second])
        plt.title("Second Transfer Probability")
        plt.ylabel("Probability")
        out = os.path.join(FIG_DIR, "second_transfer_probability.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    # Plot 3: Arrival -> first transfer gap (hours)
    if "first_transfer_in" in df.columns and "arrival_time" in df.columns:
        gap_syn = (df["first_transfer_in"] - df["arrival_time"]).dt.total_seconds() / 3600
        gap_syn = pd.to_numeric(gap_syn, errors="coerce").dropna()
        gap_syn = gap_syn[gap_syn > 0]

        if os.path.exists(MIMIC_GAP_ARRIVAL):
            gap_mimic = _load_series(MIMIC_GAP_ARRIVAL, "gap_hours")

            ks = ks_2samp(gap_mimic.dropna(), gap_syn.dropna())
            print(f"\nKS (arrival→first transfer): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

            gap_mimic_c, gap_syn_c, p99 = _clip_to_p99(gap_mimic, gap_syn)
            print(f"Plot clipping (arrival gap): <= 99th percentile of MIMIC = {p99:.2f} hours")

            plt.figure()
            plt.hist(gap_mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
            plt.hist(gap_syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
            plt.title("Arrival → First Transfer Gap (hours)")
            plt.xlabel("Hours")
            plt.ylabel("Density")
            plt.legend()
            out = os.path.join(FIG_DIR, "gap_arrival_to_first_transfer_hours.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved figure -> {out}")
        else:
            print(f"WARNING: Missing {MIMIC_GAP_ARRIVAL}. Skipping arrival gap plot.")

    # Plot 4: Transfer stay duration (hours)
    if "first_transfer_in" in df.columns and "first_transfer_out" in df.columns:
        stay_syn = (df["first_transfer_out"] - df["first_transfer_in"]).dt.total_seconds() / 3600
        stay_syn = pd.to_numeric(stay_syn, errors="coerce").dropna()
        stay_syn = stay_syn[stay_syn > 0]

        if os.path.exists(MIMIC_TRANSFER_STAY):
            stay_mimic = _load_series(MIMIC_TRANSFER_STAY, "gap_hours")

            ks = ks_2samp(stay_mimic.dropna(), stay_syn.dropna())
            print(f"\nKS (transfer stay): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

            stay_mimic_c, stay_syn_c, p99 = _clip_to_p99(stay_mimic, stay_syn)
            print(f"Plot clipping (transfer stay): <= 99th percentile of MIMIC = {p99:.2f} hours")

            plt.figure()
            plt.hist(stay_mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
            plt.hist(stay_syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
            plt.title("Transfer Stay Duration (hours)")
            plt.xlabel("Hours")
            plt.ylabel("Density")
            plt.legend()
            out = os.path.join(FIG_DIR, "gap_transfer_stay_hours.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved figure -> {out}")
        else:
            print(f"WARNING: Missing {MIMIC_TRANSFER_STAY}. Skipping transfer stay plot.")

    # Plot 5: ICU LOS (hours)
    icu_syn = pd.to_numeric(df.get("icu_los_hours", pd.Series(dtype=float)), errors="coerce").dropna()
    icu_syn = icu_syn[icu_syn > 0]

    if os.path.exists(MIMIC_ICU_LOS):
        icu_mimic = _load_series(MIMIC_ICU_LOS, "gap_hours")

        ks = ks_2samp(icu_mimic.dropna(), icu_syn.dropna())
        print(f"\nKS (ICU LOS): D={ks.statistic:.4f}, p={ks.pvalue:.4g}")

        icu_mimic_c, icu_syn_c, p99 = _clip_to_p99(icu_mimic, icu_syn)
        print(f"Plot clipping (ICU LOS): <= 99th percentile of MIMIC = {p99:.2f} hours")

        plt.figure()
        plt.hist(icu_mimic_c, bins=100, alpha=0.5, label="MIMIC", density=True)
        plt.hist(icu_syn_c, bins=100, alpha=0.5, label="Synthetic", density=True)
        plt.title("ICU Length of Stay (hours)")
        plt.xlabel("Hours")
        plt.ylabel("Density")
        plt.legend()
        out = os.path.join(FIG_DIR, "icu_los_hours.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")
    else:
        print(f"WARNING: Missing {MIMIC_ICU_LOS}. Skipping ICU LOS plot.")

    # Plot 6: Arrival hour-of-day distribution
    if "arrival_time" in df.columns:
        mimic_hour_probs, mimic_weekday_probs = _load_mimic_arrival_temporal_distributions()

        syn_hour_probs = (
            df["arrival_time"].dt.hour.value_counts(normalize=True).sort_index().to_dict()
        )
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
            [mimic_hour_probs[h] for h in range(24)],
            width=0.4,
            label="MIMIC"
        )
        plt.xticks(range(24))
        plt.xlabel("Hour of day")
        plt.ylabel("Probability")
        plt.title("Arrival Hour-of-Day Distribution: Synthetic vs MIMIC")
        plt.legend()
        out = os.path.join(FIG_DIR, "arrival_hour_distribution.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

        # Plot 7: Arrival weekday distribution
        syn_weekday_probs = (
            df["arrival_time"].dt.weekday.value_counts(normalize=True).sort_index().to_dict()
        )
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
            [mimic_weekday_probs[d] for d in range(7)],
            width=0.4,
            label="MIMIC"
        )
        plt.xticks(range(7), weekday_labels)
        plt.xlabel("Weekday")
        plt.ylabel("Probability")
        plt.title("Arrival Weekday Distribution: Synthetic vs MIMIC")
        plt.legend()
        out = os.path.join(FIG_DIR, "arrival_weekday_distribution.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {out}")

    print("\nValidation completed")
    print(f"Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()