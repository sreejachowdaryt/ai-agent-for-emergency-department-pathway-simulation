# src/extract_ed_timing_from_mimic_iv_ed.py
"""
Extract ED-specific timing and disposition distributions from MIMIC-IV-ED edstays.

What is derived from MIMIC-IV-ED:
- arrival hour-of-day probabilities
- arrival weekday probabilities
- ED LOS (overall)
- ED LOS for admitted cases
- ED LOS for discharged-home cases
- ED disposition probabilities

These are used to model the ED phase:
Arrival -> Triage/Doctor -> Decision -> Discharge OR Boarding -> ED departure

Important:
edstays intime->outtime gives total ED stay, not pure boarding wait.
"""

import os
import pandas as pd

from mimic_paths import get_mimic_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "Synthetic_dataset", "data")
os.makedirs(OUT_DIR, exist_ok=True)

MIMIC_DIR = os.path.join(REPO_ROOT, "Reference_mimic_iii")


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _save_series(series: pd.Series, filename: str, col_name: str):
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    out_path = os.path.join(OUT_DIR, filename)
    pd.DataFrame({col_name: series}).to_csv(out_path, index=False)
    print(f"Saved {len(series)} rows -> {out_path}")


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("REPO_ROOT =", REPO_ROOT)
    print("OUT_DIR =", OUT_DIR)
    print("MIMIC_DIR =", MIMIC_DIR)

    paths = get_mimic_paths(MIMIC_DIR)

    if not paths.get("EDSTAYS"):
        raise FileNotFoundError(
            "Could not find edstays.csv. Please add MIMIC-IV-ED edstays.csv under the reference folder."
        )

    ed = pd.read_csv(paths["EDSTAYS"])
    ed.columns = ed.columns.str.strip().str.lower()

    required_cols = {"subject_id", "stay_id", "intime", "outtime", "disposition"}
    missing = required_cols - set(ed.columns)
    if missing:
        raise KeyError(f"edstays.csv missing required columns: {missing}")

    ed["intime"] = _to_dt(ed["intime"])
    ed["outtime"] = _to_dt(ed["outtime"])
    ed = ed.dropna(subset=["intime", "outtime", "disposition"]).copy()

    ed["disposition"] = ed["disposition"].astype(str).str.strip().str.upper()
    ed["ed_los_hours"] = (ed["outtime"] - ed["intime"]).dt.total_seconds() / 3600
    ed = ed[ed["ed_los_hours"] > 0].copy()

    # ---------------------------------------------------------
    # Arrival hour-of-day probabilities
    # ---------------------------------------------------------
    hour_probs = (
        ed["intime"].dt.hour.value_counts(normalize=True).sort_index().reset_index()
    )
    hour_probs.columns = ["hour", "probability"]
    hour_probs.to_csv(os.path.join(OUT_DIR, "mimic_ed_arrival_hour_probabilities.csv"), index=False)
    print("Saved -> mimic_ed_arrival_hour_probabilities.csv")

    # ---------------------------------------------------------
    # Arrival weekday probabilities
    # ---------------------------------------------------------
    weekday_probs = (
        ed["intime"].dt.weekday.value_counts(normalize=True).sort_index().reset_index()
    )
    weekday_probs.columns = ["weekday", "probability"]
    weekday_probs.to_csv(os.path.join(OUT_DIR, "mimic_ed_arrival_weekday_probabilities.csv"), index=False)
    print("Saved -> mimic_ed_arrival_weekday_probabilities.csv")

    # ---------------------------------------------------------
    # Disposition probabilities
    # ---------------------------------------------------------
    disp_probs = ed["disposition"].value_counts(normalize=True).reset_index()
    disp_probs.columns = ["disposition", "probability"]
    disp_probs.to_csv(os.path.join(OUT_DIR, "mimic_ed_disposition_probabilities.csv"), index=False)
    print("Saved -> mimic_ed_disposition_probabilities.csv")

    # ---------------------------------------------------------
    # ED LOS overall
    # ---------------------------------------------------------
    _save_series(ed["ed_los_hours"], "mimic_ed_los_hours.csv", "ed_los_hours")

    # ---------------------------------------------------------
    # ED LOS for admitted cases
    # ---------------------------------------------------------
    admitted = ed[ed["disposition"] == "ADMITTED"].copy()
    if not admitted.empty:
        _save_series(admitted["ed_los_hours"], "mimic_ed_los_admitted_hours.csv", "ed_los_hours")
    else:
        print("WARNING: No ADMITTED rows found in edstays.")

    # ---------------------------------------------------------
    # ED LOS for home/discharged cases
    # ---------------------------------------------------------
    home = ed[ed["disposition"] == "HOME"].copy()
    if not home.empty:
        _save_series(home["ed_los_hours"], "mimic_ed_los_home_hours.csv", "ed_los_hours")
    else:
        print("WARNING: No HOME rows found in edstays.")

    print("\nDone. ED timing and disposition distributions extracted from MIMIC-IV-ED edstays.")


if __name__ == "__main__":
    main()