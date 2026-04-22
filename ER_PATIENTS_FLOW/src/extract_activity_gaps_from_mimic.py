# src/extract_activity_gaps_from_mimic.py
"""
Extract MIMIC-III transfer-based distributions for modelling the post-ED inpatient pathway.

This script:
- Uses TRANSFERS and ADMISSIONS tables from MIMIC-III
- Restricts transfer-based analysis to EMERGENCY admissions to reflect ED-originated pathways
- Derives:
  - probability of a second careunit transfer
  - first careunit distribution
  - second careunit distribution
  - conditional transition probabilities between careunits
  - careunit stay durations
  - time gap between first and second transfer
  - time gap from last careunit to hospital discharge

These distributions are used to simulate the admitted patient pathway after ED departure.
Other attributes (e.g., admission_type) in the synthetic dataset may still include all admission categories.

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


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )
    return df


def _save_series(series: pd.Series, filename: str, col_name: str):
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    out_path = os.path.join(OUT_DIR, filename)
    pd.DataFrame({col_name: series}).to_csv(out_path, index=False)
    print(f"Saved {len(series)} rows -> {out_path}")


def _build_probability_dict(series: pd.Series) -> dict[str, float]:
    s = series.astype(str).str.strip().str.upper()
    s = s[s.notna() & (s != "") & (s != "NAN")]
    if s.empty:
        return {}
    probs = s.value_counts(normalize=True).to_dict()
    return {str(k): float(v) for k, v in probs.items()}


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("REPO_ROOT =", REPO_ROOT)
    print("OUT_DIR =", OUT_DIR)
    print("MIMIC_DIR =", MIMIC_DIR)

    paths = get_mimic_paths(MIMIC_DIR)

    adm = pd.read_csv(
        paths["ADMISSIONS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE", "DISCHARGE_LOCATION"]
    )
    adm = _normalize_cols(adm)
    adm["ADMITTIME"] = _to_dt(adm["ADMITTIME"])
    adm["DISCHTIME"] = _to_dt(adm["DISCHTIME"])
    adm = adm.dropna(subset=["HADM_ID", "ADMITTIME", "DISCHTIME"]).copy()

    tr = pd.read_csv(
        paths["TRANSFERS"],
        usecols=["SUBJECT_ID", "HADM_ID", "INTIME", "OUTTIME", "CURR_CAREUNIT"]
    )
    tr = _normalize_cols(tr)
    tr["INTIME"] = _to_dt(tr["INTIME"])
    tr["OUTTIME"] = _to_dt(tr["OUTTIME"])
    tr = tr.dropna(subset=["HADM_ID", "INTIME", "CURR_CAREUNIT"]).copy()
    tr = tr.sort_values(["HADM_ID", "INTIME"])

    # Restrict to EMERGENCY admissions for ED-like realism
    adm_em = adm.loc[
        adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().eq("EMERGENCY"),
        "HADM_ID"
    ].dropna().unique()
    adm_em_set = set(adm_em)

    tr_em = tr[tr["HADM_ID"].isin(adm_em_set)].copy()
    adm_em_df = adm[adm["HADM_ID"].isin(adm_em_set)].copy()

    tr_em["CURR_CAREUNIT"] = tr_em["CURR_CAREUNIT"].astype(str).str.strip().str.upper()
    tr_em = tr_em[
        tr_em["CURR_CAREUNIT"].notna() &
        (tr_em["CURR_CAREUNIT"] != "") &
        (tr_em["CURR_CAREUNIT"] != "NAN")
    ].copy()

    # ---------------------------------------------------------
    # p_second_transfer
    # ---------------------------------------------------------
    total_flow_adm = tr_em["HADM_ID"].nunique()
    if total_flow_adm == 0:
        p_second_transfer = 0.0
        print("WARNING: No EMERGENCY admissions found in TRANSFERS.")
    else:
        careunit_counts = tr_em.groupby("HADM_ID")["CURR_CAREUNIT"].nunique()
        p_second_transfer = float((careunit_counts >= 2).mean())

    pd.DataFrame([{
        "p_second_transfer": p_second_transfer,
        "used_emergency_filter": True
    }]).to_csv(os.path.join(OUT_DIR, "mimic_branch_probabilities.csv"), index=False)

    print(f"Saved p_second_transfer={p_second_transfer:.6f} -> mimic_branch_probabilities.csv")

    # ---------------------------------------------------------
    # first careunit distribution
    # ---------------------------------------------------------
    first_careunits = tr_em.groupby("HADM_ID")["CURR_CAREUNIT"].first()
    first_probs = _build_probability_dict(first_careunits)
    pd.DataFrame(
        [{"careunit": k, "probability": v} for k, v in first_probs.items()]
    ).to_csv(os.path.join(OUT_DIR, "mimic_first_careunit_probabilities.csv"), index=False)
    print("Saved -> mimic_first_careunit_probabilities.csv")

    # ---------------------------------------------------------
    # second careunit overall distribution
    # ---------------------------------------------------------
    transfer_sequences = tr_em.groupby("HADM_ID")["CURR_CAREUNIT"].apply(list)
    second_careunits = transfer_sequences[
        transfer_sequences.apply(lambda x: len(x) >= 2)
    ].apply(lambda x: x[1])

    second_probs = _build_probability_dict(second_careunits)
    pd.DataFrame(
        [{"careunit": k, "probability": v} for k, v in second_probs.items()]
    ).to_csv(os.path.join(OUT_DIR, "mimic_second_careunit_probabilities.csv"), index=False)
    print("Saved -> mimic_second_careunit_probabilities.csv")

    # ---------------------------------------------------------
    # conditional transition probs first -> second careunit
    # ---------------------------------------------------------
    rows = []
    paired_sequences = transfer_sequences[
        transfer_sequences.apply(lambda x: len(x) >= 2)
    ]

    transition_counts = {}
    for seq in paired_sequences:
        first_cu = str(seq[0]).strip().upper()
        second_cu = str(seq[1]).strip().upper()
        transition_counts.setdefault(first_cu, {})
        transition_counts[first_cu][second_cu] = transition_counts[first_cu].get(second_cu, 0) + 1

    for first_cu, second_map in transition_counts.items():
        total = sum(second_map.values())
        for second_cu, count in second_map.items():
            rows.append({
                "first_careunit": first_cu,
                "second_careunit": second_cu,
                "probability": count / total
            })

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "mimic_second_careunit_transition_probabilities.csv"),
        index=False
    )
    print("Saved -> mimic_second_careunit_transition_probabilities.csv")

    # ---------------------------------------------------------
    # careunit stay duration
    # ---------------------------------------------------------
    tr_stay = tr_em.dropna(subset=["OUTTIME"]).copy()
    careunit_stay_h = (tr_stay["OUTTIME"] - tr_stay["INTIME"]).dt.total_seconds() / 3600
    _save_series(careunit_stay_h, "mimic_gap_careunit_stay_hours.csv", "gap_hours")

    # ---------------------------------------------------------
    # first transfer out -> second transfer in
    # ---------------------------------------------------------
    tr2 = tr_em.groupby("HADM_ID").head(2).copy()
    tr2["rank"] = tr2.groupby("HADM_ID").cumcount() + 1

    t1 = tr2[tr2["rank"] == 1][["HADM_ID", "INTIME", "OUTTIME"]].rename(
        columns={"INTIME": "t1_in", "OUTTIME": "t1_out"}
    )
    t2 = tr2[tr2["rank"] == 2][["HADM_ID", "INTIME"]].rename(
        columns={"INTIME": "t2_in"}
    )
    t12 = t1.merge(t2, on="HADM_ID", how="inner")

    primary_gap = (
        t12.dropna(subset=["t1_out", "t2_in"])
        .assign(gap_hours=lambda df: (df["t2_in"] - df["t1_out"]).dt.total_seconds() / 3600)
    )
    primary_gap = primary_gap[primary_gap["gap_hours"] > 0]["gap_hours"]

    if primary_gap.empty:
        print("Primary transfer1->transfer2 gap empty. Using fallback (t2_in - t1_in).")
        fallback_gap = (
            t12.dropna(subset=["t1_in", "t2_in"])
            .assign(gap_hours=lambda df: (df["t2_in"] - df["t1_in"]).dt.total_seconds() / 3600)
        )
        fallback_gap = fallback_gap[fallback_gap["gap_hours"] > 0]["gap_hours"]
        gap_between_transfers_h = fallback_gap
    else:
        gap_between_transfers_h = primary_gap

    _save_series(
        gap_between_transfers_h,
        "mimic_gap_between_transfer1_out_and_transfer2_in_hours.csv",
        "gap_hours"
    )

    # ---------------------------------------------------------
    # last careunit -> hospital discharge
    # ---------------------------------------------------------
    last_tr_out = tr_em.dropna(subset=["OUTTIME"]).groupby("HADM_ID")["OUTTIME"].max().reset_index()
    last_tr_out = last_tr_out.rename(columns={"OUTTIME": "last_time"})

    gD = adm_em_df.merge(last_tr_out[["HADM_ID", "last_time"]], on="HADM_ID", how="inner").dropna(subset=["last_time"])
    gap_last_to_discharge_h = (gD["DISCHTIME"] - gD["last_time"]).dt.total_seconds() / 3600
    _save_series(
        gap_last_to_discharge_h,
        "mimic_gap_last_careunit_to_discharge_hours.csv",
        "gap_hours"
    )

    print("\nDone. Post-ED transfer distributions extracted from MIMIC-III.")


if __name__ == "__main__":
    main()