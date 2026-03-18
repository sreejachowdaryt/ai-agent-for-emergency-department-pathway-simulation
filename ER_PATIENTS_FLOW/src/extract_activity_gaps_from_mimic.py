# src/extract_activity_gaps_from_mimic.py (Time gap between the previous activity and the next activity)
import os
import pandas as pd

from mimic_paths import get_mimic_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ER_PATIENTS_FLOW/src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../ER_PATIENTS_FLOW
REPO_ROOT = os.path.dirname(PROJECT_ROOT)              # .../ai-agent-for-emergency-department-pathway-simulation

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
    # IMPORTANT: adjust if your folder name differs
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("REPO_ROOT =", REPO_ROOT)
    print("OUT_DIR =", OUT_DIR)
    print("MIMIC_DIR =", MIMIC_DIR)
   
    paths = get_mimic_paths(MIMIC_DIR)

    # -------------------------
    # ADMISSIONS
    # -------------------------
    adm = pd.read_csv(
        paths["ADMISSIONS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE"]
    )
    adm["ADMITTIME"] = _to_dt(adm["ADMITTIME"])
    adm["DISCHTIME"] = _to_dt(adm["DISCHTIME"])
    adm = adm.dropna(subset=["HADM_ID", "ADMITTIME", "DISCHTIME"])

    # -------------------------
    # TRANSFERS
    # -------------------------
    tr = pd.read_csv(
        paths["TRANSFERS"],
        usecols=["SUBJECT_ID", "HADM_ID", "INTIME", "OUTTIME", "CURR_CAREUNIT"]
    )
    tr["INTIME"] = _to_dt(tr["INTIME"])
    tr["OUTTIME"] = _to_dt(tr["OUTTIME"])
    tr = tr.dropna(subset=["HADM_ID", "INTIME"])
    tr = tr.sort_values(["HADM_ID", "INTIME"])

    # -------------------------
    # ICUSTAYS
    # -------------------------
    icu = pd.read_csv(
        paths["ICUSTAYS"],
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]
    )
    icu["INTIME"] = _to_dt(icu["INTIME"])
    icu["OUTTIME"] = _to_dt(icu["OUTTIME"])
    icu = icu.dropna(subset=["HADM_ID", "INTIME", "OUTTIME"])

    # -------------------------
    # CALLOUT
    # -------------------------
    callout = pd.read_csv(
        paths["CALLOUT"],
        usecols=["SUBJECT_ID", "HADM_ID", "CREATETIME", "ACKNOWLEDGETIME"]
    )
    callout["CREATETIME"] = _to_dt(callout["CREATETIME"])
    callout["ACKNOWLEDGETIME"] = _to_dt(callout["ACKNOWLEDGETIME"])
    callout = callout.dropna(subset=["HADM_ID"])

    # =========================================================
    # Branch probabilities (for the process tree) - ED-like cohort
    # =========================================================

    def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.upper()
            .str.replace(" ", "_")
        )
        return df

    adm = normalize_cols(adm)
    tr  = normalize_cols(tr)
    icu = normalize_cols(icu)

    # -------------------------
    # Make sure required columns exist
    # -------------------------
    required_tr = {"HADM_ID", "INTIME", "CURR_CAREUNIT"}
    missing_tr = required_tr - set(tr.columns)
    if missing_tr:
        raise KeyError(f"TRANSFERS missing columns: {missing_tr}. Found: {list(tr.columns)}")

    has_adm_type = "ADMISSION_TYPE" in adm.columns
    if not has_adm_type:
        print("WARNING: ADMISSION_TYPE not found in ADMISSIONS. Using ALL admissions (no EMERGENCY filter).")
    else:
        # If present, show counts for sanity
        print("ADMISSION_TYPE values (top 10):")
        print(adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().value_counts().head(10))

    # -------------------------
    # 1) Define ED-admitted proxy cohort
    # -------------------------
    if has_adm_type:
        adm_em = adm.loc[
            adm["ADMISSION_TYPE"].astype(str).str.strip().str.upper().eq("EMERGENCY"),
            "HADM_ID"
        ].dropna().unique()
        adm_em_set = set(adm_em)
    else:
        adm_em_set = set(adm["HADM_ID"].dropna().unique())

    # -------------------------
    # 2) Transfers restricted to cohort
    # -------------------------
    tr_em = tr.loc[tr["HADM_ID"].isin(adm_em_set)].copy()
    tr_em = tr_em.dropna(subset=["HADM_ID", "INTIME"])
    tr_em = tr_em.sort_values(["HADM_ID", "INTIME"])
    tr_em["CURR_CAREUNIT"] = tr_em["CURR_CAREUNIT"].astype(str).str.strip().str.upper()
    tr_em = tr_em[tr_em["CURR_CAREUNIT"].notna() & (tr_em["CURR_CAREUNIT"] != "NAN")]

    total_flow_adm = tr_em["HADM_ID"].nunique()

    p_second_transfer = 0.0
    p_icu_raw = 0.0

    icu_units = {"MICU", "SICU", "CCU", "CSRU", "TSICU"}

    if total_flow_adm == 0:
        print("WARNING: No admissions found in TRANSFERS after cohort filtering.")
    else:
        # p_second_transfer: admission visited >= 2 distinct care units
        careunit_counts = tr_em.groupby("HADM_ID")["CURR_CAREUNIT"].nunique()
        p_second_transfer = float((careunit_counts >= 2).mean())

        # ---------------------------------------------------------
        # p_icu_raw: escalation to ICU after a non-ICU start
        # (computed from TRANSFERS sequence; no need to rely on ICUSTAYS)
        # ---------------------------------------------------------
        first_unit = tr_em.groupby("HADM_ID")["CURR_CAREUNIT"].first()
        starts_non_icu = first_unit[~first_unit.isin(icu_units)].index
        den = len(starts_non_icu)

        icu_hadm_set = set(icu.loc[icu["HADM_ID"].notna(), "HADM_ID"].unique())

        num_escalate = sum((hadm in icu_hadm_set) for hadm in starts_non_icu)
        p_icu_raw = (num_escalate / den) if den else 0.0

        # Debug Statements
        print("ICU escalation diagnostics (ED-like cohort):")
        print(" admissions with transfers:", total_flow_adm)
        print(" admissions starting non-ICU:", den)
        print(" of those, later ICU careunit:", num_escalate)
        print(" p_icu_raw:", round(p_icu_raw, 6))
        
        print("Top CURR_CAREUNIT values (EMERGENCY):")
        print(tr_em["CURR_CAREUNIT"].astype(str).str.strip().str.upper().value_counts().head(15))

    # ---------------------------------------------------------
    # Option 2 (hybrid): calibrate ICU probability for general ED realism
    # ---------------------------------------------------------
    TARGET_ICU = 0.15  # justify + sensitivity (e.g., 0.05, 0.10, 0.15)
    p_icu = min(p_icu_raw, TARGET_ICU)

    # 3) Save probabilities
    pd.DataFrame([{
        "p_second_transfer": p_second_transfer,
        "p_icu_raw": p_icu_raw,
        "p_icu": p_icu,
        "target_icu_cap": TARGET_ICU,
        "used_emergency_filter": bool(has_adm_type)
    }]).to_csv(os.path.join(OUT_DIR, "mimic_branch_probabilities.csv"), index=False)

    print("Saved branch probabilities -> Synthetic_dataset/data/mimic_branch_probabilities.csv")
    print(f"p_second_transfer={p_second_transfer:.6f}, p_icu_raw={p_icu_raw:.6f}, p_icu={p_icu:.6f}")

    # =========================================================
    # GAP A: arrival (ADMITTIME) -> first transfer INTIME (hours)
    # =========================================================
    first_tr = tr.groupby("HADM_ID", as_index=False).first()[["HADM_ID", "INTIME"]]
    gA = adm.merge(first_tr, on="HADM_ID", how="inner")
    gap_arrival_to_first_in_h = (gA["INTIME"] - gA["ADMITTIME"]).dt.total_seconds() / 3600
    _save_series(gap_arrival_to_first_in_h,
                 "mimic_gap_arrival_to_first_transfer_in_hours.csv",
                 "gap_hours")

    # =========================================================
    # GAP B: transfer stay length OUTTIME - INTIME (hours)
    # =========================================================
    tr_stay = tr.dropna(subset=["OUTTIME"]).copy()
    gap_transfer_stay_h = (tr_stay["OUTTIME"] - tr_stay["INTIME"]).dt.total_seconds() / 3600
    _save_series(gap_transfer_stay_h,
                 "mimic_gap_transfer_stay_hours.csv",
                 "gap_hours")

    # =========================================================
    # GAP C: transfer1 OUTTIME -> transfer2 INTIME (hours)
    # (only admissions with >=2 transfers)
    # =========================================================
    tr2 = tr.groupby("HADM_ID").head(2).copy()

    # rank within admission
    tr2["rank"] = tr2.groupby("HADM_ID").cumcount() + 1

    t1 = tr2[tr2["rank"] == 1][["HADM_ID", "INTIME", "OUTTIME"]].rename(
        columns={"INTIME": "t1_in", "OUTTIME": "t1_out"}
    )

    t2 = tr2[tr2["rank"] == 2][["HADM_ID", "INTIME"]].rename(
        columns={"INTIME": "t2_in"}
    )

    t12 = t1.merge(t2, on="HADM_ID", how="inner")

    # --- primary gap: t2_in - t1_out
    primary_gap = (
        t12.dropna(subset=["t1_out", "t2_in"])
        .assign(gap_hours=lambda df:
                (df["t2_in"] - df["t1_out"]).dt.total_seconds() / 3600)
    )

    primary_gap = primary_gap[primary_gap["gap_hours"] > 0]["gap_hours"]

    # --- fallback if primary is empty
    if primary_gap.empty:
        print("Primary transfer1->transfer2 gap empty. Using fallback (t2_in - t1_in).")
        fallback_gap = (
            t12.dropna(subset=["t1_in", "t2_in"])
            .assign(gap_hours=lambda df:
                    (df["t2_in"] - df["t1_in"]).dt.total_seconds() / 3600)
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

    # =========================================================
    # GAP D: ICU LOS (hours)
    # =========================================================
    icu_los_h = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / 3600
    _save_series(icu_los_h,
                 "mimic_gap_icu_los_hours.csv",
                 "gap_hours")

    # =========================================================
    # GAP E: last activity -> discharge (hours)
    # last activity approximated by max(last transfer out, last icu out)
    # =========================================================
    last_tr_out = tr.dropna(subset=["OUTTIME"]).groupby("HADM_ID")["OUTTIME"].max()
    last_icu_out = icu.groupby("HADM_ID")["OUTTIME"].max()

    last = pd.DataFrame({
        "last_transfer_out": last_tr_out,
        "last_icu_out": last_icu_out
    }).reset_index()

    last["last_time"] = last[["last_transfer_out", "last_icu_out"]].max(axis=1)

    gE = adm.merge(last[["HADM_ID", "last_time"]], on="HADM_ID", how="inner").dropna(subset=["last_time"])
    gap_last_to_discharge_h = (gE["DISCHTIME"] - gE["last_time"]).dt.total_seconds() / 3600
    _save_series(gap_last_to_discharge_h,
                 "mimic_gap_last_activity_to_discharge_hours.csv",
                 "gap_hours")

    # =========================================================
    # GAP F: discharge -> callout create (minutes)
    # =========================================================
    gF = adm.merge(callout[["HADM_ID", "CREATETIME"]], on="HADM_ID", how="inner").dropna(subset=["CREATETIME"])
    gap_discharge_to_callout_min = (gF["CREATETIME"] - gF["DISCHTIME"]).dt.total_seconds() / 60
    _save_series(gap_discharge_to_callout_min,
                 "mimic_gap_discharge_to_callout_create_minutes.csv",
                 "gap_minutes")

    # =========================================================
    # GAP G: callout create -> acknowledge (minutes)
    # =========================================================
    gG = callout.dropna(subset=["CREATETIME", "ACKNOWLEDGETIME"])
    gap_callout_ack_min = (gG["ACKNOWLEDGETIME"] - gG["CREATETIME"]).dt.total_seconds() / 60
    _save_series(gap_callout_ack_min,
                 "mimic_gap_callout_create_to_ack_minutes.csv",
                 "gap_minutes")

    print("\nDone. You can now generate synthetic ED cases using these gap distributions.")

if __name__ == "__main__":
    main()
