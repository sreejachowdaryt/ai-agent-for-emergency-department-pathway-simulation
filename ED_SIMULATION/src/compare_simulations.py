# src/compare_simulations.py
"""
Compare baseline, rule-based, and hybrid ML ED simulation results.

This script performs a three-way comparison of the final dataset-driven
simulation models:

- Baseline model:
  discrete-event simulation with FIFO boarding
- Rule-based model:
  severity-based boarding priority and escalation
- Hybrid ML model:
  POCT-based assessment acceleration combined with rule-based boarding

Outputs:
1. comparison_table.csv
   Contains replication-level summary comparisons including:
   - mean and standard deviation
   - absolute differences between models
   - paired t-test p-values
   - statistical significance flags

2. nhs_compliance_comparison.csv
   Contains replication-level NHS 4-hour compliance values and
   compliance improvements across the three models

Comparison metrics include:
- assessment waiting time
- boarding waiting time
- total ED length of stay (LOS)
- NHS 4-hour compliance
- annual patient outcome counts
- boarding wait by severity

This script supports the final evaluation stage of the project by
quantifying whether AI-based interventions improve ED performance
relative to the baseline model.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

BASELINE_SUMMARY = "../data/simulation_summary.csv"
AI_SUMMARY       = "../data/simulation_ai_summary.csv"
ML_SUMMARY       = "../data/simulation_ml_summary.csv"

BASELINE_LOG     = "../data/simulation_patient_log.csv"
AI_LOG           = "../data/simulation_ai_patient_log.csv"
ML_LOG           = "../data/simulation_ml_patient_log.csv"

OUT_DIR          = "../data"
COMPARISON_OUT   = os.path.join(OUT_DIR, "comparison_table.csv")
COMPLIANCE_OUT   = os.path.join(OUT_DIR, "nhs_compliance_comparison.csv")


def convert_time_cols_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert simulation summary time columns from hours to minutes
    when needed.
    """
    df = df.copy()
    for col in df.columns:
        if any(x in col for x in ["wait", "los"]) and df[col].dropna().mean() < 100:
            df[col] = df[col] * 60
    return df


def compute_compliance(log_path: str) -> pd.Series:
    """
    Compute NHS 4-hour compliance per replication from patient log.

    total_los is converted to minutes if needed, then compared to 240 min.
    """
    df = pd.read_csv(log_path)

    if "total_los" not in df.columns:
        raise ValueError(f"'total_los' not found in {log_path}")

    if df["total_los"].dropna().mean() < 100:
        df["total_los"] = df["total_los"] * 60

    df["compliant"] = df["total_los"].apply(
        lambda x: 1 if pd.notna(x) and x <= 240 else 0
    )

    return (
        df.groupby("replication")
          .apply(lambda g: g["compliant"].mean() * 100)
          .reset_index(name="compliance_pct")["compliance_pct"]
    )


def per_severity_boarding(log_path: str, severities: list[str]):
    """
    Compute per-severity boarding wait mean and std across replications.

    Uses replication-level means (not patient-level pooling) so the
    reported std reflects variability across the 10 replications.
    """
    df = pd.read_csv(log_path)

    if "boarding_wait" not in df.columns:
        raise ValueError(f"'boarding_wait' not found in {log_path}")

    if df["boarding_wait"].dropna().mean() < 100:
        df["boarding_wait"] = df["boarding_wait"] * 60

    sev_means, sev_stds = [], []

    for sev in severities:
        rep_means = []
        for _, grp in df.groupby("replication"):
            sub = grp[
                (grp["severity"] == sev) &
                (grp["boarding_wait"].notna()) &
                (grp["outcome"].isin(["admission", "transferred"]))
            ]["boarding_wait"]

            if len(sub) > 0:
                rep_means.append(sub.mean())

        sev_means.append(float(np.mean(rep_means)) if rep_means else 0.0)
        sev_stds.append(float(np.std(rep_means)) if rep_means else 0.0)

    return sev_means, sev_stds


def paired_ttest(a: np.ndarray, b: np.ndarray):
    """
    Paired t-test between two replication-level metric vectors.
    """
    return stats.ttest_rel(a, b)


def run_comparison():
    print("Loading simulation summaries...")

    baseline = convert_time_cols_to_minutes(pd.read_csv(BASELINE_SUMMARY))
    ai       = convert_time_cols_to_minutes(pd.read_csv(AI_SUMMARY))
    ml       = convert_time_cols_to_minutes(pd.read_csv(ML_SUMMARY))

    b_comp = compute_compliance(BASELINE_LOG).values
    a_comp = compute_compliance(AI_LOG).values
    m_comp = compute_compliance(ML_LOG).values

    print(f"  Baseline replications:   {len(baseline)}")
    print(f"  Rule-based replications: {len(ai)}")
    print(f"  Hybrid ML replications:  {len(ml)}")
    print()

    rows = []

    # ----------------------------------------------------------
    # TIME METRICS
    # ----------------------------------------------------------
    time_metrics = {
        "assessment_wait_mean": "Assessment Wait Mean (min)",
        "assessment_wait_p95":  "Assessment Wait p95 (min)",
        "boarding_wait_mean":   "Boarding Wait Mean (min)",
        "boarding_wait_p95":    "Boarding Wait p95 (min)",
        "total_los_mean":       "Total LOS Mean (min)",
        "total_los_p95":        "Total LOS p95 (min)",
    }

    for col, label in time_metrics.items():
        if col not in baseline.columns:
            continue

        b_vals = baseline[col].values.astype(float)
        a_vals = ai[col].values.astype(float) if col in ai.columns else np.zeros(len(b_vals))
        m_vals = ml[col].values.astype(float) if col in ml.columns else np.zeros(len(b_vals))

        _, p_ba = paired_ttest(b_vals, a_vals)
        _, p_bm = paired_ttest(b_vals, m_vals)
        _, p_am = paired_ttest(a_vals, m_vals)

        rows.append({
            "Metric":                  label,
            "Baseline Mean":           f"{b_vals.mean():.2f}",
            "Baseline Std":            f"{b_vals.std():.2f}",
            "Rule-Based Mean":         f"{a_vals.mean():.2f}",
            "Rule-Based Std":          f"{a_vals.std():.2f}",
            "Hybrid ML Mean":          f"{m_vals.mean():.2f}",
            "Hybrid ML Std":           f"{m_vals.std():.2f}",
            "Δ (Base→Rule)":           f"{a_vals.mean() - b_vals.mean():+.2f}",
            "Δ (Base→Hybrid)":         f"{m_vals.mean() - b_vals.mean():+.2f}",
            "p (Base vs Rule)":        f"{p_ba:.4f}",
            "p (Base vs Hybrid)":      f"{p_bm:.4f}",
            "p (Rule vs Hybrid)":      f"{p_am:.4f}",
            "Sig Base→Rule (p<.05)":   "Yes" if p_ba < 0.05 else "No",
            "Sig Base→Hybrid (p<.05)": "Yes" if p_bm < 0.05 else "No",
            "Sig Rule→Hybrid (p<.05)": "Yes" if p_am < 0.05 else "No",
        })

    # ----------------------------------------------------------
    # NHS 4-HOUR COMPLIANCE
    # ----------------------------------------------------------
    _, p_ba_c = paired_ttest(b_comp, a_comp)
    _, p_bm_c = paired_ttest(b_comp, m_comp)
    _, p_am_c = paired_ttest(a_comp, m_comp)

    rows.append({
        "Metric":                  "NHS 4-Hour Compliance (%)",
        "Baseline Mean":           f"{b_comp.mean():.2f}",
        "Baseline Std":            f"{b_comp.std():.2f}",
        "Rule-Based Mean":         f"{a_comp.mean():.2f}",
        "Rule-Based Std":          f"{a_comp.std():.2f}",
        "Hybrid ML Mean":          f"{m_comp.mean():.2f}",
        "Hybrid ML Std":           f"{m_comp.std():.2f}",
        "Δ (Base→Rule)":           f"{a_comp.mean() - b_comp.mean():+.2f}",
        "Δ (Base→Hybrid)":         f"{m_comp.mean() - b_comp.mean():+.2f}",
        "p (Base vs Rule)":        f"{p_ba_c:.4f}",
        "p (Base vs Hybrid)":      f"{p_bm_c:.4f}",
        "p (Rule vs Hybrid)":      f"{p_am_c:.4f}",
        "Sig Base→Rule (p<.05)":   "Yes" if p_ba_c < 0.05 else "No",
        "Sig Base→Hybrid (p<.05)": "Yes" if p_bm_c < 0.05 else "No",
        "Sig Rule→Hybrid (p<.05)": "Yes" if p_am_c < 0.05 else "No",
    })

    # ----------------------------------------------------------
    # COUNT METRICS
    # ----------------------------------------------------------
    count_metrics = {
        "n_patients":    "Patients per Year",
        "n_discharge":   "Discharged",
        "n_admission":   "Admitted",
        "n_transferred": "Transferred",
    }

    for col, label in count_metrics.items():
        if col not in baseline.columns:
            continue

        b_vals = baseline[col].values.astype(float)
        a_vals = ai[col].values.astype(float) if col in ai.columns else np.zeros(len(b_vals))
        m_vals = ml[col].values.astype(float) if col in ml.columns else np.zeros(len(b_vals))

        _, p_ba = paired_ttest(b_vals, a_vals)
        _, p_bm = paired_ttest(b_vals, m_vals)
        _, p_am = paired_ttest(a_vals, m_vals)

        rows.append({
            "Metric":                  label,
            "Baseline Mean":           f"{b_vals.mean():.1f}",
            "Baseline Std":            f"{b_vals.std():.1f}",
            "Rule-Based Mean":         f"{a_vals.mean():.1f}",
            "Rule-Based Std":          f"{a_vals.std():.1f}",
            "Hybrid ML Mean":          f"{m_vals.mean():.1f}",
            "Hybrid ML Std":           f"{m_vals.std():.1f}",
            "Δ (Base→Rule)":           f"{a_vals.mean() - b_vals.mean():+.1f}",
            "Δ (Base→Hybrid)":         f"{m_vals.mean() - b_vals.mean():+.1f}",
            "p (Base vs Rule)":        f"{p_ba:.4f}",
            "p (Base vs Hybrid)":      f"{p_bm:.4f}",
            "p (Rule vs Hybrid)":      f"{p_am:.4f}",
            "Sig Base→Rule (p<.05)":   "Yes" if p_ba < 0.05 else "No",
            "Sig Base→Hybrid (p<.05)": "Yes" if p_bm < 0.05 else "No",
            "Sig Rule→Hybrid (p<.05)": "Yes" if p_am < 0.05 else "No",
        })

    # ----------------------------------------------------------
    # BOARDING WAIT BY SEVERITY
    # Baseline uses actual per-severity means from patient log.
    # Rule-based and Hybrid use summary outputs.
    # ----------------------------------------------------------
    severities = ["critical", "high", "medium", "low"]

    print("Computing per-severity boarding waits from patient logs...")
    b_sev_means, b_sev_stds = per_severity_boarding(BASELINE_LOG, severities)

    for i, sev in enumerate(severities):
        col = f"boarding_wait_{sev}_mean"

        a_vals = ai[col].values.astype(float) if col in ai.columns else np.zeros(10)
        m_vals = ml[col].values.astype(float) if col in ml.columns else np.zeros(10)

        rows.append({
            "Metric":                  f"Boarding Wait — {sev.title()} (min)",
            "Baseline Mean":           f"{b_sev_means[i]:.1f}",
            "Baseline Std":            f"{b_sev_stds[i]:.1f}",
            "Rule-Based Mean":         f"{a_vals.mean():.2f}",
            "Rule-Based Std":          f"{a_vals.std():.2f}",
            "Hybrid ML Mean":          f"{m_vals.mean():.2f}",
            "Hybrid ML Std":           f"{m_vals.std():.2f}",
            "Δ (Base→Rule)":           f"{a_vals.mean() - b_sev_means[i]:+.1f}",
            "Δ (Base→Hybrid)":         f"{m_vals.mean() - b_sev_means[i]:+.1f}",
            "p (Base vs Rule)":        "—",
            "p (Base vs Hybrid)":      "—",
            "p (Rule vs Hybrid)":      "—",
            "Sig Base→Rule (p<.05)":   "—",
            "Sig Base→Hybrid (p<.05)": "—",
            "Sig Rule→Hybrid (p<.05)": "—",
        })

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(COMPARISON_OUT, index=False)

    comp_df = pd.DataFrame({
        "replication":         list(range(1, len(b_comp) + 1)),
        "baseline_pct":        b_comp,
        "rule_based_pct":      a_comp,
        "hybrid_ml_pct":       m_comp,
        "improvement_rule":    a_comp - b_comp,
        "improvement_hybrid":  m_comp - b_comp,
    })
    comp_df.to_csv(COMPLIANCE_OUT, index=False)

    # ----------------------------------------------------------
    # PRINT
    # ----------------------------------------------------------
    print(f"\n{'='*88}")
    print("  THREE-WAY COMPARISON: BASELINE vs RULE-BASED vs HYBRID ML MODEL")
    print(f"{'='*88}")
    print(
        f"\n  {'Metric':<32} {'Baseline':>10} {'Rule-Based':>12} {'Hybrid ML':>12}"
        f"  {'p(B→R)':>8}  {'p(B→H)':>8}  {'p(R→H)':>8}"
    )
    print(
        f"  {'-'*32} {'-'*10} {'-'*12} {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}"
    )

    for row in rows:
        if row["p (Base vs Rule)"] == "—":
            print(
                f"  {row['Metric']:<32} "
                f"{row['Baseline Mean']:>10} "
                f"{row['Rule-Based Mean']:>12} "
                f"{row['Hybrid ML Mean']:>12}  "
                f"{'—':>8}  {'—':>8}  {'—':>8}"
            )
            continue

        sig_r = "*" if row["Sig Base→Rule (p<.05)"] == "Yes" else " "
        sig_h = "*" if row["Sig Base→Hybrid (p<.05)"] == "Yes" else " "
        sig_a = "*" if row["Sig Rule→Hybrid (p<.05)"] == "Yes" else " "

        print(
            f"  {row['Metric']:<32} "
            f"{row['Baseline Mean']:>10} "
            f"{row['Rule-Based Mean']:>12} "
            f"{row['Hybrid ML Mean']:>12}  "
            f"{row['p (Base vs Rule)']:>7}{sig_r}  "
            f"{row['p (Base vs Hybrid)']:>7}{sig_h}  "
            f"{row['p (Rule vs Hybrid)']:>7}{sig_a}"
        )

    print("\n  * = statistically significant (paired t-test, p<0.05, n=10)")

    print("\n  NHS 4-Hour Compliance Summary:")
    print(f"    Baseline:    {b_comp.mean():.1f}% ± {b_comp.std():.1f}%")
    print(
        f"    Rule-Based:  {a_comp.mean():.1f}% ± {a_comp.std():.1f}%  "
        f"({a_comp.mean()-b_comp.mean():+.1f}pp, p={p_ba_c:.4f})"
    )
    print(
        f"    Hybrid ML:   {m_comp.mean():.1f}% ± {m_comp.std():.1f}%  "
        f"({m_comp.mean()-b_comp.mean():+.1f}pp, p={p_bm_c:.4f})"
    )
    print(
        f"    Rule vs Hybrid: p={p_am_c:.4f} "
        f"({'significant' if p_am_c < 0.05 else 'not significant'})"
    )

    print(f"\n  Comparison table → {COMPARISON_OUT}")
    print(f"  Compliance data  → {COMPLIANCE_OUT}")

    return comparison_df


if __name__ == "__main__":
    run_comparison()