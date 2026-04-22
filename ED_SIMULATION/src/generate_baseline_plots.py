# generate_baseline_plots.py
"""
Baseline ED Simulation — Figure Generation Script
Generates three dissertation figures from the baseline simulation for Chapter 4.

Updated for final dataset-driven baseline model:
  - assessment_wait_mean used instead of separate triage/doctor waits
  - outcomes are discharge / admission / transferred
  - third figure changed from boarding-wait histogram to total LOS histogram
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_PATH = "../data/simulation_summary.csv"
PATIENT_PATH = "../data/simulation_patient_log.csv"
FIGURES_DIR  = "../figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

C_ASSESS   = "#2E86AB"
C_BOARDING = "#F18F01"
C_NHS      = "#D62828"
C_GRID     = "#E8E8E8"
C_PURPLE   = "#A23B72"
C_LOS      = "#4CAF50"

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_AXIS  = {"fontsize": 11}
FONT_ANNOT = 9
DPI        = 150


def style_axes(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(labelsize=10)


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


print("Loading simulation outputs...")

if not os.path.exists(SUMMARY_PATH):
    print(f"ERROR: {SUMMARY_PATH} not found. Run ed_simulation.py first.")
    sys.exit(1)

if not os.path.exists(PATIENT_PATH):
    print(f"ERROR: {PATIENT_PATH} not found. Run ed_simulation.py first.")
    sys.exit(1)

summary  = pd.read_csv(SUMMARY_PATH)
patients = pd.read_csv(PATIENT_PATH)

print(f"  Summary:  {len(summary)} replications")
print(f"  Patients: {len(patients):,} records")

# Convert hours → minutes where needed
for col in summary.columns:
    if any(x in col for x in ["wait", "los"]) and summary[col].dropna().mean() < 100:
        summary[col] = summary[col] * 60

for col in ["assessment_wait", "boarding_wait", "total_los"]:
    if col in patients.columns and patients[col].dropna().mean() < 100:
        patients[col] = patients[col] * 60

# NHS compliance per replication
if "nhs_compliance" not in summary.columns:
    comp = (
        patients[patients["total_los"].notna()]
        .groupby("replication")
        .apply(lambda g: (g["total_los"] <= 240).mean() * 100)
        .reset_index(name="nhs_compliance")
    )
    summary = summary.merge(comp, on="replication", how="left")

reps = summary["replication"].values
print()

# ============================================================
# FIGURE 1 — WAITING TIME BY STAGE
# ============================================================

print("Generating Figure 1: Waiting time by stage...")

stages  = ["Assessment Wait\n(triage + doctor combined)",
           "Boarding Wait\n(non-discharge patients)"]
means   = [summary["assessment_wait_mean"].mean(),
           summary["boarding_wait_mean"].mean()]
stds    = [summary["assessment_wait_mean"].std(),
           summary["boarding_wait_mean"].std()]
colours = [C_ASSESS, C_BOARDING]

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

bars = ax.bar(
    stages, means, color=colours, width=0.45,
    yerr=stds, capsize=6,
    error_kw={"elinewidth": 1.5, "ecolor": "#555555"},
    zorder=3, edgecolor="white", linewidth=0.5
)

for bar, mean, std in zip(bars, means, stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 8,
        f"{mean:.1f} min",
        ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold"
    )

ax.axhline(
    y=15, color=C_NHS, linestyle="--", linewidth=1.2, zorder=2,
    label="NHS 15-min benchmark for Assessment Wait"
)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

ax.set_title(
    "Mean Waiting Time Analysis by Pathway Stage\n(Baseline Simulation, 10 Replications)",
    **FONT_TITLE
)
ax.set_ylabel("Mean Waiting Time (minutes)", **FONT_AXIS)
ax.set_xlabel("Pathway Stage", **FONT_AXIS)

ax.annotate(
    "Boarding wait applies only to\nadmitted/transferred patients",
    xy=(0.98, 0.97), xycoords="axes fraction",
    ha="right", va="top", fontsize=8, color="#666666", style="italic"
)

fig.tight_layout()
save_fig(fig, "baseline_waiting_times.png")

# ============================================================
# FIGURE 2 — NHS 4-HOUR COMPLIANCE
# ============================================================

print("Generating Figure 2: NHS 4-hour compliance...")

compliance = summary["nhs_compliance"].values
mean_comp  = compliance.mean()

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

ax.plot(
    reps, compliance, marker="o", color=C_ASSESS, linewidth=2,
    markersize=8, zorder=3, label="4-hour compliance (%)"
)
ax.fill_between(
    reps, compliance, 95, where=(compliance < 95),
    alpha=0.15, color=C_NHS, label="Gap to NHS target"
)
ax.axhline(
    y=95, color=C_NHS, linestyle="--", linewidth=1.5,
    label="NHS target (95%)", zorder=2
)
ax.axhline(
    y=mean_comp, color=C_PURPLE, linestyle=":", linewidth=1.5,
    label=f"Mean compliance ({mean_comp:.1f}%)", zorder=2
)

for r, c in zip(reps, compliance):
    ax.text(
        r, c + 0.4, f"{c:.1f}%", ha="center", va="bottom",
        fontsize=FONT_ANNOT, color=C_ASSESS, fontweight="bold"
    )

ax.set_title(
    "NHS 4-Hour Target Compliance per Replication\n(Baseline Simulation)",
    **FONT_TITLE
)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("4-Hour Compliance (%)", **FONT_AXIS)
ax.set_xticks(reps)
ax.set_ylim(60, 100)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9, loc="lower right")

fig.tight_layout()
save_fig(fig, "baseline_nhs_compliance.png")

# ============================================================
# FIGURE 3 — TOTAL LOS HISTOGRAM (Replication 1, trimmed)
# ============================================================

print("Generating Figure 3: Total LOS histogram (trimmed)...")

los = patients[
    (patients["replication"] == 1) &
    (patients["total_los"].notna()) &
    (patients["total_los"] > 0)
]["total_los"]

# Trim extreme outliers (use p99 or p95)
p99 = los.quantile(0.99)
los_trimmed = los[los <= p99]

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

ax.hist(
    los_trimmed, bins=50,
    edgecolor="white", linewidth=0.3,
    alpha=0.85, zorder=3
)

# NHS threshold
ax.axvline(
    x=240, color=C_NHS, linestyle="--", linewidth=1.5,
    label="NHS 4-hour threshold (240 min)", zorder=4
)

# Mean / median from FULL data (important)
ax.axvline(
    x=los.mean(), color="#333333", linestyle="-", linewidth=1.5,
    label=f"Mean ({los.mean():.0f} min)", zorder=4
)
ax.axvline(
    x=los.median(), color=C_PURPLE, linestyle="-.", linewidth=1.5,
    label=f"Median ({los.median():.0f} min)", zorder=4
)

ax.set_xlim(0, p99)  # 👈 KEY FIX

ax.set_title(
    "Total ED Length of Stay Distribution — Baseline (Replication 1)\n",
    **FONT_TITLE
)
ax.set_xlabel("Total Length of Stay (minutes)", **FONT_AXIS)
ax.set_ylabel("Number of Patients", **FONT_AXIS)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

ax.text(
    0.97, 0.97,
    f"n = {len(los):,}\n"
    f"Displayed ≤ p99 ({p99:.0f} min)\n"
    f"Mean: {los.mean():.0f} min\n"
    f"Median: {los.median():.0f} min\n"
    f"p95: {los.quantile(0.95):.0f} min",
    transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#CCCCCC", alpha=0.9)
)

fig.tight_layout()
save_fig(fig, "baseline_total_los_histogram.png")

print()
print("=" * 50)
print("  Baseline figures generated (3 of 3)")
print("=" * 50)
for fname, desc in [
    ("baseline_waiting_times.png",       "Fig 1 — Waiting time by stage      → Chapter 4"),
    ("baseline_nhs_compliance.png",      "Fig 2 — NHS 4-hour compliance      → Chapter 4"),
    ("baseline_total_los_histogram.png", "Fig 3 — Total LOS distribution     → Chapter 4"),
]:
    status = "✓" if os.path.exists(os.path.join(FIGURES_DIR, fname)) else "✗ MISSING"
    print(f"  {status}  {desc}")
print("=" * 50)