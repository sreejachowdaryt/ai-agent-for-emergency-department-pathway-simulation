# generate_baseline_plots.py
"""
Baseline ED Simulation — Figure Generation Script
===================================================
Generates all dissertation figures from the baseline simulation outputs.

Reads:
    ../data/simulation_summary.csv    — per-replication aggregate metrics
    ../data/simulation_patient_log.csv — per-patient records (all replications)

Saves all figures to:
    ../figures/                       — alongside process_tree.png and dfg.png

Figures produced:
    Fig 1 — baseline_waiting_times.png
             Bar chart: mean waiting time by stage (triage / doctor / boarding)
             with error bars showing std across replications.

    Fig 2 — baseline_los_boxplot.png
             Box plot: total LOS distribution per replication.
             Shows central tendency, spread, and outlier tail.

    Fig 3 — baseline_nhs_compliance.png
             Line chart: NHS 4-hour target compliance per replication.
             Reference line at 95% target.

    Fig 4 — baseline_boarding_histogram.png
             Histogram: boarding wait distribution for non-discharge patients
             across all replications combined.

    Fig 5 — baseline_boarding_variance.png
             Scatter + line: boarding wait mean per replication.
             Illustrates inter-replication variance.

    Fig 6 — baseline_outcome_distribution.png
             Grouped bar chart: outcome counts per replication
             (discharge / admission / ICU).

    Fig 7 — baseline_summary_heatmap.png
             Heatmap: key metrics across all 10 replications.
             Provides a compact visual summary for the dissertation.

Usage:
    python generate_baseline_plots.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

SUMMARY_PATH    = "../data/simulation_summary.csv"
PATIENT_PATH    = "../data/simulation_patient_log.csv"
FIGURES_DIR     = "../figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------
# STYLE CONSTANTS
# ----------------------------------------------------------

# Consistent colour palette across all figures
C_TRIAGE   = "#2E86AB"   # blue
C_DOCTOR   = "#A23B72"   # purple
C_BOARDING = "#F18F01"   # amber
C_DISCHARGE = "#44BBA4"  # teal
C_ADMISSION = "#E94F37"  # red
C_ICU       = "#393E41"  # dark grey
C_NHS       = "#D62828"  # NHS red for target lines
C_GRID      = "#E8E8E8"

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_AXIS   = {"fontsize": 11}
FONT_TICK   = 10
FONT_ANNOT  = 9

DPI         = 150
FIG_W       = 9
FIG_H       = 5

def style_axes(ax):
    """Apply consistent styling to an axes object."""
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(labelsize=FONT_TICK)

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

print("Loading simulation outputs...")

if not os.path.exists(SUMMARY_PATH):
    print(f"ERROR: {SUMMARY_PATH} not found. Run ed_simulation.py first.")
    sys.exit(1)

if not os.path.exists(PATIENT_PATH):
    print(f"ERROR: {PATIENT_PATH} not found. Run ed_simulation.py first.")
    sys.exit(1)

summary = pd.read_csv(SUMMARY_PATH)
patients = pd.read_csv(PATIENT_PATH)

print(f"  Summary:  {len(summary)} replications")
print(f"  Patients: {len(patients):,} records")
print()

# Convert time metrics from hours to minutes in summary
time_cols = [c for c in summary.columns if any(x in c for x in ["wait", "los"])]
for col in time_cols:
    if summary[col].mean() < 100:   # still in hours (< 100 means hours not minutes)
        summary[col] = summary[col] * 60

# Convert patient log times to minutes
for col in ["triage_wait", "doctor_wait", "boarding_wait", "total_los"]:
    if col in patients.columns:
        if patients[col].dropna().mean() < 100:
            patients[col] = patients[col] * 60

# Compute NHS 4-hour compliance per replication from patient log
# (more accurate than summary if not already computed)
if "nhs_compliance" not in summary.columns:
    comp = (
        patients[patients["total_los"].notna()]
        .groupby("replication")
        .apply(lambda g: (g["total_los"] <= 240).mean() * 100)
        .reset_index(name="nhs_compliance")
    )
    summary = summary.merge(comp, on="replication", how="left")

reps = summary["replication"].values

# ============================================================
# FIGURE 1 — WAITING TIME BY STAGE (Bar chart with error bars)
# ============================================================

print("Generating Figure 1: Waiting time by stage...")

stages      = ["Triage Wait", "Doctor Wait", "Boarding Wait"]
means       = [
    summary["triage_wait_mean"].mean(),
    summary["doctor_wait_mean"].mean(),
    summary["boarding_wait_mean"].mean(),
]
stds        = [
    summary["triage_wait_mean"].std(),
    summary["doctor_wait_mean"].std(),
    summary["boarding_wait_mean"].std(),
]
colours     = [C_TRIAGE, C_DOCTOR, C_BOARDING]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
style_axes(ax)

bars = ax.bar(stages, means, color=colours, width=0.5,
              yerr=stds, capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#555555"},
              zorder=3, edgecolor="white", linewidth=0.5)

# Annotate bar tops
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 8,
            f"{mean:.1f} min",
            ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold")

ax.set_title("Mean Waiting Time by Pathway Stage\n(Baseline Simulation, 10 Replications)",
             **FONT_TITLE)
ax.set_ylabel("Mean Waiting Time (minutes)", **FONT_AXIS)
ax.set_xlabel("Pathway Stage", **FONT_AXIS)

# Add NHS triage target reference line (15 min)
ax.axhline(y=15, color=C_NHS, linestyle="--", linewidth=1.2, zorder=2,
           label="NHS triage target (15 min)")
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

# Note about boarding scale
ax.annotate("Note: boarding wait axis includes\nnon-discharge patients only",
            xy=(0.98, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color="#666666",
            style="italic")

fig.tight_layout()
save_fig(fig, "baseline_waiting_times.png")

# ============================================================
# FIGURE 2 — TOTAL LOS BOXPLOT (per replication)
# ============================================================

print("Generating Figure 2: Total LOS box plot...")

los_by_rep = [
    patients[patients["replication"] == r]["total_los"].dropna().values
    for r in sorted(patients["replication"].unique())
]

fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H))
style_axes(ax)

bp = ax.boxplot(los_by_rep,
                patch_artist=True,
                notch=False,
                showfliers=True,
                flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                markerfacecolor="#AAAAAA", markeredgecolor="none"),
                medianprops=dict(color="#D62828", linewidth=2),
                whiskerprops=dict(color="#555555", linewidth=1),
                capprops=dict(color="#555555", linewidth=1.5),
                boxprops=dict(linewidth=0))

for patch in bp["boxes"]:
    patch.set_facecolor("#2E86AB")
    patch.set_alpha(0.7)

# NHS 4-hour line
ax.axhline(y=240, color=C_NHS, linestyle="--", linewidth=1.5,
           label="NHS 4-hour target (240 min)", zorder=4)

ax.set_title("Total ED Length of Stay Distribution by Replication\n(Baseline Simulation)",
             **FONT_TITLE)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("Total ED LOS (minutes)", **FONT_AXIS)
ax.set_xticklabels([f"Rep {r}" for r in sorted(patients["replication"].unique())],
                   rotation=30, ha="right", fontsize=9)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

# Annotate median across all reps
all_los = patients["total_los"].dropna()
ax.text(0.02, 0.97,
        f"Overall median: {all_los.median():.0f} min\n"
        f"Overall mean: {all_los.mean():.0f} min\n"
        f"p95: {all_los.quantile(0.95):.0f} min",
        transform=ax.transAxes, va="top", fontsize=FONT_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9))

fig.tight_layout()
save_fig(fig, "baseline_los_boxplot.png")

# ============================================================
# FIGURE 3 — NHS 4-HOUR COMPLIANCE (Line chart per replication)
# ============================================================

print("Generating Figure 3: NHS 4-hour compliance...")

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
style_axes(ax)

compliance = summary["nhs_compliance"].values

ax.plot(reps, compliance, marker="o", color=C_TRIAGE, linewidth=2,
        markersize=8, zorder=3, label="4-hour compliance (%)")

# Shade between line and target
ax.fill_between(reps, compliance, 95, where=(compliance < 95),
                alpha=0.15, color=C_NHS, label="Gap to NHS target")

# NHS 95% target
ax.axhline(y=95, color=C_NHS, linestyle="--", linewidth=1.5,
           label="NHS target (95%)", zorder=2)

# Mean line
mean_comp = compliance.mean()
ax.axhline(y=mean_comp, color=C_DOCTOR, linestyle=":", linewidth=1.5,
           label=f"Mean compliance ({mean_comp:.1f}%)", zorder=2)

# Annotate each point
for r, c in zip(reps, compliance):
    ax.text(r, c + 0.4, f"{c:.1f}%", ha="center", va="bottom",
            fontsize=FONT_ANNOT, color=C_TRIAGE, fontweight="bold")

ax.set_title("NHS 4-Hour Target Compliance per Replication\n(Baseline Simulation)",
             **FONT_TITLE)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("4-Hour Compliance (%)", **FONT_AXIS)
ax.set_xticks(reps)
ax.set_ylim(60, 100)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9, loc="lower right")

fig.tight_layout()
save_fig(fig, "baseline_nhs_compliance.png")

# ============================================================
# FIGURE 4 — BOARDING WAIT HISTOGRAM
# ============================================================

print("Generating Figure 4: Boarding wait histogram...")

boarding = patients[
    (patients["outcome"].isin(["admission", "icu"])) &
    (patients["boarding_wait"].notna()) &
    (patients["boarding_wait"] > 0)
]["boarding_wait"]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
style_axes(ax)

n, bins, patches = ax.hist(boarding, bins=60, color=C_BOARDING,
                            edgecolor="white", linewidth=0.3,
                            alpha=0.85, zorder=3)

# Colour bars beyond 4-hour threshold red
four_hour = 240
for patch, left in zip(patches, bins[:-1]):
    if left >= four_hour:
        patch.set_facecolor(C_NHS)
        patch.set_alpha(0.7)

# Reference lines
ax.axvline(x=four_hour, color=C_NHS, linestyle="--", linewidth=1.5,
           label=f"4-hour mark (240 min)", zorder=4)
ax.axvline(x=boarding.mean(), color="#333333", linestyle="-", linewidth=1.5,
           label=f"Mean ({boarding.mean():.0f} min)", zorder=4)
ax.axvline(x=boarding.median(), color=C_DOCTOR, linestyle="-.", linewidth=1.5,
           label=f"Median ({boarding.median():.0f} min)", zorder=4)

ax.set_title("Boarding Wait Time Distribution\n(Non-Discharge Patients, All Replications Combined)",
             **FONT_TITLE)
ax.set_xlabel("Boarding Wait Time (minutes)", **FONT_AXIS)
ax.set_ylabel("Number of Patients", **FONT_AXIS)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

# Stats box
ax.text(0.97, 0.97,
        f"n = {len(boarding):,}\n"
        f"Mean: {boarding.mean():.0f} min\n"
        f"Median: {boarding.median():.0f} min\n"
        f"p95: {boarding.quantile(0.95):.0f} min\n"
        f"Max: {boarding.max():.0f} min",
        transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9))

fig.tight_layout()
save_fig(fig, "baseline_boarding_histogram.png")

# ============================================================
# FIGURE 5 — BOARDING WAIT VARIANCE (per replication)
# ============================================================

print("Generating Figure 5: Boarding wait variance across replications...")

board_means = summary["boarding_wait_mean"].values
board_p95   = summary["boarding_wait_p95"].values

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
style_axes(ax)

# P95 shaded area
ax.fill_between(reps, board_means, board_p95,
                alpha=0.15, color=C_BOARDING, label="Mean to p95 range")

# P95 line
ax.plot(reps, board_p95, marker="s", color=C_BOARDING, linewidth=1.5,
        markersize=6, linestyle="--", alpha=0.7, label="p95 boarding wait")

# Mean line
ax.plot(reps, board_means, marker="o", color=C_BOARDING, linewidth=2.5,
        markersize=8, zorder=3, label="Mean boarding wait")

# Overall mean reference
overall_mean = board_means.mean()
ax.axhline(y=overall_mean, color="#555555", linestyle=":",
           linewidth=1.2, zorder=2,
           label=f"Grand mean ({overall_mean:.0f} min)")

# 4-hour reference
ax.axhline(y=240, color=C_NHS, linestyle="--", linewidth=1.2,
           zorder=2, label="4-hour mark (240 min)")

# Annotate mean points
for r, m in zip(reps, board_means):
    ax.text(r, m + 15, f"{m:.0f}", ha="center", va="bottom",
            fontsize=8, color=C_BOARDING, fontweight="bold")

ax.set_title("Boarding Wait: Mean and p95 per Replication\n(Baseline Simulation — Illustrates Inter-Replication Variance)",
             **FONT_TITLE)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("Boarding Wait Time (minutes)", **FONT_AXIS)
ax.set_xticks(reps)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9, loc="upper right")

fig.tight_layout()
save_fig(fig, "baseline_boarding_variance.png")

# ============================================================
# FIGURE 6 — OUTCOME DISTRIBUTION (per replication)
# ============================================================

print("Generating Figure 6: Outcome distribution per replication...")

x       = np.arange(len(reps))
width   = 0.25

fig, ax = plt.subplots(figsize=(FIG_W + 1, FIG_H))
style_axes(ax)

b1 = ax.bar(x - width, summary["n_discharge"], width, label="Discharge",
            color=C_DISCHARGE, zorder=3, edgecolor="white")
b2 = ax.bar(x,          summary["n_admission"], width, label="Admission (Ward)",
            color=C_ADMISSION, zorder=3, edgecolor="white")
b3 = ax.bar(x + width,  summary["n_icu"],       width, label="ICU",
            color=C_ICU, zorder=3, edgecolor="white")

# Mean lines
ax.axhline(y=summary["n_discharge"].mean(), color=C_DISCHARGE,
           linestyle="--", linewidth=1, alpha=0.8)
ax.axhline(y=summary["n_admission"].mean(), color=C_ADMISSION,
           linestyle="--", linewidth=1, alpha=0.8)
ax.axhline(y=summary["n_icu"].mean(), color=C_ICU,
           linestyle="--", linewidth=1, alpha=0.8)

ax.set_title("Patient Outcome Distribution per Replication\n(Baseline Simulation)",
             **FONT_TITLE)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("Number of Patients", **FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels([f"Rep {r}" for r in reps], rotation=30, ha="right", fontsize=9)
ax.legend(fontsize=FONT_ANNOT, framealpha=0.9)

# Summary text
total_mean = summary["n_patients"].mean()
ax.text(0.98, 0.97,
        f"Mean total/rep: {total_mean:,.0f}\n"
        f"Discharge: {summary['n_discharge'].mean()/total_mean*100:.1f}%\n"
        f"Admission: {summary['n_admission'].mean()/total_mean*100:.1f}%\n"
        f"ICU: {summary['n_icu'].mean()/total_mean*100:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=FONT_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9))

fig.tight_layout()
save_fig(fig, "baseline_outcome_distribution.png")

# ============================================================
# FIGURE 7 — SUMMARY HEATMAP (metrics × replications)
# ============================================================

print("Generating Figure 7: Summary heatmap...")

# Select the key numeric metrics for the heatmap
heatmap_cols = {
    "Triage Wait\nMean (min)":    "triage_wait_mean",
    "Doctor Wait\nMean (min)":    "doctor_wait_mean",
    "Boarding Wait\nMean (min)":  "boarding_wait_mean",
    "Boarding Wait\np95 (min)":   "boarding_wait_p95",
    "Total LOS\nMean (min)":      "total_los_mean",
    "Total LOS\np95 (min)":       "total_los_p95",
    "NHS 4-hr\nCompliance (%)":   "nhs_compliance",
}

labels  = list(heatmap_cols.keys())
data    = np.array([summary[col].values for col in heatmap_cols.values()])

# Normalise each row (metric) independently for colour scaling
data_norm = np.zeros_like(data, dtype=float)
for i in range(data.shape[0]):
    row_min, row_max = data[i].min(), data[i].max()
    if row_max > row_min:
        data_norm[i] = (data[i] - row_min) / (row_max - row_min)
    else:
        data_norm[i] = 0.5

fig, ax = plt.subplots(figsize=(FIG_W + 2, len(labels) * 0.7 + 1.5))

im = ax.imshow(data_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

# Annotate cells with actual values
for i in range(len(labels)):
    for j in range(len(reps)):
        val = data[i, j]
        text_color = "white" if data_norm[i, j] > 0.6 else "#333333"
        label_str = f"{val:.1f}%" if "Compliance" in labels[i] else f"{val:.0f}"
        ax.text(j, i, label_str, ha="center", va="center",
                fontsize=8.5, color=text_color, fontweight="bold")

ax.set_xticks(np.arange(len(reps)))
ax.set_xticklabels([f"Rep {r}" for r in reps], fontsize=9)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=9)

ax.set_title("Baseline Simulation — Key Metrics Across All Replications\n(Colour intensity = relative value within each metric row)",
             **FONT_TITLE)

# Add grid lines between cells
ax.set_xticks(np.arange(-0.5, len(reps), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

fig.tight_layout()
save_fig(fig, "baseline_summary_heatmap.png")

# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 55)
print("  All figures generated successfully")
print("=" * 55)
print(f"  Output directory: {os.path.abspath(FIGURES_DIR)}")
print()
print("  Figures produced:")
figs = [
    ("baseline_waiting_times.png",      "Fig 1 — Waiting time by stage"),
    ("baseline_los_boxplot.png",         "Fig 2 — Total LOS box plot"),
    ("baseline_nhs_compliance.png",      "Fig 3 — NHS 4-hour compliance"),
    ("baseline_boarding_histogram.png",  "Fig 4 — Boarding wait histogram"),
    ("baseline_boarding_variance.png",   "Fig 5 — Boarding wait variance"),
    ("baseline_outcome_distribution.png","Fig 6 — Outcome distribution"),
    ("baseline_summary_heatmap.png",     "Fig 7 — Summary heatmap"),
]
for fname, desc in figs:
    full = os.path.join(FIGURES_DIR, fname)
    status = "✓" if os.path.exists(full) else "✗ MISSING"
    print(f"  {status}  {desc}")
    print(f"       → {full}")
print()
print("  Insert figures into dissertation chapters 3 and 4")
print("  at the [Figure X.X — Insert ...] placeholder locations.")
print("=" * 55)