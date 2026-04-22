# generate_comparison_plots.py
"""
Final Comparison Figures — Dataset-Driven Three-Way Simulation Comparison

Recommended figure set for dissertation:

1. Baseline vs Rule-Based:
   Boarding wait by severity
   -> demonstrates effect of severity-based priority + boarding escalation

2. Baseline vs Hybrid:
   Waiting time by stage
   -> demonstrates reduction in assessment bottleneck through POCT

3. All 3 models:
   NHS 4-hour compliance per replication
   -> system-level performance comparison

4. All 3 models:
   Summary comparison table
   -> compact overall comparison of key metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_SUMMARY = "../data/simulation_summary.csv"
AI_SUMMARY       = "../data/simulation_ai_summary.csv"
ML_SUMMARY       = "../data/simulation_ml_summary.csv"

BASELINE_LOG     = "../data/simulation_patient_log.csv"
AI_LOG           = "../data/simulation_ai_patient_log.csv"
ML_LOG           = "../data/simulation_ml_patient_log.csv"

FIGURES_DIR      = "../figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

C_BASELINE = "#2E86AB"
C_AI       = "#14A085"
C_ML       = "#E67E22"
C_NHS      = "#D62828"
C_GRID     = "#E8E8E8"
C_NAVY     = "#0A2342"
C_PURPLE   = "#A23B72"

FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_AXIS  = {"fontsize": 11}
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


def per_rep_compliance(log_path):
    df = pd.read_csv(log_path)

    # Convert hours -> minutes if needed
    if df["total_los"].dropna().mean() < 100:
        df["total_los"] = df["total_los"] * 60

    df["compliant"] = df["total_los"].apply(
        lambda x: 1 if pd.notna(x) and x <= 240 else 0
    )

    return (
        df.groupby("replication")
          .apply(lambda g: g["compliant"].mean() * 100)
          .values
    )


def per_severity_boarding_from_log(log_path, severities):
    df = pd.read_csv(log_path)

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


# ----------------------------------------------------------
# LOAD
# ----------------------------------------------------------

print("Loading simulation outputs...")

b = pd.read_csv(BASELINE_SUMMARY)
a = pd.read_csv(AI_SUMMARY)
m = pd.read_csv(ML_SUMMARY)

# Convert hours -> minutes for time columns
for df in [b, a, m]:
    for col in df.columns:
        if any(x in col for x in ["wait", "los"]) and df[col].dropna().mean() < 100:
            df[col] = df[col] * 60

b_comp = per_rep_compliance(BASELINE_LOG)
a_comp = per_rep_compliance(AI_LOG)
m_comp = per_rep_compliance(ML_LOG)
reps   = list(range(1, len(b_comp) + 1))

print(f"  Baseline compliance:   {b_comp.mean():.1f}%")
print(f"  Rule-based compliance: {a_comp.mean():.1f}%")
print(f"  Hybrid compliance:     {m_comp.mean():.1f}%")
print()

# ============================================================
# FIGURE 1 — BASELINE vs RULE-BASED: BOARDING WAIT BY SEVERITY
# ============================================================

print("Fig 1: Baseline vs Rule-Based — boarding wait by severity...")

severities = ["critical", "high", "medium", "low"]
sev_labels = ["Critical", "High", "Medium", "Low"]

# Baseline from log (actual per-severity means under FIFO)
b_sev_means, b_sev_stds = per_severity_boarding_from_log(BASELINE_LOG, severities)

# Rule-based from summary
a_sev_means, a_sev_stds = [], []
for sev in severities:
    col = f"boarding_wait_{sev}_mean"
    a_sev_means.append(a[col].mean() if col in a.columns else 0.0)
    a_sev_stds.append(a[col].std() if col in a.columns else 0.0)

x = np.arange(len(severities))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
style_axes(ax)

bars_b = ax.bar(
    x - width/2, b_sev_means, width,
    label="Baseline FIFO (actual per-severity mean)",
    color=C_BASELINE, yerr=b_sev_stds, capsize=5,
    error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    zorder=3, edgecolor="white", alpha=0.85
)

bars_a = ax.bar(
    x + width/2, a_sev_means, width,
    label="Rule-Based Agent",
    color=C_AI, yerr=a_sev_stds, capsize=5,
    error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    zorder=3, edgecolor="white"
)

for bar, val, std in zip(bars_b, b_sev_means, b_sev_stds):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + std + 6,
        f"{val:.0f}",
        ha="center", va="bottom",
        fontsize=9.5, color=C_BASELINE, fontweight="bold"
    )

for bar, val, std in zip(bars_a, a_sev_means, a_sev_stds):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + std + 6,
        f"{val:.0f}",
        ha="center", va="bottom",
        fontsize=9.5, color=C_AI, fontweight="bold"
    )

# Add baseline overall mean as reference line
baseline_overall = b["boarding_wait_mean"].mean()
ax.axhline(
    y=baseline_overall,
    color=C_BASELINE,
    linestyle="--",
    linewidth=1.6,
    alpha=0.55,
    zorder=2,
    label=f"Baseline overall mean ({baseline_overall:.0f} min)"
)

ax.set_title(
    "Boarding Wait by Patient Severity — Baseline vs Rule-Based Agent\n"
    "(Rule-based priority protects higher-acuity patients at the boarding stage)",
    y=1.08,
    **FONT_TITLE
)

ax.set_xlabel("Patient Severity", **FONT_AXIS)
ax.set_ylabel("Mean Boarding Wait (minutes)", **FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(sev_labels, fontsize=11)

ax.legend(
    fontsize=9,
    framealpha=0.9,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=3
)

plt.subplots_adjust(top=0.82)

save_fig(fig, "comparison_boarding_by_severity.png")

# ============================================================
# FIGURE 2 — ALL 3 MODELS: ASSESSMENT WAIT COMPARISON
# ============================================================

print("Fig 2: All models — assessment wait comparison...")

labels = ["Baseline", "Rule-Based", "Hybrid ML + Rule-Based"]
means  = [
    b["assessment_wait_mean"].mean(),
    a["assessment_wait_mean"].mean(),
    m["assessment_wait_mean"].mean(),
]
stds   = [
    b["assessment_wait_mean"].std(),
    a["assessment_wait_mean"].std(),
    m["assessment_wait_mean"].std(),
]
colors = [C_BASELINE, C_AI, C_ML]

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

bars = ax.bar(
    x, means, width,
    color=colors,
    yerr=stds, capsize=6,
    error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    zorder=3, edgecolor="white"
)

for bar, val, std, colour in zip(bars, means, stds, colors):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + std + 2,
        f"{val:.1f}",
        ha="center", va="bottom",
        fontsize=10, color=colour, fontweight="bold"
    )

ax.axhline(
    y=15, color=C_NHS, linestyle="--", linewidth=1.4,
    label="NHS 15-min benchmark", zorder=2
)

ax.set_title(
    "Assessment Wait Mean — Three-Way Comparison\n"
    "(Hybrid model reduces the primary upstream bottleneck)",
    **FONT_TITLE
)
ax.set_ylabel("Mean Assessment Wait (minutes)", **FONT_AXIS)
ax.set_xlabel("Simulation Model", **FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "comparison_hybrid_waiting_times.png")

# ============================================================
# FIGURE 3 — ALL 3 MODELS: BOARDING WAIT MEAN COMPARISON
# ============================================================

print("Fig 3: All models — boarding wait mean comparison...")

labels = ["Baseline", "Rule-Based", "Hybrid ML + Rule-Based"]
means  = [
    b["boarding_wait_mean"].mean(),
    a["boarding_wait_mean"].mean(),
    m["boarding_wait_mean"].mean(),
]
stds   = [
    b["boarding_wait_mean"].std(),
    a["boarding_wait_mean"].std(),
    m["boarding_wait_mean"].std(),
]
colors = [C_BASELINE, C_AI, C_ML]

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

bars = ax.bar(
    x, means, width,
    color=colors,
    yerr=stds, capsize=6,
    error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    zorder=3, edgecolor="white"
)

for bar, val, std, colour in zip(bars, means, stds, colors):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + std + 8,
        f"{val:.1f}",
        ha="center", va="bottom",
        fontsize=10, color=colour, fontweight="bold"
    )

ax.axhline(
    y=240, color=C_NHS, linestyle="--", linewidth=1.4,
    label="NHS 4-hour threshold (240 min)", zorder=2
)

ax.set_title(
    "Boarding Wait Mean — Three-Way Comparison\n"
    "(Upstream acceleration increases downstream pressure)",
    **FONT_TITLE
)
ax.set_ylabel("Mean Boarding Wait (minutes)", **FONT_AXIS)
ax.set_xlabel("Simulation Model", **FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "comparison_boarding_wait_mean.png")


# ============================================================
# FIGURE 4 — TOTAL LOS MEAN: THREE-WAY COMPARISON
# ============================================================

print("Fig 3: All models — total LOS mean comparison...")

labels = ["Baseline", "Rule-Based", "Hybrid ML + Rule-Based"]
means  = [
    b["total_los_mean"].mean(),
    a["total_los_mean"].mean(),
    m["total_los_mean"].mean(),
]
stds   = [
    b["total_los_mean"].std(),
    a["total_los_mean"].std(),
    m["total_los_mean"].std(),
]
colors = [C_BASELINE, C_AI, C_ML]

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(9, 5))
style_axes(ax)

bars = ax.bar(
    x, means, width,
    color=colors,
    yerr=stds, capsize=6,
    error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    zorder=3, edgecolor="white"
)

for bar, val, std, colour in zip(bars, means, stds, colors):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + std + 6,
        f"{val:.1f}",
        ha="center", va="bottom",
        fontsize=10, color=colour, fontweight="bold"
    )

ax.axhline(
    y=240, color=C_NHS, linestyle="--", linewidth=1.4,
    label="NHS 4-hour threshold (240 min)", zorder=2
)

ax.set_title(
    "Total ED Length of Stay Mean — Three-Way Comparison\n"
    "(Hybrid model produces the strongest overall system-level improvement)",
    **FONT_TITLE
)
ax.set_ylabel("Mean Total LOS (minutes)", **FONT_AXIS)
ax.set_xlabel("Simulation Model", **FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "comparison_total_los_mean.png")


# ============================================================
# FIGURE 5 — ALL 3 MODELS: NHS 4-HOUR COMPLIANCE
# ============================================================

print("Fig 3: All models — NHS 4-hour compliance...")

fig, ax = plt.subplots(figsize=(11, 5))
style_axes(ax)

ax.plot(
    reps, b_comp, marker="o", color=C_BASELINE, linewidth=2,
    markersize=7, label=f"Baseline (mean {b_comp.mean():.1f}%)", zorder=3
)
ax.plot(
    reps, a_comp, marker="s", color=C_AI, linewidth=2,
    markersize=7, label=f"Rule-Based (mean {a_comp.mean():.1f}%)", zorder=3
)
ax.plot(
    reps, m_comp, marker="^", color=C_ML, linewidth=2,
    markersize=7, label=f"Hybrid ML + Rule-Based (mean {m_comp.mean():.1f}%)", zorder=3
)

ax.axhline(
    y=95, color=C_NHS, linestyle="--", linewidth=1.8,
    label="NHS target (95%)", zorder=2
)

ax.set_title(
    "NHS 4-Hour Target Compliance — Three-Way Comparison\n"
    "(10 Replications, Dataset-Driven Simulation)",
    **FONT_TITLE
)
ax.set_xlabel("Replication", **FONT_AXIS)
ax.set_ylabel("4-Hour Compliance (%)", **FONT_AXIS)
ax.set_xticks(reps)
ax.set_ylim(60, 100)
ax.legend(fontsize=10, framealpha=0.9, loc="lower right")

fig.tight_layout()
save_fig(fig, "comparison_nhs_compliance.png")

# ============================================================
# FIGURE 6 — SUMMARY TABLE
# ============================================================

print("Fig 4: Summary comparison table...")

summary_data = [
    ["Metric",                    "Baseline",          "Rule-Based",        "Hybrid ML",          "Best"],
    ["Assessment wait (mean)",
     f"{b['assessment_wait_mean'].mean():.1f} min",
     f"{a['assessment_wait_mean'].mean():.1f} min",
     f"{m['assessment_wait_mean'].mean():.1f} min",
     "Hybrid ML"],
    ["Boarding wait (mean)",
     f"{b['boarding_wait_mean'].mean():.0f} min",
     f"{a['boarding_wait_mean'].mean():.0f} min",
     f"{m['boarding_wait_mean'].mean():.0f} min",
     "Rule-Based"],
    ["Total LOS (mean)",
     f"{b['total_los_mean'].mean():.0f} min",
     f"{a['total_los_mean'].mean():.0f} min",
     f"{m['total_los_mean'].mean():.0f} min",
     "Hybrid ML"],
    ["NHS 4-hr compliance",
     f"{b_comp.mean():.1f}%",
     f"{a_comp.mean():.1f}%",
     f"{m_comp.mean():.1f}%",
     "Hybrid ML"],
    ["Boarding wait — critical",
     f"{b_sev_means[0]:.0f} min",
     f"{a_sev_means[0]:.0f} min",
     f"{m['boarding_wait_critical_mean'].mean():.0f} min" if "boarding_wait_critical_mean" in m.columns else "—",
     "Rule-Based / Hybrid"],
]

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.axis("off")

col_widths = [0.30, 0.17, 0.17, 0.17, 0.19]
col_starts = [0.00, 0.30, 0.47, 0.64, 0.81]
col_colors = ["#ffffff", C_BASELINE, C_AI, C_ML, C_NAVY]
row_colors = ["#EBF3FB", "#FFFFFF"]

for j, (header, w, x0, hc) in enumerate(zip(summary_data[0], col_widths, col_starts, col_colors)):
    ax.add_patch(plt.Rectangle((x0, 0.82), w - 0.01, 0.16,
                               color=hc if j > 0 else C_NAVY, zorder=2))
    ax.text(x0 + w/2, 0.90, header, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            transform=ax.transAxes)

for i, row in enumerate(summary_data[1:]):
    y_top = 0.82 - (i + 1) * 0.12
    bg = row_colors[i % 2]
    for j, (cell, w, x0) in enumerate(zip(row, col_widths, col_starts)):
        ax.add_patch(plt.Rectangle((x0, y_top), w - 0.01, 0.115,
                                   color=bg, zorder=1))
        color = "black"
        if j == 4:
            if "Hybrid" in str(cell):
                color = C_ML
            elif "Rule" in str(cell):
                color = C_AI
        ax.text(
            x0 + (0.04 if j == 0 else w/2),
            y_top + 0.055,
            cell,
            ha=("left" if j == 0 else "center"),
            va="center",
            fontsize=8.8,
            color=color,
            transform=ax.transAxes,
            fontweight="bold" if j == 4 else "normal"
        )

ax.set_title(
    "Summary Comparison — Baseline vs Rule-Based vs Hybrid ML Model\n"
    "(Mean across 10 replications)",
    fontsize=12, fontweight="bold", pad=12, y=1.02
)

fig.tight_layout()
save_fig(fig, "comparison_summary_table.png")

print()
print("=" * 60)
print("  Final comparison figures generated.")
print("=" * 60)
for fname in [
    "comparison_boarding_by_severity.png",
    "comparison_hybrid_waiting_times.png"
    "comparison_boarding_wait_mean.png",
    "comparison_total_los_mean.png",
    "comparison_nhs_compliance.png",
    "comparison_summary_table.png",
]:
    status = "✓" if os.path.exists(os.path.join(FIGURES_DIR, fname)) else "✗"
    print(f"  {status}  {fname}")
print("=" * 60)