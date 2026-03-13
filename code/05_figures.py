# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pyyaml",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", app_title="OHCA-RL Pre-Training Figures")


# ── Cell 1: Setup, Load Data, Helpers ─────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import numpy as np
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from pathlib import Path

    from utils import setup_logging
    logger = setup_logging("05_figures")

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)
    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as _f:
        ohca_config = yaml.safe_load(_f)

    site_name = config["site_name"]
    intermediate_dir = project_root / "output" / "intermediate"
    final_dir = project_root / "output" / "final"
    fig_dir = final_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    bucketed_df = pd.read_parquet(intermediate_dir / "wide_df_bucketed.parquet")
    bucketed_df["hospitalization_id"] = bucketed_df["hospitalization_id"].astype(str)
    summary_df = pd.read_parquet(intermediate_dir / "hospitalization_summary.parquet")
    summary_df["hospitalization_id"] = summary_df["hospitalization_id"].astype(str)

    logger.info("Loaded bucketed_df: %d rows, summary_df: %d rows", len(bucketed_df), len(summary_df))

    # ── Plot style ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.2,
    })

    # ── Colors ──
    COLOR_SURV = "#2196F3"      # blue
    COLOR_DEAD = "#E53935"      # red
    COLOR_ALL = "#555555"       # grey
    ACTION_COLORS = ["#66BB6A", "#FFA726", "#42A5F5", "#EF5350"]  # stay, increase, decrease, stop
    ACTION_LABELS = {0: "Stay", 1: "Increase", 2: "Decrease", 3: "Stop"}

    # ── Helpers ──
    def plot_median_iqr(df, col, ax, color, label, time_col="time_bucket"):
        """Plot median line with IQR shading for a numeric column over time."""
        _stats = df.groupby(time_col)[col].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        _stats.columns = ["median", "q25", "q75"]
        _stats = _stats.dropna(subset=["median"])
        if _stats.empty:
            return
        ax.plot(_stats.index, _stats["median"], color=color, label=label, linewidth=1.5)
        ax.fill_between(_stats.index, _stats["q25"], _stats["q75"], color=color, alpha=0.15)

    def plot_pct_over_time(df, col, ax, color, label, time_col="time_bucket"):
        """Plot % of patients with col==1 at each time bucket."""
        _pct = df.groupby(time_col)[col].mean() * 100
        ax.plot(_pct.index, _pct.values, color=color, label=label, linewidth=1.5)

    def add_day_lines(ax, max_hours=120):
        """Add vertical dashed lines at 24h intervals."""
        for _h in range(24, max_hours + 1, 24):
            ax.axvline(_h, color="grey", linestyle="--", alpha=0.3, linewidth=0.5)

    def save_fig(fig, name):
        """Save figure as PDF."""
        fig.savefig(fig_dir / f"{name}.pdf")
        logger.info("Saved %s.pdf", name)
        plt.close(fig)

    # Drop runt bucket 120 (partial edge bucket with very few patients)
    bucketed_df = bucketed_df[bucketed_df["time_bucket"] < 120]

    # Subgroups
    surv_df = bucketed_df[bucketed_df["ever_died"] == 0]
    dead_df = bucketed_df[bucketed_df["ever_died"] == 1]

    mo.md(f"""
    ## Step 05: Pre-Training Figures

    | Setting | Value |
    |---------|-------|
    | **Site** | {site_name} |
    | **Patients** | {bucketed_df['hospitalization_id'].nunique():,} |
    | **Survivors** | {surv_df['hospitalization_id'].nunique():,} |
    | **Non-survivors** | {dead_df['hospitalization_id'].nunique():,} |
    | **Output** | `{fig_dir}` |
    """)
    return (
        ACTION_COLORS,
        ACTION_LABELS,
        COLOR_ALL,
        COLOR_DEAD,
        COLOR_SURV,
        add_day_lines,
        bucketed_df,
        dead_df,
        fig_dir,
        final_dir,
        logger,
        mo,
        mticker,
        np,
        ohca_config,
        pd,
        plot_median_iqr,
        plot_pct_over_time,
        plt,
        save_fig,
        site_name,
        sns,
        summary_df,
        surv_df,
    )


# ── Cell 2: Figure 1 — Missingness Heatmap ───────────────────────────
@app.cell
def _(final_dir, logger, mo, np, pd, plt, save_fig, sns):
    logger.info("Figure 1: Missingness heatmap...")

    _comparison = pd.read_csv(final_dir / "missingness_comparison.csv")

    # Categorize variables by type
    def _var_group(v):
        if v.startswith("vital_"):
            return "Vitals"
        elif v.startswith("lab_"):
            return "Labs"
        elif v.startswith("med_cont_") or v.startswith("med_int_"):
            return "Meds"
        elif v.startswith("resp_"):
            return "Resp"
        else:
            return "Other"

    _comparison["group"] = _comparison["variable"].apply(_var_group)
    _comparison = _comparison.sort_values(["group", "variable"])

    # Use patient-level missingness
    _before_col = "patient_missing_pct_before"
    _after_col = "patient_missing_pct_after"

    # Filter to variables with non-zero missingness in at least one phase
    _has_miss = _comparison[(_comparison[_before_col] > 0) | (_comparison[_after_col] > 0)].copy()
    if _has_miss.empty:
        mo.md("### Figure 1: No missingness to display")
    else:
        _labels = _has_miss["variable"].tolist()
        _groups = _has_miss["group"].tolist()

        _data = np.column_stack([
            _has_miss[_before_col].values,
            _has_miss[_after_col].values,
        ])

        fig1, ax1 = plt.subplots(figsize=(6, max(4, len(_labels) * 0.3)))
        _im = ax1.imshow(_data, cmap="Reds", aspect="auto", vmin=0, vmax=100)

        # Annotate cells
        for _i in range(len(_labels)):
            for _j in range(2):
                _val = _data[_i, _j]
                _color = "white" if _val > 60 else "black"
                ax1.text(_j, _i, f"{_val:.0f}", ha="center", va="center",
                         fontsize=7, color=_color)

        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Before\nImputation", "After\nImputation"])
        ax1.set_yticks(range(len(_labels)))
        ax1.set_yticklabels(_labels, fontsize=8)

        # Add group separators
        _prev_group = None
        for _i, _g in enumerate(_groups):
            if _prev_group is not None and _g != _prev_group:
                ax1.axhline(_i - 0.5, color="black", linewidth=1)
            _prev_group = _g

        _cbar = fig1.colorbar(_im, ax=ax1, shrink=0.8)
        _cbar.set_label("Patient-Level Missingness (%)")
        ax1.set_title("Missingness Before vs After Imputation")
        fig1.tight_layout()

        save_fig(fig1, "fig1_missingness_heatmap")
        mo.md("### Figure 1: Missingness Heatmap\n\n![](figures/fig1_missingness_heatmap.pdf)")
    return


# ── Cell 3: Figure 2 — Vasopressor/NEE Temporal ──────────────────────
@app.cell
def _(COLOR_DEAD, COLOR_SURV, add_day_lines, dead_df, logger, mo, plot_median_iqr, plot_pct_over_time, plt, save_fig, surv_df):
    logger.info("Figure 2: Vasopressor/NEE temporal...")

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel A: % on vasopressors
    plot_pct_over_time(surv_df, "on_vaso", ax2a, COLOR_SURV, "Survivors")
    plot_pct_over_time(dead_df, "on_vaso", ax2a, COLOR_DEAD, "Non-Survivors")
    ax2a.set_ylabel("Patients on Vasopressors (%)")
    ax2a.set_title("A. Vasopressor Utilization Over 120 Hours")
    ax2a.legend(frameon=False)
    add_day_lines(ax2a)

    # Panel B: Median NEE with IQR
    # Only among patients ON vasopressors (NEE > 0)
    _surv_on = surv_df[surv_df["med_cont_nee"] > 0]
    _dead_on = dead_df[dead_df["med_cont_nee"] > 0]
    plot_median_iqr(_surv_on, "med_cont_nee", ax2b, COLOR_SURV, "Survivors")
    plot_median_iqr(_dead_on, "med_cont_nee", ax2b, COLOR_DEAD, "Non-Survivors")
    ax2b.set_ylabel("NEE (mcg/kg/min)")
    ax2b.set_xlabel("Hours Since First Event")
    ax2b.set_title("B. Norepinephrine Equivalent (Among Patients on Vasopressors)")
    ax2b.legend(frameon=False)
    add_day_lines(ax2b)

    ax2b.set_xlim(0, 120)
    fig2.tight_layout()
    save_fig(fig2, "fig2_vasopressor_nee")
    mo.md("### Figure 2: Vasopressor/NEE\n\n![](figures/fig2_vasopressor_nee.pdf)")
    return


# ── Cell 4: Figure 3 — Treatment Timelines ───────────────────────────
@app.cell
def _(COLOR_DEAD, COLOR_SURV, add_day_lines, dead_df, logger, mo, plot_pct_over_time, plt, save_fig, surv_df):
    logger.info("Figure 3: Treatment timelines...")

    _treatments = [
        ("on_vaso", "Vasopressors"),
        ("on_imv", "Invasive Mechanical Ventilation"),
        ("on_crrt", "CRRT"),
    ]

    fig3, axes3 = plt.subplots(len(_treatments), 1, figsize=(10, 3 * len(_treatments)), sharex=True)

    for _i, (_col, _title) in enumerate(_treatments):
        _ax = axes3[_i]
        plot_pct_over_time(surv_df, _col, _ax, COLOR_SURV, "Survivors")
        plot_pct_over_time(dead_df, _col, _ax, COLOR_DEAD, "Non-Survivors")
        _ax.set_ylabel("Patients (%)")
        _ax.set_title(_title)
        _ax.legend(frameon=False, loc="upper right")
        add_day_lines(_ax)

    axes3[-1].set_xlabel("Hours Since First Event")
    axes3[-1].set_xlim(0, 120)
    fig3.suptitle("Organ Support Utilization Over 120 Hours", fontsize=14, y=1.01)
    fig3.tight_layout()
    save_fig(fig3, "fig3_treatment_timelines")
    mo.md("### Figure 3: Treatment Timelines\n\n![](figures/fig3_treatment_timelines.pdf)")
    return


# ── Cell 5: Figure 4 — SOFA Trajectory ───────────────────────────────
@app.cell
def _(COLOR_DEAD, COLOR_SURV, logger, mo, np, pd, plt, save_fig, sns, summary_df):
    logger.info("Figure 4: SOFA trajectory...")

    # Identify SOFA total columns (sofa_0_24, sofa_24_48, etc.)
    _sofa_cols = sorted([c for c in summary_df.columns
                          if c.startswith("sofa_") and c.count("_") == 2
                          and not any(s in c for s in ["cv", "coag", "liver", "resp", "cns", "renal"])])

    if not _sofa_cols:
        mo.md("### Figure 4: No SOFA data available")
    else:
        # Melt to long format for boxplot
        _sofa_long = []
        for _col in _sofa_cols:
            _parts = _col.split("_")
            _window = f"{_parts[1]}–{_parts[2]}h"
            _tmp = summary_df[["hospitalization_id", "ever_died", _col]].dropna(subset=[_col]).copy()
            _tmp = _tmp.rename(columns={_col: "sofa"})
            _tmp["window"] = _window
            _sofa_long.append(_tmp)
        _sofa_long = pd.concat(_sofa_long, ignore_index=True)
        _sofa_long["group"] = _sofa_long["ever_died"].map({0: "Survivors", 1: "Non-Survivors"})

        # Preserve window order
        _window_order = [f"{c.split('_')[1]}–{c.split('_')[2]}h" for c in _sofa_cols]

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=_sofa_long, x="window", y="sofa", hue="group",
            order=_window_order,
            palette={"Survivors": COLOR_SURV, "Non-Survivors": COLOR_DEAD},
            fliersize=2, linewidth=0.8, ax=ax4,
        )
        ax4.set_xlabel("Time Window")
        ax4.set_ylabel("SOFA Score")
        ax4.set_title("SOFA Score Trajectory by 24-Hour Window")
        ax4.legend(title="", frameon=False)
        fig4.tight_layout()
        save_fig(fig4, "fig4_sofa_trajectory")
        mo.md("### Figure 4: SOFA Trajectory\n\n![](figures/fig4_sofa_trajectory.pdf)")
    return


# ── Cell 6: Figure 5 — Vital Signs ───────────────────────────────────
@app.cell
def _(COLOR_DEAD, COLOR_SURV, add_day_lines, dead_df, logger, mo, plot_median_iqr, plt, save_fig, surv_df):
    logger.info("Figure 5: Vital signs...")

    _vitals = [
        ("vital_heart_rate", "Heart Rate (bpm)"),
        ("vital_map", "MAP (mmHg)"),
        ("vital_temp_c", "Temperature (°C)"),
        ("vital_respiratory_rate", "Respiratory Rate (/min)"),
    ]

    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes5 = axes5.flatten()

    for _i, (_col, _ylabel) in enumerate(_vitals):
        _ax = axes5[_i]
        plot_median_iqr(surv_df, _col, _ax, COLOR_SURV, "Survivors")
        plot_median_iqr(dead_df, _col, _ax, COLOR_DEAD, "Non-Survivors")
        _ax.set_ylabel(_ylabel)
        _ax.legend(frameon=False, fontsize=9)
        add_day_lines(_ax)
        _ax.set_xlim(0, 120)

    axes5[2].set_xlabel("Hours Since First Event")
    axes5[3].set_xlabel("Hours Since First Event")
    fig5.suptitle("Key Vital Signs Over 120 Hours", fontsize=14)
    fig5.tight_layout()
    save_fig(fig5, "fig5_vital_signs")
    mo.md("### Figure 5: Vital Signs\n\n![](figures/fig5_vital_signs.pdf)")
    return


# ── Cell 7: Figure 6 — Action Distribution Over Time ─────────────────
@app.cell
def _(ACTION_COLORS, ACTION_LABELS, add_day_lines, bucketed_df, logger, mo, np, pd, plt, save_fig):
    logger.info("Figure 6: Action distribution over time...")

    # Compute action proportions per time bucket
    _action_props = bucketed_df.groupby("time_bucket")["action"].value_counts(normalize=True).unstack(fill_value=0)
    # Ensure all actions present
    for _a in range(4):
        if _a not in _action_props.columns:
            _action_props[_a] = 0
    _action_props = _action_props[[0, 1, 2, 3]]

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.stackplot(
        _action_props.index,
        [_action_props[_a].values for _a in range(4)],
        labels=[ACTION_LABELS[_a] for _a in range(4)],
        colors=ACTION_COLORS,
        alpha=0.85,
    )
    ax6.set_ylabel("Proportion")
    ax6.set_xlabel("Hours Since First Event")
    ax6.set_title("Action Distribution Over 120 Hours")
    ax6.set_xlim(0, 120)
    ax6.set_ylim(0, 1)
    ax6.legend(loc="upper right", frameon=False)
    add_day_lines(ax6)
    fig6.tight_layout()
    save_fig(fig6, "fig6_action_distribution")
    mo.md("### Figure 6: Action Distribution\n\n![](figures/fig6_action_distribution.pdf)")
    return


# ── Cell 8: Figure 7 — Lactate & Key Labs ────────────────────────────
@app.cell
def _(COLOR_DEAD, COLOR_SURV, add_day_lines, dead_df, logger, mo, plot_median_iqr, plt, save_fig, surv_df):
    logger.info("Figure 7: Lactate & key labs...")

    _labs = [
        ("lab_lactate", "Lactate (mmol/L)"),
        ("lab_creatinine", "Creatinine (mg/dL)"),
        ("lab_hemoglobin", "Hemoglobin (g/dL)"),
    ]

    fig7, axes7 = plt.subplots(len(_labs), 1, figsize=(10, 3 * len(_labs)), sharex=True)

    for _i, (_col, _ylabel) in enumerate(_labs):
        _ax = axes7[_i]
        plot_median_iqr(surv_df, _col, _ax, COLOR_SURV, "Survivors")
        plot_median_iqr(dead_df, _col, _ax, COLOR_DEAD, "Non-Survivors")
        _ax.set_ylabel(_ylabel)
        _ax.legend(frameon=False)
        add_day_lines(_ax)

    axes7[-1].set_xlabel("Hours Since First Event")
    axes7[-1].set_xlim(0, 120)
    fig7.suptitle("Key Lab Trajectories Over 120 Hours", fontsize=14, y=1.01)
    fig7.tight_layout()
    save_fig(fig7, "fig7_labs")
    mo.md("### Figure 7: Lab Trajectories\n\n![](figures/fig7_labs.pdf)")
    return


# ── Cell 9: Summary ──────────────────────────────────────────────────
@app.cell
def _(fig_dir, logger, mo):
    _pdfs = sorted(fig_dir.glob("*.pdf"))
    logger.info("All figures saved: %d PDFs in %s", len(_pdfs), fig_dir)

    _rows = "\n".join(f"    | `{p.name}` |" for p in _pdfs)
    mo.md(f"""
    ## Figures Complete

    | File |
    |------|
{_rows}

    **Location**: `{fig_dir}`
    """)
    return


if __name__ == "__main__":
    app.run()
