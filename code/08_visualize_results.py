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
#     "tabulate",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="OHCA-RL Post-Training Visualizations")


# ── Cell 0: Imports & Config ──────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import seaborn as sns
    from pathlib import Path

    from utils import setup_logging
    logger = setup_logging("08_visualize_results")

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    site_name = config["site_name"]
    intermediate_dir = project_root / "output" / "intermediate"
    training_dir = intermediate_dir / "training"
    final_dir = project_root / "output" / "final"
    fig_dir = final_dir / "figures"
    ext_val_dir = final_dir / "external_validation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot style (match 05_figures.py) ──
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

    # ── Action colors (PI encoding: 0=Increase, 1=Decrease, 2=Stop, 3=Stay) ──
    ACTION_LABELS = {0: "Increase", 1: "Decrease", 2: "Stop", 3: "Stay"}
    ACTION_COLORS = {
        0: "#E53935",   # red — escalation
        1: "#42A5F5",   # blue — de-escalation
        2: "#66BB6A",   # green — liberation
        3: "#FFA726",   # orange — maintenance
    }

    # CPC endpoint colors
    CPC_COLORS = {
        "CPC1_2": "#22c55e",  # green — good neurological outcome
        "CPC3": "#facc15",    # yellow
        "CPC4": "#f97316",    # orange
        "CPC5": "#ef4444",    # red — death
    }

    # ── Save helper (PDF + PNG) ──
    def save_fig(fig, name):
        """Save figure as PDF (300dpi) and PNG (150dpi for dashboard)."""
        fig.savefig(fig_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        logger.info("Saved %s.pdf and %s.png", name, name)
        plt.close(fig)

    mo.md(f"# OHCA-RL Post-Training Visualizations\n**Site:** {site_name}")
    return (
        ACTION_COLORS,
        ACTION_LABELS,
        CPC_COLORS,
        BoundaryNorm,
        ListedColormap,
        config,
        ext_val_dir,
        fig_dir,
        final_dir,
        intermediate_dir,
        logger,
        mo,
        mpatches,
        mticker,
        np,
        pd,
        plt,
        save_fig,
        site_name,
        sns,
        training_dir,
    )


# ── Cell 1: Load All Data ────────────────────────────────────────────
@app.cell
def _(ext_val_dir, intermediate_dir, logger, mo, np, pd, training_dir):
    # Training outputs
    history_df = pd.read_csv(training_dir / "training_history.csv")
    action_summary = pd.read_csv(training_dir / "test_action_summary.csv")
    coef_summary = pd.read_csv(training_dir / "coef_summary.csv")
    bin_summary = pd.read_csv(training_dir / "bin_summary.csv")

    with open(training_dir / "action_remap.json") as _f:
        import json as _json
        action_remap = _json.load(_f)

    # Test predictions for timelines and confusion matrix
    _pred_path = training_dir / "test_with_predictions.parquet"
    if _pred_path.exists():
        pred_df = pd.read_parquet(_pred_path)
        pred_df["hospitalization_id"] = pred_df["hospitalization_id"].astype(str)
        has_predictions = True
        logger.info("Loaded test_with_predictions: %d rows", len(pred_df))
    else:
        pred_df = pd.DataFrame()
        has_predictions = False
        logger.warning("test_with_predictions.parquet not found — timeline/confusion plots will be skipped")

    # Hospitalization summary (for CPC outcomes)
    summary_df = pd.read_parquet(intermediate_dir / "hospitalization_summary.parquet")
    summary_df["hospitalization_id"] = summary_df["hospitalization_id"].astype(str)

    # STROBE counts
    _strobe_path = pd.read_csv(intermediate_dir.parent / "final" / "strobe_counts.csv")
    strobe_df = _strobe_path

    # External validation outputs (optional)
    has_ext_val = ext_val_dir.exists()
    if has_ext_val:
        try:
            ext_action_summary = pd.read_csv(ext_val_dir / "action_summary.csv")
            ext_coef_summary = pd.read_csv(ext_val_dir / "coef_summary.csv")
            ext_bin_summary = pd.read_csv(ext_val_dir / "bin_summary.csv")
            logger.info("Loaded external validation outputs")
        except FileNotFoundError:
            has_ext_val = False

    if not has_ext_val:
        ext_action_summary = pd.DataFrame()
        ext_coef_summary = pd.DataFrame()
        ext_bin_summary = pd.DataFrame()

    mo.md(f"""
    ## Data Loaded

    | Dataset | Rows |
    |---------|------|
    | Training history | {len(history_df)} epochs |
    | Action summary | {len(action_summary)} actions |
    | Test predictions | {len(pred_df) if has_predictions else 'N/A'} |
    | External validation | {'Available' if has_ext_val else 'Not available'} |
    """)
    return (
        action_summary,
        bin_summary,
        coef_summary,
        ext_action_summary,
        ext_bin_summary,
        ext_coef_summary,
        has_ext_val,
        has_predictions,
        history_df,
        pred_df,
        strobe_df,
        summary_df,
    )


# ── Cell 2: Fig 8 — Training Loss Curves ─────────────────────────────
@app.cell
def _(history_df, logger, mo, np, plt, save_fig):
    logger.info("Figure 8: Training loss curves...")

    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    _epochs = history_df["epoch"]

    # Panel A: Loss curves
    ax8a.plot(_epochs, history_df["train_loss"], color="#1565C0", label="Train Loss", linewidth=1.5)
    ax8a.plot(_epochs, history_df["test_loss"], color="#E53935", label="Test Loss", linewidth=1.5)

    # Mark best epoch (lowest test loss)
    _best_idx = history_df["test_loss"].idxmin()
    _best_epoch = int(history_df.loc[_best_idx, "epoch"])
    _best_loss = history_df.loc[_best_idx, "test_loss"]
    ax8a.axvline(_best_epoch, color="grey", linestyle="--", alpha=0.5, linewidth=1)
    ax8a.scatter([_best_epoch], [_best_loss], color="#E53935", zorder=5, s=60, marker="*")
    ax8a.annotate(f"Best: epoch {_best_epoch}\nloss={_best_loss:.3f}",
                  xy=(_best_epoch, _best_loss), xytext=(_best_epoch + 1, _best_loss * 1.05),
                  fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="grey"))

    ax8a.set_ylabel("Loss")
    ax8a.set_title("A. Training and Test Loss")
    ax8a.legend(frameon=False)

    # Panel B: Q-value means
    ax8b.plot(_epochs, history_df["train_q_mean"], color="#1565C0", label="Train Q-mean", linewidth=1.5)
    ax8b.plot(_epochs, history_df["test_q_mean"], color="#E53935", label="Test Q-mean", linewidth=1.5)
    if "train_target_mean" in history_df.columns:
        ax8b.plot(_epochs, history_df["train_target_mean"], color="#1565C0", linestyle="--",
                  alpha=0.5, label="Train Target-mean")
        ax8b.plot(_epochs, history_df["test_target_mean"], color="#E53935", linestyle="--",
                  alpha=0.5, label="Test Target-mean")
    ax8b.axvline(_best_epoch, color="grey", linestyle="--", alpha=0.5, linewidth=1)
    ax8b.set_ylabel("Q-value Mean")
    ax8b.set_xlabel("Epoch")
    ax8b.set_title("B. Q-Value and Target Means")
    ax8b.legend(frameon=False, fontsize=9)

    fig8.tight_layout()
    save_fig(fig8, "fig8_training_curves")

    mo.md("### Figure 8: Training Loss Curves\n\n![](figures/fig8_training_curves.png)")
    return


# ── Cell 3: Fig 9 — Action Distribution (Diverging Bar Chart) ────────
@app.cell
def _(ACTION_COLORS, ACTION_LABELS, action_summary, logger, mo, np, pd, plt, save_fig, site_name):
    logger.info("Figure 9: Action distribution comparison...")

    # Compute percentages
    _total_pred = action_summary["pred_count"].sum()
    _total_obs = action_summary["obs_count"].sum()

    _actions = list(range(4))
    _pred_pcts = []
    _obs_pcts = []
    _labels = []

    for _a in _actions:
        _row = action_summary[action_summary.index == _a] if "action_name" not in action_summary.columns else action_summary.iloc[_a:_a+1]
        if len(action_summary) > _a:
            _pred_pcts.append(action_summary.iloc[_a]["pred_count"] / _total_pred * 100)
            _obs_pcts.append(action_summary.iloc[_a]["obs_count"] / _total_obs * 100)
        else:
            _pred_pcts.append(0)
            _obs_pcts.append(0)
        _labels.append(ACTION_LABELS[_a])

    _x = np.arange(len(_actions))
    _width = 0.6

    fig9, ax9 = plt.subplots(figsize=(8, 6))

    # RL Agent (above zero)
    _colors = [ACTION_COLORS[_a] for _a in _actions]
    _bars_rl = ax9.bar(_x, _pred_pcts, _width, color=_colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    # Clinicians (below zero)
    _bars_clin = ax9.bar(_x, [-p for p in _obs_pcts], _width, color=_colors, alpha=0.5, edgecolor="white", linewidth=0.5)

    # Add percentage labels
    for _i, (_bar, _pct) in enumerate(zip(_bars_rl, _pred_pcts)):
        ax9.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.8,
                 f"{_pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for _i, (_bar, _pct) in enumerate(zip(_bars_clin, _obs_pcts)):
        ax9.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() - 0.8,
                 f"{_pct:.1f}%", ha="center", va="top", fontsize=10, fontweight="bold")

    # Divider and labels
    ax9.axhline(0, color="black", linewidth=1)
    ax9.set_xticks(_x)
    ax9.set_xticklabels(_labels, fontsize=11)

    # Y-axis: symmetric
    _max_pct = max(max(_pred_pcts), max(_obs_pcts)) * 1.2
    ax9.set_ylim(-_max_pct, _max_pct)
    ax9.set_ylabel("Percent (%)")

    # Agent labels
    ax9.text(-0.5, _max_pct * 0.85, "RL Agent", fontsize=13, fontweight="bold", color="#333")
    ax9.text(-0.5, -_max_pct * 0.85, "Clinicians", fontsize=13, fontweight="bold", color="#333")

    # Custom y-tick labels (absolute values)
    _yticks = ax9.get_yticks()
    ax9.set_yticks(_yticks)
    ax9.set_yticklabels([f"{abs(y):.0f}" for y in _yticks])

    ax9.set_title(f"Vasopressor Action Distribution: RL Agent vs Clinician — {site_name}")
    fig9.tight_layout()
    save_fig(fig9, "fig9_action_distribution")

    # Save CSV for multi-site aggregation
    _csv_rows = []
    for _a in _actions:
        _csv_rows.append({
            "action": _a, "action_label": ACTION_LABELS[_a],
            "rl_count": int(action_summary.iloc[_a]["pred_count"]),
            "rl_pct": _pred_pcts[_a],
            "clinician_count": int(action_summary.iloc[_a]["obs_count"]),
            "clinician_pct": _obs_pcts[_a],
            "site": site_name,
        })
    pd.DataFrame(_csv_rows).to_csv(fig_dir / "fig9_action_distribution.csv", index=False)

    mo.md("### Figure 9: Action Distribution (RL vs Clinician)\n\n![](figures/fig9_action_distribution.png)")
    return


# ── Cell 4: Fig 10 — Patient Timeline Heatmaps ───────────────────────
@app.cell
def _(ACTION_COLORS, ACTION_LABELS, BoundaryNorm, CPC_COLORS, ListedColormap, has_predictions, logger, mo, mpatches, np, plt, pred_df, save_fig, site_name, summary_df):
    if not has_predictions:
        mo.md("### Figure 10: Skipped (no test_with_predictions.parquet)")
    else:
        logger.info("Figure 10: Patient timeline heatmaps...")

        # Merge CPC outcome
        _pred_with_cpc = pred_df.merge(
            summary_df[["hospitalization_id", "cpc", "ever_died"]],
            on="hospitalization_id", how="left",
        )

        # Get per-patient trajectory length and sample 100 patients
        _traj_len = _pred_with_cpc.groupby("hospitalization_id")["time_bucket"].count().reset_index()
        _traj_len.columns = ["hospitalization_id", "n_steps"]
        _traj_len = _traj_len[_traj_len["n_steps"] >= 12]  # at least 12 hours
        _sample_ids = _traj_len.sample(n=min(100, len(_traj_len)), random_state=42)
        _sample_ids = _sample_ids.sort_values("n_steps", ascending=False)["hospitalization_id"].tolist()

        # Build heatmap matrices
        _max_t = int(_pred_with_cpc["time_bucket"].max()) + 1
        _n_patients = len(_sample_ids)

        _physician_matrix = np.full((_n_patients, _max_t), np.nan)
        _rl_matrix = np.full((_n_patients, _max_t), np.nan)
        _cpc_list = []
        _last_bucket = []

        for _i, _pid in enumerate(_sample_ids):
            _patient = _pred_with_cpc[_pred_with_cpc["hospitalization_id"] == _pid].sort_values("time_bucket")
            for _, _row in _patient.iterrows():
                _t = int(_row["time_bucket"])
                if _t < _max_t:
                    _physician_matrix[_i, _t] = _row["action"]
                    _rl_matrix[_i, _t] = _row["pred_action"]
            _cpc_list.append(_patient["cpc"].iloc[0] if len(_patient) > 0 else "unknown")
            _last_bucket.append(int(_patient["time_bucket"].max()))

        # Colormap: 4 actions
        _cmap = ListedColormap([ACTION_COLORS[_a] for _a in range(4)])
        _norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

        fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(14, 10))

        for _ax, _matrix, _title in [
            (ax10a, _physician_matrix, "Clinician Actions"),
            (ax10b, _rl_matrix, "RL Recommended Actions"),
        ]:
            _ax.imshow(_matrix, aspect="auto", cmap=_cmap, norm=_norm,
                       interpolation="nearest", origin="upper")

            # Add CPC endpoint markers
            for _i, (_cpc, _lb) in enumerate(zip(_cpc_list, _last_bucket)):
                _color = CPC_COLORS.get(_cpc, "#888888")
                _ax.plot(_lb, _i, marker="D", color=_color, markersize=3, markeredgecolor="black",
                         markeredgewidth=0.3)

            _ax.set_ylabel("Patients (sorted by trajectory length)")
            _ax.set_title(_title, fontsize=12, fontweight="bold")
            _ax.set_xlim(-0.5, min(_max_t, 120) - 0.5)

            # Day lines
            for _h in range(24, 121, 24):
                _ax.axvline(_h - 0.5, color="white", linewidth=0.5, alpha=0.5)

        ax10b.set_xlabel("Hours Since First Event")

        # Build legends
        _action_patches = [mpatches.Patch(color=ACTION_COLORS[_a], label=ACTION_LABELS[_a]) for _a in range(4)]
        _cpc_markers = [plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=CPC_COLORS[c],
                                    markersize=6, label=c, markeredgecolor="black", markeredgewidth=0.5)
                        for c in ["CPC1_2", "CPC3", "CPC4", "CPC5"]]

        _legend1 = ax10a.legend(handles=_action_patches, loc="upper right", title="Action",
                                frameon=True, fontsize=8, title_fontsize=9)
        ax10a.add_artist(_legend1)
        ax10b.legend(handles=_cpc_markers, loc="upper right", title="Outcome",
                     frameon=True, fontsize=8, title_fontsize=9)

        fig10.suptitle(f"Patient Action Timelines — {site_name}", fontsize=14, y=1.01)
        fig10.tight_layout()
        save_fig(fig10, "fig10_patient_timelines")

        mo.md("### Figure 10: Patient Timeline Heatmaps\n\n![](figures/fig10_patient_timelines.png)")
    return


# ── Cell 5: Fig 11 — Agreement-Outcome Bar Chart ─────────────────────
@app.cell
def _(bin_summary, ext_bin_summary, has_ext_val, logger, mo, np, plt, save_fig, site_name):
    logger.info("Figure 11: Agreement-outcome relationship...")

    _n_bins = len(bin_summary)
    _bin_labels = bin_summary["agreement_bin"].tolist()
    _cpc_means = bin_summary["mean_cpc_ord_good"].tolist()
    _n_hosps = bin_summary["n_hosp"].tolist()

    # Color gradient: red (low agreement) → green (high agreement)
    _bin_colors = ["#ef4444", "#f97316", "#facc15", "#22c55e"]
    if _n_bins != 4:
        _bin_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, _n_bins))

    fig11, ax11 = plt.subplots(figsize=(8, 5))

    if has_ext_val and len(ext_bin_summary) > 0:
        _width = 0.35
        _x = np.arange(_n_bins)
        _bars1 = ax11.bar(_x - _width / 2, _cpc_means, _width, color=_bin_colors, alpha=0.9,
                          label="Training Site", edgecolor="white")
        _ext_cpc = ext_bin_summary["mean_cpc_ord_good"].tolist()
        _ext_n = ext_bin_summary["n_hosp"].tolist()
        _bars2 = ax11.bar(_x + _width / 2, _ext_cpc, _width, color=_bin_colors, alpha=0.5,
                          label="External Validation", edgecolor="white", hatch="//")
        for _i, (_bar, _n) in enumerate(zip(_bars1, _n_hosps)):
            ax11.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.03,
                      f"n={_n}", ha="center", va="bottom", fontsize=8)
        for _i, (_bar, _n) in enumerate(zip(_bars2, _ext_n)):
            ax11.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.03,
                      f"n={_n}", ha="center", va="bottom", fontsize=8)
        ax11.legend(frameon=False)
    else:
        _x = np.arange(_n_bins)
        _bars = ax11.bar(_x, _cpc_means, 0.6, color=_bin_colors, edgecolor="white", linewidth=0.5)
        for _i, (_bar, _n) in enumerate(zip(_bars, _n_hosps)):
            ax11.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.03,
                      f"n={_n}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax11.set_xticks(_x if has_ext_val and len(ext_bin_summary) > 0 else np.arange(_n_bins))
    ax11.set_xticklabels(_bin_labels, fontsize=10)
    ax11.set_xlabel("RL-Clinician Agreement Rate")
    ax11.set_ylabel("Mean CPC Ordinal Score\n(higher = better neurological outcome)")
    ax11.set_title(f"Agreement-Outcome Relationship — {site_name}")
    ax11.set_ylim(0, max(_cpc_means) * 1.3)
    fig11.tight_layout()
    save_fig(fig11, "fig11_agreement_outcome")

    mo.md("### Figure 11: Agreement-Outcome\n\n![](figures/fig11_agreement_outcome.png)")
    return


# ── Cell 6: Fig 12 — Concordance OR Forest Plot ──────────────────────
@app.cell
def _(coef_summary, ext_coef_summary, has_ext_val, logger, mo, np, plt, save_fig, site_name):
    logger.info("Figure 12: Concordance OR forest plot...")

    # Parse OR data
    _or = coef_summary.iloc[0]["odds_ratio"] if "odds_ratio" in coef_summary.columns else coef_summary.iloc[0].get("OR", 1)
    _ci_low = coef_summary.iloc[0].get("or_95ci_low", coef_summary.iloc[0].get("OR_95CI_low", 1))
    _ci_high = coef_summary.iloc[0].get("or_95ci_high", coef_summary.iloc[0].get("OR_95CI_high", 1))
    _pval = coef_summary.iloc[0].get("p_value", coef_summary.iloc[0].get("pvalue", 0))

    _labels = [f"Training ({site_name})"]
    _ors = [_or]
    _lows = [_ci_low]
    _highs = [_ci_high]
    _pvals = [_pval]

    if has_ext_val and len(ext_coef_summary) > 0:
        _ext_or = ext_coef_summary.iloc[0].get("odds_ratio", ext_coef_summary.iloc[0].get("OR", 1))
        _ext_ci_low = ext_coef_summary.iloc[0].get("or_95ci_low", ext_coef_summary.iloc[0].get("OR_95CI_low", 1))
        _ext_ci_high = ext_coef_summary.iloc[0].get("or_95ci_high", ext_coef_summary.iloc[0].get("OR_95CI_high", 1))
        _ext_pval = ext_coef_summary.iloc[0].get("p_value", ext_coef_summary.iloc[0].get("pvalue", 0))
        _labels.append("External Validation")
        _ors.append(_ext_or)
        _lows.append(_ext_ci_low)
        _highs.append(_ext_ci_high)
        _pvals.append(_ext_pval)

    fig12, ax12 = plt.subplots(figsize=(8, max(3, len(_labels) * 1.5 + 1)))

    _y = np.arange(len(_labels))
    _colors = ["#1565C0", "#E53935"]

    for _i, (_label, _o, _lo, _hi, _p) in enumerate(zip(_labels, _ors, _lows, _highs, _pvals)):
        ax12.errorbar(_o, _i, xerr=[[_o - _lo], [_hi - _o]], fmt="o", color=_colors[_i],
                      markersize=10, capsize=5, linewidth=2, capthick=2)
        _pstr = f"p < 0.001" if _p < 0.001 else f"p = {_p:.3f}"
        ax12.text(_hi + (_hi - _lo) * 0.1, _i,
                  f"OR = {_o:.1f} ({_lo:.1f}–{_hi:.1f}), {_pstr}",
                  va="center", fontsize=10)

    ax12.axvline(1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax12.set_yticks(_y)
    ax12.set_yticklabels(_labels, fontsize=11)
    ax12.set_xlabel("Odds Ratio (Agreement Rate → Better CPC)")
    ax12.set_title("Concordance-Outcome: Ordinal Logistic Regression")
    ax12.set_xscale("log")
    ax12.invert_yaxis()

    fig12.tight_layout()
    save_fig(fig12, "fig12_concordance_or")

    mo.md("### Figure 12: Concordance OR Forest Plot\n\n![](figures/fig12_concordance_or.png)")
    return


# ── Cell 7: Fig 13 — Updated CONSORT Diagram ─────────────────────────
@app.cell
def _(logger, mo, np, plt, save_fig, site_name, strobe_df):
    logger.info("Figure 13: Updated CONSORT diagram...")

    # Parse STROBE counts
    def _get_count(counter_name):
        _row = strobe_df[strobe_df["counter"] == counter_name]
        return int(_row["value"].iloc[0]) if len(_row) > 0 else 0

    _n_cardiac = _get_count("1_cardiac_arrest_encounters")
    _n_ohca = _get_count("2_ohca_encounters")
    _n_excluded_ihca = _get_count("2_excluded_ihca_unknown")
    _n_first = _get_count("3_first_encounter_patients")
    _n_excluded_repeat = _get_count("3_excluded_repeat_encounters")
    _n_icu = _get_count("4_icu_admitted_patients")
    _n_excluded_no_icu = _get_count("4_excluded_no_icu")
    _n_vaso = _get_count("5_vaso_patients")
    _n_excluded_no_vaso = _get_count("5_excluded_no_vaso")
    _n_vaso_surv = _get_count("5_vaso_survivors")
    _n_vaso_dead = _get_count("5_vaso_non_survivors")

    # Build CONSORT diagram
    fig13, ax13 = plt.subplots(figsize=(10, 14))
    ax13.set_xlim(0, 10)
    ax13.set_ylim(0, 16)
    ax13.axis("off")

    _box_style = dict(boxstyle="round,pad=0.5", facecolor="#1565C0", edgecolor="#0D47A1", alpha=0.9)
    _excl_style = dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2", edgecolor="#E53935", alpha=0.8)
    _final_style = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32", alpha=0.9)

    # Step 1: All Cardiac Arrest
    ax13.text(5, 15, f"All Cardiac Arrest Encounters\nn = {_n_cardiac:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="white", bbox=_box_style)

    # Arrow
    ax13.annotate("", xy=(5, 13.5), xytext=(5, 14.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Step 2: OHCA
    ax13.text(5, 13, f"OHCA (Present on Admission)\nn = {_n_ohca:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="white", bbox=_box_style)
    ax13.text(8.5, 14, f"Excluded: IHCA/Unknown\nn = {_n_excluded_ihca:,}",
              ha="center", va="center", fontsize=9, color="#B71C1C", bbox=_excl_style)
    ax13.annotate("", xy=(7, 14), xytext=(6.3, 14),
                  arrowprops=dict(arrowstyle="->", color="#E53935", lw=1))

    # Arrow
    ax13.annotate("", xy=(5, 11.5), xytext=(5, 12.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Step 3: First Encounter
    ax13.text(5, 11, f"First Encounter per Patient\nn = {_n_first:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="white", bbox=_box_style)
    ax13.text(8.5, 12, f"Excluded: Repeat\nn = {_n_excluded_repeat:,}",
              ha="center", va="center", fontsize=9, color="#B71C1C", bbox=_excl_style)
    ax13.annotate("", xy=(7, 12), xytext=(6.3, 12),
                  arrowprops=dict(arrowstyle="->", color="#E53935", lw=1))

    # Arrow
    ax13.annotate("", xy=(5, 9.5), xytext=(5, 10.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Step 4: ICU Admitted
    ax13.text(5, 9, f"ICU Admitted\nn = {_n_icu:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="white", bbox=_box_style)
    ax13.text(8.5, 10, f"Excluded: No ICU\nn = {_n_excluded_no_icu:,}",
              ha="center", va="center", fontsize=9, color="#B71C1C", bbox=_excl_style)
    ax13.annotate("", xy=(7, 10), xytext=(6.3, 10),
                  arrowprops=dict(arrowstyle="->", color="#E53935", lw=1))

    # Arrow
    ax13.annotate("", xy=(5, 7.5), xytext=(5, 8.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Step 5: Vasopressor Population
    ax13.text(5, 7, f"Vasopressor Population\nn = {_n_vaso:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="white", bbox=_box_style)
    ax13.text(8.5, 8, f"Excluded: No Vasopressors\nn = {_n_excluded_no_vaso:,}",
              ha="center", va="center", fontsize=9, color="#B71C1C", bbox=_excl_style)
    ax13.annotate("", xy=(7, 8), xytext=(6.3, 8),
                  arrowprops=dict(arrowstyle="->", color="#E53935", lw=1))

    # Arrow to outcomes
    ax13.annotate("", xy=(3.5, 5.5), xytext=(5, 6.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax13.annotate("", xy=(6.5, 5.5), xytext=(5, 6.3),
                  arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Final outcomes
    ax13.text(3.5, 5, f"Survivors\nn = {_n_vaso_surv:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="#1B5E20", bbox=_final_style)
    _dead_style = dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#C62828", alpha=0.9)
    ax13.text(6.5, 5, f"Non-Survivors\nn = {_n_vaso_dead:,}",
              ha="center", va="center", fontsize=11, fontweight="bold",
              color="#B71C1C", bbox=_dead_style)

    # Title
    ax13.set_title(f"CONSORT Flow Diagram — {site_name}", fontsize=14, fontweight="bold", pad=20)

    fig13.tight_layout()
    save_fig(fig13, "fig13_consort_updated")

    mo.md("### Figure 13: Updated CONSORT Diagram\n\n![](figures/fig13_consort_updated.png)")
    return


# ── Cell 8: Fig 14 — Action Confusion Matrix ─────────────────────────
@app.cell
def _(ACTION_LABELS, has_predictions, logger, mo, np, pd, plt, pred_df, save_fig, site_name, sns):
    if not has_predictions:
        mo.md("### Figure 14: Skipped (no test_with_predictions.parquet)")
    else:
        logger.info("Figure 14: Action confusion matrix...")

        # Build confusion matrix
        _actions = list(range(4))
        _conf = np.zeros((4, 4), dtype=int)
        for _, _row in pred_df.iterrows():
            _a_obs = int(_row["action"])
            _a_pred = int(_row["pred_action"])
            if 0 <= _a_obs < 4 and 0 <= _a_pred < 4:
                _conf[_a_obs, _a_pred] += 1

        _conf_pct = _conf / _conf.sum() * 100
        _conf_row_pct = _conf / _conf.sum(axis=1, keepdims=True) * 100

        fig14, ax14 = plt.subplots(figsize=(7, 6))

        # Heatmap with percentage labels
        sns.heatmap(_conf_row_pct, annot=False, fmt=".1f", cmap="Blues",
                    xticklabels=[ACTION_LABELS[_a] for _a in _actions],
                    yticklabels=[ACTION_LABELS[_a] for _a in _actions],
                    ax=ax14, vmin=0, vmax=100, linewidths=0.5, linecolor="white")

        # Add count + percentage annotations
        for _i in range(4):
            for _j in range(4):
                _count = _conf[_i, _j]
                _pct = _conf_row_pct[_i, _j]
                _color = "white" if _pct > 50 else "black"
                ax14.text(_j + 0.5, _i + 0.5, f"{_count:,}\n({_pct:.1f}%)",
                          ha="center", va="center", fontsize=9, color=_color, fontweight="bold")

        ax14.set_xlabel("RL Recommended Action", fontsize=12)
        ax14.set_ylabel("Clinician Action", fontsize=12)
        ax14.set_title(f"Action Agreement Matrix — {site_name}")

        fig14.tight_layout()
        save_fig(fig14, "fig14_action_confusion_matrix")

        mo.md("### Figure 14: Action Confusion Matrix\n\n![](figures/fig14_action_confusion_matrix.png)")
    return


# ── Cell 9: Summary Statistics & Manifest ─────────────────────────────
@app.cell
def _(bin_summary, coef_summary, fig_dir, final_dir, has_predictions, history_df, logger, mo, np, pd, pred_df, site_name):
    logger.info("Building summary statistics...")

    # Gather key metrics
    _best_idx = history_df["test_loss"].idxmin()
    _best_epoch = int(history_df.loc[_best_idx, "epoch"])
    _best_test_loss = float(history_df.loc[_best_idx, "test_loss"])
    _best_train_loss = float(history_df.loc[_best_idx, "train_loss"])

    _or = coef_summary.iloc[0].get("odds_ratio", coef_summary.iloc[0].get("OR", None))
    _pval = coef_summary.iloc[0].get("p_value", coef_summary.iloc[0].get("pvalue", None))

    _agreement = None
    if has_predictions and len(pred_df) > 0:
        _agreement = float((pred_df["action"] == pred_df["pred_action"]).mean())

    _summary_rows = [
        {"metric": "Site", "value": site_name},
        {"metric": "Total epochs", "value": str(len(history_df))},
        {"metric": "Best epoch", "value": str(_best_epoch)},
        {"metric": "Best train loss", "value": f"{_best_train_loss:.4f}"},
        {"metric": "Best test loss", "value": f"{_best_test_loss:.4f}"},
        {"metric": "Overall agreement", "value": f"{_agreement:.1%}" if _agreement else "N/A"},
        {"metric": "Concordance OR", "value": f"{_or:.1f}" if _or else "N/A"},
        {"metric": "Concordance p-value", "value": f"{_pval:.2e}" if _pval else "N/A"},
    ]
    _summary = pd.DataFrame(_summary_rows)
    _summary.to_csv(final_dir / "training_results_summary.csv", index=False)

    # File manifest
    _figs = sorted(fig_dir.glob("fig[8-9]_*.p*")) + sorted(fig_dir.glob("fig1[0-4]_*.p*"))
    _fig_list = "\n".join(f"    | `{p.name}` |" for p in _figs)

    logger.info("All post-training figures saved: %d files", len(_figs))

    mo.md(f"""
    ## Post-Training Results Summary

    {_summary.to_markdown(index=False)}

    ## Generated Figures

    | File |
    |------|
{_fig_list}

    **Location**: `{fig_dir}`
    """)
    return


if __name__ == "__main__":
    app.run()
