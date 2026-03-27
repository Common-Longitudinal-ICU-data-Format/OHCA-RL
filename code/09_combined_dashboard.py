# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="OHCA-RL Multi-Site Dashboard")


# ── Cell 0: Imports & Site Discovery ──────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import base64
    import io
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils import setup_logging
    logger = setup_logging("09_combined_dashboard")

    project_root = Path(__file__).parent.parent.resolve()
    all_site_dir = project_root / "all_site_data"
    output_dir = project_root / "output" / "final"
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover sites
    site_names = sorted([
        d.name for d in all_site_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    # Detect training site by presence of training_config.json
    training_site = None
    for _s in site_names:
        if (all_site_dir / _s / "training_config.json").exists():
            training_site = _s
            break

    # Move training site to end so it always appears last in tables/charts
    if training_site and training_site in site_names:
        site_names.remove(training_site)
        site_names.append(training_site)

    SITE_COLORS = {
        "ucmc": "#1565C0",
        "rush": "#E53935",
        "emory": "#43A047",
        "nu": "#757575",
        "ucsf": "#FF9800",
    }
    SITE_LABELS = {
        "ucmc": "UCMC (Training)",
        "rush": "Rush",
        "emory": "Emory",
        "nu": "NU",
        "ucsf": "UCSF",
    }
    ACTION_LABELS = {0: "Increase", 1: "Decrease", 2: "Stop", 3: "Stay"}
    ACTION_COLORS = {
        0: "#E53935",
        1: "#42A5F5",
        2: "#66BB6A",
        3: "#F8BBD0",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.2,
    })

    logger.info("Discovered sites: %s (training=%s)", site_names, training_site)
    mo.md(f"# Multi-Site Combined Dashboard\n**Sites:** {', '.join(site_names)}\n**Training site:** {training_site}")
    return (
        ACTION_COLORS, ACTION_LABELS, SITE_COLORS, SITE_LABELS,
        all_site_dir, base64, io, json, logger, mo, np, os, output_dir,
        pd, plt, project_root, site_names, training_site,
    )


# ── Cell 1: Load All Site Data ────────────────────────────────────────
@app.cell
def _(all_site_dir, base64, json, logger, mo, pd, site_names):
    def fig_to_data_uri(path):
        if not path.exists():
            return None
        with open(path, "rb") as _f:
            _data = base64.b64encode(_f.read()).decode("utf-8")
        return f"data:image/png;base64,{_data}"

    def safe_read_csv(path):
        if path.exists():
            return pd.read_csv(path)
        return None

    def safe_read_json(path):
        if path.exists():
            with open(path, "r") as _f:
                return json.load(_f)
        return None

    site_data = {}
    for _site in site_names:
        _dir = all_site_dir / _site
        _d = {}

        # Root-level CSVs
        for _key in ["table1_ohca_long", "table1_ohca_vaso_long",
                      "strobe_counts", "action_distribution",
                      "feature_summary", "missingness_comparison"]:
            _d[_key] = safe_read_csv(_dir / f"{_key}.csv")

        # Training-only files (UCMC)
        for _key in ["training_history", "training_results_summary",
                      "test_action_summary", "bin_summary", "coef_summary"]:
            _d[_key] = safe_read_csv(_dir / f"{_key}.csv")
        _d["training_config"] = safe_read_json(_dir / "training_config.json")

        # External validation
        _ext_dir = _dir / "external_validation"
        _d["ext_val_metadata"] = safe_read_json(_ext_dir / "evaluation_metadata.json")
        _d["ext_val_coef"] = safe_read_csv(_ext_dir / "coef_summary.csv")
        _d["ext_val_adjusted_coef"] = safe_read_csv(_ext_dir / "adjusted_coef_summary.csv")
        _d["ext_val_bin"] = safe_read_csv(_ext_dir / "bin_summary.csv")
        _d["ext_val_action"] = safe_read_csv(_ext_dir / "action_summary.csv")

        # Figures (root)
        _d["figures"] = {}
        _fig_dir = _dir / "figures"
        if _fig_dir.exists():
            for _png in sorted(_fig_dir.glob("*.png")):
                _d["figures"][_png.stem] = fig_to_data_uri(_png)

        # External validation figures
        _d["ext_val_figures"] = {}
        _ext_fig_dir = _ext_dir / "figures"
        if _ext_fig_dir.exists():
            for _png in sorted(_ext_fig_dir.glob("*.png")):
                _d["ext_val_figures"][_png.stem] = fig_to_data_uri(_png)

        site_data[_site] = _d
        _n_figs = sum(1 for v in _d["figures"].values() if v)
        _n_ext_figs = sum(1 for v in _d["ext_val_figures"].values() if v)
        logger.info("Loaded %s: %d figs, %d ext_val figs, ext_val=%s",
                     _site, _n_figs, _n_ext_figs,
                     "yes" if _d["ext_val_metadata"] else "no")

    mo.md(f"Loaded data for **{len(site_data)}** sites: {', '.join(site_data.keys())}")
    return (site_data, fig_to_data_uri, safe_read_csv, safe_read_json)


# ── Cell 2: Generate Combined Figures ─────────────────────────────────
@app.cell
def _(
    ACTION_COLORS, ACTION_LABELS, SITE_COLORS, SITE_LABELS,
    all_site_dir, base64, io, logger, mo, np, pd, plt, site_data, site_names,
):
    def fig_to_base64(fig):
        _buf = io.BytesIO()
        fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
        _buf.seek(0)
        _data = base64.b64encode(_buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{_data}"

    # ── Fig A: Multi-Site Forest Plot (per-10pp OR) — Unadjusted vs Adjusted ──
    from matplotlib.lines import Line2D

    # Collect data for both unadjusted and adjusted (core)
    _forest_rows = []
    for _site in site_names:
        _color = SITE_COLORS.get(_site, "#666")
        _label = SITE_LABELS.get(_site, _site.upper())

        # Unadjusted
        _coef = site_data[_site].get("ext_val_coef")
        if _coef is not None and len(_coef) > 0:
            _row = _coef.iloc[0]
            _forest_rows.append({
                "site": _site, "label": _label, "model": "Unadjusted",
                "or_10pp": float(_row["or_10pp"]),
                "ci_low": float(_row["or_10pp_ci_low"]),
                "ci_high": float(_row["or_10pp_ci_high"]),
                "pval": float(_row["p_value"]),
                "color": _color,
            })

        # Adjusted (core) — from adjusted_coef_summary.csv
        _adj_coef = site_data[_site].get("ext_val_adjusted_coef")
        if _adj_coef is not None and len(_adj_coef) > 0:
            _m2 = _adj_coef[
                (_adj_coef["term"] == "agreement_rate")
                & (_adj_coef["model"] == "M2: Adjusted (core)")
            ]
            if len(_m2) > 0:
                _row = _m2.iloc[0]
                _forest_rows.append({
                    "site": _site, "label": _label, "model": "Adjusted",
                    "or_10pp": float(_row["or_10pp"]),
                    "ci_low": float(_row["or_10pp_ci_low"]),
                    "ci_high": float(_row["or_10pp_ci_high"]),
                    "pval": float(_row["p_value"]),
                    "color": _color,
                })

    _has_adjusted = any(r["model"] == "Adjusted" for r in _forest_rows)

    # Build y-positions: pair unadjusted/adjusted on same row per site
    _seen_sites = []
    for _r in _forest_rows:
        if _r["site"] not in _seen_sites:
            _seen_sites.append(_r["site"])

    if _has_adjusted:
        # Compact layout: one row per site, unadjusted + adjusted share the row
        _y_positions = []
        _y_labels = []
        _y_pos = 0
        for _site in _seen_sites:
            _site_rows = [r for r in _forest_rows if r["site"] == _site]
            _site_label = _site_rows[0]["label"]
            _y_labels.append(_site_label)
            _y_positions.append(_y_pos)
            _offset = 0.12  # vertical offset within the row
            for _r in _site_rows:
                if _r["model"] == "Unadjusted":
                    _r["y"] = _y_pos - _offset
                else:
                    _r["y"] = _y_pos + _offset
            _y_pos += 1

        _fig_a, _ax_a = plt.subplots(figsize=(12, max(3, len(_seen_sites) * 1.4 + 1)))

        for _r in _forest_rows:
            _fmt = "o" if _r["model"] == "Unadjusted" else "D"
            _fill = "white" if _r["model"] == "Unadjusted" else _r["color"]
            _ms = 9 if _r["model"] == "Unadjusted" else 10
            _ax_a.errorbar(
                _r["or_10pp"], _r["y"],
                xerr=[[_r["or_10pp"] - _r["ci_low"]], [_r["ci_high"] - _r["or_10pp"]]],
                fmt=_fmt, color=_r["color"], markerfacecolor=_fill,
                markersize=_ms, capsize=5, linewidth=1.8, capthick=1.8,
                markeredgewidth=2,
            )
            _pstr = "p < 0.001" if _r["pval"] < 0.001 else f"p = {_r['pval']:.3f}"
            _ax_a.text(
                _r["ci_high"] + 0.01, _r["y"],
                f" {_r['or_10pp']:.2f} [{_r['ci_low']:.2f}\u2013{_r['ci_high']:.2f}]",
                va="center", fontsize=9,
            )

        _ax_a.axvline(1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        _ax_a.set_yticks(_y_positions)
        _ax_a.set_yticklabels(_y_labels, fontsize=11, fontweight="bold")
        # Add light horizontal dividers between sites
        for _yp in _y_positions[:-1]:
            _ax_a.axhline(_yp + 0.5, color="#e0e0e0", linewidth=0.5)

        _legend_handles = [
            Line2D([0], [0], marker="o", color="grey", markerfacecolor="white",
                   markersize=9, markeredgewidth=2, linestyle="None", label="Unadjusted"),
            Line2D([0], [0], marker="D", color="grey", markerfacecolor="grey",
                   markersize=10, markeredgewidth=2, linestyle="None",
                   label="Adjusted (age, sex, SOFA)"),
        ]
        _ax_a.legend(handles=_legend_handles, loc="upper left", fontsize=10,
                     frameon=True, fancybox=True, shadow=False)
    else:
        # Fallback: unadjusted only (no adjusted data available)
        _y_positions = list(range(len(_forest_rows)))
        _y_labels = [r["label"] for r in _forest_rows]
        for _i, _r in enumerate(_forest_rows):
            _r["y"] = _i

        _fig_a, _ax_a = plt.subplots(figsize=(10, max(3, len(_forest_rows) * 1.5 + 1)))
        for _r in _forest_rows:
            _ax_a.errorbar(
                _r["or_10pp"], _r["y"],
                xerr=[[_r["or_10pp"] - _r["ci_low"]], [_r["ci_high"] - _r["or_10pp"]]],
                fmt="D", color=_r["color"],
                markersize=10, capsize=6, linewidth=2, capthick=2,
            )
            _pstr = "p < 0.001" if _r["pval"] < 0.001 else f"p = {_r['pval']:.3f}"
            _ax_a.text(
                _r["ci_high"] + 0.02, _r["y"],
                f"  {_r['or_10pp']:.2f} [{_r['ci_low']:.2f}\u2013{_r['ci_high']:.2f}], {_pstr}",
                va="center", fontsize=10,
            )
        _ax_a.axvline(1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        _ax_a.set_yticks(_y_positions)
        _ax_a.set_yticklabels(_y_labels, fontsize=11)

    _ax_a.set_xlabel("Odds Ratio per 10pp Agreement Increase", fontsize=11)
    _ax_a.set_title("Multi-Site Concordance: Agreement Rate \u2192 Better CPC Outcome",
                     fontsize=13, fontweight="bold")
    _ax_a.invert_yaxis()
    _fig_a.tight_layout()
    _forest_uri = fig_to_base64(_fig_a)

    # ── Fig B: Combined Agreement-Outcome ──
    _bin_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    _n_bins = len(_bin_labels)
    _sites_with_bins = [s for s in site_names if site_data[s].get("ext_val_bin") is not None]
    _n_sites_b = len(_sites_with_bins)

    _fig_b, _ax_b = plt.subplots(figsize=(10, 6))
    _width_b = 0.8 / max(_n_sites_b, 1)
    _x_b = np.arange(_n_bins)

    _site_idx_b = 0
    for _site in _sites_with_bins:
        _bins_df = site_data[_site]["ext_val_bin"]
        _cpc_vals = _bins_df["mean_cpc_ord_good"].tolist()
        _n_hosps = _bins_df["n_hosp"].tolist()
        _offset = (_site_idx_b - (_n_sites_b - 1) / 2) * _width_b
        _bars = _ax_b.bar(_x_b + _offset, _cpc_vals, _width_b,
                          label=SITE_LABELS.get(_site, _site.upper()),
                          color=SITE_COLORS.get(_site, "#666"),
                          alpha=0.85, edgecolor="white")
        for _bar, _n in zip(_bars, _n_hosps):
            _ax_b.text(_bar.get_x() + _bar.get_width() / 2,
                       _bar.get_height() + 0.02,
                       f"n={_n}", ha="center", va="bottom", fontsize=7)
        _site_idx_b += 1

    _ax_b.set_xticks(_x_b)
    _ax_b.set_xticklabels(_bin_labels)
    _ax_b.set_xlabel("RL-Clinician Agreement Rate")
    _ax_b.set_ylabel("Mean CPC Ordinal Score (higher = better)")
    _ax_b.set_title("Agreement-Outcome Relationship Across Sites",
                     fontsize=13, fontweight="bold")
    _ax_b.legend(frameon=True, fontsize=10)
    _ax_b.set_ylim(0)
    _fig_b.tight_layout()
    _agreement_uri = fig_to_base64(_fig_b)

    # ── Fig C: Combined Action Distribution (1x3 subplots) ──
    _sites_with_actions = [s for s in site_names
                           if site_data[s].get("ext_val_action") is not None]
    _n_c = len(_sites_with_actions)

    _fig_c, _axes_c = plt.subplots(1, _n_c, figsize=(5 * _n_c, 5), sharey=True)
    if _n_c == 1:
        _axes_c = [_axes_c]

    for _idx, _site in enumerate(_sites_with_actions):
        _ax = _axes_c[_idx]
        _adf = site_data[_site]["ext_val_action"]
        _total_pred = _adf["pred_count"].sum()
        _total_obs = _adf["obs_count"].sum()
        _actions = sorted(_adf.index.tolist())
        _pred_pcts = [(_adf.loc[a, "pred_count"] / _total_pred * 100) for a in _actions]
        _obs_pcts = [(_adf.loc[a, "obs_count"] / _total_obs * 100) for a in _actions]
        _x_c = np.arange(4)
        _w_c = 0.6
        _colors_c = [ACTION_COLORS[a] for a in _actions]
        _act_labels = [ACTION_LABELS[a] for a in _actions]

        _ax.bar(_x_c, _pred_pcts, _w_c, color=_colors_c, alpha=0.9, edgecolor="white")
        _ax.bar(_x_c, [-p for p in _obs_pcts], _w_c, color=_colors_c, alpha=0.5, edgecolor="white")

        for _j, _pct in enumerate(_pred_pcts):
            _ax.text(_j, _pct + 0.5, f"{_pct:.1f}%", ha="center", fontsize=8, fontweight="bold")
        for _j, _pct in enumerate(_obs_pcts):
            _ax.text(_j, -_pct - 0.5, f"{_pct:.1f}%", ha="center", va="top", fontsize=8, fontweight="bold")

        _ax.axhline(0, color="black", linewidth=1)
        _ax.set_xticks(_x_c)
        _ax.set_xticklabels(_act_labels, fontsize=9)
        _ax.set_title(SITE_LABELS.get(_site, _site.upper()), fontsize=11, fontweight="bold")

        _max_pct = max(max(_pred_pcts), max(_obs_pcts)) * 1.3
        _ax.set_ylim(-_max_pct, _max_pct)
        _yticks = _ax.get_yticks()
        _ax.set_yticks(_yticks)
        _ax.set_yticklabels([f"{abs(y):.0f}" for y in _yticks])

        if _idx == 0:
            _ax.set_ylabel("Percent (%)")

    _fig_c.suptitle("Vasopressor Action Distribution: RL Agent vs Clinician",
                     fontsize=13, fontweight="bold", y=1.02)
    _fig_c.tight_layout(rect=[0.05, 0, 1, 1])  # reserve 5% left margin for labels
    _fig_c.text(0.01, 0.72, "RL Agent \u2191", fontsize=10, fontweight="bold",
                color="#333", va="center", ha="left", rotation=90)
    _fig_c.text(0.01, 0.28, "\u2193 Clinicians", fontsize=10, fontweight="bold",
                color="#333", va="center", ha="left", rotation=90)
    _action_dist_uri = fig_to_base64(_fig_c)

    # ── Fig D & E: Combined Missingness Heatmaps (Before & After) ──
    import seaborn as sns

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

    # Collect missingness data from all sites
    _miss_frames_before = []
    _miss_frames_after = []
    for _site in site_names:
        _mc = site_data[_site].get("missingness_comparison")
        if _mc is None:
            continue
        _mc = _mc.copy()
        _miss_frames_before.append(
            _mc[["variable", "patient_missing_pct_before"]].rename(
                columns={"patient_missing_pct_before": _site}
            )
        )
        _miss_frames_after.append(
            _mc[["variable", "patient_missing_pct_after"]].rename(
                columns={"patient_missing_pct_after": _site}
            )
        )

    _miss_before_uri = None
    _miss_after_uri = None

    if _miss_frames_before:
        # Merge all sites on variable name
        from functools import reduce
        _before_merged = reduce(lambda l, r: pd.merge(l, r, on="variable", how="outer"),
                                _miss_frames_before).fillna(0)
        _after_merged = reduce(lambda l, r: pd.merge(l, r, on="variable", how="outer"),
                               _miss_frames_after).fillna(0)

        # Add group for sorting, filter to non-zero missingness
        _before_merged["group"] = _before_merged["variable"].apply(_var_group)
        _after_merged["group"] = _after_merged["variable"].apply(_var_group)

        _site_cols = [s for s in site_names if s in _before_merged.columns]

        # Before: keep rows where any site has >0 missingness
        _mask_before = _before_merged[_site_cols].max(axis=1) > 0
        _before_plot = _before_merged[_mask_before].sort_values(["group", "variable"]).copy()

        # After: keep rows where any site has >0 missingness
        _mask_after = _after_merged[_site_cols].max(axis=1) > 0
        _after_plot = _after_merged[_mask_after].sort_values(["group", "variable"]).copy()

        # Helper to build a multi-site missingness heatmap
        def _build_missingness_heatmap(plot_df, title):
            _vars = plot_df["variable"].tolist()
            _groups = plot_df["group"].tolist()
            _data = plot_df[_site_cols].values  # rows=variables, cols=sites
            _site_labels = [SITE_LABELS.get(s, s.upper()) for s in _site_cols]

            _fig, _ax = plt.subplots(figsize=(max(5, len(_site_cols) * 2.5), max(5, len(_vars) * 0.32)))
            _im = _ax.imshow(_data, cmap="Reds", aspect="auto", vmin=0, vmax=100)

            # Annotate cells
            for _i in range(len(_vars)):
                for _j in range(len(_site_cols)):
                    _val = _data[_i, _j]
                    _color = "white" if _val > 60 else "black"
                    _ax.text(_j, _i, f"{_val:.0f}", ha="center", va="center",
                             fontsize=8, color=_color, fontweight="bold" if _val > 0 else "normal")

            _ax.set_xticks(range(len(_site_labels)))
            _ax.set_xticklabels(_site_labels, fontsize=10, fontweight="bold")
            _ax.xaxis.set_ticks_position("top")
            _ax.xaxis.set_label_position("top")
            _ax.set_yticks(range(len(_vars)))
            _ax.set_yticklabels(_vars, fontsize=8)

            # Group separators
            _prev = None
            for _i, _g in enumerate(_groups):
                if _prev is not None and _g != _prev:
                    _ax.axhline(_i - 0.5, color="black", linewidth=1.5)
                _prev = _g

            _cbar = _fig.colorbar(_im, ax=_ax, shrink=0.7, pad=0.02)
            _cbar.set_label("Patient-Level Missingness (%)", fontsize=9)
            _ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
            _fig.tight_layout()
            return _fig

        if len(_before_plot) > 0:
            _fig_d = _build_missingness_heatmap(
                _before_plot, "Patient-Level Missingness Before Imputation"
            )
            _miss_before_uri = fig_to_base64(_fig_d)

        if len(_after_plot) > 0:
            _fig_e = _build_missingness_heatmap(
                _after_plot, "Patient-Level Missingness After Imputation"
            )
            _miss_after_uri = fig_to_base64(_fig_e)

    # ── Fig F: Cross-Site Clinical Trajectories (hourly) ──
    # Each panel: median line per site with IQR ribbon (continuous) or proportion (binary)
    _traj_panels = [
        # (csv_file, variable_filter, y_col, title, ylabel, is_proportion)
        ("fig5_vital_signs.csv", "vital_heart_rate", "median", "Heart Rate", "bpm", False),
        ("fig5_vital_signs.csv", "vital_map", "median", "MAP", "mmHg", False),
        ("fig7_labs.csv", "lab_lactate", "median", "Lactate", "mmol/L", False),
        ("fig2_vasopressor_nee.csv", "med_cont_nee", "median", "NEE Dose (patients on vaso)", "mcg/kg/min", False),
        ("fig3_treatment_timelines.csv", "on_vaso", "proportion", "On Vasopressors", "%", True),
        ("fig3_treatment_timelines.csv", "on_imv", "proportion", "On Mechanical Ventilation", "%", True),
    ]

    _fig_f, _axes_f = plt.subplots(3, 2, figsize=(14, 12))
    _axes_flat = _axes_f.flatten()

    for _p_idx, (_csv, _var, _ycol, _title, _ylabel, _is_pct) in enumerate(_traj_panels):
        _ax = _axes_flat[_p_idx]
        for _site in site_names:
            _csv_path = all_site_dir / _site / "figures" / _csv
            if not _csv_path.exists():
                continue
            _df = pd.read_csv(_csv_path)
            _df = _df[(_df["group"] == "overall") & (_df["variable"] == _var)].copy()
            if len(_df) == 0:
                continue
            _df = _df.sort_values("time_bucket")
            _x = _df["time_bucket"].values
            _y = _df[_ycol].astype(float).values
            if _is_pct:
                _y = _y * 100  # proportion → %

            _color = SITE_COLORS.get(_site, "#666")
            _label = SITE_LABELS.get(_site, _site.upper())
            _ax.plot(_x, _y, color=_color, label=_label, linewidth=1.5, alpha=0.85)

            # IQR ribbon for continuous variables
            if not _is_pct and "q25" in _df.columns and "q75" in _df.columns:
                _q25 = pd.to_numeric(_df["q25"], errors="coerce").values
                _q75 = pd.to_numeric(_df["q75"], errors="coerce").values
                _valid = ~(np.isnan(_q25) | np.isnan(_q75))
                if _valid.any():
                    _ax.fill_between(_x[_valid], _q25[_valid], _q75[_valid],
                                     color=_color, alpha=0.10)

        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _ax.set_ylabel(_ylabel, fontsize=9)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _p_idx >= 4:  # bottom row
            _ax.set_xlabel("Hours from ICU Admission", fontsize=9)

    # Single shared legend from first panel
    _handles, _labels_leg = _axes_flat[0].get_legend_handles_labels()
    _fig_f.legend(_handles, _labels_leg, loc="upper center", ncol=len(site_names),
                  fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    _fig_f.suptitle("Clinical Trajectories Across Sites (Overall Cohort)",
                     fontsize=14, fontweight="bold", y=1.05)
    _fig_f.tight_layout(rect=[0, 0, 1, 0.98])
    _clinical_dist_uri = fig_to_base64(_fig_f)

    combined_figures = {
        "forest_plot": _forest_uri,
        "agreement_outcome": _agreement_uri,
        "action_distribution": _action_dist_uri,
        "missingness_before": _miss_before_uri,
        "missingness_after": _miss_after_uri,
        "clinical_distributions": _clinical_dist_uri,
    }

    logger.info("Generated %d combined figures", len([v for v in combined_figures.values() if v]))
    mo.md(f"Generated **{len([v for v in combined_figures.values() if v])}** combined comparison figures")
    return (combined_figures, fig_to_base64)


# ── Cell 3: Build Combined Overview HTML ──────────────────────────────
@app.cell
def _(combined_figures, logger, mo, np, pd, site_data, site_names, training_site, SITE_LABELS):
    # ── Summary Statistics Table ──
    _summary_rows = []
    for _site in site_names:
        _strobe = site_data[_site].get("strobe_counts")
        _meta = site_data[_site].get("ext_val_metadata") or {}
        _coef = site_data[_site].get("ext_val_coef")
        _role = "Training" if _site == training_site else "Validation"

        def _get_strobe(counter):
            if _strobe is None:
                return "N/A"
            _match = _strobe[_strobe["counter"] == counter]
            return f"{int(_match.iloc[0]['value']):,}" if len(_match) > 0 else "N/A"

        def _get_strobe_pct(counter):
            if _strobe is None:
                return "N/A"
            _match = _strobe[_strobe["counter"] == counter]
            return f"{_match.iloc[0]['value']:.1f}%" if len(_match) > 0 else "N/A"

        _n_icu = _get_strobe("4_icu_admitted_patients")
        _n_vaso = _get_strobe("5_vaso_patients")
        _mort = _get_strobe_pct("5_vaso_mortality_pct")
        _n_trans = f"{_meta.get('n_transitions', 0):,}" if _meta else "N/A"
        _agree = f"{_meta.get('overall_agreement', 0):.1%}" if _meta else "N/A"

        # OR per 10pp from coef_summary
        if _coef is not None and len(_coef) > 0:
            _r = _coef.iloc[0]
            _or_str = f"{_r['or_10pp']:.2f} [{_r['or_10pp_ci_low']:.2f}\u2013{_r['or_10pp_ci_high']:.2f}]"
            _p = float(_r["p_value"])
            _p_str = "< 0.001" if _p < 0.001 else f"{_p:.3f}"
        else:
            _or_str = "N/A"
            _p_str = "N/A"

        _summary_rows.append(f"""<tr>
            <td><strong>{SITE_LABELS.get(_site, _site.upper())}</strong></td>
            <td>{_role}</td>
            <td>{_n_icu}</td>
            <td>{_n_vaso}</td>
            <td>{_mort}</td>
            <td>{_n_trans}</td>
            <td>{_agree}</td>
            <td>{_or_str}</td>
            <td>{_p_str}</td>
        </tr>""")

    _summary_table = f"""
    <table class="data-table">
        <tr>
            <th>Site</th><th>Role</th><th>N ICU</th><th>N Vaso</th>
            <th>Mortality</th><th>Patient-Hours</th><th>Agreement</th>
            <th>OR per 10pp [95% CI]</th><th>p-value</th>
        </tr>
        {"".join(_summary_rows)}
    </table>"""

    # ── Concordance Summary Table (Unadjusted vs Adjusted) ──
    _conc_rows = []
    for _site in site_names:
        _label = SITE_LABELS.get(_site, _site.upper())
        _meta = site_data[_site].get("ext_val_metadata") or {}
        _coef = site_data[_site].get("ext_val_coef")
        _adj_coef = site_data[_site].get("ext_val_adjusted_coef")

        _n = f"{_meta.get('n_patients', 'N/A'):,}" if _meta else "N/A"
        _agree = f"{_meta.get('overall_agreement', 0):.1%}" if _meta else "N/A"

        # Unadjusted
        if _coef is not None and len(_coef) > 0:
            _r = _coef.iloc[0]
            _unadj_or = f"{_r['or_10pp']:.2f} [{_r['or_10pp_ci_low']:.2f}\u2013{_r['or_10pp_ci_high']:.2f}]"
            _unadj_p = "< 0.001" if float(_r["p_value"]) < 0.001 else f"{float(_r['p_value']):.3f}"
        else:
            _unadj_or = "N/A"
            _unadj_p = "N/A"

        # Adjusted (core)
        if _adj_coef is not None and len(_adj_coef) > 0:
            _m2 = _adj_coef[
                (_adj_coef["term"] == "agreement_rate")
                & (_adj_coef["model"] == "M2: Adjusted (core)")
            ]
            if len(_m2) > 0:
                _r2 = _m2.iloc[0]
                _adj_or = f"{_r2['or_10pp']:.2f} [{_r2['or_10pp_ci_low']:.2f}\u2013{_r2['or_10pp_ci_high']:.2f}]"
                _adj_p = "< 0.001" if float(_r2["p_value"]) < 0.001 else f"{float(_r2['p_value']):.3f}"
            else:
                _adj_or = "\u2014"
                _adj_p = "\u2014"
        else:
            _adj_or = "\u2014"
            _adj_p = "\u2014"

        _conc_rows.append(f"""<tr>
            <td><strong>{_label}</strong></td>
            <td>{_n}</td>
            <td>{_agree}</td>
            <td>{_unadj_or}</td>
            <td>{_unadj_p}</td>
            <td>{_adj_or}</td>
            <td>{_adj_p}</td>
        </tr>""")

    _concordance_table = f"""
    <table class="data-table">
        <tr>
            <th rowspan="2">Site</th>
            <th rowspan="2">N Patients</th>
            <th rowspan="2">Agreement</th>
            <th colspan="2" style="text-align:center; border-bottom:2px solid #fff;">Unadjusted</th>
            <th colspan="2" style="text-align:center; border-bottom:2px solid #fff;">Adjusted (age, sex, SOFA)</th>
        </tr>
        <tr>
            <th>OR/10pp [95% CI]</th><th>p</th>
            <th>OR/10pp [95% CI]</th><th>p</th>
        </tr>
        {"".join(_conc_rows)}
    </table>
    <p style="font-size: 11px; color: #666; margin-top: 4px;">
        OR/10pp = odds ratio per 10 percentage-point increase in RL-clinician agreement.
        Adjusted model controls for age at admission, sex, and admission SOFA score (first 24h).
    </p>"""

    # ── CONSORT Flow Table ──
    _consort_counters = [
        ("1_all_cardiac_arrest_patients", "All cardiac arrest patients"),
        ("2_ohca_patients", "OHCA patients (present on admission)"),
        ("3_first_encounter_patients", "First encounter per patient"),
        ("4_icu_admitted_patients", "ICU admitted"),
        ("5_vaso_patients", "Vasopressor patients (study cohort)"),
    ]

    _consort_rows = []
    for _counter, _label in _consort_counters:
        _row = [f"<td>{_label}</td>"]
        _total = 0
        for _site in site_names:
            _strobe = site_data[_site].get("strobe_counts")
            if _strobe is not None:
                _match = _strobe[_strobe["counter"] == _counter]
                _val = int(_match.iloc[0]["value"]) if len(_match) > 0 else 0
            else:
                _val = 0
            _row.append(f"<td>{_val:,}</td>")
            _total += _val
        _row.append(f"<td><strong>{_total:,}</strong></td>")
        _consort_rows.append("<tr>" + "".join(_row) + "</tr>")

    _consort_header = "<tr><th>Cohort Step</th>" + "".join(
        f"<th>{SITE_LABELS.get(_s, _s.upper())}</th>" for _s in site_names) + "<th>Total</th></tr>"
    _consort_table = f'<table class="data-table">{_consort_header}{"".join(_consort_rows)}</table>'

    # ── CPC Distribution Table ──
    _cpc_levels = ["CPC1_2", "CPC3", "CPC4", "CPC5"]
    _cpc_labels = {
        "CPC1_2": "CPC 1-2 (Good neurological outcome)",
        "CPC3": "CPC 3 (Severe disability)",
        "CPC4": "CPC 4 (Vegetative state)",
        "CPC5": "CPC 5 (Dead)",
    }

    _cpc_rows = []
    for _cpc in _cpc_levels:
        _row = [f"<td>{_cpc_labels[_cpc]}</td>"]
        _total_n = 0
        _total_denom = 0
        for _site in site_names:
            _t1 = site_data[_site].get("table1_ohca_vaso_long")
            if _t1 is not None:
                _match = _t1[
                    (_t1["subgroup"] == "Overall") &
                    (_t1["variable"].str.strip() == "CPC") &
                    (_t1["level"].fillna("").astype(str).str.strip() == _cpc)
                ]
                if len(_match) > 0:
                    _n = int(_match.iloc[0]["n"])
                    _tot = int(_match.iloc[0]["total"])
                    _pct = _n / _tot * 100 if _tot > 0 else 0
                    _row.append(f"<td>{_n} ({_pct:.1f}%)</td>")
                    _total_n += _n
                    _total_denom += _tot
                else:
                    _row.append("<td>N/A</td>")
            else:
                _row.append("<td>N/A</td>")
        _total_pct = _total_n / _total_denom * 100 if _total_denom > 0 else 0
        _row.append(f"<td><strong>{_total_n} ({_total_pct:.1f}%)</strong></td>")
        _cpc_rows.append("<tr>" + "".join(_row) + "</tr>")

    _cpc_header = "<tr><th>CPC Category</th>" + "".join(
        f"<th>{SITE_LABELS.get(_s, _s.upper())}</th>" for _s in site_names) + "<th>Total</th></tr>"
    _cpc_table = f'<table class="data-table">{_cpc_header}{"".join(_cpc_rows)}</table>'

    # ── Combined Table 1 (side-by-side) ──
    # Variables to exclude from Table 1
    _t1_exclude_vars = {
        "Minimum GCS (over stay)",
        "Median RASS",
    }

    # Categorical levels to exclude per variable (keep only common categories)
    _t1_exclude_levels = {
        "Race": {"american indian or alaska native", "native hawaiian or other pacific islander", "other", "unknown"},
        "Ethnicity": {"unknown"},
        "Sex": {"unknown"},
    }

    def _format_stat(row):
        _st = str(row.get("stat_type", "")).strip()
        if _st == "count":
            return str(int(row["n"]))
        elif _st == "continuous":
            return f"{row['median']:.1f} [{row['q25']:.1f}, {row['q75']:.1f}]"
        elif _st in ("categorical", "binary"):
            _pct = row["n"] / row["total"] * 100 if row["total"] > 0 else 0
            return f"{int(row['n'])} ({_pct:.1f}%)"
        return str(row.get("n", ""))

    def _make_display_label(row):
        """For binary rows (yes/no), show the variable name directly.
           For categorical levels, indent them under the parent variable."""
        _var = str(row["variable"]).strip()
        _level = str(row.get("level", "")).strip() if pd.notna(row.get("level")) else ""
        _stype = str(row.get("stat_type", "")).strip()
        if _stype == "binary":
            # Show variable name (e.g., "Ever CRRT, n (%)") — not the "yes" level
            return _var
        elif _level:
            return f"&nbsp;&nbsp;{_level}"
        else:
            return _var

    _all_t1_rows = []
    for _site in site_names:
        _df = site_data[_site].get("table1_ohca_vaso_long")
        if _df is None:
            continue
        _overall = _df[_df["subgroup"] == "Overall"].copy()
        # Filter out excluded variables
        _overall = _overall[~_overall["variable"].str.strip().isin(_t1_exclude_vars)]
        # Filter out excluded categorical levels
        _overall = _overall[~_overall.apply(
            lambda r: str(r.get("level", "")).strip().lower()
            in _t1_exclude_levels.get(str(r["variable"]).strip(), set()),
            axis=1,
        )]
        _overall["display"] = _overall.apply(_format_stat, axis=1)
        _overall["display_label"] = _overall.apply(_make_display_label, axis=1)
        _overall["variable_clean"] = _overall["variable"].str.strip()
        _overall["level_clean"] = _overall["level"].fillna("").astype(str).str.strip()
        _overall["site_key"] = _site
        _all_t1_rows.append(_overall[["variable_clean", "level_clean", "display_label", "display", "site_key"]])

    if _all_t1_rows:
        _combined_t1 = pd.concat(_all_t1_rows, ignore_index=True)
        _pivot = _combined_t1.pivot_table(
            index=["variable_clean", "level_clean", "display_label"],
            columns="site_key",
            values="display",
            aggfunc="first"
        ).reset_index()

        # Preserve original row order from first site with data
        _first_site_with_data = next(
            (s for s in site_names if site_data[s].get("table1_ohca_vaso_long") is not None), None
        )
        if _first_site_with_data is not None:
            _first_df = site_data[_first_site_with_data]["table1_ohca_vaso_long"]
            _order = _first_df[_first_df["subgroup"] == "Overall"][["variable", "level"]].copy()
            _order = _order[~_order["variable"].str.strip().isin(_t1_exclude_vars)]
            _order = _order[~_order.apply(
                lambda r: str(r.get("level", "")).strip().lower()
                in _t1_exclude_levels.get(str(r["variable"]).strip(), set()),
                axis=1,
            )]
            _order["variable_clean"] = _order["variable"].str.strip()
            _order["level_clean"] = _order["level"].fillna("").astype(str).str.strip()
            _order = _order.drop_duplicates(subset=["variable_clean", "level_clean"])
            _order["sort_key"] = range(len(_order))
            _pivot = _pivot.merge(_order[["variable_clean", "level_clean", "sort_key"]],
                                  on=["variable_clean", "level_clean"], how="left")
            _pivot = _pivot.sort_values("sort_key").drop(columns=["sort_key"])

        _t1_header = "<tr><th>Variable</th>" + "".join(
            f"<th>{SITE_LABELS.get(s, s.upper())}</th>" for s in site_names) + "</tr>"
        _t1_body = []
        for _, _row in _pivot.iterrows():
            _cells = f"<td>{_row['display_label']}</td>"
            for _site in site_names:
                _val = _row.get(_site, "N/A")
                _cells += f"<td>{_val if pd.notna(_val) else 'N/A'}</td>"
            _t1_body.append(f"<tr>{_cells}</tr>")
        _table1_html = f'<table class="data-table">{_t1_header}{"".join(_t1_body)}</table>'
    else:
        _table1_html = "<p><em>Table 1 data not available</em></p>"

    # ── Helper for figure embedding ──
    def _img(uri, alt="", width="100%"):
        if uri:
            return f'<img src="{uri}" alt="{alt}" style="max-width:{width}; height:auto; margin: 10px 0;">'
        return f'<p class="missing"><em>Figure not available: {alt}</em></p>'

    # ── Assemble Overview HTML ──
    overview_html = f"""
    <div class="section">
        <h2>Study Overview</h2>
        {_summary_table}
    </div>

    <div class="section">
        <h2>Cohort Flow</h2>
        {_consort_table}
    </div>

    <div class="section">
        <h2>CPC Distribution (Vasopressor Cohort)</h2>
        {_cpc_table}
    </div>

    <div class="section">
        <h2>Baseline Characteristics (Vasopressor Cohort)</h2>
        {_table1_html}
    </div>

    <div class="section">
        <h2>Multi-Site Concordance</h2>

        <h3>Concordance OR Forest Plot (per 10pp Agreement Increase)</h3>
        {_img(combined_figures.get("forest_plot"), "Multi-Site Forest Plot", "90%")}

        <h3>Concordance Summary</h3>
        {_concordance_table}

        <h3>Agreement-Outcome Relationship</h3>
        {_img(combined_figures.get("agreement_outcome"), "Agreement-Outcome", "90%")}

        <h3>Action Distribution Comparison</h3>
        {_img(combined_figures.get("action_distribution"), "Action Distribution", "100%")}
    </div>

    <div class="section">
        <h2>Supplemental: Clinical Trajectories</h2>
        {_img(combined_figures.get("clinical_distributions"), "Clinical Trajectories", "100%")}
    </div>

    <div class="section">
        <h2>Supplemental: Data Missingness</h2>
        <p>Patient-level missingness (%) across sites, before and after forward-fill imputation.
        Values represent the percentage of patients with <strong>no data</strong> for that variable across all time buckets.</p>

        <h3>Before Imputation</h3>
        {_img(combined_figures.get("missingness_before"), "Missingness Before Imputation", "90%")}

        <h3>After Imputation</h3>
        {_img(combined_figures.get("missingness_after"), "Missingness After Imputation", "90%")}
    </div>
    """

    logger.info("Built combined overview HTML")
    mo.md("Built **Combined Overview** tab content")
    return (overview_html,)


# ── Cell 4: Build Per-Site HTML Sections ──────────────────────────────
@app.cell
def _(logger, mo, pd, site_data, site_names, training_site, SITE_LABELS):
    def _img_tag(data_uri, alt="", width="100%"):
        if data_uri:
            return f'<img src="{data_uri}" alt="{alt}" style="max-width:{width}; height:auto; margin: 10px 0;">'
        return f'<p class="missing"><em>Figure not available: {alt}</em></p>'

    def _csv_to_html(df, max_rows=50):
        if df is None or len(df) == 0:
            return "<p><em>Not available</em></p>"
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df.to_html(index=False, classes="data-table", border=0)

    _pretraining_figs = [
        ("fig1_missingness_heatmap", "Missingness Heatmap"),
        ("fig2_vasopressor_nee", "Vasopressor/NEE Temporal"),
        ("fig3_treatment_timelines", "Treatment Timelines"),
        ("fig4_sofa_trajectory", "SOFA Trajectory"),
        ("fig5_vital_signs", "Vital Signs"),
        ("fig6_action_distribution", "Clinician Action Distribution"),
        ("fig7_labs", "Lab Trajectories"),
    ]

    _ext_val_figs = [
        ("fig_patient_timelines", "Patient Timelines (RL vs Clinician)"),
        ("fig_action_distribution", "Action Distribution (RL vs Clinician)"),
        ("fig_action_confusion_matrix", "Action Confusion Matrix"),
        ("fig_agreement_outcome", "Agreement-Outcome Relationship"),
        ("fig_concordance_or", "Concordance OR"),
    ]

    per_site_html = {}
    for _site in site_names:
        _d = site_data[_site]
        _sections = []

        # ── Pre-Training Figures ──
        _pre = '<div class="section"><h2>Pre-Training Figures</h2>'
        for _fig_name, _title in _pretraining_figs:
            _uri = _d["figures"].get(_fig_name)
            _pre += f"<h3>{_title}</h3>{_img_tag(_uri, _title)}"
        _pre += "</div>"
        _sections.append(_pre)

        # ── Training (UCMC only) ──
        if _site == training_site:
            _train = '<div class="section"><h2>Training Results</h2>'
            _uri8 = _d["figures"].get("fig8_training_curves")
            _train += f"<h3>Training Loss Curves</h3>{_img_tag(_uri8, 'Training Curves')}"
            _train += f"<h3>Training History</h3>{_csv_to_html(_d.get('training_history'))}"
            _uri9 = _d["figures"].get("fig9_action_distribution")
            _train += f"<h3>Action Distribution (RL vs Clinician)</h3>{_img_tag(_uri9, 'Action Dist')}"
            _train += f"<h3>Action Summary</h3>{_csv_to_html(_d.get('test_action_summary'))}"
            _train += "</div>"
            _sections.append(_train)

            # Evaluation tables (test set)
            _eval = '<div class="section"><h2>Model Evaluation (Test Set)</h2>'
            _eval += f"<h3>Agreement Bins</h3>{_csv_to_html(_d.get('bin_summary'))}"
            _eval += f"<h3>Ordinal Logistic Regression</h3>{_csv_to_html(_d.get('coef_summary'))}"
            _eval += "</div>"
            _sections.append(_eval)

        # ── External Validation (all sites) ──
        _meta = _d.get("ext_val_metadata")
        if _meta:
            _p_val = _meta.get("ordinal_pvalue", 1)
            _p_str = "< 0.001" if _p_val < 0.001 else f"{_p_val:.3e}"
            _ext = f"""<div class="section">
            <h2>External Validation</h2>
            <table class="data-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Training site</td><td>{_meta.get('training_site', 'N/A')}</td></tr>
                <tr><td>N patients</td><td>{_meta.get('n_patients', 'N/A'):,}</td></tr>
                <tr><td>N transitions (patient-hours)</td><td>{_meta.get('n_transitions', 'N/A'):,}</td></tr>
                <tr><td>Overall Agreement</td><td>{_meta.get('overall_agreement', 0):.1%}</td></tr>
                <tr><td>Concordance OR (full unit)</td><td>{_meta.get('ordinal_or', 0):.1f}</td></tr>
                <tr><td>p-value</td><td>{_p_str}</td></tr>
            </table>"""

            for _fig_name, _title in _ext_val_figs:
                _uri = _d["ext_val_figures"].get(_fig_name)
                _ext += f"<h3>{_title}</h3>{_img_tag(_uri, _title)}"

            _ext += f"<h3>Action Summary</h3>{_csv_to_html(_d.get('ext_val_action'))}"
            _ext += f"<h3>Agreement Bins</h3>{_csv_to_html(_d.get('ext_val_bin'))}"
            _ext += f"<h3>Ordinal Logistic Regression</h3>{_csv_to_html(_d.get('ext_val_coef'))}"
            _ext += "</div>"
            _sections.append(_ext)

        per_site_html[_site] = "\n".join(_sections)
        logger.info("Built per-site HTML for %s: %d sections", _site, len(_sections))

    mo.md(f"Built per-site HTML for **{len(per_site_html)}** sites")
    return (per_site_html,)


# ── Cell 5: Assemble Full HTML Dashboard ──────────────────────────────
@app.cell
def _(logger, mo, overview_html, pd, per_site_html, site_names, SITE_LABELS):
    _tabs = [("overview", "Combined Overview", overview_html)]
    for _site in site_names:
        _label = SITE_LABELS.get(_site, _site.upper())
        _tabs.append((_site, _label, per_site_html.get(_site, "<p>No data</p>")))

    _tab_buttons = "\n".join(
        f'        <button class="tab-btn{" active" if _i == 0 else ""}" '
        f'onclick="switchTab(\'{_tid}\')" id="btn-{_tid}">{_label}</button>'
        for _i, (_tid, _label, _) in enumerate(_tabs)
    )

    _tab_contents = "\n".join(
        f'    <div class="tab-content{" active" if _i == 0 else ""}" id="tab-{_tid}">\n{_content}\n    </div>'
        for _i, (_tid, _, _content) in enumerate(_tabs)
    )

    _site_list = ", ".join(s.upper() for s in site_names)

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OHCA-RL Multi-Site Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #1565C0, #0D47A1);
            color: white;
            padding: 25px 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .header h1 {{ font-size: 24px; font-weight: 600; }}
        .header p {{ font-size: 13px; opacity: 0.85; margin-top: 5px; }}
        .tab-bar {{
            background: white;
            padding: 0 40px;
            display: flex;
            gap: 0;
            border-bottom: 2px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            overflow-x: auto;
        }}
        .tab-btn {{
            padding: 14px 24px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab-btn:hover {{ color: #1565C0; background: #f5f8ff; }}
        .tab-btn.active {{
            color: #1565C0;
            border-bottom-color: #1565C0;
            font-weight: 600;
        }}
        .content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 40px;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        .section h2 {{
            color: #1565C0;
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e3f2fd;
        }}
        .section h3 {{
            color: #333;
            font-size: 15px;
            margin: 25px 0 10px 0;
        }}
        .section img {{
            display: block;
            margin: 15px auto;
            border: 1px solid #eee;
            border-radius: 4px;
        }}
        table, .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            font-size: 12px;
            table-layout: auto;
        }}
        th {{
            background: #1565C0;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 500;
            border: 1px solid #1256A0;
            white-space: nowrap;
        }}
        td {{
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            vertical-align: top;
            overflow-wrap: break-word;
            word-wrap: break-word;
        }}
        td:first-child {{
            min-width: 200px;
            white-space: normal;
        }}
        tr:nth-child(even) {{ background: #fafafa; }}
        tr:hover {{ background: #f0f7ff; }}
        .missing {{ color: #999; font-style: italic; }}
        hr {{ border: none; border-top: 1px solid #eee; }}
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 11px;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OHCA-RL Multi-Site Results Dashboard</h1>
        <p>{_site_list}</p>
    </div>

    <div class="tab-bar">
{_tab_buttons}
    </div>

    <div class="content">
{_tab_contents}
    </div>

    <div class="footer">
        <p>OHCA-RL: Out-of-Hospital Cardiac Arrest Reinforcement Learning | Multi-Site Dashboard</p>
    </div>

    <script>
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
        document.getElementById('btn-' + tabId).classList.add('active');
    }}
    </script>
</body>
</html>"""

    logger.info("Assembled multi-site dashboard: %d chars", len(dashboard_html))
    mo.md(f"Dashboard assembled: **{len(dashboard_html):,}** characters, **{len(_tabs)}** tabs")
    return (dashboard_html,)


# ── Cell 6: Save Dashboard ────────────────────────────────────────────
@app.cell
def _(dashboard_html, logger, mo, output_dir):
    _path = output_dir / "ohca_rl_multisite_dashboard.html"
    with open(_path, "w", encoding="utf-8") as _f:
        _f.write(dashboard_html)

    _size_mb = _path.stat().st_size / 1024 / 1024
    logger.info("Saved multi-site dashboard: %s (%.1f MB)", _path, _size_mb)

    mo.md(f"""
    ## Dashboard Saved

    | | |
    |---|---|
    | **Path** | `{_path}` |
    | **Size** | {_size_mb:.1f} MB |

    Open in a browser to view the interactive multi-site tabbed dashboard.
    """)
    return


if __name__ == "__main__":
    app.run()
