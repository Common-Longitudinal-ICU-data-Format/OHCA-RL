# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "tabulate",
#     "pyyaml",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", app_title="OHCA-RL Table 1 & Pre-Training Summary")


# ── Cell 1: Setup & Load ────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import logging
    import pandas as pd
    import numpy as np
    import yaml
    from pathlib import Path

    from utils import setup_logging
    logger = setup_logging("04_create_tableone")

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as _f:
        ohca_config = yaml.safe_load(_f)

    intermediate_dir = project_root / "output" / "intermediate"
    final_dir = project_root / "output" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    site_name = config["site_name"]

    # Load inputs from pipeline
    patient_static = pd.read_parquet(intermediate_dir / "patient_static.parquet")
    patient_static["hospitalization_id"] = patient_static["hospitalization_id"].astype(str)

    summary_df = pd.read_parquet(intermediate_dir / "hospitalization_summary.parquet")
    summary_df["hospitalization_id"] = summary_df["hospitalization_id"].astype(str)

    bucketed_df = pd.read_parquet(intermediate_dir / "wide_df_bucketed.parquet")
    bucketed_df["hospitalization_id"] = bucketed_df["hospitalization_id"].astype(str)

    logger.info("Loaded: patient_static=%d, summary=%d, bucketed=%d rows",
                len(patient_static), len(summary_df), len(bucketed_df))

    mo.md(f"""
    ## Step 04: Table 1 & Pre-Training Summary

    | Input | Rows |
    |-------|------|
    | **patient_static** | {len(patient_static):,} |
    | **hospitalization_summary** | {len(summary_df):,} |
    | **wide_df_bucketed** | {len(bucketed_df):,} |
    | **Site** | {site_name} |
    """)
    return (
        bucketed_df,
        final_dir,
        logger,
        mo,
        np,
        ohca_config,
        patient_static,
        pd,
        site_name,
        summary_df,
    )


# ── Cell 2: Formatting Helpers ──────────────────────────────────────────
@app.cell
def _():
    def fmt_median_iqr(series):
        """Format as 'median [Q1, Q3]'."""
        valid = series.dropna()
        if len(valid) == 0:
            return "—"
        med = valid.median()
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"

    def fmt_n_pct(count, total):
        """Format as 'n (pct%)'."""
        if total == 0:
            return "0 (0.0%)"
        return f"{count} ({count / total * 100:.1f}%)"

    def fmt_mean_sd(series):
        """Format as 'mean ± SD'."""
        valid = series.dropna()
        if len(valid) == 0:
            return "—"
        return f"{valid.mean():.1f} ± {valid.std():.1f}"
    return fmt_mean_sd, fmt_median_iqr, fmt_n_pct


# ── Cell 3: Prepare t1_df & Build Table 1 Function ────────────────────
@app.cell
def _(fmt_median_iqr, fmt_n_pct, logger, mo, np, pd, patient_static, site_name, summary_df):
    logger.info("Building Table 1...")

    # Merge static demographics with summary flags
    t1_df = summary_df.merge(
        patient_static[["hospitalization_id", "age_at_admission",
                         "sex_category", "race_category", "ethnicity_category"]],
        on="hospitalization_id", how="left", suffixes=("", "_static"),
    )
    # Derive ever_crrt from total_crrt_hours
    if "total_crrt_hours" in t1_df.columns:
        t1_df["ever_crrt"] = (t1_df["total_crrt_hours"] > 0).astype(int)

    # Compute Hospital LOS (days) from patient_static
    if "admission_dttm" in patient_static.columns and "discharge_dttm" in patient_static.columns:
        _los = patient_static[["hospitalization_id", "admission_dttm", "discharge_dttm"]].copy()
        _los["hospital_los_days"] = (
            (pd.to_datetime(_los["discharge_dttm"]) - pd.to_datetime(_los["admission_dttm"]))
            .dt.total_seconds() / 86400
        )
        t1_df = t1_df.merge(_los[["hospitalization_id", "hospital_los_days"]],
                             on="hospitalization_id", how="left")

    # Compute ICU LOS (days) from total_icu_hours
    if "total_icu_hours" in t1_df.columns:
        t1_df["icu_los_days"] = t1_df["total_icu_hours"] / 24

    # Use static demographics if summary doesn't have them
    for _col in ["age_at_admission", "sex_category", "race_category", "ethnicity_category"]:
        _static_col = f"{_col}_static"
        if _static_col in t1_df.columns:
            t1_df[_col] = t1_df[_col].fillna(t1_df[_static_col])
            t1_df = t1_df.drop(columns=[_static_col])

    # ── Reusable Table 1 builder ──
    def build_table1(df_in, site, sofa_cols):
        """Build Table 1 for a given population DataFrame.
        Returns (table1_df, long_df).
        """
        _survivors = df_in[df_in["ever_died"] == 0]
        _non_survivors = df_in[df_in["ever_died"] == 1]
        _subgroups = {"Overall": df_in, "Survivors": _survivors, "Non-Survivors": _non_survivors}
        _sg_names = list(_subgroups.keys())
        _long_rows = []
        _rows = []

        def _add_header(text):
            _rows.append([f"**{text}**", ""] + [""] * len(_sg_names))

        def _add_count_row(label):
            _row = [label, ""]
            for _sn, _sd in _subgroups.items():
                _row.append(str(len(_sd)))
                _long_rows.append({"variable": label, "level": "", "subgroup": _sn,
                                   "stat_type": "count", "n": len(_sd), "total": len(_sd)})
            _rows.append(_row)

        def _add_continuous(label, col_name):
            _row = [label, ""]
            for _sn, _sd in _subgroups.items():
                if col_name in _sd.columns:
                    _row.append(fmt_median_iqr(_sd[col_name]))
                    _v = _sd[col_name].dropna()
                    _long_rows.append({
                        "variable": label, "level": "", "subgroup": _sn,
                        "stat_type": "continuous", "n": len(_v), "total": len(_sd),
                        "median": float(_v.median()) if len(_v) > 0 else None,
                        "q25": float(_v.quantile(0.25)) if len(_v) > 0 else None,
                        "q75": float(_v.quantile(0.75)) if len(_v) > 0 else None,
                    })
                else:
                    _row.append("—")
            _rows.append(_row)

        def _add_categorical(label, col_name, levels):
            for _level in levels:
                _row = [f"{label}, n (%)", str(_level)]
                for _sn, _sd in _subgroups.items():
                    if col_name not in _sd.columns:
                        _row.append("—")
                        continue
                    _count = int((_sd[col_name] == _level).sum())
                    _total = len(_sd)
                    _row.append(fmt_n_pct(_count, _total))
                    _long_rows.append({"variable": label, "level": str(_level), "subgroup": _sn,
                                       "stat_type": "categorical", "n": _count, "total": _total})
                _rows.append(_row)

        def _add_binary(label, col_name):
            _row = [label, ""]
            for _sn, _sd in _subgroups.items():
                if col_name not in _sd.columns:
                    _row.append("—")
                    continue
                _count = int(_sd[col_name].sum())
                _total = len(_sd)
                _row.append(fmt_n_pct(_count, _total))
                _long_rows.append({"variable": label, "level": "yes", "subgroup": _sn,
                                   "stat_type": "binary", "n": _count, "total": _total})
            _rows.append(_row)

        # ── Counts ──
        _add_count_row("N")

        # ── Demographics ──
        _add_header("Demographics")
        _add_continuous("Age, median [IQR]", "age_at_admission")
        _sex_levels = sorted(df_in["sex_category"].dropna().unique().tolist())
        _add_categorical("Sex", "sex_category", _sex_levels)
        _race_levels = sorted(df_in["race_category"].dropna().unique().tolist())
        _add_categorical("Race", "race_category", _race_levels)
        _eth_levels = sorted(df_in["ethnicity_category"].dropna().unique().tolist())
        _add_categorical("Ethnicity", "ethnicity_category", _eth_levels)

        # ── CPC Outcome ──
        _add_header("Neurological Outcome (CPC)")
        _cpc_levels = ["CPC1_2", "CPC3", "CPC4", "CPC5"]
        _add_categorical("CPC", "cpc", _cpc_levels)

        # ── SOFA Scores ──
        _add_header("SOFA Scores")
        for _col in sofa_cols:
            _parts = _col.split("_")
            _label = f"SOFA {_parts[1]}–{_parts[2]}h"
            _add_continuous(_label, _col)

        # ── Treatment Characteristics ──
        _add_header("Treatment")
        _add_binary("Ever vasopressors, n (%)", "ever_vaso")
        _add_binary("Ever IMV, n (%)", "ever_imv")
        _add_continuous("Total ICU hours", "total_icu_hours")
        _add_continuous("Total ED hours", "total_ed_hours")
        _add_continuous("Total vasopressor hours", "total_vaso_hours")
        _add_continuous("Total IMV hours", "total_imv_hours")
        _add_binary("Ever CRRT, n (%)", "ever_crrt")
        _add_continuous("Max NEE (mcg/kg/min)", "max_nee")
        if "max_lactate" in df_in.columns:
            _add_continuous("Max lactate", "max_lactate")
        _add_continuous("Total observation hours", "total_hours")

        # ── Neurological Assessments ──
        _add_header("Neurological Assessments")
        if "min_gcs" in df_in.columns:
            _add_continuous("Minimum GCS (over stay)", "min_gcs")
        if "median_gcs" in df_in.columns:
            _add_continuous("Median GCS", "median_gcs")
        if "median_rass" in df_in.columns:
            _add_continuous("Median RASS", "median_rass")

        # ── Outcomes ──
        _add_header("Outcomes")
        _add_binary("In-hospital mortality, n (%)", "ever_died")
        _add_continuous("Hospital LOS (days)", "hospital_los_days")
        _add_continuous("ICU LOS (days)", "icu_los_days")

        # Build DataFrames
        _table = pd.DataFrame(_rows, columns=["Variable", "Level"] + _sg_names)
        _long = pd.DataFrame(_long_rows)
        _long["site"] = site
        return _table, _long

    # Identify SOFA total columns
    sofa_total_cols = sorted([c for c in summary_df.columns
                               if c.startswith("sofa_") and c.count("_") == 2
                               and not any(s in c for s in ["cv", "coag", "liver", "resp", "cns", "renal"])])

    # Check for unmatched discharge categories (CPC = "exclude")
    _exclude_mask = t1_df["cpc"] == "exclude"
    if _exclude_mask.any():
        _unmatched = t1_df.loc[_exclude_mask, "discharge_category"].unique().tolist()
        logger.warning("CPC 'exclude': %d patients with unmatched discharge_category: %s",
                        _exclude_mask.sum(), _unmatched)

    # Build full-cohort Table 1
    table1, long_df = build_table1(t1_df, site_name, sofa_total_cols)
    logger.info("Table 1 (full cohort): %d rows, %d patients", len(table1), len(t1_df))

    mo.md("### Table 1 Preview (Full Cohort)\n\n" + table1.to_markdown(index=False))
    return build_table1, long_df, sofa_total_cols, table1, t1_df


# ── Cell 4: Save Table 1 ──────────────────────────────────────────────
@app.cell
def _(final_dir, logger, long_df, mo, pd, site_name, table1):
    # CSV (formatted)
    _csv_path = final_dir / "table1_ohca.csv"
    table1.to_csv(_csv_path, index=False)

    # Long format (machine-readable, for multi-site aggregation)
    _long_path = final_dir / "table1_ohca_long.csv"
    long_df.to_csv(_long_path, index=False)

    # HTML
    _html_path = final_dir / "table1_ohca.html"
    _data_rows = table1[~table1["Variable"].str.startswith("**")].copy()

    _html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Table 1 - OHCA-RL Cohort</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
        th {{ background: #2196F3; color: white; padding: 12px; text-align: left;
              border: 1px solid #ddd; }}
        td {{ padding: 10px; border: 1px solid #ddd; vertical-align: top; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f0f0; }}
        .section-header {{ background: #e3f2fd !important; font-weight: bold; }}
        .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd;
                   font-size: 11px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Table 1: OHCA-RL Cohort — {site_name}</h1>
        <p><em>Continuous variables: median [Q1, Q3]</em></p>
        <p><em>Categorical variables: n (%)</em></p>
        {table1.to_html(index=False, escape=False, classes='table')}
        <div class="footer">
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

    with open(_html_path, "w", encoding="utf-8") as _f:
        _f.write(_html)

    logger.info("Saved Table 1: CSV=%s, long=%s, HTML=%s", _csv_path, _long_path, _html_path)

    mo.md(f"""
    ### Table 1 Saved

    | Format | Path |
    |--------|------|
    | CSV | `{_csv_path}` |
    | Long CSV | `{_long_path}` |
    | HTML | `{_html_path}` |
    """)
    return


# ── Cell 4b: Vaso-Only Table 1 ────────────────────────────────────────────
@app.cell
def _(build_table1, final_dir, logger, mo, pd, site_name, sofa_total_cols, t1_df):
    logger.info("Building vaso-only Table 1...")

    t1_df_vaso = t1_df[t1_df["ever_vaso"] == 1].copy()
    table1_vaso, long_df_vaso = build_table1(t1_df_vaso, site_name, sofa_total_cols)
    logger.info("Table 1 (vaso cohort): %d rows, %d patients", len(table1_vaso), len(t1_df_vaso))

    # Save CSV
    table1_vaso.to_csv(final_dir / "table1_ohca_vaso.csv", index=False)
    long_df_vaso.to_csv(final_dir / "table1_ohca_vaso_long.csv", index=False)

    # Save HTML
    _html_vaso = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Table 1 - OHCA-RL Vasopressor Cohort</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #E53935; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
        th {{ background: #E53935; color: white; padding: 12px; text-align: left;
              border: 1px solid #ddd; }}
        td {{ padding: 10px; border: 1px solid #ddd; vertical-align: top; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f0f0; }}
        .section-header {{ background: #ffebee !important; font-weight: bold; }}
        .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd;
                   font-size: 11px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Table 1: OHCA-RL Vasopressor Cohort — {site_name}</h1>
        <p><em>Population: Patients who received vasopressors during ICU stay (training cohort)</em></p>
        <p><em>Continuous variables: median [Q1, Q3] &bull; Categorical variables: n (%)</em></p>
        {table1_vaso.to_html(index=False, escape=False, classes='table')}
        <div class="footer">
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
    with open(final_dir / "table1_ohca_vaso.html", "w", encoding="utf-8") as _f:
        _f.write(_html_vaso)

    logger.info("Saved vaso-only Table 1: CSV, long CSV, HTML")

    mo.md(f"""
    ### Vaso-Only Table 1

    **Vasopressor patients**: {len(t1_df_vaso):,} / {len(t1_df):,} ({len(t1_df_vaso)/len(t1_df)*100:.1f}%)

    {table1_vaso.to_markdown(index=False)}
    """)
    return


# ── Cell 4c: Update STROBE Counts with Vaso Population ──────────────────
@app.cell
def _(final_dir, logger, mo, pd, site_name, t1_df):
    _strobe_path = final_dir / "strobe_counts.csv"
    if _strobe_path.exists():
        _strobe_df = pd.read_csv(_strobe_path)

        # Remove any prior vaso rows (idempotent)
        _strobe_df = _strobe_df[~_strobe_df["counter"].str.startswith("5_")]

        _n_vaso = int(t1_df["ever_vaso"].sum())
        _n_no_vaso = len(t1_df) - _n_vaso
        _n_vaso_surv = int(t1_df.loc[t1_df["ever_vaso"] == 1, "ever_died"].eq(0).sum())
        _n_vaso_dead = int(t1_df.loc[t1_df["ever_vaso"] == 1, "ever_died"].eq(1).sum())

        _new_rows = pd.DataFrame([
            {"counter": "5_vaso_patients", "value": _n_vaso, "site": site_name},
            {"counter": "5_excluded_no_vaso", "value": _n_no_vaso, "site": site_name},
            {"counter": "5_vaso_survivors", "value": _n_vaso_surv, "site": site_name},
            {"counter": "5_vaso_non_survivors", "value": _n_vaso_dead, "site": site_name},
            {"counter": "5_vaso_mortality_pct", "value": round(_n_vaso_dead / _n_vaso * 100, 1) if _n_vaso > 0 else 0, "site": site_name},
        ])
        _strobe_df = pd.concat([_strobe_df, _new_rows], ignore_index=True)
        _strobe_df.to_csv(_strobe_path, index=False)
        logger.info("Updated strobe_counts.csv with vaso population counts")

        mo.md(f"""
        ### STROBE Updated

        | Step | Count |
        |------|-------|
        | ICU Admitted | {len(t1_df):,} |
        | **Vasopressor patients** | **{_n_vaso:,}** |
        | Excluded (no vasopressors) | {_n_no_vaso:,} |
        | Vaso — Survivors | {_n_vaso_surv:,} |
        | Vaso — Non-Survivors | {_n_vaso_dead:,} |
        | Vaso — Mortality | {_n_vaso_dead / _n_vaso * 100:.1f}% |
        """)
    else:
        logger.warning("strobe_counts.csv not found — skipping STROBE update")
        mo.md("**Warning**: `strobe_counts.csv` not found. Run step 00 first.")
    return


# ── Cell 5: Pre-Training Summary — Action Distribution ──────────────────
@app.cell
def _(bucketed_df, fmt_n_pct, logger, mo, np, pd):
    logger.info("Computing pre-training summaries...")

    # Action distribution
    _action_labels = {0: "Stay", 1: "Increase", 2: "Decrease", 3: "Stop"}
    _action_counts = bucketed_df["action"].value_counts().sort_index()
    _n_total = len(bucketed_df)

    _action_rows = []
    for _a, _label in _action_labels.items():
        _count = _action_counts.get(_a, 0)
        _action_rows.append({
            "Action": _a, "Label": _label, "Count": f"{_count:,}",
            "Pct": f"{_count / _n_total * 100:.1f}%",
        })
    action_dist = pd.DataFrame(_action_rows)

    # Action distribution by survival
    _action_by_surv = bucketed_df.groupby(["ever_died", "action"]).size().unstack(fill_value=0)
    _action_by_surv.index = _action_by_surv.index.map({0: "Survivors", 1: "Non-Survivors"})
    _action_by_surv.columns = [_action_labels.get(c, c) for c in _action_by_surv.columns]
    # Convert to percentages
    _action_pct = _action_by_surv.div(_action_by_surv.sum(axis=1), axis=0) * 100

    mo.md(f"""
    ### Pre-Training: Action Distribution

    **Overall**:

    {action_dist.to_markdown(index=False)}

    **By Survival (% of row)**:

    {_action_pct.round(1).to_markdown()}
    """)
    return (action_dist,)


# ── Cell 6: Pre-Training Summary — Feature Distributions ────────────────
@app.cell
def _(bucketed_df, fmt_median_iqr, logger, mo, np, pd):
    # Key features to summarize before training
    _feature_groups = {
        "Vitals": [c for c in bucketed_df.columns if c.startswith("vital_")],
        "Labs": [c for c in bucketed_df.columns if c.startswith("lab_")],
        "Meds (continuous)": [c for c in bucketed_df.columns if c.startswith("med_cont_")],
        "Respiratory": [c for c in bucketed_df.columns if c.startswith("resp_") and bucketed_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]],
        "Assessments": [c for c in bucketed_df.columns if c.startswith("assess_")],
    }

    _feat_rows = []
    _n_total = len(bucketed_df)
    for _group, _cols in _feature_groups.items():
        for _col in sorted(_cols):
            _valid = bucketed_df[_col].dropna()
            if len(_valid) == 0:
                continue
            _feat_rows.append({
                "Group": _group,
                "Feature": _col,
                "N non-null": f"{len(_valid):,}",
                "Median [IQR]": fmt_median_iqr(bucketed_df[_col]),
                "Min": f"{_valid.min():.2f}",
                "Max": f"{_valid.max():.2f}",
                "% Zero": f"{(_valid == 0).mean() * 100:.1f}%",
                "% NaN": f"{bucketed_df[_col].isna().mean() * 100:.1f}%",
            })
    feature_summary = pd.DataFrame(_feat_rows)

    logger.info("Feature summary: %d features across %d groups",
                len(feature_summary), len(_feature_groups))

    mo.md("### Pre-Training: Feature Distributions\n\n" + feature_summary.to_markdown(index=False))
    return (feature_summary,)


# ── Cell 7: Pre-Training Summary — Trajectory Stats ────────────────────
@app.cell
def _(bucketed_df, fmt_median_iqr, logger, mo, np, pd):
    # Per-patient trajectory characteristics
    _traj = bucketed_df.groupby("hospitalization_id").agg(
        n_buckets=("time_bucket", "count"),
        max_bucket=("time_bucket", "max"),
        n_action_changes=("action", lambda x: (x.diff().fillna(0) != 0).sum()),
        n_vaso_hours=("on_vaso", "sum"),
        n_imv_hours=("on_imv", "sum"),
        ever_died=("ever_died", "first"),
    ).reset_index()

    _surv_traj = _traj[_traj["ever_died"] == 0]
    _dead_traj = _traj[_traj["ever_died"] == 1]

    mo.md(f"""
    ### Pre-Training: Trajectory Characteristics

    | Metric | Overall | Survivors | Non-Survivors |
    |--------|---------|-----------|---------------|
    | **N patients** | {len(_traj):,} | {len(_surv_traj):,} | {len(_dead_traj):,} |
    | **Trajectory length (buckets)** | {fmt_median_iqr(_traj['n_buckets'])} | {fmt_median_iqr(_surv_traj['n_buckets'])} | {fmt_median_iqr(_dead_traj['n_buckets'])} |
    | **Action changes per trajectory** | {fmt_median_iqr(_traj['n_action_changes'])} | {fmt_median_iqr(_surv_traj['n_action_changes'])} | {fmt_median_iqr(_dead_traj['n_action_changes'])} |
    | **Vasopressor hours** | {fmt_median_iqr(_traj['n_vaso_hours'])} | {fmt_median_iqr(_surv_traj['n_vaso_hours'])} | {fmt_median_iqr(_dead_traj['n_vaso_hours'])} |
    | **IMV hours** | {fmt_median_iqr(_traj['n_imv_hours'])} | {fmt_median_iqr(_surv_traj['n_imv_hours'])} | {fmt_median_iqr(_dead_traj['n_imv_hours'])} |
    """)
    return


# ── Cell 8: Pre-Training Summary — Missingness Check ───────────────────
@app.cell
def _(bucketed_df, logger, mo, np, pd):
    # Final missingness check on the bucketed df that will feed training
    _feature_cols = [
        c for c in bucketed_df.columns
        if c not in ("hospitalization_id", "time_bucket", "event_dttm",
                      "discharge_category", "cpc", "action",
                      "adt_location_category", "resp_device_name",
                      "resp_device_category", "resp_mode_name",
                      "resp_mode_category", "resp_vent_brand_name")
    ]
    _numeric_cols = [c for c in _feature_cols if bucketed_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.int8]]

    _miss = bucketed_df[_numeric_cols].isna().mean() * 100
    _any_missing = _miss[_miss > 0]

    if len(_any_missing) == 0:
        _status = "All numeric features have **zero missingness** — ready for training."
    else:
        _status = f"**{len(_any_missing)} features** still have missing values:\n\n" + _any_missing.round(2).to_markdown()

    mo.md(f"""
    ### Pre-Training: Missingness Check

    - **Numeric features checked**: {len(_numeric_cols)}
    - {_status}
    """)
    return


# ── Cell 9: Save Pre-Training Summary ──────────────────────────────────
@app.cell
def _(action_dist, feature_summary, final_dir, logger, mo, pd):
    # Save summaries
    action_dist.to_csv(final_dir / "action_distribution.csv", index=False)
    feature_summary.to_csv(final_dir / "feature_summary.csv", index=False)

    logger.info("Saved pre-training summaries to %s", final_dir)

    mo.md(f"""
    ### Pre-Training Summaries Saved

    | File | Path |
    |------|------|
    | Action distribution | `{final_dir / 'action_distribution.csv'}` |
    | Feature summary | `{final_dir / 'feature_summary.csv'}` |
    """)
    return


if __name__ == "__main__":
    app.run()
