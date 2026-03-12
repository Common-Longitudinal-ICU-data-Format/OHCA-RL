# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "duckdb",
#     "pyarrow",
#     "matplotlib",
#     "tabulate",
#     "pytz",
#     "pyyaml",
#     "numpy",
#     "clifpy==0.3.8",
#     "sqlglot",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


# ── Cell 1: Setup & Config ──────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import numpy as np
    import duckdb
    import yaml
    from pathlib import Path

    import clifpy
    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category

    # Project root
    project_root = Path(__file__).parent.parent.resolve()

    # Load site config
    with open(project_root / "config" / "config.json", "r") as f:
        config = json.load(f)

    tables_path = config["tables_path"]
    file_type = config["file_type"]
    site_name = config["site_name"]
    timezone = config["timezone"]

    # Load OHCA variable config
    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as f:
        ohca_config = yaml.safe_load(f)

    # Action inference config
    action_config = ohca_config["action_inference"]
    INTERVAL_MIN = action_config["interval_minutes"]
    NEE_TOLERANCE = action_config["nee_change_tolerance"]
    TIME_WINDOW_HRS = action_config["time_window_hours"]

    # Output directories
    intermediate_dir = project_root / "output" / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = project_root / "output" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load cohort
    cohort_path = intermediate_dir / "cohort_ohca_icu.parquet"
    cohort_df = pd.read_parquet(cohort_path)
    cohort_hosp_ids = cohort_df["hospitalization_id"].astype(str).unique().tolist()

    mo.md(f"""
    ## Step 02: MAR-Informed Medication Processing & Action Inference

    | Setting | Value |
    |---------|-------|
    | **Site** | `{site_name}` |
    | **Cohort size** | {len(cohort_hosp_ids):,} hospitalizations |
    | **Interval** | {INTERVAL_MIN} minutes |
    | **NEE tolerance** | {NEE_TOLERANCE} mcg/kg/min |
    | **Time window** | first vital to +{TIME_WINDOW_HRS}h |
    """)
    return (
        ClifOrchestrator,
        INTERVAL_MIN,
        NEE_TOLERANCE,
        TIME_WINDOW_HRS,
        apply_outlier_handling,
        cohort_df,
        cohort_hosp_ids,
        convert_dose_units_by_med_category,
        duckdb,
        file_type,
        figures_dir,
        intermediate_dir,
        mo,
        np,
        ohca_config,
        pd,
        tables_path,
        timezone,
    )


# ── Cell 2: Load raw continuous meds (with mar_action_category) ─────────
@app.cell
def _(
    ClifOrchestrator,
    cohort_hosp_ids,
    file_type,
    mo,
    ohca_config,
    pd,
    tables_path,
    timezone,
):
    # Load raw meds — we need mar_action_category which step 01 discards
    clif = ClifOrchestrator(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
    )

    # Only vasoactive meds (NEE components)
    _vaso_meds = list(ohca_config["nee_coefficients"].keys())

    clif.load_table(
        "medication_admin_continuous",
        filters={
            "hospitalization_id": cohort_hosp_ids,
            "med_category": _vaso_meds,
        },
    )

    raw_meds = clif.medication_admin_continuous.df.copy()
    raw_meds["hospitalization_id"] = raw_meds["hospitalization_id"].astype(str)
    raw_meds["med_category"] = raw_meds["med_category"].str.lower()

    # Ensure mar_action_category exists and is lowercase
    if "mar_action_category" in raw_meds.columns:
        raw_meds["mar_action_category"] = raw_meds["mar_action_category"].str.lower()
    else:
        raw_meds["mar_action_category"] = None

    _n_raw = len(raw_meds)
    _n_hosp = raw_meds["hospitalization_id"].nunique()
    _mar_dist = raw_meds["mar_action_category"].value_counts(dropna=False).to_dict()

    mo.md(f"""
    ## Raw Vasoactive Medications

    | Metric | Value |
    |--------|-------|
    | **Raw rows** | {_n_raw:,} |
    | **Hospitalizations with vasoactives** | {_n_hosp:,} |
    | **MAR action distribution** | {_mar_dist} |
    | **Meds loaded** | {', '.join(sorted(raw_meds['med_category'].unique()))} |
    """)
    return clif, raw_meds


# ── Cell 3: Deduplication ────────────────────────────────────────────────
@app.cell
def _(duckdb, mo, raw_meds):
    # Dedup: one record per (hospitalization_id, admin_dttm, med_category)
    # Priority: real dose events > verify/other > stop > going
    _n_before = len(raw_meds)

    deduped_meds = duckdb.sql("""
    SELECT *
    FROM raw_meds
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY hospitalization_id, admin_dttm, med_category
        ORDER BY
            -- Dedup priority (highest number = kept first)
            CASE WHEN mar_action_category IS NULL THEN 10
                 WHEN mar_action_category IN ('verify', 'other') THEN 9
                 WHEN mar_action_category = 'stop' THEN 8
                 WHEN mar_action_category = 'going' THEN 7
                 ELSE 1 END,
            -- Prefer positive doses
            CASE WHEN med_dose > 0 THEN 1 ELSE 2 END,
            -- Largest dose wins ties
            med_dose DESC
    ) = 1
    ORDER BY hospitalization_id, med_category, admin_dttm
    """).df()

    _n_after = len(deduped_meds)
    _n_removed = _n_before - _n_after

    mo.md(f"""
    ## Deduplication

    | Metric | Value |
    |--------|-------|
    | **Before** | {_n_before:,} |
    | **After** | {_n_after:,} |
    | **Removed** | {_n_removed:,} ({_n_removed/_n_before*100:.1f}%) |
    """)
    return (deduped_meds,)


# ── Cell 4: MAR-informed dose correction ─────────────────────────────────
@app.cell
def _(deduped_meds, mo):
    # If mar_action_category is 'stop', set med_dose to 0
    meds_corrected = deduped_meds.copy()

    _stop_mask = meds_corrected["mar_action_category"] == "stop"
    _n_stops = _stop_mask.sum()
    meds_corrected.loc[_stop_mask, "med_dose"] = 0.0

    # Drop rows with null doses
    _n_before = len(meds_corrected)
    meds_corrected = meds_corrected[meds_corrected["med_dose"].notna()].copy()
    # Keep dose >= 0 (stops are 0, active are positive)
    meds_corrected = meds_corrected[meds_corrected["med_dose"] >= 0].copy()
    _n_after = len(meds_corrected)

    mo.md(f"""
    ## MAR-Informed Dose Correction

    | Metric | Value |
    |--------|-------|
    | **Stop events → dose=0** | {_n_stops:,} |
    | **Null/negative doses dropped** | {_n_before - _n_after:,} |
    | **Rows remaining** | {_n_after:,} |
    """)
    return (meds_corrected,)


# ── Cell 5: Unit conversion ──────────────────────────────────────────────
@app.cell
def _(
    apply_outlier_handling,
    clif,
    convert_dose_units_by_med_category,
    meds_corrected,
    mo,
    ohca_config,
    pd,
):
    # Load vitals for weight-based conversion
    # (reuse clif instance — vitals needed for weight lookup)
    clif.load_table(
        "vitals",
        filters={
            "hospitalization_id": meds_corrected["hospitalization_id"].unique().tolist(),
            "vital_category": ["weight_kg"],
        },
    )
    _vitals_df = clif.vitals.df.copy()
    _vitals_df["hospitalization_id"] = _vitals_df["hospitalization_id"].astype(str)
    _vitals_df["vital_category"] = _vitals_df["vital_category"].str.lower()

    # Clean preferred unit names (clifpy bug: doesn't clean these internally)
    from clifpy.utils.unit_converter import _clean_dose_unit_formats, _clean_dose_unit_names

    _raw_preferred = ohca_config.get("meds_continuous_preferred_units", {})
    # Only keep vasoactive meds (the ones we loaded)
    _nee_meds = list(ohca_config["nee_coefficients"].keys())
    _raw_preferred = {k: v for k, v in _raw_preferred.items() if k in _nee_meds}

    _pref_series = pd.Series(list(_raw_preferred.values()))
    _pref_series = _clean_dose_unit_names(_clean_dose_unit_formats(_pref_series))
    _vaso_preferred = dict(zip(_raw_preferred.keys(), _pref_series))

    # Convert
    try:
        meds_converted, _report = convert_dose_units_by_med_category(
            meds_corrected,
            vitals_df=_vitals_df,
            preferred_units=_vaso_preferred,
            override=True,
        )
        if "med_dose_converted" in meds_converted.columns:
            meds_converted["med_dose"] = meds_converted["med_dose_converted"]
        _conv_msg = f"Converted {len(meds_converted):,} rows"
    except Exception as _e:
        meds_converted = meds_corrected.copy()
        _conv_msg = f"Conversion skipped: {_e}"

    # Apply outlier handling after conversion
    # We need to temporarily put this back into a clifpy table object
    # Instead, just apply manual capping from clifpy's known ranges
    _n_before_outlier = len(meds_converted)

    mo.md(f"""
    ## Unit Conversion

    | Metric | Value |
    |--------|-------|
    | **{_conv_msg}** | |
    | **Rows after conversion** | {len(meds_converted):,} |
    """)
    return (meds_converted,)


# ── Cell 6: Compute NEE at event level ────────────────────────────────────
@app.cell
def _(duckdb, meds_converted, mo, np, ohca_config, pd):
    # Pivot meds to wide format per (hospitalization_id, admin_dttm)
    # Then compute NEE from individual vasopressor doses
    _nee_coefficients = ohca_config["nee_coefficients"]

    _meds_pivot = meds_converted.pivot_table(
        index=["hospitalization_id", "admin_dttm"],
        columns="med_category",
        values="med_dose",
        aggfunc="first",
    ).reset_index()

    # Compute NEE
    _nee = pd.Series(0.0, index=_meds_pivot.index)
    _any_present = pd.Series(False, index=_meds_pivot.index)
    for _med, _coeff in _nee_coefficients.items():
        if _med in _meds_pivot.columns:
            _nee += _coeff * _meds_pivot[_med].fillna(0)
            _any_present |= _meds_pivot[_med].notna()
    _nee[~_any_present] = np.nan

    nee_events = pd.DataFrame({
        "hospitalization_id": _meds_pivot["hospitalization_id"],
        "event_dttm": _meds_pivot["admin_dttm"],
        "nee": _nee,
    }).dropna(subset=["nee"]).copy()

    # Also keep individual med doses for debugging
    for _med in _nee_coefficients:
        if _med in _meds_pivot.columns:
            nee_events[f"dose_{_med}"] = _meds_pivot[_med].values

    _n_events = len(nee_events)
    _n_hosp = nee_events["hospitalization_id"].nunique()

    mo.md(f"""
    ## NEE at Event Level

    | Metric | Value |
    |--------|-------|
    | **NEE events** | {_n_events:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **NEE median** | {nee_events['nee'].median():.4f} |
    | **NEE mean** | {nee_events['nee'].mean():.4f} |
    | **NEE max** | {nee_events['nee'].max():.4f} |
    | **NEE=0 events** | {(nee_events['nee'] == 0).sum():,} |
    """)
    return (nee_events,)


# ── Cell 7: Build hourly grid + LOCF → starting dose ─────────────────────
@app.cell
def _(
    TIME_WINDOW_HRS,
    cohort_df,
    duckdb,
    intermediate_dir,
    mo,
    nee_events,
    pd,
):
    # Get time boundaries per patient:
    # start = first recorded vital; end = start + 120h (capped)
    wide_for_bounds = pd.read_parquet(intermediate_dir / "wide_df.parquet")
    wide_for_bounds["hospitalization_id"] = wide_for_bounds["hospitalization_id"].astype(str)

    # First and last vital per hospitalization
    time_bounds = duckdb.sql(f"""
    WITH vital_times AS (
        SELECT hospitalization_id,
               MIN(event_dttm) as first_vital,
               MAX(event_dttm) as last_vital
        FROM wide_for_bounds
        GROUP BY hospitalization_id
    )
    SELECT
        hospitalization_id,
        DATE_TRUNC('hour', first_vital) as start_hour,
        -- Cap at first_vital + TIME_WINDOW_HRS
        LEAST(
            DATE_TRUNC('hour', last_vital),
            DATE_TRUNC('hour', first_vital + INTERVAL '{TIME_WINDOW_HRS} hours')
        ) as end_hour
    FROM vital_times
    """).df()

    # Generate hourly grid: one row per (hospitalization_id, hour)
    hourly_grid = duckdb.sql("""
    SELECT
        hospitalization_id,
        unnest(generate_series(start_hour, end_hour, INTERVAL '1 hour')) as hour
    FROM time_bounds
    ORDER BY hospitalization_id, hour
    """).df()

    # Step A: Pre-aggregate NEE events into hourly buckets
    nee_hourly_agg = duckdb.sql("""
    SELECT
        hospitalization_id,
        DATE_TRUNC('hour', event_dttm) as event_hour,
        MAX(nee) as nee_max_in_hour,
        LAST(nee ORDER BY event_dttm) as nee_last_in_hour,
        COUNT(*) as n_events_in_hour
    FROM nee_events
    GROUP BY hospitalization_id, DATE_TRUNC('hour', event_dttm)
    """).df()

    # Step B: Join grid with pre-aggregated hourly NEE
    hourly_nee = duckdb.sql("""
    SELECT
        g.hospitalization_id,
        g.hour,
        a.nee_max_in_hour,
        a.nee_last_in_hour,
        COALESCE(a.n_events_in_hour, 0) as n_events_in_hour
    FROM hourly_grid g
    LEFT JOIN nee_hourly_agg a
        ON g.hospitalization_id = a.hospitalization_id
        AND g.hour = a.event_hour
    ORDER BY g.hospitalization_id, g.hour
    """).df()

    # Step C: Compute nee_start via LOCF using pandas (fast forward-fill)
    # nee_start = the last known NEE value from a PREVIOUS hour
    # First, get nee_last_in_hour as the "known value" for each hour
    # Then shift forward by 1 and forward-fill within each hospitalization
    hourly_nee["nee_start"] = (
        hourly_nee.groupby("hospitalization_id")["nee_last_in_hour"]
        .shift(1)  # previous hour's last NEE
    )
    hourly_nee["nee_start"] = (
        hourly_nee.groupby("hospitalization_id")["nee_start"]
        .ffill()  # LOCF: carry forward across gaps
    )

    # nee_end = last event in hour if exists, else LOCF from nee_start
    hourly_nee["nee_end"] = hourly_nee["nee_last_in_hour"].fillna(
        hourly_nee["nee_start"]
    )
    # Fill remaining NaN with 0 (no vasopressor data = not on vasopressors)
    hourly_nee["nee_start"] = hourly_nee["nee_start"].fillna(0.0)
    hourly_nee["nee_end"] = hourly_nee["nee_end"].fillna(0.0)
    hourly_nee["nee_max_in_hour"] = hourly_nee["nee_max_in_hour"].fillna(0.0)

    _n_rows = len(hourly_nee)
    _n_hosp = hourly_nee["hospitalization_id"].nunique()
    _n_hours_median = hourly_nee.groupby("hospitalization_id").size().median()

    mo.md(f"""
    ## Hourly Grid + LOCF

    | Metric | Value |
    |--------|-------|
    | **Hourly rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Median hours/patient** | {_n_hours_median:.0f} |
    | **Rows with events** | {(hourly_nee['n_events_in_hour'] > 0).sum():,} |
    | **Rows with LOCF only** | {(hourly_nee['n_events_in_hour'] == 0).sum():,} |
    """)
    return (hourly_nee,)


# ── Cell 8: Infer actions from hourly NEE ─────────────────────────────────
@app.cell
def _(NEE_TOLERANCE, hourly_nee, mo, np):
    # Action inference:
    #   0 = stay     (no meaningful NEE change)
    #   1 = increase (NEE goes up >= tolerance, or new med started)
    #   2 = decrease (NEE goes down >= tolerance, NEE still > 0)
    #   3 = stop     (NEE drops to 0 from > 0)

    actions = hourly_nee.copy()

    _nee_delta = actions["nee_end"] - actions["nee_start"]

    # dose_inc: any dose in hour > starting dose AND starting dose > 0
    _dose_inc = (actions["nee_max_in_hour"] > actions["nee_start"]) & (
        actions["nee_start"] > 0
    )
    # new_med: any dose in hour > 0 AND starting dose == 0
    _new_med = (actions["nee_max_in_hour"] > 0) & (actions["nee_start"] == 0)

    # Default: stay
    actions["action"] = 0

    # Stop: NEE drops to 0 from > 0
    _stop = (actions["nee_end"] == 0) & (actions["nee_start"] > 0)
    actions.loc[_stop, "action"] = 3

    # Increase: NEE goes up by >= tolerance OR new med started OR dose increased
    _increase = (
        (_nee_delta >= NEE_TOLERANCE) | _new_med | _dose_inc
    ) & ~_stop  # stop takes priority
    actions.loc[_increase, "action"] = 1

    # Decrease: NEE goes down by >= tolerance but stays > 0
    _decrease = (
        (_nee_delta <= -NEE_TOLERANCE) & (actions["nee_end"] > 0)
    ) & ~_stop & ~_increase
    actions.loc[_decrease, "action"] = 2

    # Add readable labels
    _action_labels = {0: "stay", 1: "increase", 2: "decrease", 3: "stop"}
    actions["action_label"] = actions["action"].map(_action_labels)

    # NEE delta for reference
    actions["nee_delta"] = _nee_delta

    # Action distribution
    _dist = actions["action_label"].value_counts()
    _dist_pct = actions["action_label"].value_counts(normalize=True) * 100

    mo.md(f"""
    ## Action Inference (tolerance={NEE_TOLERANCE})

    | Action | Count | % |
    |--------|-------|---|
    | **Stay** | {_dist.get('stay', 0):,} | {_dist_pct.get('stay', 0):.1f}% |
    | **Increase** | {_dist.get('increase', 0):,} | {_dist_pct.get('increase', 0):.1f}% |
    | **Decrease** | {_dist.get('decrease', 0):,} | {_dist_pct.get('decrease', 0):.1f}% |
    | **Stop** | {_dist.get('stop', 0):,} | {_dist_pct.get('stop', 0):.1f}% |
    """)
    return (actions,)


# ── Cell 9: Validation ───────────────────────────────────────────────────
@app.cell
def _(actions, duckdb, mo):
    # Validation checks
    _checks = []

    # 1. No gaps in hourly sequence per hospitalization
    _gap_check = duckdb.sql("""
    WITH lagged AS (
        SELECT *,
            LAG(hour) OVER (PARTITION BY hospitalization_id ORDER BY hour) as prev_hour
        FROM actions
    )
    SELECT COUNT(*) as gap_count
    FROM lagged
    WHERE prev_hour IS NOT NULL
      AND hour - prev_hour > INTERVAL '1 hour'
    """).df()
    _n_gaps = _gap_check["gap_count"].iloc[0]
    _checks.append(("No hourly gaps", _n_gaps == 0, f"{_n_gaps} gaps found"))

    # 2. Stop requires nee_start > 0
    _bad_stops = ((actions["action"] == 3) & (actions["nee_start"] == 0)).sum()
    _checks.append(("Stop requires nee_start > 0", _bad_stops == 0, f"{_bad_stops} invalid stops"))

    # 3. All hospitalizations have at least 1 hour
    _min_hours = actions.groupby("hospitalization_id").size().min()
    _checks.append(("All patients have >= 1 hour", _min_hours >= 1, f"min={_min_hours}"))

    # 4. Action values are 0-3
    _valid_actions = actions["action"].isin([0, 1, 2, 3]).all()
    _checks.append(("Actions in {0,1,2,3}", _valid_actions, ""))

    _check_table = "\n".join([
        f"| {'PASS' if passed else 'FAIL'} | {name} | {detail} |"
        for name, passed, detail in _checks
    ])

    mo.md(f"""
    ## Validation

    | Status | Check | Detail |
    |--------|-------|--------|
    {_check_table}
    """)
    return


# ── Cell 10: Save output ─────────────────────────────────────────────────
@app.cell
def _(actions, figures_dir, intermediate_dir, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select output columns
    _output_cols = [
        "hospitalization_id", "hour",
        "nee_start", "nee_end", "nee_max_in_hour", "nee_delta",
        "n_events_in_hour",
        "action", "action_label",
    ]
    output_df = actions[_output_cols].copy()

    # Save
    _out_path = intermediate_dir / "hourly_nee_actions.parquet"
    output_df.to_parquet(_out_path, index=False)
    _file_size_mb = _out_path.stat().st_size / (1024 * 1024)

    # Plot action distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: Action distribution bar chart
    _dist = output_df["action_label"].value_counts()
    _colors = {"stay": "steelblue", "increase": "coral", "decrease": "mediumseagreen", "stop": "gold"}
    _order = ["stay", "increase", "decrease", "stop"]
    _vals = [_dist.get(a, 0) for a in _order]
    axes[0].bar(_order, _vals, color=[_colors[a] for a in _order], edgecolor="white")
    axes[0].set_title("Action Distribution", fontsize=13)
    axes[0].set_ylabel("Count", fontsize=11)
    for _i, _v in enumerate(_vals):
        axes[0].text(_i, _v + max(_vals)*0.01, f"{_v:,}", ha="center", fontsize=9)

    # Panel 2: NEE end distribution (among hours with NEE > 0)
    _nee_pos = output_df.loc[output_df["nee_end"] > 0, "nee_end"]
    if len(_nee_pos) > 0:
        _cap = _nee_pos.quantile(0.99)
        axes[1].hist(_nee_pos[_nee_pos <= _cap], bins=60, color="steelblue",
                     edgecolor="white", linewidth=0.3)
        axes[1].axvline(_nee_pos.median(), color="red", linestyle="--",
                        label=f"median={_nee_pos.median():.3f}")
        axes[1].legend(fontsize=10)
    axes[1].set_title("Hourly NEE Distribution (NEE > 0)", fontsize=13)
    axes[1].set_xlabel("NEE (mcg/kg/min equivalent)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)

    plt.tight_layout()
    fig.savefig(figures_dir / "step02_action_distribution.png", dpi=150, bbox_inches="tight")

    mo.md(f"""
    ## Output Saved

    | Metric | Value |
    |--------|-------|
    | **File** | `{_out_path}` |
    | **Size** | {_file_size_mb:.2f} MB |
    | **Rows** | {len(output_df):,} |
    | **Hospitalizations** | {output_df['hospitalization_id'].nunique():,} |
    | **Figure** | `{figures_dir / 'step02_action_distribution.png'}` |
    """)
    return


# ── Cell 11: Spot-check patient trajectories ──────────────────────────────
@app.cell
def _(actions, mo, pd):
    # Show a few patient trajectories for manual inspection
    # Pick patients with diverse action patterns
    _patients_with_actions = (
        actions[actions["action"] != 0]
        .groupby("hospitalization_id")["action"]
        .nunique()
        .sort_values(ascending=False)
    )

    if len(_patients_with_actions) > 0:
        # Top 3 patients with most action diversity
        _sample_ids = _patients_with_actions.head(3).index.tolist()

        _tables = []
        for _pid in _sample_ids:
            _pt = actions[actions["hospitalization_id"] == _pid].copy()
            # Show only hours with events or action changes
            _interesting = _pt[
                (_pt["action"] != 0) | (_pt["n_events_in_hour"] > 0)
            ].head(15)
            _tables.append(f"### Patient: {_pid}\n\n" +
                          _interesting[["hour", "nee_start", "nee_end", "nee_delta",
                                       "action_label", "n_events_in_hour"]].to_markdown(index=False))

        _spot_check = "\n\n".join(_tables)
    else:
        _spot_check = "No patients with non-stay actions found."

    mo.md(f"""
    ## Spot-Check: Patient Trajectories

    {_spot_check}
    """)
    return


if __name__ == "__main__":
    app.run()
