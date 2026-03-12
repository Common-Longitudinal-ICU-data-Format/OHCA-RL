# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "duckdb",
#     "pyarrow",
#     "tabulate",
#     "pyyaml",
#     "numpy",
#     "clifpy==0.3.8",
#     "sqlglot",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


# ── Cell 1: Setup ────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import numpy as np
    import yaml
    from pathlib import Path

    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.sofa import REQUIRED_SOFA_CATEGORIES_BY_TABLE

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as _f:
        ohca_config = yaml.safe_load(_f)

    tables_path = config["tables_path"]
    file_type = config["file_type"]
    timezone = config["timezone"]

    intermediate_dir = project_root / "output" / "intermediate"
    TIME_WINDOW_HRS = ohca_config["action_inference"]["time_window_hours"]  # 120

    # Load cohort
    cohort_df = pd.read_parquet(intermediate_dir / "cohort_ohca_icu.parquet")
    cohort_hosp_ids = cohort_df["hospitalization_id"].astype(str).unique().tolist()

    # 24h windows: 0-24, 24-48, 48-72, 72-96, 96-120
    n_windows = TIME_WINDOW_HRS // 24

    mo.md(f"""
    ## Step 03: SOFA Score Calculator (Time-Varying)

    | Setting | Value |
    |---------|-------|
    | **Cohort** | {len(cohort_hosp_ids):,} hospitalizations |
    | **Windows** | {n_windows} × 24h (0–{TIME_WINDOW_HRS}h) |
    """)
    return (
        ClifOrchestrator,
        REQUIRED_SOFA_CATEGORIES_BY_TABLE,
        cohort_df,
        cohort_hosp_ids,
        config,
        file_type,
        intermediate_dir,
        mo,
        n_windows,
        np,
        pd,
        tables_path,
        timezone,
    )


# ── Cell 2: Load tables + convert units (once) ──────────────────────────
@app.cell
def _(
    ClifOrchestrator,
    REQUIRED_SOFA_CATEGORIES_BY_TABLE,
    cohort_hosp_ids,
    file_type,
    mo,
    tables_path,
    timezone,
):
    # 1. Init ClifOrchestrator
    co = ClifOrchestrator(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
    )

    # 2. Load required SOFA tables (once — covers all 120h)
    sofa_cats = REQUIRED_SOFA_CATEGORIES_BY_TABLE

    co.load_table("labs", filters={
        "hospitalization_id": cohort_hosp_ids,
        "lab_category": sofa_cats["labs"],
    })
    co.load_table("vitals", filters={
        "hospitalization_id": cohort_hosp_ids,
        "vital_category": sofa_cats["vitals"],
    })
    co.load_table("patient_assessments", filters={
        "hospitalization_id": cohort_hosp_ids,
        "assessment_category": sofa_cats["patient_assessments"],
    })
    co.load_table("medication_admin_continuous", filters={
        "hospitalization_id": cohort_hosp_ids,
        "med_category": sofa_cats["medication_admin_continuous"],
    })
    co.load_table("respiratory_support", filters={
        "hospitalization_id": cohort_hosp_ids,
    })

    # 3. Clean medication data (remove null doses)
    _med_df = co.medication_admin_continuous.df.copy()
    _med_df = _med_df[_med_df["med_dose"].notna()]
    _med_df = _med_df[_med_df["med_dose_unit"].notna()]
    _med_df = _med_df[~_med_df["med_dose_unit"].astype(str).str.lower().isin(["nan", "none", ""])]
    co.medication_admin_continuous.df = _med_df

    # 4. Convert medication units (SOFA needs mcg/kg/min)
    _sofa_preferred_units = {
        "norepinephrine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "dopamine": "mcg/kg/min",
        "dobutamine": "mcg/kg/min",
    }
    co.convert_dose_units_for_continuous_meds(
        preferred_units=_sofa_preferred_units,
        override=True,
    )

    # 5. Filter to successful conversions
    if hasattr(co.medication_admin_continuous, "df_converted"):
        _med_conv = co.medication_admin_continuous.df_converted.copy()
        if "_convert_status" in _med_conv.columns:
            _med_conv = _med_conv[_med_conv["_convert_status"] == "success"]
            co.medication_admin_continuous.df_converted = _med_conv

    mo.md("### Tables loaded and medication units converted.")
    return co, sofa_cats


# ── Cell 3: Compute SOFA for each 24h window ────────────────────────────
@app.cell
def _(
    co,
    cohort_df,
    intermediate_dir,
    mo,
    n_windows,
    np,
    pd,
    sofa_cats,
):
    # Required med columns for SOFA cardiovascular scoring
    _required_med_cols = [
        "norepinephrine_mcg_kg_min",
        "epinephrine_mcg_kg_min",
        "dopamine_mcg_kg_min",
        "dobutamine_mcg_kg_min",
    ]

    _all_sofa = []

    for _w in range(n_windows):
        _start_h = _w * 24
        _end_h = (_w + 1) * 24

        print(f"Computing SOFA for window {_w}: {_start_h}–{_end_h}h ...")

        # Build cohort time window
        _window_cohort = pd.DataFrame({
            "hospitalization_id": cohort_df["hospitalization_id"].astype(str),
            "start_time": pd.to_datetime(cohort_df["admission_dttm"]) + pd.Timedelta(hours=_start_h),
            "end_time": pd.to_datetime(cohort_df["admission_dttm"]) + pd.Timedelta(hours=_end_h),
        })

        # Create wide dataset for this window
        co.create_wide_dataset(
            category_filters=sofa_cats,
            cohort_df=_window_cohort,
            return_dataframe=True,
        )

        # Add missing med columns
        for _col in _required_med_cols:
            if _col not in co.wide_df.columns:
                co.wide_df[_col] = None

        # Compute SOFA
        _scores = co.compute_sofa_scores(
            wide_df=co.wide_df,
            id_name="hospitalization_id",
            fill_na_scores_with_zero=True,
            remove_outliers=True,
            create_new_wide_df=False,
        )

        # Tag with window info
        _scores["sofa_window"] = _w
        _scores["sofa_window_start_h"] = _start_h
        _scores["sofa_window_end_h"] = _end_h

        _all_sofa.append(_scores)
        print(f"  → {len(_scores)} patients, mean SOFA total = {_scores['sofa_total'].mean():.2f}")

    # Stack all windows
    sofa_all = pd.concat(_all_sofa, ignore_index=True)

    # Save
    _out_path = intermediate_dir / "sofa_scores.parquet"
    sofa_all.to_parquet(_out_path, index=False)

    # Summary stats per window
    _sofa_cols = [c for c in sofa_all.columns if c.startswith("sofa_") and c not in
                  ("sofa_window", "sofa_window_start_h", "sofa_window_end_h")]
    _summary = sofa_all.groupby("sofa_window")[_sofa_cols].agg(["mean", "count"]).round(2)

    # Simplified summary table
    _window_summary = sofa_all.groupby("sofa_window").agg(
        n_patients=("hospitalization_id", "nunique"),
        sofa_mean=("sofa_total", "mean"),
        sofa_median=("sofa_total", "median"),
        sofa_std=("sofa_total", "std"),
    ).round(2)

    mo.md(f"""
    ## SOFA Scores Computed (Time-Varying)

    | Metric | Value |
    |--------|-------|
    | **Total rows** | {len(sofa_all):,} |
    | **Windows** | {n_windows} × 24h |
    | **Saved to** | `{_out_path}` |

    ### Per-Window Summary

    {_window_summary.to_markdown()}
    """)
    return


if __name__ == "__main__":
    app.run()
