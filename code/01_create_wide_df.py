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
#     "sqlglot",
#     "clifpy==0.3.8",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import logging
    import pandas as pd
    import numpy as np
    import yaml
    from pathlib import Path

    from clifpy import (
        Vitals,
        Labs,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        RespiratorySupport,
        CrrtTherapy,
        PatientAssessments,
        Adt,
    )
    from clifpy.utils.outlier_handler import apply_outlier_handling

    from utils import (
        build_weight_table, convert_med_doses, compute_nee,
        categorize_device_from_tracheostomy, categorize_device, impute_fio2,
        setup_logging,
    )

    logger = setup_logging("01_create_wide_df")

    # Project root
    project_root = Path(__file__).parent.parent.resolve()

    # Load site config
    config_path = project_root / "config" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    tables_path = config["tables_path"]
    file_type = config["file_type"]
    site_name = config["site_name"]
    timezone = config["timezone"]

    # Load OHCA variable config
    ohca_config_path = project_root / "config" / "ohca_rl_config.yaml"
    with open(ohca_config_path, "r") as f:
        ohca_config = yaml.safe_load(f)

    # Output directories
    intermediate_dir = project_root / "output" / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Verify cohort file
    cohort_path = intermediate_dir / "cohort_ohca_icu.parquet"
    cohort_exists = cohort_path.exists()
    data_exists = Path(tables_path).exists()

    mo.md(f"""
    ## Setup & Configuration

    | Setting | Value |
    |---------|-------|
    | **Site** | `{site_name}` |
    | **Data path** | `{tables_path}` |
    | **Data exists** | {'Yes' if data_exists else '**NO**'} |
    | **Cohort file** | `{cohort_path}` |
    | **Cohort exists** | {'Yes' if cohort_exists else '**NO — run 00_cohort_identification.py first**'} |
    """)
    return (
        Adt,
        CrrtTherapy,
        Labs,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        PatientAssessments,
        RespiratorySupport,
        Vitals,
        apply_outlier_handling,
        build_weight_table,
        categorize_device,
        categorize_device_from_tracheostomy,
        cohort_path,
        compute_nee,
        convert_med_doses,
        file_type,
        impute_fio2,
        intermediate_dir,
        mo,
        ohca_config,
        pd,
        tables_path,
        timezone,
    )


@app.cell
def _(cohort_path, mo, ohca_config, pd):
    # Load cohort from step 00
    cohort_df = pd.read_parquet(cohort_path)
    cohort_hosp_ids = cohort_df["hospitalization_id"].astype(str).unique().tolist()

    # Variable lists from config
    labs_of_interest = ohca_config["labs_of_interest"]
    vitals_of_interest = ohca_config["vitals_of_interest"]
    meds_cont_of_interest = ohca_config["meds_continuous_of_interest"]
    meds_int_of_interest = ohca_config["meds_intermittent_of_interest"]
    assessments_of_interest = ohca_config["assessments_of_interest"]

    mo.md(f"""
    ## Cohort

    | Metric | Value |
    |--------|-------|
    | **Cohort size** | {len(cohort_hosp_ids):,} hospitalizations |
    | **Labs of interest** | {len(labs_of_interest)} |
    | **Vitals of interest** | {len(vitals_of_interest)} |
    | **Meds (continuous)** | {len(meds_cont_of_interest)} |
    | **Meds (intermittent)** | {len(meds_int_of_interest)} |
    | **Assessments** | {len(assessments_of_interest)} |
    """)
    return (
        assessments_of_interest,
        cohort_df,
        cohort_hosp_ids,
        labs_of_interest,
        meds_cont_of_interest,
        meds_int_of_interest,
        vitals_of_interest,
    )


@app.cell
def _(
    Vitals,
    apply_outlier_handling,
    build_weight_table,
    cohort_hosp_ids,
    file_type,
    intermediate_dir,
    mo,
    tables_path,
    timezone,
    vitals_of_interest,
):
    # Load vitals via individual clifpy loader (filtered to cohort + categories at source)
    vitals_tbl = Vitals.from_file(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
        filters={
            "hospitalization_id": cohort_hosp_ids,
            "vital_category": vitals_of_interest,
        },
    )

    # Apply clifpy outlier handling
    apply_outlier_handling(vitals_tbl)

    # Extract cleaned DataFrame
    vitals_df = vitals_tbl.df.copy()
    vitals_df["hospitalization_id"] = vitals_df["hospitalization_id"].astype(str)
    vitals_df["vital_category"] = vitals_df["vital_category"].str.lower()

    # Build weight lookup table and save as intermediate
    weight_df = build_weight_table(vitals_df)
    weight_df.to_parquet(intermediate_dir / "weight_lookup.parquet", index=False)

    _n_weight_patients = weight_df["hospitalization_id"].nunique()
    _n_weight_rows = len(weight_df)

    # Pivot: narrow → wide
    vitals_wide = vitals_df.pivot_table(
        index=["hospitalization_id", "recorded_dttm"],
        columns="vital_category",
        values="vital_value",
        aggfunc="first",
    ).reset_index()

    # Flatten column names with prefix
    vitals_wide.columns = [
        f"vital_{c}" if c not in ("hospitalization_id", "recorded_dttm") else c
        for c in vitals_wide.columns
    ]
    vitals_wide = vitals_wide.rename(columns={"recorded_dttm": "event_dttm"})

    # Recalculate MAP from SBP/DBP (raw MAP can have artifact values)
    if "vital_sbp" in vitals_wide.columns and "vital_dbp" in vitals_wide.columns:
        _has_bp = vitals_wide["vital_sbp"].notna() & vitals_wide["vital_dbp"].notna()
        vitals_wide["vital_map"] = np.where(
            _has_bp,
            (vitals_wide["vital_sbp"] + 2 * vitals_wide["vital_dbp"]) / 3,
            np.nan,
        )
        logger.info("Recalculated MAP = (SBP + 2*DBP)/3 for %d rows", _has_bp.sum())

    _n_rows = len(vitals_wide)
    _n_hosp = vitals_wide["hospitalization_id"].nunique()
    _vital_cols = [c for c in vitals_wide.columns if c.startswith("vital_")]

    mo.md(f"""
    ## Checkpoint: Vitals (clifpy outlier-handled + pivoted)

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Vital columns** | {len(_vital_cols)}: {', '.join(_vital_cols)} |
    | **Weight lookup** | {_n_weight_rows:,} measurements from {_n_weight_patients:,} patients |

    {vitals_wide[_vital_cols].describe().to_markdown()}
    """)
    return vitals_wide, weight_df


@app.cell
def _(
    Labs,
    apply_outlier_handling,
    cohort_hosp_ids,
    file_type,
    labs_of_interest,
    mo,
    tables_path,
    timezone,
):
    # Load labs via individual clifpy loader (filtered to cohort + categories at source)
    labs_tbl = Labs.from_file(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
        filters={
            "hospitalization_id": cohort_hosp_ids,
            "lab_category": labs_of_interest,
        },
    )

    # Apply clifpy outlier handling
    apply_outlier_handling(labs_tbl)

    # Extract cleaned DataFrame
    labs_df = labs_tbl.df.copy()
    labs_df["hospitalization_id"] = labs_df["hospitalization_id"].astype(str)
    labs_df["lab_category"] = labs_df["lab_category"].str.lower()

    # Pivot
    labs_wide = labs_df.pivot_table(
        index=["hospitalization_id", "lab_result_dttm"],
        columns="lab_category",
        values="lab_value_numeric",
        aggfunc="first",
    ).reset_index()

    labs_wide.columns = [
        f"lab_{c}" if c not in ("hospitalization_id", "lab_result_dttm") else c
        for c in labs_wide.columns
    ]
    labs_wide = labs_wide.rename(columns={"lab_result_dttm": "event_dttm"})

    # Keep only columns of interest
    _labs_keep = [
        "hospitalization_id", "event_dttm",
        "lab_bicarbonate", "lab_bun", "lab_calcium_total", "lab_chloride",
        "lab_creatinine", "lab_glucose_serum", "lab_hemoglobin", "lab_lactate",
        "lab_magnesium", "lab_pco2_arterial", "lab_ph_arterial", "lab_po2_arterial",
        "lab_potassium", "lab_so2_arterial", "lab_sodium",
    ]
    labs_wide = labs_wide[[c for c in _labs_keep if c in labs_wide.columns]]

    _n_rows = len(labs_wide)
    _n_hosp = labs_wide["hospitalization_id"].nunique()
    _lab_cols = [c for c in labs_wide.columns if c.startswith("lab_")]

    mo.md(f"""
    ## Checkpoint: Labs (clifpy outlier-handled + pivoted)

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Lab columns** | {len(_lab_cols)}: {', '.join(_lab_cols)} |

    {labs_wide[_lab_cols].describe().to_markdown()}
    """)
    return (labs_wide,)


@app.cell
def _(
    MedicationAdminContinuous,
    apply_outlier_handling,
    cohort_hosp_ids,
    compute_nee,
    convert_med_doses,
    file_type,
    meds_cont_of_interest,
    mo,
    ohca_config,
    pd,
    tables_path,
    timezone,
    weight_df,
):
    # Load continuous meds via individual clifpy loader (filtered to cohort + categories at source)
    meds_cont_tbl = MedicationAdminContinuous.from_file(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
        filters={
            "hospitalization_id": cohort_hosp_ids,
            "med_category": meds_cont_of_interest,
        },
    )

    # Apply clifpy outlier handling (caps implausible doses per med/unit thresholds)
    apply_outlier_handling(meds_cont_tbl)

    meds_cont_df = meds_cont_tbl.df.copy()
    meds_cont_df["hospitalization_id"] = meds_cont_df["hospitalization_id"].astype(str)
    meds_cont_df["med_category"] = meds_cont_df["med_category"].str.lower()

    # Filter to non null doses only
    meds_cont_df = meds_cont_df[
        meds_cont_df["med_dose"].notna()
        & (meds_cont_df["med_dose"] >= 0)
    ].copy()

    # Unit conversion using local pipeline (utils.py)
    _cont_preferred = ohca_config.get("meds_continuous_preferred_units", {})
    meds_cont_df, _cont_counts = convert_med_doses(
        meds_cont_df, weight_df, _cont_preferred
    )

    # Use converted dose where available
    if "med_dose_converted" in meds_cont_df.columns:
        meds_cont_df["med_dose"] = meds_cont_df["med_dose_converted"]

    _ts_col = "admin_dttm"

    # Pivot
    meds_cont_wide = meds_cont_df.pivot_table(
        index=["hospitalization_id", _ts_col],
        columns="med_category",
        values="med_dose",
        aggfunc="first",
    ).reset_index()

    meds_cont_wide.columns = [
        f"med_cont_{c}" if c not in ("hospitalization_id", _ts_col) else c
        for c in meds_cont_wide.columns
    ]
    meds_cont_wide = meds_cont_wide.rename(columns={_ts_col: "event_dttm"})
    if meds_cont_wide["event_dttm"].dtype == object:
        meds_cont_wide["event_dttm"] = pd.to_datetime(
            meds_cont_wide["event_dttm"], utc=True
        )

    # NEE is computed post-ffill in step 03 (not here) to avoid aggregation artifacts

    # Keep only columns of interest
    _meds_cont_keep = [
        "hospitalization_id", "event_dttm",
        "med_cont_angiotensin", "med_cont_cisatracurium",
        "med_cont_dexmedetomidine", "med_cont_dobutamine", "med_cont_dopamine",
        "med_cont_epinephrine", "med_cont_isoproterenol", "med_cont_lorazepam",
        "med_cont_midazolam", "med_cont_milrinone", "med_cont_norepinephrine",
        "med_cont_phenylephrine", "med_cont_propofol", "med_cont_vasopressin",
    ]
    meds_cont_wide = meds_cont_wide[[c for c in _meds_cont_keep if c in meds_cont_wide.columns]]

    _n_rows = len(meds_cont_wide)
    _n_hosp = meds_cont_wide["hospitalization_id"].nunique()
    _med_cols = [c for c in meds_cont_wide.columns if c.startswith("med_cont_")]

    mo.md(f"""
    ## Checkpoint: Continuous Meds (local unit-converted + pivoted)

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Med columns** | {len(_med_cols)}: {', '.join(_med_cols)} |

    {meds_cont_wide[_med_cols].describe().to_markdown()}
    """)
    return meds_cont_df, meds_cont_wide


@app.cell
def _(intermediate_dir, meds_cont_df):
    meds_save_int_path = intermediate_dir / "meds_cont_df.parquet"
    meds_cont_df.to_parquet(meds_save_int_path, index=False)
    return


@app.cell
def _(
    MedicationAdminIntermittent,
    apply_outlier_handling,
    cohort_hosp_ids,
    convert_med_doses,
    file_type,
    meds_int_of_interest,
    mo,
    ohca_config,
    pd,
    tables_path,
    timezone,
    weight_df,
):
    try:
        # Load intermittent meds via individual clifpy loader (filtered to cohort + categories at source)
        meds_int_tbl = MedicationAdminIntermittent.from_file(
            data_directory=tables_path,
            filetype=file_type,
            timezone=timezone,
            filters={
                "hospitalization_id": cohort_hosp_ids,
                "med_category": meds_int_of_interest,
            },
        )

        # Apply clifpy outlier handling (caps implausible doses)
        apply_outlier_handling(meds_int_tbl)

        _int_df = meds_int_tbl.df.copy()
        _int_df["hospitalization_id"] = _int_df["hospitalization_id"].astype(str)
        _int_df["med_category"] = _int_df["med_category"].str.lower()

        # Filter to valid doses only
        _int_df = _int_df[
            _int_df["med_dose"].notna()
        ].copy()

        # Unit conversion using local pipeline
        _int_preferred = ohca_config.get("meds_intermittent_preferred_units", {})
        if len(_int_df) > 0 and _int_preferred:
            _int_df, _int_counts = convert_med_doses(
                _int_df, weight_df, _int_preferred
            )
            if "med_dose_converted" in _int_df.columns:
                _int_df["med_dose"] = _int_df["med_dose_converted"]

        _ts_col = "admin_dttm"

        meds_int_wide = _int_df.pivot_table(
            index=["hospitalization_id", _ts_col],
            columns="med_category",
            values="med_dose",
            aggfunc="first",
        ).reset_index()

        meds_int_wide.columns = [
            f"med_int_{c}" if c not in ("hospitalization_id", _ts_col) else c
            for c in meds_int_wide.columns
        ]
        meds_int_wide = meds_int_wide.rename(columns={_ts_col: "event_dttm"})
        if meds_int_wide["event_dttm"].dtype == object:
            meds_int_wide["event_dttm"] = pd.to_datetime(
                meds_int_wide["event_dttm"], utc=True
            )

        # Keep only columns of interest
        _meds_int_keep = [
            "hospitalization_id", "event_dttm",
            "med_int_diazepam", "med_int_lorazepam", "med_int_valproate",
        ]
        meds_int_wide = meds_int_wide[[c for c in _meds_int_keep if c in meds_int_wide.columns]]

        _n_rows = len(meds_int_wide)
        _n_hosp = meds_int_wide["hospitalization_id"].nunique()
        _int_cols = [c for c in meds_int_wide.columns if c.startswith("med_int_")]

        _msg = f"""
        ## Checkpoint: Intermittent Meds (Anti-convulsants)

        | Metric | Value |
        |--------|-------|
        | **Rows** | {_n_rows:,} |
        | **Hospitalizations** | {_n_hosp:,} |
        | **Columns** | {', '.join(_int_cols)} |
        """
    except Exception as _e:
        meds_int_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
        _msg = f"""
        ## Checkpoint: Intermittent Meds

        Table not available or error: {_e}. Skipping.
        """

    mo.md(_msg)
    return (meds_int_wide,)


@app.cell
def _(
    RespiratorySupport,
    apply_outlier_handling,
    categorize_device,
    categorize_device_from_tracheostomy,
    cohort_hosp_ids,
    file_type,
    impute_fio2,
    mo,
    tables_path,
    timezone,
):
    # Load respiratory support via individual clifpy loader (filtered to cohort at source)
    resp_tbl = RespiratorySupport.from_file(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
        filters={"hospitalization_id": cohort_hosp_ids},
    )

    # Apply clifpy outlier handling BEFORE waterfall
    apply_outlier_handling(resp_tbl)

    # --- PRE-WATERFALL: device categorization ---
    # Infer device_category from tracheostomy + LPM (image logic)
    resp_tbl.df = categorize_device_from_tracheostomy(resp_tbl.df)
    # Infer device_category from mode, fio2, lpm, peep, tidal_volume
    resp_tbl.df = categorize_device(resp_tbl.df)

    # Apply waterfall — fills missing FiO2, infers IMV/NIPPV, forward-fills params
    resp_tbl = resp_tbl.waterfall()

    # --- POST-WATERFALL: FiO2 imputation for remaining gaps ---
    resp_tbl.df = impute_fio2(resp_tbl.df)

    # Extract processed DataFrame
    resp_df = resp_tbl.df.copy()
    resp_df["hospitalization_id"] = resp_df["hospitalization_id"].astype(str)

    # Rename columns with resp_ prefix
    _rename = {}
    for _col in resp_df.columns:
        if _col not in ("hospitalization_id", "recorded_dttm"):
            _rename[_col] = f"resp_{_col}"
    resp_wide = resp_df.rename(columns=_rename)
    resp_wide = resp_wide.rename(columns={"recorded_dttm": "event_dttm"})

    # Keep only columns of interest
    _resp_keep = [
        "hospitalization_id", "event_dttm",
        "resp_device_name", "resp_device_category", "resp_mode_name",
        "resp_mode_category", "resp_vent_brand_name", "resp_artificial_airway",
        "resp_tracheostomy", "resp_fio2_set", "resp_lpm_set",
        "resp_tidal_volume_set", "resp_resp_rate_set", "resp_peep_set",
        "resp_tidal_volume_obs", "resp_resp_rate_obs",
    ]
    resp_wide = resp_wide[[c for c in _resp_keep if c in resp_wide.columns]]

    _n_rows = len(resp_wide)
    _n_hosp = resp_wide["hospitalization_id"].nunique()
    _resp_cols = [c for c in resp_wide.columns if c.startswith("resp_")]

    mo.md(f"""
    ## Checkpoint: Respiratory Support (clifpy waterfall + outlier-handled)

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Columns** | {len(_resp_cols)} |

    Columns: {', '.join(_resp_cols[:15])}{'...' if len(_resp_cols) > 15 else ''}
    """)
    return (resp_wide,)


@app.cell
def _(CrrtTherapy, cohort_hosp_ids, file_type, mo, pd, tables_path, timezone):
    try:
        # Load CRRT via individual clifpy loader (filtered to cohort at source)
        crrt_tbl = CrrtTherapy.from_file(
            data_directory=tables_path,
            filetype=file_type,
            timezone=timezone,
            filters={"hospitalization_id": cohort_hosp_ids},
        )

        crrt_df = crrt_tbl.df.copy()
        crrt_df["hospitalization_id"] = crrt_df["hospitalization_id"].astype(str)
        crrt_df["on_crrt"] = 1

        _rename = {}
        for _col in crrt_df.columns:
            if _col not in ("hospitalization_id", "recorded_dttm", "on_crrt"):
                _rename[_col] = f"crrt_{_col}"
        crrt_wide = crrt_df.rename(columns=_rename)
        crrt_wide = crrt_wide.rename(columns={"recorded_dttm": "event_dttm"})

        # Keep only columns of interest
        _crrt_keep = [
            "hospitalization_id", "event_dttm",
            "on_crrt", "crrt_crrt_mode_name", "crrt_crrt_mode_category",
        ]
        crrt_wide = crrt_wide[[c for c in _crrt_keep if c in crrt_wide.columns]]

        _n_rows = len(crrt_wide)
        _n_hosp = crrt_wide["hospitalization_id"].nunique()
        _msg = f"""
        ## Checkpoint: CRRT

        | Metric | Value |
        |--------|-------|
        | **Rows** | {_n_rows:,} |
        | **Hospitalizations with CRRT** | {_n_hosp:,} |
        """
    except Exception as _e:
        crrt_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
        _msg = f"""
        ## Checkpoint: CRRT

        Table not available: {_e}. Skipping.
        """

    mo.md(_msg)
    return (crrt_wide,)


@app.cell
def _(
    PatientAssessments,
    apply_outlier_handling,
    assessments_of_interest,
    cohort_hosp_ids,
    file_type,
    mo,
    pd,
    tables_path,
    timezone,
):
    try:
        # Load patient assessments via clifpy loader (filtered to cohort + categories)
        assess_tbl = PatientAssessments.from_file(
            data_directory=tables_path,
            filetype=file_type,
            timezone=timezone,
            filters={
                "hospitalization_id": cohort_hosp_ids,
                "assessment_category": assessments_of_interest,
            },
        )

        # Apply clifpy outlier handling (gcs_total: 3-15, RASS: -5 to +4)
        apply_outlier_handling(assess_tbl)

        # Extract cleaned DataFrame
        assess_df = assess_tbl.df.copy()
        assess_df["hospitalization_id"] = assess_df["hospitalization_id"].astype(str)
        assess_df["assessment_category"] = assess_df["assessment_category"].str.lower()

        # Pivot: narrow → wide on (hospitalization_id, recorded_dttm)
        assess_wide = assess_df.pivot_table(
            index=["hospitalization_id", "recorded_dttm"],
            columns="assessment_category",
            values="numerical_value",
            aggfunc="first",
        ).reset_index()

        # Flatten columns with assess_ prefix
        assess_wide.columns = [
            f"assess_{c}" if c not in ("hospitalization_id", "recorded_dttm") else c
            for c in assess_wide.columns
        ]
        assess_wide = assess_wide.rename(columns={"recorded_dttm": "event_dttm"})

        # Ensure datetime is tz-aware
        if assess_wide["event_dttm"].dtype == object:
            assess_wide["event_dttm"] = pd.to_datetime(
                assess_wide["event_dttm"], utc=True
            )

        # Keep only columns of interest
        _assess_keep = [
            "hospitalization_id", "event_dttm",
            "assess_gcs_total", "assess_rass",
        ]
        assess_wide = assess_wide[[c for c in _assess_keep if c in assess_wide.columns]]

        _n_rows = len(assess_wide)
        _n_hosp = assess_wide["hospitalization_id"].nunique()
        _n_cohort = len(cohort_hosp_ids)
        _assess_cols = [c for c in assess_wide.columns if c.startswith("assess_")]

        # --- PI Missingness Report ---
        _miss_lines = []
        for _ac in _assess_cols:
            _patients_with_data = assess_wide.loc[
                assess_wide[_ac].notna(), "hospitalization_id"
            ].nunique()
            _pct_patients = _patients_with_data / _n_cohort * 100
            _total_obs = assess_wide[_ac].notna().sum()
            _med_per_patient = (
                assess_wide.loc[assess_wide[_ac].notna()]
                .groupby("hospitalization_id")[_ac]
                .count()
                .median()
            ) if _patients_with_data > 0 else 0
            _miss_lines.append(
                f"| **{_ac}** | {_patients_with_data:,} / {_n_cohort:,} "
                f"({_pct_patients:.1f}%) | {_total_obs:,} | {_med_per_patient:.0f} |"
            )

        # --- Charting Frequency (median hours between consecutive observations) ---
        _freq_lines = []
        for _ac in _assess_cols:
            _obs_times = (
                assess_wide.loc[assess_wide[_ac].notna(), ["hospitalization_id", "event_dttm"]]
                .sort_values(["hospitalization_id", "event_dttm"])
            )
            if len(_obs_times) > 1:
                _gaps = _obs_times.groupby("hospitalization_id")["event_dttm"].diff()
                _median_gap_h = _gaps.dt.total_seconds().dropna().median() / 3600
                _freq_lines.append(f"| **{_ac}** | {_median_gap_h:.1f} |")
            else:
                _freq_lines.append(f"| **{_ac}** | N/A |")

        # --- Missingness by Time Window (% patients with ≥1 obs in each window) ---
        # Use first assessment per patient as t0 reference
        _patient_t0 = (
            assess_wide.groupby("hospitalization_id")["event_dttm"]
            .min()
            .reset_index()
            .rename(columns={"event_dttm": "_t0"})
        )
        _with_t0 = assess_wide.merge(_patient_t0, on="hospitalization_id")
        _with_t0["_hours_since_t0"] = (
            (_with_t0["event_dttm"] - _with_t0["_t0"]).dt.total_seconds() / 3600
        )

        _windows = [(0, 24), (24, 48), (48, 72), (72, 120)]
        _window_lines = []
        for _ac in _assess_cols:
            _cells = []
            for _ws, _we in _windows:
                _window_df = _with_t0[
                    (_with_t0["_hours_since_t0"] >= _ws) & (_with_t0["_hours_since_t0"] < _we)
                ]
                _pts_in_window = _window_df["hospitalization_id"].nunique()
                _pts_with_obs = _window_df.loc[
                    _window_df[_ac].notna(), "hospitalization_id"
                ].nunique()
                _pct = (_pts_with_obs / _pts_in_window * 100) if _pts_in_window > 0 else 0
                _cells.append(f"{_pts_with_obs}/{_pts_in_window} ({_pct:.0f}%)")
            _window_lines.append(f"| **{_ac}** | {' | '.join(_cells)} |")

        _msg = f"""
        ## Checkpoint: Patient Assessments (GCS, RASS)

        | Metric | Value |
        |--------|-------|
        | **Rows** | {_n_rows:,} |
        | **Hospitalizations** | {_n_hosp:,} |
        | **Columns** | {', '.join(_assess_cols)} |

        ### PI Missingness Report

        | Variable | Patients with data | Total obs | Median obs/patient |
        |----------|-------------------|-----------|-------------------|
        {chr(10).join(_miss_lines)}

        ### Charting Frequency

        | Variable | Median hours between obs |
        |----------|--------------------------|
        {chr(10).join(_freq_lines)}

        ### Missingness by Time Window (patients with ≥1 obs, relative to first assessment)

        | Variable | 0–24h | 24–48h | 48–72h | 72–120h |
        |----------|-------|--------|--------|---------|
        {chr(10).join(_window_lines)}

        ### Value Distributions
        {assess_wide[_assess_cols].describe().to_markdown()}
        """
    except Exception as _e:
        assess_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
        _msg = f"""
        ## Checkpoint: Patient Assessments

        Table not available or error: {_e}. Skipping.
        """

    mo.md(_msg)
    return (assess_wide,)


@app.cell
def _(Adt, cohort_hosp_ids, file_type, mo, tables_path, timezone):
    # Load ADT via individual clifpy loader (filtered to cohort at source)
    adt_tbl = Adt.from_file(
        data_directory=tables_path,
        filetype=file_type,
        timezone=timezone,
        filters={"hospitalization_id": cohort_hosp_ids},
    )

    adt_df = adt_tbl.df.copy()
    adt_df["hospitalization_id"] = adt_df["hospitalization_id"].astype(str)

    # Lowercase category columns
    if "location_category" in adt_df.columns:
        adt_df["location_category"] = adt_df["location_category"].str.lower()

    adt_wide = adt_df.rename(
        columns={
            "in_dttm": "event_dttm",
            "location_category": "adt_location_category",
        }
    )
    if "out_dttm" in adt_wide.columns:
        adt_wide = adt_wide.rename(columns={"out_dttm": "adt_out_dttm"})

    # Drop other columns that aren't needed
    _keep = ["hospitalization_id", "event_dttm", "adt_location_category"]
    if "adt_out_dttm" in adt_wide.columns:
        _keep.append("adt_out_dttm")
    adt_wide = adt_wide[[c for c in _keep if c in adt_wide.columns]]

    _n_rows = len(adt_wide)
    _n_hosp = adt_wide["hospitalization_id"].nunique()

    mo.md(f"""
    ## Checkpoint: ADT Location

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    """)
    return (adt_wide,)


@app.cell
def _(
    adt_wide,
    assess_wide,
    crrt_wide,
    intermediate_dir,
    labs_wide,
    meds_cont_wide,
    meds_int_wide,
    mo,
    resp_wide,
    vitals_wide,
):
    # Outer merge all tables on (hospitalization_id, event_dttm)
    wide_df = vitals_wide.copy()

    for _df_name, _df in [
        ("labs", labs_wide),
        ("meds_cont", meds_cont_wide),
        ("meds_int", meds_int_wide),
        ("resp", resp_wide),
        ("crrt", crrt_wide),
        ("assess", assess_wide),
        ("adt", adt_wide),
    ]:
        if len(_df) > 0 and "event_dttm" in _df.columns:
            wide_df = wide_df.merge(
                _df,
                on=["hospitalization_id", "event_dttm"],
                how="outer",
                suffixes=("", f"_{_df_name}_dup"),
            )

    # Sort
    wide_df = wide_df.sort_values(["hospitalization_id", "event_dttm"]).reset_index(
        drop=True
    )

    # Summary
    _n_rows = len(wide_df)
    _n_hosp = wide_df["hospitalization_id"].nunique()
    _n_cols = len(wide_df.columns)

    # Column groups
    _vital_cols = [c for c in wide_df.columns if c.startswith("vital_")]
    _lab_cols = [c for c in wide_df.columns if c.startswith("lab_")]
    _med_cont_cols = [c for c in wide_df.columns if c.startswith("med_cont_")]
    _med_int_cols = [c for c in wide_df.columns if c.startswith("med_int_")]
    _resp_cols = [c for c in wide_df.columns if c.startswith("resp_")]
    _crrt_cols = [c for c in wide_df.columns if c.startswith("crrt_")]
    _assess_cols = [c for c in wide_df.columns if c.startswith("assess_")]
    _adt_cols = [c for c in wide_df.columns if c.startswith("adt_")]

    def _group_miss(cols):
        if not cols:
            return "N/A"
        return f"{wide_df[cols].isna().mean().mean() * 100:.1f}%"

    # Save t0 mapping (first event_dttm per patient) for SOFA alignment
    _t0_map = (
        wide_df.groupby("hospitalization_id")["event_dttm"]
        .min()
        .reset_index()
        .rename(columns={"event_dttm": "t0_dttm"})
    )
    _t0_map.to_parquet(intermediate_dir / "t0_mapping.parquet", index=False)
    logger.info("Saved t0_mapping.parquet (%d patients)", len(_t0_map))

    # Save
    _output_path = intermediate_dir / "wide_df.parquet"
    wide_df.to_parquet(_output_path, index=False)
    _file_size_mb = _output_path.stat().st_size / (1024 * 1024)

    mo.md(f"""
    ## Wide DataFrame — Final Output

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Total columns** | {_n_cols} |
    | **Saved to** | `{_output_path}` |
    | **File size** | {_file_size_mb:.1f} MB |

    ### Column Groups

    | Group | Count | Avg Missing |
    |-------|-------|-------------|
    | Vitals (`vital_*`) | {len(_vital_cols)} | {_group_miss(_vital_cols)} |
    | Labs (`lab_*`) | {len(_lab_cols)} | {_group_miss(_lab_cols)} |
    | Meds continuous (`med_cont_*`) | {len(_med_cont_cols)} | {_group_miss(_med_cont_cols)} |
    | Meds intermittent (`med_int_*`) | {len(_med_int_cols)} | {_group_miss(_med_int_cols)} |
    | Respiratory (`resp_*`) | {len(_resp_cols)} | {_group_miss(_resp_cols)} |
    | CRRT (`crrt_*`) | {len(_crrt_cols)} | {_group_miss(_crrt_cols)} |
    | Assessments (`assess_*`) | {len(_assess_cols)} | {_group_miss(_assess_cols)} |
    | ADT (`adt_*`) | {len(_adt_cols)} | {_group_miss(_adt_cols)} |

    ### All Columns
    {', '.join(sorted(wide_df.columns.tolist()))}
    """)
    return


if __name__ == "__main__":
    app.run()
