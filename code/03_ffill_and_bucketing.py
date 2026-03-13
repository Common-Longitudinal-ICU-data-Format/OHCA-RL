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
app = marimo.App(width="medium")


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
    logger = setup_logging("03_ffill_and_bucketing")

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as _f:
        ohca_config = yaml.safe_load(_f)

    intermediate_dir = project_root / "output" / "intermediate"
    final_dir = project_root / "output" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Load raw wide_df from step 01
    wide_df = pd.read_parquet(intermediate_dir / "wide_df.parquet")
    wide_df["hospitalization_id"] = wide_df["hospitalization_id"].astype(str)
    logger.info("Loaded wide_df: %d rows, %d hospitalizations", len(wide_df), wide_df["hospitalization_id"].nunique())

    # Load static patient-level df (demographics, death_dttm, discharge info) from step 00
    patient_static = pd.read_parquet(intermediate_dir / "patient_static.parquet")
    patient_static["hospitalization_id"] = patient_static["hospitalization_id"].astype(str)
    death_time_df = patient_static[["hospitalization_id", "death_dttm"]].copy()
    logger.info("Loaded patient_static: %d patients, death_dttm available for %d",
                len(patient_static), patient_static["death_dttm"].notna().sum())

    # Config
    bucket_minutes = ohca_config["action_inference"]["interval_minutes"]  # 60
    bucket_hours = bucket_minutes / 60  # 1.0
    time_window_hours = ohca_config["action_inference"]["time_window_hours"]  # 120

    # Reusable missingness helper
    def compute_missingness(df, cols):
        """Per-variable patient-level and row-level missingness."""
        _n_hosp = df["hospitalization_id"].nunique()
        _n_rows = len(df)
        _rows = []
        for _c in cols:
            if _c not in df.columns:
                continue
            _na_rows = df[_c].isna().sum()
            _patients_with_data = df.loc[df[_c].notna(), "hospitalization_id"].nunique()
            _rows.append({
                "variable": _c,
                "row_missing_pct": round(_na_rows / _n_rows * 100, 1),
                "patient_missing_pct": round((1 - _patients_with_data / _n_hosp) * 100, 1),
            })
        return pd.DataFrame(_rows)

    # Identify column groups
    vital_cols = [c for c in wide_df.columns if c.startswith("vital_")]
    lab_cols = [c for c in wide_df.columns if c.startswith("lab_")]
    med_cont_cols = [c for c in wide_df.columns if c.startswith("med_cont_")]
    med_int_cols = [c for c in wide_df.columns if c.startswith("med_int_")]
    resp_numeric_cols = [
        c for c in wide_df.columns
        if c in (
            "resp_fio2_set", "resp_peep_set", "resp_lpm_set",
            "resp_tidal_volume_set", "resp_resp_rate_set",
            "resp_tidal_volume_obs", "resp_resp_rate_obs",
        )
    ]
    resp_categorical_cols = [
        c for c in wide_df.columns
        if c in (
            "resp_device_name", "resp_device_category", "resp_mode_name",
            "resp_mode_category", "resp_vent_brand_name",
        )
    ]
    # Drop resp_artificial_airway — high missingness, redundant with device/mode info
    if "resp_artificial_airway" in wide_df.columns:
        wide_df = wide_df.drop(columns=["resp_artificial_airway"])
    # Binary resp indicators — treat as numeric, not categorical
    resp_binary_cols = [
        c for c in wide_df.columns
        if c in ("resp_tracheostomy",)
    ]
    assess_cols = [c for c in wide_df.columns if c.startswith("assess_")]

    _n_rows = len(wide_df)
    _n_hosp = wide_df["hospitalization_id"].nunique()

    mo.md(f"""
    ## Step 02: Forward-Fill & Time Bucketing

    | Setting | Value |
    |---------|-------|
    | **Input rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Bucket size** | {bucket_minutes} min |
    | **Time window** | {time_window_hours}h |
    | **Vital cols** | {len(vital_cols)} |
    | **Lab cols** | {len(lab_cols)} |
    | **Med cont cols** | {len(med_cont_cols)} |
    | **Med int cols** | {len(med_int_cols)} |
    | **Resp numeric cols** | {len(resp_numeric_cols)} |
    | **Resp categorical cols** | {len(resp_categorical_cols)} |
    | **Assess cols** | {len(assess_cols)} |
    """)
    return (
        assess_cols,
        bucket_hours,
        compute_missingness,
        death_time_df,
        final_dir,
        intermediate_dir,
        lab_cols,
        logger,
        med_cont_cols,
        med_int_cols,
        mo,
        np,
        ohca_config,
        patient_static,
        pd,
        resp_binary_cols,
        resp_categorical_cols,
        resp_numeric_cols,
        time_window_hours,
        vital_cols,
        wide_df,
    )


# ── Cell 2: Missingness BEFORE ffill ──────────────────────────────────
@app.cell
def _(
    assess_cols,
    compute_missingness,
    final_dir,
    lab_cols,
    logger,
    med_cont_cols,
    med_int_cols,
    mo,
    pd,
    resp_binary_cols,
    resp_categorical_cols,
    resp_numeric_cols,
    vital_cols,
    wide_df,
):
    _all_numeric_cols = (
        vital_cols + lab_cols + med_cont_cols + med_int_cols
        + resp_numeric_cols + resp_binary_cols + assess_cols
    )
    _all_cols = _all_numeric_cols + resp_categorical_cols
    if "on_crrt" in wide_df.columns:
        _all_cols = _all_cols + ["on_crrt"]
    if "adt_location_category" in wide_df.columns:
        _all_cols = _all_cols + ["adt_location_category"]

    miss_before = compute_missingness(wide_df, _all_cols)
    miss_before.to_csv(final_dir / "missingness_before_ffill.csv", index=False)
    logger.info("Saved missingness_before_ffill.csv (%d variables)", len(miss_before))

    mo.md(
        "### Missingness BEFORE Forward-Fill\n\n"
        + miss_before.to_markdown(index=False)
    )
    return (miss_before,)


# ── Cell 3: Forward-Fill ────────────────────────────────────────────────
@app.cell
def _(
    assess_cols,
    lab_cols,
    logger,
    med_cont_cols,
    med_int_cols,
    mo,
    np,
    ohca_config,
    pd,
    resp_binary_cols,
    resp_categorical_cols,
    resp_numeric_cols,
    vital_cols,
    wide_df,
):
    ffilled = wide_df.copy()
    ffilled["event_dttm"] = pd.to_datetime(ffilled["event_dttm"], utc=True)

    # ── Pre-ffill: remove sentinel/placeholder and clamp outliers ──
    if "lab_glucose_serum" in ffilled.columns:
        _sentinel_mask = ffilled["lab_glucose_serum"] <= 1
        _n_sentinel = _sentinel_mask.sum()
        if _n_sentinel > 0:
            ffilled.loc[_sentinel_mask, "lab_glucose_serum"] = np.nan
            logger.info("Removed %d glucose sentinel values (<=1) before ffill", _n_sentinel)

    if "lab_ph_arterial" in ffilled.columns:
        _ph_oob = (ffilled["lab_ph_arterial"] < 6.5) | (ffilled["lab_ph_arterial"] > 7.8)
        _n_ph_oob = _ph_oob.sum()
        if _n_ph_oob > 0:
            ffilled.loc[_ph_oob, "lab_ph_arterial"] = np.nan
            logger.info("Removed %d pH outliers (outside 6.5-7.8) before ffill", _n_ph_oob)

    # ── Normal values for fallback imputation (from config) ──
    _lab_normal_values = {
        f"lab_{k}": v
        for k, v in ohca_config["labs_normal_values"].items()
    }

    # --- 1. Vitals: unlimited ffill + bfill for height/weight/temp ---
    logger.info("Step 1: Forward-filling vitals (unlimited)...")
    for _col in vital_cols:
        _before = ffilled[_col].isna().sum()
        ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].ffill()
        _after = ffilled[_col].isna().sum()
        if _before != _after:
            logger.info("  %s: %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)

    # Bfill leading NAs for slow-changing vitals
    _bfill_vitals = [c for c in vital_cols if any(
        v in c for v in ("height", "weight", "temp")
    )]
    if _bfill_vitals:
        logger.info("Step 1b: Backward-filling leading NAs for %s", _bfill_vitals)
        for _col in _bfill_vitals:
            _before = ffilled[_col].isna().sum()
            ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].bfill()
            _after = ffilled[_col].isna().sum()
            if _before != _after:
                logger.info("  %s (bfill): %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)

    # 1c. Vitals remaining: patient median → cohort median
    logger.info("Step 1c: Filling remaining vital NaN with patient median → cohort median...")
    for _col in vital_cols:
        _before = ffilled[_col].isna().sum()
        if _before == 0:
            continue
        # Patient median
        _patient_median = ffilled.groupby("hospitalization_id")[_col].transform("median")
        ffilled[_col] = ffilled[_col].fillna(_patient_median)
        # Cohort median for patients with zero measurements
        _cohort_median = ffilled[_col].median()
        ffilled[_col] = ffilled[_col].fillna(_cohort_median)
        _after = ffilled[_col].isna().sum()
        logger.info("  %s: %d NaN → %d NaN (patient+cohort median)", _col, _before, _after)

    # --- 2. Labs: time-limited ffill (12h max), then normal-value fallback ---
    logger.info("Step 2: Forward-filling labs (12h time-limited)...")
    _lab_max_delta = pd.Timedelta("12h")
    _available_lab_cols = [c for c in lab_cols if c in ffilled.columns]

    if _available_lab_cols:
        for _col in _available_lab_cols:
            _before = ffilled[_col].isna().sum()

            # Vectorized time-limited ffill
            _notna_mask = ffilled[_col].notna()
            _last_valid_time = ffilled["event_dttm"].where(_notna_mask)
            _last_valid_time = _last_valid_time.groupby(
                ffilled["hospitalization_id"]
            ).ffill()
            _time_gap = ffilled["event_dttm"] - _last_valid_time
            _filled_vals = ffilled.groupby("hospitalization_id")[_col].ffill()
            ffilled[_col] = _filled_vals.where(_time_gap <= _lab_max_delta)

            _after = ffilled[_col].isna().sum()
            if _before != _after:
                logger.info("  %s: %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)

    # 2b. Labs remaining: normal-value imputation
    logger.info("Step 2b: Filling remaining lab NaN with normal values...")
    for _col in _available_lab_cols:
        _before = ffilled[_col].isna().sum()
        if _before == 0:
            continue
        _normal = _lab_normal_values.get(_col)
        if _normal is not None:
            ffilled[_col] = ffilled[_col].fillna(_normal)
            logger.info("  %s: %d NaN → 0 NaN (filled with normal=%.2f)", _col, _before, _normal)
        else:
            # Fallback: cohort median for labs not in normal_values dict
            _cohort_med = ffilled[_col].median()
            ffilled[_col] = ffilled[_col].fillna(_cohort_med)
            logger.info("  %s: %d NaN → 0 NaN (filled with cohort median=%.2f)", _col, _before, _cohort_med)

    # --- 2c. Assessments (GCS, RASS): time-limited ffill, NO default imputation ---
    # GCS and RASS are charted q4-6h. 8h ffill covers one missed charting period.
    # CRITICAL: Do NOT impute missing values — OHCA patients are deeply comatose (GCS 3)
    # and heavily sedated (RASS -5). Normal-value imputation (GCS=15, RASS=0) would be
    # clinically incorrect for this population. NaN stays NaN.
    _assess_ffill_hours = ohca_config.get("assessments_ffill_hours", 8)
    _assess_max_delta = pd.Timedelta(hours=_assess_ffill_hours)
    _available_assess_cols = [c for c in assess_cols if c in ffilled.columns]
    if _available_assess_cols:
        logger.info("Step 2c: Forward-filling assessments (%dh time-limited, NO default imputation)...",
                    _assess_ffill_hours)
        for _col in _available_assess_cols:
            _before = ffilled[_col].isna().sum()

            # Vectorized time-limited ffill (same pattern as labs)
            _notna_mask = ffilled[_col].notna()
            _last_valid_time = ffilled["event_dttm"].where(_notna_mask)
            _last_valid_time = _last_valid_time.groupby(
                ffilled["hospitalization_id"]
            ).ffill()
            _time_gap = ffilled["event_dttm"] - _last_valid_time
            _filled_vals = ffilled.groupby("hospitalization_id")[_col].ffill()
            ffilled[_col] = _filled_vals.where(_time_gap <= _assess_max_delta)

            _after = ffilled[_col].isna().sum()
            if _before != _after:
                logger.info("  %s: %d NaN → %d NaN (filled %d, %dh limit)",
                            _col, _before, _after, _before - _after, _assess_ffill_hours)

        logger.info("  Assessments: remaining NaN left as NaN (no safe default for OHCA)")

    # --- 3a. Continuous meds: unlimited ffill between observations, time-limited after last ---
    # These are infusion RATES (mcg/kg/min) — rate stays constant between charting events.
    # Logic:
    #   - Between recorded values: unlimited ffill (infusion is running)
    #   - After LAST observation per patient:
    #       - If last value = 0: drug was explicitly stopped → fillna(0)
    #       - If last value > 0: ffill for up to Xh (configurable), then 0
    #         (drug likely discontinued but nurse didn't chart the stop)
    #   - Before first observation: fillna(0) (drug not yet started)
    _med_trailing_hours = ohca_config["action_inference"].get("med_ffill_trailing_hours", 4)
    logger.info("Step 3a: Forward-filling CONTINUOUS meds (unlimited between obs, %dh trailing)...", _med_trailing_hours)
    _med_trailing_limit = pd.Timedelta(hours=_med_trailing_hours)
    _cont_cols = [c for c in med_cont_cols if c in ffilled.columns]
    for _col in _cont_cols:
        _before = ffilled[_col].isna().sum()

        # 1. Unlimited ffill (fills between AND after observations)
        _filled = ffilled.groupby("hospitalization_id")[_col].ffill()

        # 2. Find last observation time and value per patient for this med
        _notna = ffilled[_col].notna()
        _last_obs_time = (
            ffilled["event_dttm"].where(_notna)
            .groupby(ffilled["hospitalization_id"]).transform("max")
        )
        _last_obs_value = (
            ffilled[_col].where(_notna)
            .groupby(ffilled["hospitalization_id"]).transform("last")
        )

        # 3. Rows after last observation where last value > 0: apply time limit
        _after_last = ffilled["event_dttm"] > _last_obs_time
        _time_since_last = ffilled["event_dttm"] - _last_obs_time
        _beyond_limit = _after_last & (_last_obs_value > 0) & (_time_since_last > _med_trailing_limit)
        _filled = _filled.where(~_beyond_limit)

        # 4. Fill remaining NaN with 0 (before first obs, or after time limit)
        ffilled[_col] = _filled.fillna(0)
        _after = ffilled[_col].isna().sum()
        _n_trailing_clipped = _beyond_limit.sum()
        logger.info("  %s: %d NaN → %d NaN (trailing clipped: %d)", _col, _before, _after, _n_trailing_clipped)

    # --- 3a-post. Compute NEE from ffilled vasopressor doses ---
    # NEE is computed here (post-ffill, pre-bucketing) so that each row has a
    # self-consistent NEE reflecting all ffilled component doses at that timestamp.
    _nee_coefs = ohca_config["nee_coefficients"]
    ffilled["med_cont_nee"] = sum(
        ffilled[f"med_cont_{med}"].fillna(0) * coef
        for med, coef in _nee_coefs.items()
        if f"med_cont_{med}" in ffilled.columns
    )
    logger.info("Computed NEE from ffilled doses: %d non-zero rows", (ffilled["med_cont_nee"] > 0).sum())

    # --- 3b. Intermittent meds: NO ffill — bolus doses are discrete events ---
    # These are single-dose administrations (mg). The dose exists only at the
    # moment of administration; between boluses the dose is 0.
    logger.info("Step 3b: Intermittent meds (no ffill, fillna(0))...")
    _int_cols = [c for c in med_int_cols if c in ffilled.columns]
    for _col in _int_cols:
        _before = ffilled[_col].isna().sum()
        ffilled[_col] = ffilled[_col].fillna(0)
        logger.info("  %s: %d NaN → 0 NaN (fillna(0), no ffill — bolus)", _col, _before)

    # --- 4. Respiratory categoricals: unlimited ffill FIRST (needed for device groups) ---
    logger.info("Step 4: Forward-filling resp categoricals (unlimited)...")
    _available_resp_cat = [c for c in resp_categorical_cols if c in ffilled.columns]
    for _col in _available_resp_cat:
        _before = ffilled[_col].isna().sum()
        ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].ffill()
        _after = ffilled[_col].isna().sum()
        if _before != _after:
            logger.info("  %s: %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)

    # 4b. Bfill resp categoricals for leading NAs
    logger.info("Step 4b: Backward-filling leading resp categorical NAs...")
    for _col in _available_resp_cat:
        _before = ffilled[_col].isna().sum()
        if _before == 0:
            continue
        ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].bfill()
        _after = ffilled[_col].isna().sum()
        if _before != _after:
            logger.info("  %s (bfill): %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)

    # --- 5. Respiratory numerics: ffill within device_group, then device-conditional fill ---
    logger.info("Step 5: Forward-filling resp numerics (within device_group)...")
    _available_resp_num = [c for c in resp_numeric_cols if c in ffilled.columns]
    if _available_resp_num and "resp_device_category" in ffilled.columns:
        # Create device groups from ALREADY-FFILLED device_category
        _dc = ffilled["resp_device_category"].astype(str)
        ffilled["_device_group"] = (
            ffilled.groupby("hospitalization_id")[_dc.name]
            .transform(lambda x: (x != x.shift()).cumsum())
        )
        for _col in _available_resp_num:
            _before = ffilled[_col].isna().sum()
            ffilled[_col] = ffilled.groupby(
                ["hospitalization_id", "_device_group"]
            )[_col].ffill()
            _after = ffilled[_col].isna().sum()
            if _before != _after:
                logger.info("  %s: %d NaN → %d NaN (filled %d)", _col, _before, _after, _before - _after)
        ffilled = ffilled.drop(columns=["_device_group"])

        # 5b. Device-conditional imputation for remaining resp NaN
        logger.info("Step 5b: Device-conditional imputation for resp numerics...")
        _dc_lower = ffilled["resp_device_category"].str.lower().fillna("")

        # Room air: fio2=0.21, everything else=0
        _room_air = _dc_lower == "room air"
        if "resp_fio2_set" in ffilled.columns:
            _b = ffilled.loc[_room_air, "resp_fio2_set"].isna().sum()
            ffilled.loc[_room_air, "resp_fio2_set"] = ffilled.loc[_room_air, "resp_fio2_set"].fillna(0.21)
            logger.info("  fio2 (room air → 0.21): filled %d", _b - ffilled.loc[_room_air, "resp_fio2_set"].isna().sum())

        # Nasal cannula: fio2 from lpm (0.20 + 0.04*LPM), peep=0
        _nc = _dc_lower == "nasal cannula"
        if "resp_fio2_set" in ffilled.columns and "resp_lpm_set" in ffilled.columns:
            _nc_fio2_missing = _nc & ffilled["resp_fio2_set"].isna() & ffilled["resp_lpm_set"].notna()
            _imputed_fio2 = (0.20 + 0.04 * ffilled.loc[_nc_fio2_missing, "resp_lpm_set"]).clip(upper=1.0)
            ffilled.loc[_nc_fio2_missing, "resp_fio2_set"] = _imputed_fio2
            logger.info("  fio2 (NC from LPM): filled %d", _nc_fio2_missing.sum())

        # For non-vent devices: vent-specific params = 0
        _non_vent = _dc_lower.isin(["room air", "nasal cannula", "high flow nc", "trach collar", "face mask"])
        _vent_only_params = ["resp_peep_set", "resp_tidal_volume_set", "resp_resp_rate_set",
                             "resp_tidal_volume_obs"]
        for _col in _vent_only_params:
            if _col in ffilled.columns:
                _b = ffilled.loc[_non_vent, _col].isna().sum()
                ffilled.loc[_non_vent, _col] = ffilled.loc[_non_vent, _col].fillna(0)
                logger.info("  %s (non-vent → 0): filled %d", _col, _b)

        # For non-NC/non-HFNC devices: lpm = 0
        _no_lpm_devices = _dc_lower.isin(["room air", "vent"])
        if "resp_lpm_set" in ffilled.columns:
            _b = ffilled.loc[_no_lpm_devices, "resp_lpm_set"].isna().sum()
            ffilled.loc[_no_lpm_devices, "resp_lpm_set"] = ffilled.loc[_no_lpm_devices, "resp_lpm_set"].fillna(0)
            logger.info("  lpm (room air/vent → 0): filled %d", _b)

        # 5c. Any remaining resp numerics: fill with 0 (no device data = no support)
        logger.info("Step 5c: Filling remaining resp numeric NaN with 0...")
        for _col in _available_resp_num:
            _before = ffilled[_col].isna().sum()
            if _before == 0:
                continue
            ffilled[_col] = ffilled[_col].fillna(0)
            logger.info("  %s: %d NaN → 0 NaN (fillna(0))", _col, _before)

        # 5c-post. FiO2 floor: no patient breathes 0% oxygen
        if "resp_fio2_set" in ffilled.columns:
            _below_floor = (ffilled["resp_fio2_set"] < 0.21).sum()
            ffilled["resp_fio2_set"] = ffilled["resp_fio2_set"].clip(lower=0.21)
            logger.info("  resp_fio2_set: clipped %d rows to floor 0.21", _below_floor)

    else:
        logger.info("  Skipped — no resp numeric columns or resp_device_category missing")

    # 5d. Remaining resp categoricals: fillna("unknown")
    for _col in _available_resp_cat:
        _before = ffilled[_col].isna().sum()
        if _before == 0:
            continue
        ffilled[_col] = ffilled[_col].fillna("unknown")
        logger.info("  %s: %d NaN → 0 NaN (fillna unknown)", _col, _before)

    # 5e. Resp binary indicators: ffill + bfill + fillna(0)
    logger.info("Step 5e: Filling resp binary indicators...")
    for _col in resp_binary_cols:
        if _col not in ffilled.columns:
            continue
        _before = ffilled[_col].isna().sum()
        ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].ffill()
        ffilled[_col] = ffilled.groupby("hospitalization_id")[_col].bfill()
        ffilled[_col] = ffilled[_col].fillna(0).astype(int)
        logger.info("  %s: %d NaN → 0 NaN", _col, _before)

    # --- 6. CRRT: ffill between first and last recording, then fillna(0) ---
    # CRRT is a continuous therapy — if recorded at hours 5 and 10,
    # the patient was on CRRT the entire time in between.
    # Don't carry forward beyond last recording (session ended).
    logger.info("Step 6: CRRT on_crrt ffill between obs + fillna(0)...")
    if "on_crrt" in ffilled.columns:
        _before = ffilled["on_crrt"].isna().sum()

        # Capture last ORIGINAL observation time per patient BEFORE ffill
        _crrt_orig_notna = ffilled["on_crrt"].notna()
        _last_crrt_time = (
            ffilled["event_dttm"].where(_crrt_orig_notna)
            .groupby(ffilled["hospitalization_id"]).transform("max")
        )

        # ffill between observations (within patient)
        ffilled["on_crrt"] = ffilled.groupby("hospitalization_id")["on_crrt"].ffill()

        # Zero out rows AFTER last original CRRT observation → session ended
        _after_last_crrt = ffilled["event_dttm"] > _last_crrt_time
        ffilled.loc[_after_last_crrt, "on_crrt"] = 0

        # Remaining NaN (before first CRRT obs, or patients with no CRRT) → 0
        ffilled["on_crrt"] = ffilled["on_crrt"].fillna(0).astype(int)
        _after = ffilled["on_crrt"].isna().sum()
        _n_crrt_hours = (ffilled["on_crrt"] == 1).sum()
        logger.info("  on_crrt: %d NaN → %d NaN, %d rows with CRRT=1", _before, _after, _n_crrt_hours)
    # Drop CRRT columns that are ≥95% missing (crrt_mode_name, crrt_mode_category)
    _crrt_drop = [c for c in ffilled.columns if c.startswith("crrt_") and ffilled[c].isna().mean() > 0.95]
    if _crrt_drop:
        ffilled = ffilled.drop(columns=_crrt_drop)
        logger.info("  Dropped high-missing CRRT cols: %s", _crrt_drop)

    # --- 7. ADT: unlimited ffill + bfill; drop adt_out_dttm ---
    logger.info("Step 7: Forward-filling ADT location (unlimited)...")
    if "adt_out_dttm" in ffilled.columns:
        ffilled = ffilled.drop(columns=["adt_out_dttm"])
        logger.info("  Dropped adt_out_dttm (not a feature)")
    if "adt_location_category" in ffilled.columns:
        _before = ffilled["adt_location_category"].isna().sum()
        ffilled["adt_location_category"] = (
            ffilled.groupby("hospitalization_id")["adt_location_category"].ffill()
        )
        ffilled["adt_location_category"] = (
            ffilled.groupby("hospitalization_id")["adt_location_category"].bfill()
        )
        ffilled["adt_location_category"] = ffilled["adt_location_category"].fillna("unknown")
        _after = ffilled["adt_location_category"].isna().sum()
        logger.info("  adt_location_category: %d NaN → %d NaN", _before, _after)

    # Clean up infinities
    _numeric_cols = ffilled.select_dtypes(include=[np.number]).columns
    ffilled[_numeric_cols] = ffilled[_numeric_cols].replace([np.inf, -np.inf], np.nan)

    logger.info("Forward-fill complete. Final shape: %s", ffilled.shape)
    mo.md("### Forward-fill applied.")
    return (ffilled,)


# ── Cell 4: Missingness AFTER ffill ──────────────────────────────────
@app.cell
def _(
    compute_missingness,
    ffilled,
    final_dir,
    logger,
    miss_before,
    mo,
):
    _all_cols = [
        c for c in ffilled.columns
        if c not in ("hospitalization_id", "event_dttm")
    ]
    miss_after = compute_missingness(ffilled, _all_cols)
    miss_after.to_csv(final_dir / "missingness_after_ffill.csv", index=False)
    logger.info("Saved missingness_after_ffill.csv (%d variables)", len(miss_after))

    # Merge before/after for comparison
    _comparison = miss_before.merge(
        miss_after, on="variable", how="outer", suffixes=("_before", "_after"),
    )
    _comparison["row_delta"] = (
        _comparison["row_missing_pct_before"] - _comparison["row_missing_pct_after"]
    ).round(1)
    _comparison["patient_delta"] = (
        _comparison["patient_missing_pct_before"] - _comparison["patient_missing_pct_after"]
    ).round(1)
    _comparison.to_csv(final_dir / "missingness_comparison.csv", index=False)
    logger.info("Saved missingness_comparison.csv")

    mo.md(
        "### Missingness: Before vs After Forward-Fill\n\n"
        + _comparison.to_markdown(index=False)
    )
    return


# ── Cell 5: Time Bucketing ──────────────────────────────────────────────
@app.cell
def _(
    bucket_hours,
    ffilled,
    logger,
    mo,
    np,
    pd,
    time_window_hours,
):
    logger.info("Starting time bucketing (%.0f min buckets, %dh window)...", bucket_hours * 60, time_window_hours)

    bucketing_df = ffilled.copy()
    bucketing_df["event_dttm"] = pd.to_datetime(bucketing_df["event_dttm"], utc=True)

    # t0 = first event_dttm per patient (first recorded clinical data point)
    _t0 = (
        bucketing_df.groupby("hospitalization_id")["event_dttm"]
        .transform("min")
    )
    bucketing_df["t0"] = _t0

    # Hours since t0
    bucketing_df["hours_since_t0"] = (
        (bucketing_df["event_dttm"] - bucketing_df["t0"])
        .dt.total_seconds() / 3600
    )

    # Clip to time window (0 to time_window_hours)
    _before_clip = len(bucketing_df)
    bucketing_df = bucketing_df[
        bucketing_df["hours_since_t0"] < time_window_hours
    ].copy()
    _after_clip = len(bucketing_df)
    logger.info("Clipped to 0-%dh window (t0=first event): %d → %d rows (dropped %d)", time_window_hours, _before_clip, _after_clip, _before_clip - _after_clip)

    # Assign time bucket
    bucketing_df["time_bucket"] = (
        bucketing_df["hours_since_t0"] // bucket_hours
    ).astype(int)

    # --- Build aggregation dict ---
    _numeric_cols = bucketing_df.select_dtypes(include=[np.number]).columns.tolist()
    _categorical_cols = bucketing_df.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    # Exclude groupby/meta columns
    _exclude = {
        "hospitalization_id", "time_bucket", "hours_since_t0",
        "event_dttm", "t0",
    }

    _agg_dict = {}

    for _col in _numeric_cols:
        if _col in _exclude:
            continue
        if any(_col.startswith(p) for p in ("vital_", "lab_", "resp_")):
            _agg_dict[_col] = "mean"
        elif _col.startswith("med_cont_"):
            # Continuous infusion rates: use LAST value (end-of-bucket state)
            # This reflects the clinician's final decision for that hour,
            # which is what matters for action inference (NEE deltas).
            _agg_dict[_col] = "last"
        elif _col.startswith("med_int_"):
            # Intermittent bolus doses: use MAX (largest dose given that hour)
            _agg_dict[_col] = "max"
        elif _col.startswith("assess_"):
            # Assessments (GCS, RASS): use LAST value (end-of-bucket state)
            # Consistent with RL "current state" convention.
            _agg_dict[_col] = "last"
        elif _col == "on_crrt":
            _agg_dict[_col] = "max"
        else:
            _agg_dict[_col] = "mean"

    for _col in _categorical_cols:
        if _col in _exclude:
            continue
        _agg_dict[_col] = "last"

    # Keep event_dttm as min (start of bucket)
    _agg_dict["event_dttm"] = "min"

    logger.info("Aggregating %d columns into buckets...", len(_agg_dict))

    bucketed_df = (
        bucketing_df
        .groupby(["hospitalization_id", "time_bucket"])
        .agg(_agg_dict)
        .reset_index()
    )

    # --- Create dense hourly grid (one row per hour per patient, no gaps) ---
    logger.info("Creating dense hourly grid...")
    _max_bucket = bucketed_df.groupby("hospitalization_id")["time_bucket"].max()
    _dense_rows = []
    for _hid, _mb in _max_bucket.items():
        _dense_rows.append(pd.DataFrame({
            "hospitalization_id": _hid,
            "time_bucket": range(0, int(_mb) + 1),
        }))
    _dense_index = pd.concat(_dense_rows, ignore_index=True)
    _before_dense = len(bucketed_df)
    bucketed_df = _dense_index.merge(bucketed_df, on=["hospitalization_id", "time_bucket"], how="left")
    _after_dense = len(bucketed_df)
    logger.info("Dense grid: %d → %d rows (added %d empty buckets)", _before_dense, _after_dense, _after_dense - _before_dense)

    # Flag scaffold rows (added by dense grid, no original data)
    _feat_cols_for_scaffold = [
        c for c in bucketed_df.columns
        if c not in ("hospitalization_id", "time_bucket")
    ]
    bucketed_df["is_scaffold"] = bucketed_df[_feat_cols_for_scaffold].isna().all(axis=1)
    logger.info("Scaffold rows: %d / %d (%.1f%%)",
                bucketed_df["is_scaffold"].sum(), len(bucketed_df),
                100 * bucketed_df["is_scaffold"].mean())

    # --- Post-bucketing ffill: fill newly created empty rows + close gaps ---
    logger.info("Post-bucketing ffill pass to close remaining gaps...")
    _feature_cols = [
        c for c in bucketed_df.columns
        if c not in ("hospitalization_id", "time_bucket", "event_dttm")
    ]
    _num_feat = bucketed_df[_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    _cat_feat = bucketed_df[_feature_cols].select_dtypes(include=["object", "category", "str"]).columns.tolist()

    for _col in _num_feat + _cat_feat:
        _before = bucketed_df[_col].isna().sum()
        if _before == 0:
            continue
        # ffill within patient to carry forward last known value (no bfill to avoid future leak)
        bucketed_df[_col] = bucketed_df.groupby("hospitalization_id")[_col].ffill()
        _after = bucketed_df[_col].isna().sum()
        if _before != _after:
            logger.info("  %s: %d NaN → %d NaN (post-bucket ffill)", _col, _before, _after)

    # Final safety: any remaining NaN (patients with zero data for a variable)
    _remaining_nan = bucketed_df[_num_feat].isna().sum()
    _still_missing = _remaining_nan[_remaining_nan > 0]
    if len(_still_missing) > 0:
        logger.info("Final fillna for %d columns with remaining NaN...", len(_still_missing))
        for _col in _still_missing.index:
            # SKIP assessments — no safe default for OHCA population
            if _col.startswith("assess_"):
                logger.info("  %s: %d NaN left as NaN (no safe default for OHCA)", _col, _still_missing[_col])
                continue
            _med = bucketed_df[_col].median()
            bucketed_df[_col] = bucketed_df[_col].fillna(_med)
            logger.info("  %s: %d NaN filled with cohort median=%.2f", _col, _still_missing[_col], _med)
    for _col in _cat_feat:
        if bucketed_df[_col].isna().sum() > 0:
            bucketed_df[_col] = bucketed_df[_col].fillna("unknown")

    _n_rows = len(bucketed_df)
    _n_hosp = bucketed_df["hospitalization_id"].nunique()
    _n_buckets_per_hosp = bucketed_df.groupby("hospitalization_id")["time_bucket"].count()

    logger.info("Bucketing complete: %d rows, %d hospitalizations, median buckets/patient: %.0f",
                _n_rows, _n_hosp, _n_buckets_per_hosp.median())

    mo.md(f"""
    ### Time Bucketing Complete

    | Metric | Value |
    |--------|-------|
    | **Rows (bucketed)** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Bucket size** | {bucket_hours}h |
    | **Time window** | 0–{time_window_hours}h |
    | **Buckets/patient (median)** | {_n_buckets_per_hosp.median():.0f} |
    | **Buckets/patient (min–max)** | {_n_buckets_per_hosp.min()}–{_n_buckets_per_hosp.max()} |
    """)
    return (bucketed_df,)


# ── Cell 6: Action Inference (NEE-based) ─────────────────────────────────
@app.cell
def _(bucketed_df, logger, mo, np, ohca_config):
    _tol = ohca_config["action_inference"]["nee_change_tolerance"]  # 0.03
    logger.info("Step 6: Action inference from NEE deltas (tolerance=%.3f)...", _tol)

    actioned_df = bucketed_df.copy()

    # NEE at current and previous bucket (per patient)
    _nee = actioned_df.groupby("hospitalization_id")["med_cont_nee"]
    _nee_prev = _nee.shift(1)
    _delta_nee = actioned_df["med_cont_nee"] - _nee_prev

    # Action encoding:
    #   0 = stay     (|delta| < tolerance)
    #   1 = increase (delta >= tolerance, OR NEE went from 0 to >0)
    #   2 = decrease (delta <= -tolerance, NEE still > 0)
    #   3 = stop     (NEE drops to 0 from > 0)
    _action = np.full(len(actioned_df), 0, dtype=np.int8)  # default: stay

    _nee_cur = actioned_df["med_cont_nee"]
    _nee_lag = _nee_prev

    # Increase: NEE went up by >= tolerance (includes start: 0 → >0)
    _action = np.where(_delta_nee >= _tol, 1, _action)

    # Decrease: NEE went down by >= tolerance, but NEE still > 0
    _action = np.where(
        (_delta_nee <= -_tol) & (_nee_cur > 0),
        2, _action,
    )

    # Stop: NEE was > 0, now == 0
    _action = np.where(
        (_nee_lag > 0) & (_nee_cur == 0),
        3, _action,
    )

    # First bucket per patient: no previous state → stay (0)
    _first_bucket = _nee_lag.isna()
    _action = np.where(_first_bucket, 0, _action)

    actioned_df["action"] = _action

    # Log distribution
    _action_counts = actioned_df["action"].value_counts().sort_index()
    _action_labels = {0: "stay", 1: "increase", 2: "decrease", 3: "stop"}
    _action_table = "\n".join(
        f"    {k} ({_action_labels[k]}): {v:,} ({v / len(actioned_df) * 100:.1f}%)"
        for k, v in _action_counts.items()
    )
    logger.info("Action distribution:\n%s", _action_table)

    # Patients who ever had vasopressors
    _ever_vaso = actioned_df.groupby("hospitalization_id")["med_cont_nee"].max()
    _n_vaso_patients = (_ever_vaso > 0).sum()
    _n_total = len(_ever_vaso)

    mo.md(f"""
    ### Action Inference (NEE-based)

    | Setting | Value |
    |---------|-------|
    | **NEE change tolerance** | {_tol} mcg/kg/min |
    | **Patients with any vasopressors** | {_n_vaso_patients:,} / {_n_total:,} ({_n_vaso_patients / _n_total * 100:.1f}%) |

    | Action | Label | Count | % |
    |--------|-------|-------|---|
    | 0 | stay | {_action_counts.get(0, 0):,} | {_action_counts.get(0, 0) / len(actioned_df) * 100:.1f}% |
    | 1 | increase | {_action_counts.get(1, 0):,} | {_action_counts.get(1, 0) / len(actioned_df) * 100:.1f}% |
    | 2 | decrease | {_action_counts.get(2, 0):,} | {_action_counts.get(2, 0) / len(actioned_df) * 100:.1f}% |
    | 3 | stop | {_action_counts.get(3, 0):,} | {_action_counts.get(3, 0) / len(actioned_df) * 100:.1f}% |
    """)
    return (actioned_df,)


# ── Cell 7: Derived Booleans & Reward Variables ──────────────────────────
@app.cell
def _(actioned_df, bucket_hours, death_time_df, logger, mo, np, ohca_config, patient_static, pd):
    logger.info("Step 7: Adding derived booleans and reward variables...")
    enriched_df = actioned_df.copy()

    # --- Merge discharge_category from patient_static ---
    _discharge = patient_static[["hospitalization_id", "discharge_category"]].drop_duplicates(
        subset=["hospitalization_id"]
    )
    enriched_df = enriched_df.merge(_discharge, on="hospitalization_id", how="left")
    enriched_df["discharge_category"] = enriched_df["discharge_category"].str.lower().str.strip()

    # --- CPC mapping (use str.contains for partial matches like "against medical advice (ama)") ---
    _cpc_map = ohca_config["cpc_mapping"]
    enriched_df["cpc"] = "exclude"
    for _cpc_label, _categories in _cpc_map.items():
        if _cpc_label == "exclude":
            continue
        for _cat in _categories:
            _mask = enriched_df["discharge_category"].str.contains(_cat, na=False)
            enriched_df.loc[_mask, "cpc"] = _cpc_label
    _cpc_counts = enriched_df.groupby("hospitalization_id")["cpc"].first().value_counts()
    logger.info("CPC distribution (patient-level):\n%s", _cpc_counts.to_string())

    # --- Med booleans: was each med administered (>0) this hour? ---
    _med_cont_dose_cols = [
        c for c in enriched_df.columns
        if c.startswith("med_cont_") and c != "med_cont_nee"
    ]
    for _col in _med_cont_dose_cols:
        _bool_name = f"on_{_col}"  # e.g. on_med_cont_norepinephrine
        enriched_df[_bool_name] = (enriched_df[_col] > 0).astype(np.int8)

    _med_int_cols = [c for c in enriched_df.columns if c.startswith("med_int_")]
    for _col in _med_int_cols:
        _bool_name = f"on_{_col}"  # e.g. on_med_int_lorazepam
        enriched_df[_bool_name] = (enriched_df[_col] > 0).astype(np.int8)

    # --- Location booleans ---
    _loc = enriched_df["adt_location_category"].str.lower()
    enriched_df["in_icu"] = (_loc == "icu").astype(np.int8)
    enriched_df["in_ed"] = (_loc == "ed").astype(np.int8)

    # --- Death booleans ---
    # ever_died: patient-level (1 for all rows if patient's discharge = expired/hospice)
    _death_categories = [c.lower() for c in _cpc_map.get("CPC5", [])]
    _died_hosp = enriched_df.groupby("hospitalization_id")["discharge_category"].first()
    _died_set = set(
        _died_hosp[_died_hosp.apply(
            lambda x: any(d in str(x) for d in _death_categories)
        )].index
    )
    enriched_df["ever_died"] = enriched_df["hospitalization_id"].isin(_died_set).astype(np.int8)

    # is_dead: 1 at the time bucket where death occurred (and all subsequent buckets)
    # Uses death_dttm from clif_patient (preferred), fallback to discharge_dttm
    enriched_df = enriched_df.merge(death_time_df, on="hospitalization_id", how="left")
    enriched_df["death_dttm"] = pd.to_datetime(enriched_df["death_dttm"], utc=True)

    # Compute death bucket relative to t0 (same reference as time_bucket)
    _t0 = enriched_df.groupby("hospitalization_id")["event_dttm"].transform("min")
    _hours_to_death = (enriched_df["death_dttm"] - _t0).dt.total_seconds() / 3600
    _death_bucket = (_hours_to_death // bucket_hours).astype("Int64")  # nullable int

    # is_dead = 1 at death bucket and all subsequent buckets (absorbing terminal state)
    enriched_df["is_dead"] = (
        (enriched_df["ever_died"] == 1)
        & (_death_bucket.notna())
        & (enriched_df["time_bucket"] >= _death_bucket)
    ).astype(np.int8)

    # Log death timing stats
    _deaths_in_window = enriched_df.loc[enriched_df["is_dead"] == 1, "hospitalization_id"].nunique()
    _deaths_total = enriched_df.loc[enriched_df["ever_died"] == 1, "hospitalization_id"].nunique()
    logger.info("  Deaths: %d/%d occurred within %dh window", _deaths_in_window, _deaths_total, int(bucket_hours * enriched_df["time_bucket"].max()))

    # Clean up — don't keep death_dttm in final output
    enriched_df = enriched_df.drop(columns=["death_dttm"])

    # --- Vasopressor booleans ---
    enriched_df["on_vaso"] = (enriched_df["med_cont_nee"] > 0).astype(np.int8)
    enriched_df["ever_vaso"] = enriched_df.groupby("hospitalization_id")["on_vaso"].transform("max").astype(np.int8)

    # --- IMV (Invasive Mechanical Ventilation) booleans ---
    _device = enriched_df["resp_device_category"].str.lower()
    enriched_df["on_imv"] = (_device == "imv").astype(np.int8)
    enriched_df["ever_imv"] = enriched_df.groupby("hospitalization_id")["on_imv"].transform("max").astype(np.int8)

    # --- Summary ---
    _n = len(enriched_df)
    _n_hosp = enriched_df["hospitalization_id"].nunique()
    _new_bool_cols = (
        [c for c in enriched_df.columns if c.startswith("on_med_")]
        + ["in_icu", "in_ed", "ever_died", "is_dead", "on_vaso", "ever_vaso", "on_imv", "ever_imv"]
    )

    mo.md(f"""
    ### Derived Booleans & Reward Variables

    | Variable | Rows=1 | % |
    |----------|--------|---|
    | **in_icu** | {enriched_df['in_icu'].sum():,} | {enriched_df['in_icu'].mean() * 100:.1f}% |
    | **in_ed** | {enriched_df['in_ed'].sum():,} | {enriched_df['in_ed'].mean() * 100:.1f}% |
    | **on_vaso** | {enriched_df['on_vaso'].sum():,} | {enriched_df['on_vaso'].mean() * 100:.1f}% |
    | **ever_vaso** | {enriched_df['ever_vaso'].sum():,} | {enriched_df['ever_vaso'].mean() * 100:.1f}% |
    | **on_imv** | {enriched_df['on_imv'].sum():,} | {enriched_df['on_imv'].mean() * 100:.1f}% |
    | **ever_imv** | {enriched_df['ever_imv'].sum():,} | {enriched_df['ever_imv'].mean() * 100:.1f}% |
    | **ever_died** | {enriched_df['ever_died'].sum():,} | {enriched_df['ever_died'].mean() * 100:.1f}% |
    | **is_dead** | {enriched_df['is_dead'].sum():,} | {enriched_df['is_dead'].mean() * 100:.1f}% |

    **CPC distribution** (patient-level):

    | CPC | Patients | % |
    |-----|----------|---|
    | CPC1_2 | {_cpc_counts.get('CPC1_2', 0):,} | {_cpc_counts.get('CPC1_2', 0) / _n_hosp * 100:.1f}% |
    | CPC3 | {_cpc_counts.get('CPC3', 0):,} | {_cpc_counts.get('CPC3', 0) / _n_hosp * 100:.1f}% |
    | CPC4 | {_cpc_counts.get('CPC4', 0):,} | {_cpc_counts.get('CPC4', 0) / _n_hosp * 100:.1f}% |
    | CPC5 | {_cpc_counts.get('CPC5', 0):,} | {_cpc_counts.get('CPC5', 0) / _n_hosp * 100:.1f}% |
    | exclude | {_cpc_counts.get('exclude', 0):,} | {_cpc_counts.get('exclude', 0) / _n_hosp * 100:.1f}% |

    **Med booleans added**: {len([c for c in enriched_df.columns if c.startswith('on_med_')])} columns
    **Total new columns**: {len(_new_bool_cols) + 2} (booleans + cpc + discharge_category)
    """)
    return (enriched_df,)


# ── Cell 8: Save ────────────────────────────────────────────────────────
@app.cell
def _(enriched_df, intermediate_dir, logger, mo):
    _output_path = intermediate_dir / "wide_df_bucketed.parquet"
    enriched_df.to_parquet(_output_path, index=False)
    _file_size_mb = _output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved bucketed wide_df to %s (%.1f MB)", _output_path, _file_size_mb)

    # Final column summary
    _vital_cols = [c for c in enriched_df.columns if c.startswith("vital_")]
    _lab_cols = [c for c in enriched_df.columns if c.startswith("lab_")]
    _med_cont_cols = [c for c in enriched_df.columns if c.startswith("med_cont_")]
    _med_int_cols = [c for c in enriched_df.columns if c.startswith("med_int_")]
    _resp_cols = [c for c in enriched_df.columns if c.startswith("resp_")]
    _crrt_cols = [c for c in enriched_df.columns if c.startswith("crrt_") or c == "on_crrt"]
    _assess_cols = [c for c in enriched_df.columns if c.startswith("assess_")]
    _adt_cols = [c for c in enriched_df.columns if c.startswith("adt_")]
    _bool_cols = [c for c in enriched_df.columns if c.startswith("on_med_")]
    _derived_cols = ["in_icu", "in_ed", "on_vaso", "ever_vaso", "on_imv", "ever_imv",
                     "ever_died", "is_dead", "cpc", "action"]

    def _miss(cols):
        _available = [c for c in cols if c in enriched_df.columns]
        if not _available:
            return "N/A"
        return f"{enriched_df[_available].isna().mean().mean() * 100:.1f}%"

    mo.md(f"""
    ## Final Output: Bucketed Wide DataFrame

    | Metric | Value |
    |--------|-------|
    | **Rows** | {len(enriched_df):,} |
    | **Hospitalizations** | {enriched_df['hospitalization_id'].nunique():,} |
    | **Columns** | {len(enriched_df.columns)} |
    | **Saved to** | `{_output_path}` |
    | **File size** | {_file_size_mb:.1f} MB |

    ### Column Groups (post-bucketing missingness)

    | Group | Count | Missing % |
    |-------|-------|-----------|
    | Vitals | {len(_vital_cols)} | {_miss(_vital_cols)} |
    | Labs | {len(_lab_cols)} | {_miss(_lab_cols)} |
    | Meds continuous | {len(_med_cont_cols)} | {_miss(_med_cont_cols)} |
    | Meds intermittent | {len(_med_int_cols)} | {_miss(_med_int_cols)} |
    | Respiratory | {len(_resp_cols)} | {_miss(_resp_cols)} |
    | CRRT | {len(_crrt_cols)} | {_miss(_crrt_cols)} |
    | Assessments | {len(_assess_cols)} | {_miss(_assess_cols)} |
    | ADT | {len(_adt_cols)} | {_miss(_adt_cols)} |
    | Med booleans | {len(_bool_cols)} | {_miss(_bool_cols)} |
    | Derived | {len(_derived_cols)} | — |

    ### All Columns
    {', '.join(sorted(enriched_df.columns.tolist()))}
    """)
    return


# ── Cell 9: Hospitalization Summary ───────────────────────────────────────
@app.cell
def _(bucket_hours, enriched_df, intermediate_dir, logger, mo, np, patient_static, pd):
    logger.info("Step 9: Building hospitalization-level summary...")

    # --- Base: one row per hospitalization ---
    _hosp_ids = enriched_df[["hospitalization_id"]].drop_duplicates()

    # --- SOFA scores by 24h window ---
    _sofa = pd.read_parquet(intermediate_dir / "sofa_scores.parquet")
    _sofa["hospitalization_id"] = _sofa["hospitalization_id"].astype(str)

    # Pivot SOFA total into columns by window: sofa_0_24, sofa_24_48, etc.
    _sofa_pivot = _sofa.pivot_table(
        index="hospitalization_id",
        columns="sofa_window",
        values="sofa_total",
        aggfunc="first",
    )
    _sofa_pivot.columns = [
        f"sofa_{int(row['sofa_window_start_h'])}_{int(row['sofa_window_end_h'])}"
        for _, row in _sofa.drop_duplicates(subset=["sofa_window"])[
            ["sofa_window", "sofa_window_start_h", "sofa_window_end_h"]
        ].sort_values("sofa_window").iterrows()
    ]
    _sofa_pivot = _sofa_pivot.reset_index()

    # --- Patient-level flags from bucketed df ---
    _agg_dict = {
        "ever_died": ("ever_died", "first"),
        "ever_vaso": ("ever_vaso", "first"),
        "ever_imv": ("ever_imv", "first"),
        "cpc": ("cpc", "first"),
        "discharge_category": ("discharge_category", "first"),
        "total_hours": ("time_bucket", "count"),
        "total_icu_hours": ("in_icu", "sum"),
        "total_ed_hours": ("in_ed", "sum"),
        "total_vaso_hours": ("on_vaso", "sum"),
        "total_imv_hours": ("on_imv", "sum"),
        "total_crrt_hours": ("on_crrt", "sum"),
        "max_nee": ("med_cont_nee", "max"),
    }
    if "lab_lactate" in enriched_df.columns:
        _agg_dict["max_lactate"] = ("lab_lactate", "max")
    if "assess_gcs_total" in enriched_df.columns:
        _agg_dict["min_gcs"] = ("assess_gcs_total", "min")
        _agg_dict["median_gcs"] = ("assess_gcs_total", "median")
    if "assess_rass" in enriched_df.columns:
        _agg_dict["median_rass"] = ("assess_rass", "median")
    _patient_flags = enriched_df.groupby("hospitalization_id").agg(**_agg_dict).reset_index()

    # Scale hours by bucket size
    for _col in ["total_hours", "total_icu_hours", "total_ed_hours",
                 "total_vaso_hours", "total_imv_hours", "total_crrt_hours"]:
        _patient_flags[_col] = (_patient_flags[_col] * bucket_hours).astype(int)

    # --- Demographics from patient_static (built in step 00) ---
    _demo = patient_static[["hospitalization_id", "age_at_admission",
                             "sex_category", "race_category", "ethnicity_category"]].copy()

    # --- Assemble summary ---
    summary_df = (
        _hosp_ids
        .merge(_demo, on="hospitalization_id", how="left")
        .merge(_patient_flags, on="hospitalization_id", how="left")
        .merge(_sofa_pivot, on="hospitalization_id", how="left")
    )

    # Save
    _output_path = intermediate_dir / "hospitalization_summary.parquet"
    summary_df.to_parquet(_output_path, index=False)
    _file_size_mb = _output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved hospitalization_summary.parquet: %d rows, %d cols (%.1f MB)",
                len(summary_df), len(summary_df.columns), _file_size_mb)

    # --- Display ---
    _sofa_window_cols = [c for c in summary_df.columns if c.startswith("sofa_") and "_0_" in c or c.startswith("sofa_")]
    _sofa_total_cols = [c for c in summary_df.columns if c.startswith("sofa_") and c.count("_") == 2 and "cv" not in c and "coag" not in c and "liver" not in c and "resp" not in c and "cns" not in c and "renal" not in c]

    mo.md(f"""
    ### Hospitalization Summary

    | Metric | Value |
    |--------|-------|
    | **Patients** | {len(summary_df):,} |
    | **Columns** | {len(summary_df.columns)} |
    | **Saved to** | `{_output_path}` |

    **SOFA by 24h window** (median [IQR]):

    | Window | Median | IQR |
    |--------|--------|-----|
    {chr(10).join(
        f"    | **{c}** | {summary_df[c].median():.0f} | {summary_df[c].quantile(0.25):.0f}–{summary_df[c].quantile(0.75):.0f} |"
        for c in sorted(_sofa_total_cols) if c in summary_df.columns and summary_df[c].notna().sum() > 0
    )}

    **Outcomes**:

    | | N | % |
    |-|---|---|
    | **Ever died** | {summary_df['ever_died'].sum():,} | {summary_df['ever_died'].mean() * 100:.1f}% |
    | **Ever vasopressors** | {summary_df['ever_vaso'].sum():,} | {summary_df['ever_vaso'].mean() * 100:.1f}% |
    | **Ever IMV** | {summary_df['ever_imv'].sum():,} | {summary_df['ever_imv'].mean() * 100:.1f}% |

    **All columns**: {', '.join(sorted(summary_df.columns.tolist()))}
    """)
    return


if __name__ == "__main__":
    app.run()
