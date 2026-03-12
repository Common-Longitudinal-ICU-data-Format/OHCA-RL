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
#     "scikit-learn",
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
    import pickle
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    with open(project_root / "config" / "ohca_rl_config.yaml", "r") as _f:
        ohca_config = yaml.safe_load(_f)

    intermediate_dir = project_root / "output" / "intermediate"
    TIME_WINDOW_HRS = ohca_config["action_inference"]["time_window_hours"]
    reward_config = ohca_config["reward"]
    cpc_mapping = ohca_config["cpc_mapping"]

    # Load all inputs
    wide_df = pd.read_parquet(intermediate_dir / "wide_df.parquet")
    wide_df["hospitalization_id"] = wide_df["hospitalization_id"].astype(str)

    actions_df = pd.read_parquet(intermediate_dir / "hourly_nee_actions.parquet")
    actions_df["hospitalization_id"] = actions_df["hospitalization_id"].astype(str)

    cohort_df = pd.read_parquet(intermediate_dir / "cohort_ohca_icu.parquet")
    cohort_df["hospitalization_id"] = cohort_df["hospitalization_id"].astype(str)

    # Try loading SOFA (may not exist)
    _sofa_path = intermediate_dir / "sofa_scores.parquet"
    if _sofa_path.exists():
        sofa_df = pd.read_parquet(_sofa_path)
        sofa_df["hospitalization_id"] = sofa_df["hospitalization_id"].astype(str)
        _sofa_msg = f"Loaded {len(sofa_df):,} SOFA scores"
    else:
        sofa_df = pd.DataFrame(columns=["hospitalization_id"])
        _sofa_msg = "SOFA scores not available — will be NaN"

    mo.md(f"""
    ## Step 04: Prepare RL Training Data

    | Input | Rows | Hospitalizations |
    |-------|------|------------------|
    | **Wide DF** | {len(wide_df):,} | {wide_df['hospitalization_id'].nunique():,} |
    | **Actions** | {len(actions_df):,} | {actions_df['hospitalization_id'].nunique():,} |
    | **Cohort** | {len(cohort_df):,} | — |
    | **SOFA** | {_sofa_msg} | — |
    """)
    return (
        StandardScaler,
        TIME_WINDOW_HRS,
        actions_df,
        cohort_df,
        cpc_mapping,
        duckdb,
        intermediate_dir,
        mo,
        np,
        ohca_config,
        pd,
        pickle,
        reward_config,
        sofa_df,
        wide_df,
    )


# ── Cell 2: Bin wide_df to hourly + LOCF ─────────────────────────────────
@app.cell
def _(TIME_WINDOW_HRS, actions_df, duckdb, mo, np, pd, wide_df):
    # Floor wide_df timestamps to hour boundaries (UTC to avoid DST)
    _wide = wide_df.copy()
    _wide["hour"] = _wide["event_dttm"].dt.tz_convert("UTC").dt.floor("h")
    # Convert to America/Chicago to match actions_df timezone
    _wide["hour"] = _wide["hour"].dt.tz_convert("America/Chicago")

    # Get all feature columns (exclude IDs, timestamps, static)
    _exclude = {
        "hospitalization_id", "event_dttm", "hour",
        "patient_id", "discharge_category", "survival_status", "admission_dttm",
        "adt_out_dttm",
    }
    _feature_cols = [c for c in _wide.columns if c not in _exclude]

    # Aggregate to hourly: last value per (hospitalization_id, hour)
    # This matches the action table's hourly grain
    hourly_features = (
        _wide.groupby(["hospitalization_id", "hour"])[_feature_cols]
        .last()
        .reset_index()
    )

    # Forward-fill within each hospitalization (LOCF)
    hourly_features = hourly_features.sort_values(["hospitalization_id", "hour"])
    _ff_cols = [c for c in _feature_cols if c != "adt_location_category"]
    hourly_features[_ff_cols] = (
        hourly_features
        .groupby("hospitalization_id")[_ff_cols]
        .ffill()
    )
    # ADT location category also forward-filled
    if "adt_location_category" in hourly_features.columns:
        hourly_features["adt_location_category"] = (
            hourly_features
            .groupby("hospitalization_id")["adt_location_category"]
            .ffill()
        )

    _n_rows = len(hourly_features)
    _n_hosp = hourly_features["hospitalization_id"].nunique()

    mo.md(f"""
    ## Hourly State Features

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Feature columns** | {len(_feature_cols)} |
    """)
    return _feature_cols, hourly_features


# ── Cell 3: Merge state + actions + cohort + SOFA ─────────────────────────
@app.cell
def _(actions_df, cohort_df, hourly_features, mo, pd, sofa_df):
    # Normalize action hours to same tz as hourly_features
    actions_df["hour"] = pd.to_datetime(actions_df["hour"], utc=True).dt.tz_convert("America/Chicago")

    # Merge state features with actions on (hospitalization_id, hour)
    merged = actions_df.merge(
        hourly_features,
        on=["hospitalization_id", "hour"],
        how="left",  # keep all action hours, even if no state data
    )

    # Add static cohort columns
    _cohort_cols = ["hospitalization_id", "patient_id", "admission_dttm",
                    "discharge_category", "survival_status"]
    _avail = [c for c in _cohort_cols if c in cohort_df.columns]
    _cohort_static = cohort_df[_avail].drop_duplicates(subset=["hospitalization_id"])
    _cohort_static["hospitalization_id"] = _cohort_static["hospitalization_id"].astype(str)
    merged = merged.merge(_cohort_static, on="hospitalization_id", how="left")

    # Add demographics
    if "age_at_admission" in cohort_df.columns:
        _age = cohort_df[["hospitalization_id", "age_at_admission"]].drop_duplicates(
            subset=["hospitalization_id"]
        )
        _age["hospitalization_id"] = _age["hospitalization_id"].astype(str)
        merged = merged.merge(_age, on="hospitalization_id", how="left")

    # Add SOFA scores (time-varying, one per 24h window)
    if len(sofa_df) > 0 and "sofa_window" in sofa_df.columns:
        # Compute hours since admission for each row
        merged["admission_dttm"] = pd.to_datetime(merged["admission_dttm"])
        merged["_hours_since_admit"] = (
            merged["hour"] - merged["admission_dttm"]
        ).dt.total_seconds() / 3600.0
        # Map to 24h window index (0=0-24h, 1=24-48h, etc.)
        merged["sofa_window"] = (merged["_hours_since_admit"] // 24).clip(lower=0).astype(int)
        # Cap at max window in sofa_df
        _max_window = sofa_df["sofa_window"].max()
        merged["sofa_window"] = merged["sofa_window"].clip(upper=_max_window)
        # Merge on (hospitalization_id, sofa_window)
        _sofa_merge_cols = [c for c in sofa_df.columns
                           if c not in ("sofa_window_start_h", "sofa_window_end_h")]
        merged = merged.merge(
            sofa_df[_sofa_merge_cols],
            on=["hospitalization_id", "sofa_window"],
            how="left",
        )
        merged.drop(columns=["_hours_since_admit"], inplace=True)
    elif len(sofa_df) > 0:
        # Fallback: static SOFA (no window column)
        merged = merged.merge(sofa_df, on="hospitalization_id", how="left")

    _n_rows = len(merged)
    _n_hosp = merged["hospitalization_id"].nunique()

    mo.md(f"""
    ## Merged Dataset

    | Metric | Value |
    |--------|-------|
    | **Rows** | {_n_rows:,} |
    | **Hospitalizations** | {_n_hosp:,} |
    | **Columns** | {len(merged.columns)} |
    """)
    return (merged,)


# ── Cell 4: Compute rewards ──────────────────────────────────────────────
@app.cell
def _(cpc_mapping, merged, mo, np, pd, reward_config):
    # Build CPC lookup: discharge_category → CPC level
    _cpc_lookup = {}
    for _cpc_level, _dispositions in cpc_mapping.items():
        for _disp in _dispositions:
            _cpc_lookup[_disp] = _cpc_level

    # Terminal reward values
    _terminal_vals = reward_config["terminal"]

    # Map discharge_category → CPC → terminal reward
    reward_df = merged.copy()

    if "discharge_category" in reward_df.columns:
        reward_df["_dc_lower"] = reward_df["discharge_category"].str.lower().str.strip()
        reward_df["cpc_level"] = reward_df["_dc_lower"].map(_cpc_lookup)
        reward_df["terminal_reward"] = reward_df["cpc_level"].map(_terminal_vals)
        reward_df.drop(columns=["_dc_lower"], inplace=True)
    else:
        reward_df["cpc_level"] = np.nan
        reward_df["terminal_reward"] = np.nan

    # Mark terminal rows (last hour per hospitalization)
    reward_df["is_last_hour"] = False
    _last_idx = reward_df.groupby("hospitalization_id")["hour"].idxmax()
    reward_df.loc[_last_idx, "is_last_hour"] = True

    # Intermediate rewards
    _map_reward = reward_config["intermediate"]["map_above_65"]
    _lac_reward = reward_config["intermediate"]["lactate_clearing"]

    # MAP >= 65 reward
    if "vital_map" in reward_df.columns:
        reward_df["reward_map"] = np.where(
            reward_df["vital_map"] >= 65, _map_reward, 0
        )
    else:
        reward_df["reward_map"] = 0

    # Lactate clearing: lactate(t) < lactate(t-1)
    if "lab_lactate" in reward_df.columns:
        _prev_lac = reward_df.groupby("hospitalization_id")["lab_lactate"].shift(1)
        reward_df["reward_lactate"] = np.where(
            (reward_df["lab_lactate"].notna())
            & (_prev_lac.notna())
            & (reward_df["lab_lactate"] < _prev_lac),
            _lac_reward,
            0,
        )
    else:
        reward_df["reward_lactate"] = 0

    # Combined reward
    reward_df["reward_intermediate"] = reward_df["reward_map"] + reward_df["reward_lactate"]

    # Final reward: intermediate for all rows, terminal for last row
    reward_df["reward"] = reward_df["reward_intermediate"]
    reward_df.loc[reward_df["is_last_hour"], "reward"] = reward_df.loc[
        reward_df["is_last_hour"], "terminal_reward"
    ].fillna(0)

    # CPC distribution
    _cpc_dist = reward_df.drop_duplicates(subset=["hospitalization_id"])["cpc_level"].value_counts(dropna=False)

    mo.md(f"""
    ## Reward Computation

    ### CPC Distribution
    {_cpc_dist.to_markdown()}

    ### Reward Stats
    | Metric | Value |
    |--------|-------|
    | **Intermediate reward mean** | {reward_df['reward_intermediate'].mean():.3f} |
    | **Terminal reward mean** | {reward_df.loc[reward_df['is_last_hour'], 'terminal_reward'].mean():.2f} |
    | **Unmapped discharge categories** | {reward_df.drop_duplicates('hospitalization_id')['cpc_level'].isna().sum():,} |
    """)
    return (reward_df,)


# ── Cell 5: Define state features + standardize ──────────────────────────
@app.cell
def _(StandardScaler, mo, np, ohca_config, pd, reward_df):
    # Define feature groups
    _vital_features = [f"vital_{v}" for v in ohca_config["vitals_of_interest"]]
    _lab_features = [f"lab_{l}" for l in ohca_config["labs_of_interest"]]
    _med_cont_features = [f"med_cont_{m}" for m in ohca_config["meds_continuous_of_interest"]]
    _sofa_features = [c for c in reward_df.columns if c.startswith("sofa_")]

    # Binary features (not scaled)
    _binary_candidates = _med_cont_features  # med doses used as binary indicators
    binary_features = [c for c in _binary_candidates if c in reward_df.columns]

    # Continuous features (scaled)
    _continuous_candidates = _vital_features + _lab_features + _sofa_features + [
        "med_cont_nee", "nee_start", "nee_end",
    ]
    # Add age if available
    if "age_at_admission" in reward_df.columns:
        _continuous_candidates.append("age_at_admission")

    continuous_features = [c for c in _continuous_candidates if c in reward_df.columns]

    # All state features
    state_features = continuous_features + binary_features
    _available = [c for c in state_features if c in reward_df.columns]
    _missing = [c for c in state_features if c not in reward_df.columns]

    # Convert binary features to 0/1 (NaN → 0 means not on that med)
    training_df = reward_df.copy()
    for _bf in binary_features:
        if _bf in training_df.columns:
            training_df[_bf] = np.where(
                training_df[_bf].notna() & (training_df[_bf] > 0), 1, 0
            )

    mo.md(f"""
    ## State Feature Definition

    | Group | Count | Features |
    |-------|-------|----------|
    | **Vitals** | {len([c for c in _vital_features if c in reward_df.columns])} | {', '.join([c for c in _vital_features if c in reward_df.columns])} |
    | **Labs** | {len([c for c in _lab_features if c in reward_df.columns])} | {', '.join([c for c in _lab_features if c in reward_df.columns])} |
    | **Meds (binary)** | {len(binary_features)} | {', '.join(binary_features[:5])}... |
    | **SOFA** | {len(_sofa_features)} | {', '.join(_sofa_features)} |
    | **Other continuous** | {len([c for c in continuous_features if c not in _vital_features + _lab_features + _sofa_features])} | nee_start, nee_end, age |
    | **Total state features** | {len(_available)} | |
    | **Missing features** | {len(_missing)} | {', '.join(_missing) if _missing else 'none'} |
    """)
    return binary_features, continuous_features, state_features, training_df


# ── Cell 6: Train/test split + standardize ────────────────────────────────
@app.cell
def _(
    StandardScaler,
    binary_features,
    continuous_features,
    intermediate_dir,
    mo,
    np,
    pd,
    pickle,
    state_features,
    training_df,
):
    # Patient-level 80/20 split
    _all_patients = training_df["hospitalization_id"].unique()
    np.random.seed(42)
    _shuffled = np.random.permutation(_all_patients)
    _split_idx = int(len(_shuffled) * 0.8)
    _train_ids = set(_shuffled[:_split_idx])
    _test_ids = set(_shuffled[_split_idx:])

    train_df = training_df[training_df["hospitalization_id"].isin(_train_ids)].copy()
    test_df = training_df[training_df["hospitalization_id"].isin(_test_ids)].copy()

    # Standardize continuous features (fit on train only)
    _cont_available = [c for c in continuous_features if c in train_df.columns]
    scaler = StandardScaler()
    train_df[_cont_available] = scaler.fit_transform(train_df[_cont_available].fillna(0))
    test_df[_cont_available] = scaler.transform(test_df[_cont_available].fillna(0))

    # Save
    train_df.to_parquet(intermediate_dir / "train.parquet", index=False)
    test_df.to_parquet(intermediate_dir / "test.parquet", index=False)

    # Save scaler
    _scaler_path = intermediate_dir / "state_standardization.pkl"
    with open(_scaler_path, "wb") as _f:
        pickle.dump({
            "scaler": scaler,
            "continuous_features": _cont_available,
            "binary_features": [c for c in binary_features if c in train_df.columns],
            "state_features": [c for c in state_features if c in train_df.columns],
        }, _f)

    # Save training config
    import json as _json
    _config = {
        "num_state_features": len([c for c in state_features if c in train_df.columns]),
        "num_actions": 4,
        "action_labels": {0: "stay", 1: "increase", 2: "decrease", 3: "stop"},
        "train_patients": len(_train_ids),
        "test_patients": len(_test_ids),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "continuous_features": _cont_available,
        "binary_features": [c for c in binary_features if c in train_df.columns],
    }
    with open(intermediate_dir / "training_config.json", "w") as _f:
        _json.dump(_config, _f, indent=2)

    # Action distribution in train/test
    _train_dist = train_df["action_label"].value_counts()
    _test_dist = test_df["action_label"].value_counts()

    mo.md(f"""
    ## Train/Test Split + Standardization

    | Metric | Train | Test |
    |--------|-------|------|
    | **Patients** | {len(_train_ids):,} | {len(_test_ids):,} |
    | **Rows** | {len(train_df):,} | {len(test_df):,} |

    ### Action Distribution (Train)

    {_train_dist.to_markdown()}

    ### Action Distribution (Test)

    {_test_dist.to_markdown()}

    ### Files Saved
    - `{intermediate_dir / 'train.parquet'}`
    - `{intermediate_dir / 'test.parquet'}`
    - `{intermediate_dir / 'state_standardization.pkl'}`
    - `{intermediate_dir / 'training_config.json'}`
    """)
    return


if __name__ == "__main__":
    app.run()
