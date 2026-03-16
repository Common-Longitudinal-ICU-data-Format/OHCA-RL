# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "numpy",
#     "torch",
#     "statsmodels",
#     "tabulate",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="OHCA-RL DDQN Training")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import copy
    import shutil
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path

    # ── Load config ──
    with open("config/config.json") as _f:
        config = json.load(_f)

    site_name = config["site_name"]
    out_dir = Path("output/intermediate")
    final_dir = Path("output/final")
    training_dir = out_dir / "training"
    checkpoint_dir = training_dir / "checkpoints"
    shared_dir = Path("shared")

    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(shared_dir, exist_ok=True)

    mo.md(f"# OHCA-RL DDQN Training\n**Site:** {site_name}")
    return (
        DataLoader,
        Dataset,
        F,
        Path,
        checkpoint_dir,
        copy,
        final_dir,
        json,
        mo,
        nn,
        np,
        os,
        out_dir,
        pd,
        shared_dir,
        shutil,
        site_name,
        torch,
        training_dir,
    )


@app.cell
def _(mo, out_dir, pd):
    # Load bucketed time-series data
    bucketed_df = pd.read_parquet(out_dir / "wide_df_bucketed.parquet")

    # Load patient-level summary (demographics, outcomes)
    hosp_summary = pd.read_parquet(out_dir / "hospitalization_summary.parquet")

    # ── Build demographic features ──
    def build_demo_features(pa):
        pa_new = pa.copy()

        # --- Race dummies ---
        pa_new["demo_race_white"] = (
            pa_new["race_category"].str.lower() == "white"
        ).astype(int)

        pa_new["demo_race_black"] = (
            pa_new["race_category"].str.lower() == "black or african american"
        ).astype(int)

        pa_new["demo_race_asian"] = (
            pa_new["race_category"].str.lower() == "asian"
        ).astype(int)

        # --- Ethnicity dummy ---
        pa_new["demo_hispanic"] = (
            pa_new["ethnicity_category"].str.lower() == "hispanic"
        ).astype(int)

        # --- Sex dummy ---
        pa_new["demo_female"] = (
            pa_new["sex_category"].str.lower() == "female"
        ).astype(int)

        # --- Age ---
        pa_new["demo_age"] = pa_new["age_at_admission"]

        return pa_new

    pa = build_demo_features(hosp_summary)
    pa = pa[["hospitalization_id"] + [c for c in pa.columns if c.startswith("demo_")]]

    # Merge demographics into bucketed data
    df = bucketed_df.merge(pa, on="hospitalization_id", how="left")

    # Filter to vasopressor patients only
    df = df.query("ever_vaso == 1").copy()

    # ── Remap action encoding ──
    # KC pipeline    : 0=Stay, 1=Increase, 2=Decrease, 3=Stop
    # Yikuan's's code: 0=Increase, 1=Decrease, 2=Stop, 3=Stay
    ACTION_REMAP = {0: 3, 1: 0, 2: 1, 3: 2}
    df["action"] = df["action"].map(ACTION_REMAP)

    n_patients = df.hospitalization_id.nunique()
    n_rows = len(df)
    mo.md(
        f"**Data loaded:** {n_patients:,} patients, {n_rows:,} hourly rows\n\n"
        f"Action encoding remapped: pipeline→ new format "
        f"(0=Increase, 1=Decrease, 2=Stop, 3=Stay)"
    )
    return ACTION_REMAP, df


@app.cell
def _(df, mo, pd):
    # ── Respiratory mode dummies  ──
    def build_resp_mode_dummies(df_in):
        df_new = df_in.copy()
        mode = df_new["resp_mode_category"].str.lower()

        df_new["resp_mode_group"] = "unknown_or_other"

        df_new.loc[mode.isin([
            "pressure support/cpap",
            "volume support",
            "blow by"
        ]), "resp_mode_group"] = "spontaneous_or_lighter_support"

        df_new.loc[mode.isin([
            "pressure-regulated volume control",
            "assist control-volume control",
            "pressure control",
            "simv"
        ]), "resp_mode_group"] = "controlled_or_mandatory_ventilation"

        dummies = pd.get_dummies(df_new["resp_mode_group"], prefix="resp_mode")
        for _expected in ["resp_mode_spontaneous_or_lighter_support",
                          "resp_mode_controlled_or_mandatory_ventilation",
                          "resp_mode_unknown_or_other"]:
            if _expected not in dummies.columns:
                dummies[_expected] = 0
        df_new = pd.concat([df_new, dummies], axis=1)
        return df_new

    # ── Respiratory device dummies ──
    def build_resp_device_dummies(df_in):
        df_new = df_in.copy()
        device = df_new["resp_device_category"].str.lower()

        df_new["resp_device_group"] = "unknown"

        df_new.loc[device.isin([
            "nasal cannula",
            "room air"
        ]), "resp_device_group"] = "low_intensity"

        df_new.loc[device.isin([
            "face mask",
            "high flow nc",
            "nippv",
            "cpap"
        ]), "resp_device_group"] = "moderate_intensity"

        df_new.loc[device.isin([
            "imv",
            "vent",
            "trach collar"
        ]), "resp_device_group"] = "high_intensity"

        dummies = pd.get_dummies(df_new["resp_device_group"], prefix="resp_device")
        for _expected in ["resp_device_low_intensity", "resp_device_moderate_intensity",
                          "resp_device_high_intensity", "resp_device_unknown"]:
            if _expected not in dummies.columns:
                dummies[_expected] = 0
        df_new = pd.concat([df_new, dummies], axis=1)
        return df_new

    # ── Delta features ──
    def add_delta_features(df_in):
        df_out = df_in.sort_values(["hospitalization_id", "time_bucket"]).copy()
        df_out["delta_map"] = df_out.groupby("hospitalization_id")["vital_map"].diff()
        df_out["delta_lactate"] = df_out.groupby("hospitalization_id")["lab_lactate"].diff()
        df_out[["delta_map", "delta_lactate"]] = df_out[["delta_map", "delta_lactate"]].fillna(0)
        return df_out

    # Apply all feature engineering
    df_feat = build_resp_mode_dummies(df)
    df_feat = build_resp_device_dummies(df_feat)
    df_feat = add_delta_features(df_feat)

    # ── State features list ──
    state_features = [
        # Demographics
        "demo_race_white",
        "demo_race_black",
        "demo_race_asian",
        "demo_hispanic",
        "demo_female",
        "demo_age",
        # Vital signs
        "vital_dbp",
        "vital_heart_rate",
        "vital_map",
        "vital_respiratory_rate",
        "vital_sbp",
        "vital_spo2",
        "vital_temp_c",
        "vital_weight_kg",
        "delta_map",
        "delta_lactate",
        # Laboratory measurements
        "lab_bicarbonate",
        "lab_bun",
        "lab_calcium_total",
        "lab_chloride",
        "lab_creatinine",
        "lab_glucose_serum",
        "lab_hemoglobin",
        "lab_lactate",
        "lab_magnesium",
        "lab_pco2_arterial",
        "lab_ph_arterial",
        "lab_po2_arterial",
        "lab_potassium",
        "lab_so2_arterial",
        "lab_sodium",
        # Respiratory / ventilator
        "resp_fio2_set",
        "resp_peep_set",
        "resp_mode_spontaneous_or_lighter_support",
        "resp_mode_controlled_or_mandatory_ventilation",
        "resp_mode_unknown_or_other",
        "resp_device_low_intensity",
        "resp_device_moderate_intensity",
        "resp_device_high_intensity",
        "resp_device_unknown",
        # Sedation / cardiac medications
        "on_med_cont_dobutamine",
        "on_med_cont_milrinone",
        "on_med_cont_propofol",
        "on_med_cont_midazolam",
        "on_med_cont_dexmedetomidine",
        "on_med_cont_cisatracurium",
        # ICU support indicators
        "in_icu",
        "in_ed",
        "on_imv",
        "on_crrt",
        # Vasopressor exposure (current dose)
        "med_cont_nee",
    ]

    # Check for missing columns
    _missing = [c for c in state_features if c not in df_feat.columns]
    if _missing:
        mo.md(f"⚠️ **Missing state feature columns:** {_missing}")
    else:
        _nan_total = df_feat[state_features].isna().sum().sum()
        mo.md(
            f"**Feature engineering complete.** "
            f"{len(state_features)} state features defined. "
            f"Total NaN in state features: {_nan_total:,}"
        )
    return df_feat, state_features


@app.cell
def _(df_feat, mo, np):
    # ── Reward computation  ──
    def compute_rewards(df_in, cpc_map=None):
        """
        Add reward_intermediate, reward_terminal, and reward columns.

        Intermediate reward:
            r_t =
                1.0 * tanh((clip(MAP_t, 55, 75) - 65) / 10)
              + 0.5 * tanh((Lactate_{t-1} - Lactate_t) / 2)
              - 0.2 * tanh(NEE_t / 0.2)

        Then center reward_intermediate to mean 0.

        Terminal reward:
            CPC1_2 -> +100
            CPC3   -> +40
            CPC4   -> -40
            CPC5   -> -100
        """
        if cpc_map is None:
            cpc_map = {
                "CPC1_2": 100.0,
                "CPC3": 40.0,
                "CPC4": -40.0,
                "CPC5": -100.0,
            }

        df_out = df_in.sort_values(["hospitalization_id", "time_bucket"]).copy()

        # previous lactate within each hospitalization
        lactate_prev = df_out.groupby("hospitalization_id")["lab_lactate"].shift(1)
        lactate_prev = lactate_prev.fillna(df_out["lab_lactate"])

        # clip MAP so reward peaks around 65-75
        map_clipped = np.clip(df_out["vital_map"], 55.0, 75.0)

        # intermediate reward
        df_out["reward_intermediate"] = (
            1.0 * np.tanh((map_clipped - 65.0) / 10.0)
            + 0.5 * np.tanh((lactate_prev - df_out["lab_lactate"]) / 2.0)
            - 0.2 * np.tanh(df_out["med_cont_nee"] / 0.2)
        )

        # center intermediate reward
        df_out["reward_intermediate"] = (
            df_out["reward_intermediate"] - df_out["reward_intermediate"].mean()
        )

        # terminal reward only on last row of each hospitalization
        df_out["reward_terminal"] = 0.0
        is_terminal = (
            df_out.groupby("hospitalization_id")["time_bucket"].transform("max")
            == df_out["time_bucket"]
        )
        df_out.loc[is_terminal, "reward_terminal"] = (
            df_out.loc[is_terminal, "cpc"].map(cpc_map).fillna(0.0)
        )

        # total reward
        df_out["reward"] = df_out["reward_intermediate"] + df_out["reward_terminal"]

        return df_out

    df_rewards = compute_rewards(df_feat)

    mo.md(
        f"**Rewards computed.**\n\n"
        f"- Intermediate reward mean: {df_rewards['reward_intermediate'].mean():.4f} "
        f"(should be ~0 after centering)\n"
        f"- Terminal reward: applied to {(df_rewards['reward_terminal'] != 0).sum():,} terminal rows\n"
        f"- Action distribution (remapped):\n"
        f"  - 0 (Increase): {(df_rewards['action'] == 0).sum():,}\n"
        f"  - 1 (Decrease): {(df_rewards['action'] == 1).sum():,}\n"
        f"  - 2 (Stop): {(df_rewards['action'] == 2).sum():,}\n"
        f"  - 3 (Stay): {(df_rewards['action'] == 3).sum():,}"
    )
    return (df_rewards,)


@app.cell
def _(df_rewards, json, mo, np, pd, state_features, training_dir):
    # ── Train/test split ──
    def train_test_split_hospitalization(
        df_in, id_col="hospitalization_id", train_ratio=0.9, seed=42
    ):
        rng = np.random.default_rng(seed)
        hosp_ids = df_in[id_col].unique()
        rng.shuffle(hosp_ids)
        n_train = int(len(hosp_ids) * train_ratio)
        train_ids = hosp_ids[:n_train]
        test_ids = hosp_ids[n_train:]
        df_train = df_in[df_in[id_col].isin(train_ids)].copy()
        df_test = df_in[df_in[id_col].isin(test_ids)].copy()
        print("Total hospitalizations:", len(hosp_ids))
        print("Train hospitalizations:", len(train_ids))
        print("Test hospitalizations:", len(test_ids))
        print("Train rows:", len(df_train))
        print("Test rows:", len(df_test))
        return df_train, df_test

    # ── Feature type splitting ──
    def split_state_feature_types(df_in, feats):
        binary_features = []
        continuous_features = []
        for col in feats:
            vals = pd.Series(df_in[col]).dropna().unique()
            if df_in[col].dtype == bool:
                binary_features.append(col)
            elif len(vals) <= 2 and set(np.unique(vals)).issubset({0, 1, 0.0, 1.0}):
                binary_features.append(col)
            else:
                continuous_features.append(col)
        return {
            "binary_features": binary_features,
            "continuous_features": continuous_features,
        }

    # ── Fit preprocessor ──
    def fit_state_preprocessor(df_train, feats):
        feature_types = split_state_feature_types(df_train, feats)
        binary_features = feature_types["binary_features"]
        continuous_features = feature_types["continuous_features"]
        means = df_train[continuous_features].mean()
        stds = df_train[continuous_features].std()
        stds = stds.replace(0, 1.0)
        preprocessor = {
            "binary_features": binary_features,
            "continuous_features": continuous_features,
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }
        return preprocessor

    # ── Transform features  ──
    def transform_state_features(df_in, feats, preprocessor):
        df_out = df_in.copy()
        continuous_features = preprocessor["continuous_features"]
        binary_features = preprocessor["binary_features"]
        for col in continuous_features:
            mu = preprocessor["mean"][col]
            sd = preprocessor["std"][col]
            df_out[col] = ((df_out[col] - mu) / sd).astype(np.float32)
        for col in binary_features:
            df_out[col] = df_out[col].astype(np.float32)
        df_out[feats] = df_out[feats].astype(np.float32)
        return df_out

    # ── Execute split & standardization ──
    df_train_raw, df_test_raw = train_test_split_hospitalization(df_rewards)

    preprocessor = fit_state_preprocessor(df_train_raw, state_features)

    df_train_proc = transform_state_features(df_train_raw, state_features, preprocessor)
    df_test_proc = transform_state_features(df_test_raw, state_features, preprocessor)

    # ── Save preprocessor and feature list for external validation ──
    with open(training_dir / "preprocessor.json", "w") as _f:
        json.dump(preprocessor, _f, indent=2)

    with open(training_dir / "state_features.json", "w") as _f:
        json.dump(state_features, _f, indent=2)

    mo.md(
        f"**Train/test split complete.**\n\n"
        f"- Binary features: {len(preprocessor['binary_features'])}\n"
        f"- Continuous features: {len(preprocessor['continuous_features'])}\n"
        f"- Preprocessor saved to `{training_dir}/preprocessor.json`"
    )
    return df_test_proc, df_test_raw, df_train_proc, df_train_raw


@app.cell
def _(
    DataLoader,
    Dataset,
    df_test_proc,
    df_test_raw,
    df_train_proc,
    df_train_raw,
    mo,
    np,
    pd,
    state_features,
    torch,
):
    # ── Action constants  ──
    ACTION_INCREASE = 0
    ACTION_DECREASE = 1
    ACTION_STOP = 2
    ACTION_STAY = 3

    # ── Action mask  ──
    def build_action_mask(map_t, nee_t, nee_min=0.0, nee_max=1.1):
        """
        Return mask for [increase, decrease, stop, stay].
        Mask logic uses RAW clinical values.
        """
        if nee_t <= nee_min:
            return np.array([1, 0, 0, 1], dtype=np.int8)
        if map_t < 55:
            return np.array([1, 0, 0, 1], dtype=np.int8)
        if map_t > 90:
            return np.array([0, 1, 1, 1], dtype=np.int8)
        if nee_t >= nee_max:
            return np.array([0, 1, 0, 1], dtype=np.int8)
        return np.array([1, 1, 1, 1], dtype=np.int8)

    # ── Transition builder ──
    def build_transition_dataframe(
        df_state, df_mask_raw, feats,
        reward_col="reward", action_col="action",
        id_col="hospitalization_id", time_col="time_bucket",
        raw_nee_col="med_cont_nee", raw_map_col="vital_map",
        nee_min=0.0, nee_max=1.1,
    ):
        sort_cols = [id_col, time_col]
        df_state = df_state.sort_values(sort_cols).copy()
        df_mask_raw = df_mask_raw.sort_values(sort_cols).copy()

        aligned = (
            np.array_equal(df_state[id_col].to_numpy(), df_mask_raw[id_col].to_numpy())
            and np.array_equal(df_state[time_col].to_numpy(), df_mask_raw[time_col].to_numpy())
        )
        if not aligned:
            raise ValueError("df_state and df_mask_raw are not aligned after sorting.")

        done = (
            df_state.groupby(id_col)[time_col].transform("max") == df_state[time_col]
        ).astype(np.int8)

        next_state_df = (
            df_state.groupby(id_col)[feats]
            .shift(-1)
            .fillna(0.0)
            .astype(np.float32)
        )

        raw_map = df_mask_raw[raw_map_col].to_numpy()
        raw_nee = df_mask_raw[raw_nee_col].to_numpy()

        next_raw_map = (
            df_mask_raw.groupby(id_col)[raw_map_col].shift(-1).fillna(0.0).to_numpy()
        )
        next_raw_nee = (
            df_mask_raw.groupby(id_col)[raw_nee_col].shift(-1).fillna(0.0).to_numpy()
        )

        masks = np.stack([
            build_action_mask(map_t=m, nee_t=n, nee_min=nee_min, nee_max=nee_max)
            for m, n in zip(raw_map, raw_nee)
        ]).astype(np.int8)

        next_masks = np.stack([
            build_action_mask(map_t=m, nee_t=n, nee_min=nee_min, nee_max=nee_max)
            for m, n in zip(next_raw_map, next_raw_nee)
        ]).astype(np.int8)

        next_masks[done.to_numpy() == 1] = np.array([1, 1, 1, 1], dtype=np.int8)

        base_df = pd.DataFrame(
            {
                id_col: df_state[id_col].to_numpy(),
                time_col: df_state[time_col].to_numpy(),
                action_col: df_state[action_col].to_numpy(),
                reward_col: df_state[reward_col].to_numpy(),
                "done": done.to_numpy(),
            },
            index=df_state.index,
        )

        state_block = df_state[feats].astype(np.float32).copy()
        state_block.columns = [f"s_{c}" for c in feats]

        next_state_block = next_state_df.copy()
        next_state_block.columns = [f"ns_{c}" for c in feats]

        mask_block = pd.DataFrame(
            {
                "mask_increase": masks[:, ACTION_INCREASE],
                "mask_decrease": masks[:, ACTION_DECREASE],
                "mask_stop": masks[:, ACTION_STOP],
                "mask_stay": masks[:, ACTION_STAY],
                "next_mask_increase": next_masks[:, ACTION_INCREASE],
                "next_mask_decrease": next_masks[:, ACTION_DECREASE],
                "next_mask_stop": next_masks[:, ACTION_STOP],
                "next_mask_stay": next_masks[:, ACTION_STAY],
            },
            index=df_state.index,
        )

        out = pd.concat([base_df, state_block, next_state_block, mask_block], axis=1)
        return out.copy()

    # ── Numpy batch extraction ──
    def extract_numpy_batches(transition_df, feats, action_col="action", reward_col="reward"):
        state_cols = [f"s_{c}" for c in feats]
        next_state_cols = [f"ns_{c}" for c in feats]
        mask_cols = ["mask_increase", "mask_decrease", "mask_stop", "mask_stay"]
        next_mask_cols = ["next_mask_increase", "next_mask_decrease", "next_mask_stop", "next_mask_stay"]
        batch = {
            "states": transition_df[state_cols].to_numpy(dtype=np.float32),
            "actions": transition_df[action_col].to_numpy(dtype=np.int64),
            "rewards": transition_df[reward_col].to_numpy(dtype=np.float32),
            "next_states": transition_df[next_state_cols].to_numpy(dtype=np.float32),
            "dones": transition_df["done"].to_numpy(dtype=np.float32),
            "action_masks": transition_df[mask_cols].to_numpy(dtype=np.int8),
            "next_action_masks": transition_df[next_mask_cols].to_numpy(dtype=np.int8),
        }
        return batch

    # ── PyTorch Dataset  ──
    class OfflineRLDataset(Dataset):
        def __init__(self, batch_dict):
            self.states = torch.tensor(batch_dict["states"], dtype=torch.float32)
            self.actions = torch.tensor(batch_dict["actions"], dtype=torch.long)
            self.rewards = torch.tensor(batch_dict["rewards"], dtype=torch.float32)
            self.next_states = torch.tensor(batch_dict["next_states"], dtype=torch.float32)
            self.dones = torch.tensor(batch_dict["dones"], dtype=torch.float32)
            self.action_masks = torch.tensor(batch_dict["action_masks"], dtype=torch.float32)
            self.next_action_masks = torch.tensor(batch_dict["next_action_masks"], dtype=torch.float32)

        def __len__(self):
            return self.states.shape[0]

        def __getitem__(self, idx):
            return {
                "states": self.states[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "next_states": self.next_states[idx],
                "dones": self.dones[idx],
                "action_masks": self.action_masks[idx],
                "next_action_masks": self.next_action_masks[idx],
            }

    # ── DataLoader builder ──
    def build_dataloaders(train_batch, test_batch, batch_size=256, num_workers=0):
        train_dataset = OfflineRLDataset(train_batch)
        test_dataset = OfflineRLDataset(test_batch)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        )
        return train_dataset, test_dataset, train_loader, test_loader

    # ── Build transitions ──
    transition_train = build_transition_dataframe(
        df_state=df_train_proc, df_mask_raw=df_train_raw,
        feats=state_features, reward_col="reward", action_col="action",
        nee_min=0.0, nee_max=1.1,
    )

    transition_test = build_transition_dataframe(
        df_state=df_test_proc, df_mask_raw=df_test_raw,
        feats=state_features, reward_col="reward", action_col="action",
        nee_min=0.0, nee_max=1.1,
    )

    train_batch = extract_numpy_batches(
        transition_df=transition_train, feats=state_features,
        action_col="action", reward_col="reward",
    )

    test_batch = extract_numpy_batches(
        transition_df=transition_test, feats=state_features,
        action_col="action", reward_col="reward",
    )

    # Build dataloaders
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        train_batch=train_batch, test_batch=test_batch, batch_size=256,
    )

    state_dim = train_batch["states"].shape[1]

    mo.md(
        f"**Transitions built.**\n\n"
        f"- Train transitions: {len(transition_train):,}\n"
        f"- Test transitions: {len(transition_test):,}\n"
        f"- State dimension: {state_dim}"
    )
    return state_dim, test_loader, train_loader, transition_test


@app.cell
def _(F, copy, nn, np, os, pd, torch):
    # ── Q-Network ──
    class QNetwork(nn.Module):
        def __init__(self, state_dim_val, n_actions=4, hidden_dims=(256, 256), dropout=0.0):
            super().__init__()
            h1, h2 = hidden_dims
            self.net = nn.Sequential(
                nn.Linear(state_dim_val, h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    # ── Action masking  ──
    def apply_action_mask(q_values, action_masks, invalid_penalty=-1e9):
        masked_q = q_values.clone()
        masked_q[action_masks <= 0] = invalid_penalty
        return masked_q

    # ── Build DDQN components ──
    def build_ddqn_components(
        state_dim_val, n_actions=4, hidden_dims=(256, 256),
        dropout=0.0, lr=1e-3, device="cpu",
    ):
        online_net = QNetwork(
            state_dim_val=state_dim_val, n_actions=n_actions,
            hidden_dims=hidden_dims, dropout=dropout,
        ).to(device)

        target_net = QNetwork(
            state_dim_val=state_dim_val, n_actions=n_actions,
            hidden_dims=hidden_dims, dropout=dropout,
        ).to(device)

        target_net.load_state_dict(copy.deepcopy(online_net.state_dict()))
        optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

        return online_net, target_net, optimizer

    # ── DDQN target computation  ──
    @torch.no_grad()
    def compute_ddqn_targets(
        rewards, dones, next_states, next_action_masks,
        online_net, target_net, gamma=0.99,
    ):
        next_q_online = online_net(next_states)
        next_q_online_masked = apply_action_mask(next_q_online, next_action_masks)
        next_actions = next_q_online_masked.argmax(dim=1)

        next_q_target = target_net(next_states)
        next_q_target_selected = next_q_target.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)

        targets = rewards + gamma * (1.0 - dones) * next_q_target_selected
        return targets

    # ── Single training step  ──
    def train_one_step(
        batch, online_net, target_net, optimizer,
        gamma=0.99, grad_clip=5.0, device="cpu",
    ):
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        next_states = batch["next_states"].to(device)
        dones = batch["dones"].to(device)
        next_action_masks = batch["next_action_masks"].to(device)

        q_values = online_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        targets = compute_ddqn_targets(
            rewards=rewards, dones=dones, next_states=next_states,
            next_action_masks=next_action_masks,
            online_net=online_net, target_net=target_net, gamma=gamma,
        )

        loss = F.smooth_l1_loss(q_selected, targets)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), grad_clip)

        optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_selected.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    # ── Evaluation ──
    @torch.no_grad()
    def evaluate_ddqn(loader, online_net, target_net, gamma=0.99, device="cpu"):
        online_net.eval()
        target_net.eval()

        losses = []
        q_means = []
        target_means = []

        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)
            next_states = batch["next_states"].to(device)
            dones = batch["dones"].to(device)
            next_action_masks = batch["next_action_masks"].to(device)

            q_values = online_net(states)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            targets = compute_ddqn_targets(
                rewards=rewards, dones=dones, next_states=next_states,
                next_action_masks=next_action_masks,
                online_net=online_net, target_net=target_net, gamma=gamma,
            )

            loss = F.smooth_l1_loss(q_selected, targets)

            losses.append(loss.item())
            q_means.append(q_selected.mean().item())
            target_means.append(targets.mean().item())

        online_net.train()
        target_net.train()

        return {
            "loss": float(np.mean(losses)),
            "q_mean": float(np.mean(q_means)),
            "target_mean": float(np.mean(target_means)),
        }

    # ── Hard target update ──
    def hard_update_target_network(online_net, target_net):
        target_net.load_state_dict(copy.deepcopy(online_net.state_dict()))

    # ── Checkpoint saving  ──
    def save_checkpoint_fn(
        checkpoint_path, epoch, global_step,
        online_net, target_net, optimizer,
        history_df=None, extra_config=None,
    ):
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "online_state_dict": online_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if history_df is not None:
            ckpt["history"] = history_df.to_dict(orient="records")
        if extra_config is not None:
            ckpt["config"] = extra_config
        torch.save(ckpt, checkpoint_path)

    # ── Full training loop ──
    def train_ddqn(
        train_loader, test_loader, state_dim_val,
        n_actions=4, hidden_dims=(256, 256), dropout=0.0,
        lr=1e-3, gamma=0.99, n_epochs=50, target_update_every=500,
        grad_clip=5.0, device="cpu", checkpoint_dir_val="ddqn_checkpoints",
        checkpoint_every_epochs=5, min_training_epochs=20,
        early_stopping=False, early_stopping_metric="test_loss",
        early_stopping_mode="min", early_stopping_patience=8, min_delta=1e-4,
    ):
        os.makedirs(checkpoint_dir_val, exist_ok=True)

        online_net, target_net, optimizer = build_ddqn_components(
            state_dim_val=state_dim_val, n_actions=n_actions,
            hidden_dims=hidden_dims, dropout=dropout, lr=lr, device=device,
        )

        history = []
        global_step = 0

        best_metric = np.inf if early_stopping_mode == "min" else -np.inf
        best_epoch = 0
        best_state_dict = copy.deepcopy(online_net.state_dict())
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            online_net.train()
            target_net.train()

            train_losses = []
            train_q_means = []
            train_target_means = []

            for batch in train_loader:
                metrics = train_one_step(
                    batch=batch, online_net=online_net, target_net=target_net,
                    optimizer=optimizer, gamma=gamma, grad_clip=grad_clip,
                    device=device,
                )
                train_losses.append(metrics["loss"])
                train_q_means.append(metrics["q_mean"])
                train_target_means.append(metrics["target_mean"])
                global_step += 1

                if global_step % target_update_every == 0:
                    hard_update_target_network(online_net, target_net)

            train_metrics = {
                "loss": float(np.mean(train_losses)),
                "q_mean": float(np.mean(train_q_means)),
                "target_mean": float(np.mean(train_target_means)),
            }

            val_metrics = evaluate_ddqn(
                loader=test_loader, online_net=online_net, target_net=target_net,
                gamma=gamma, device=device,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_q_mean": train_metrics["q_mean"],
                "train_target_mean": train_metrics["target_mean"],
                "test_loss": val_metrics["loss"],
                "test_q_mean": val_metrics["q_mean"],
                "test_target_mean": val_metrics["target_mean"],
            }
            history.append(row)
            history_df = pd.DataFrame(history)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"test_loss={row['test_loss']:.4f} | "
                f"train_q={row['train_q_mean']:.4f} | "
                f"test_q={row['test_q_mean']:.4f}"
            )

            # best model tracking
            metric_value = row[early_stopping_metric]
            if early_stopping_mode == "min":
                improved = metric_value < (best_metric - min_delta)
            else:
                improved = metric_value > (best_metric + min_delta)

            if improved:
                best_metric = metric_value
                best_epoch = epoch
                best_state_dict = copy.deepcopy(online_net.state_dict())
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": online_net.state_dict(),
                        "target_state_dict": target_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "history": history_df.to_dict(orient="records"),
                    },
                    os.path.join(checkpoint_dir_val, "best_model.pt"),
                )
            else:
                patience_counter += 1

            # periodic checkpoint
            if epoch % checkpoint_every_epochs == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": online_net.state_dict(),
                        "target_state_dict": target_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": history_df.to_dict(orient="records"),
                    },
                    os.path.join(checkpoint_dir_val, f"checkpoint_epoch_{epoch}.pt"),
                )

            # optional early stopping
            if early_stopping:
                if epoch >= min_training_epochs and patience_counter >= early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered."
                        f"\nBest epoch = {best_epoch}"
                        f"\nBest {early_stopping_metric} = {best_metric:.6f}"
                    )
                    break

        # load best model before return
        best_ckpt_path = os.path.join(checkpoint_dir_val, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            online_net.load_state_dict(best_ckpt["model_state_dict"])
            if "target_state_dict" in best_ckpt:
                target_net.load_state_dict(best_ckpt["target_state_dict"])
            else:
                hard_update_target_network(online_net, target_net)
        else:
            online_net.load_state_dict(best_state_dict)
            hard_update_target_network(online_net, target_net)

        # save last model too
        history_df = pd.DataFrame(history)
        torch.save(
            {
                "epoch": int(history_df["epoch"].max()) if len(history_df) > 0 else 0,
                "model_state_dict": online_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history_df.to_dict(orient="records"),
            },
            os.path.join(checkpoint_dir_val, "last_model.pt"),
        )

        return online_net, target_net, history_df

    # ── Prediction  ──
    @torch.no_grad()
    def predict_actions(loader, online_net, device="cpu"):
        online_net.eval()
        pred_actions = []
        pred_q_values = []
        observed_actions = []
        rewards = []
        dones = []

        for batch in loader:
            states = batch["states"].to(device)
            action_masks = batch["action_masks"].to(device)

            q_values = online_net(states)
            q_values_masked = apply_action_mask(q_values, action_masks)
            actions_pred = q_values_masked.argmax(dim=1)

            pred_actions.append(actions_pred.cpu().numpy())
            pred_q_values.append(q_values.cpu().numpy())
            observed_actions.append(batch["actions"].cpu().numpy())
            rewards.append(batch["rewards"].cpu().numpy())
            dones.append(batch["dones"].cpu().numpy())

        out = {
            "pred_actions": np.concatenate(pred_actions),
            "pred_q_values": np.concatenate(pred_q_values),
            "observed_actions": np.concatenate(observed_actions),
            "rewards": np.concatenate(rewards),
            "dones": np.concatenate(dones),
        }
        return out

    # ── Prediction summary ──
    def summarize_predictions(pred_dict):
        pred_actions = pred_dict["pred_actions"]
        observed_actions = pred_dict["observed_actions"]

        pred_counts = pd.Series(pred_actions).value_counts().sort_index()
        obs_counts = pd.Series(observed_actions).value_counts().sort_index()

        summary_df = pd.DataFrame({
            "pred_count": pred_counts,
            "obs_count": obs_counts,
        }).fillna(0).astype(int)

        action_name_map = {
            0: "increase",
            1: "decrease",
            2: "stop",
            3: "stay",
        }
        summary_df["action_name"] = summary_df.index.map(action_name_map)

        agreement = float((pred_actions == observed_actions).mean())

        return summary_df, agreement

    return predict_actions, summarize_predictions, train_ddqn


@app.cell
def _(
    checkpoint_dir,
    json,
    mo,
    site_name,
    state_dim,
    test_loader,
    torch,
    train_ddqn,
    train_loader,
    training_dir,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Training hyperparameters
    _training_config = {
        "site_name": site_name,
        "state_dim": state_dim,
        "n_actions": 4,
        "hidden_dims": [256, 256],
        "dropout": 0.0,
        "lr": 1e-3,
        "gamma": 0.99,
        "n_epochs": 50,
        "target_update_every": 500,
        "grad_clip": 5.0,
        "batch_size": 256,
        "checkpoint_every_epochs": 5,
        "min_training_epochs": 20,
        "early_stopping": True,
        "early_stopping_patience": 8,
        "device": device,
    }

    # Save training config
    with open(training_dir / "training_config.json", "w") as _f:
        json.dump(_training_config, _f, indent=2)

    # Train
    online_net, target_net, history_df = train_ddqn(
        train_loader=train_loader,
        test_loader=test_loader,
        state_dim_val=state_dim,
        n_actions=4,
        hidden_dims=(256, 256),
        dropout=0.0,
        lr=1e-3,
        gamma=0.99,
        n_epochs=_training_config["n_epochs"],
        target_update_every=500,
        grad_clip=5.0,
        device=device,
        checkpoint_dir_val=str(checkpoint_dir),
        checkpoint_every_epochs=5,
        min_training_epochs=20,
        early_stopping=True,
        early_stopping_patience=8,
    )

    # Save training history
    history_df.to_csv(training_dir / "training_history.csv", index=False)

    mo.md(
        f"**Training complete.**\n\n"
        f"- Best test loss: {history_df['test_loss'].min():.4f} "
        f"(epoch {history_df.loc[history_df['test_loss'].idxmin(), 'epoch']:.0f})\n"
        f"- Final train loss: {history_df['train_loss'].iloc[-1]:.4f}\n"
        f"- Checkpoints saved to `{checkpoint_dir}`"
    )
    return device, online_net


@app.cell
def _(
    checkpoint_dir,
    device,
    mo,
    online_net,
    os,
    predict_actions,
    summarize_predictions,
    test_loader,
    torch,
    training_dir,
):
    # Load best checkpoint explicitly
    _best_ckpt_path = os.path.join(str(checkpoint_dir), "best_model.pt")
    _best_ckpt = torch.load(_best_ckpt_path, map_location=device, weights_only=False)
    online_net.load_state_dict(_best_ckpt["model_state_dict"])
    online_net.to(device)
    online_net.eval()

    print(f"Loaded best checkpoint from epoch {_best_ckpt['epoch']}")

    # Predict actions on test set
    test_pred = predict_actions(
        loader=test_loader, online_net=online_net, device=device,
    )

    # Summarize
    pred_summary_df, agreement = summarize_predictions(test_pred)

    # Save action summary
    pred_summary_df.to_csv(training_dir / "test_action_summary.csv")

    # Save test set with predictions for downstream visualization (08_visualize_results.py)
    _eval_df = transition_test[["hospitalization_id", "time_bucket", "action"]].copy()
    _eval_df["pred_action"] = test_pred["pred_actions"]
    _eval_df["reward"] = test_pred["rewards"]
    _eval_df["done"] = test_pred["dones"]
    _eval_df.to_parquet(training_dir / "test_with_predictions.parquet", index=False)
    print(f"Saved test_with_predictions.parquet: {len(_eval_df)} rows")

    mo.md(
        f"**Test set evaluation:**\n\n"
        f"- Action agreement: {agreement:.1%}\n\n"
        f"{pred_summary_df.to_markdown()}"
    )
    return (test_pred,)


@app.cell
def _(df_test_raw, mo, np, pd, test_pred, training_dir, transition_test):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # ── Patient agreement dataset  ──
    def build_patient_agreement_dataset(
        trans_test, t_pred, raw_test,
        id_col="hospitalization_id", time_col="time_bucket",
    ):
        df_eval = trans_test.copy().reset_index(drop=True)
        if len(df_eval) != len(t_pred["pred_actions"]):
            raise ValueError("transition_test and test_pred lengths do not match.")

        df_eval["pred_action"] = t_pred["pred_actions"]
        df_eval["agree"] = (df_eval["pred_action"] == df_eval["action"]).astype(int)

        outcome_df = (
            raw_test.sort_values([id_col, time_col])
            .groupby(id_col)
            .tail(1)[[id_col, "cpc"]]
            .copy()
        )

        cpc_map_good = {"CPC1_2": 4, "CPC3": 3, "CPC4": 2, "CPC5": 1}
        outcome_df["cpc_ord_good"] = outcome_df["cpc"].map(cpc_map_good)

        patient_df = (
            df_eval.groupby(id_col)
            .agg(agreement_rate=("agree", "mean"), n_steps=("agree", "size"))
            .reset_index()
        )

        patient_df = patient_df.merge(
            outcome_df[[id_col, "cpc", "cpc_ord_good"]], on=id_col, how="left",
        )
        return patient_df

    # ── Ordinal logistic regression  ──
    def fit_ordinal_agreement_model(patient_df):
        X = patient_df[["agreement_rate"]].copy()
        y = patient_df["cpc_ord_good"].copy()
        model = OrderedModel(endog=y, exog=X, distr="logit")
        result = model.fit(method="bfgs", disp=False)
        return result

    # ── Summarize ordinal result  ──
    def summarize_ordinal_result(result, predictor_name="agreement_rate"):
        coef = result.params[predictor_name]
        se = result.bse[predictor_name]
        pval = result.pvalues[predictor_name]
        or_value = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        # Per-10pp OR (clinically reported: odds change per 10pp increase)
        or_10pp = np.exp(coef * 0.10)
        or_10pp_ci_low = np.exp((coef - 1.96 * se) * 0.10)
        or_10pp_ci_high = np.exp((coef + 1.96 * se) * 0.10)
        summary_df = pd.DataFrame({
            "term": [predictor_name],
            "coef": [coef],
            "std_err": [se],
            "p_value": [pval],
            "odds_ratio": [or_value],
            "or_95ci_low": [ci_low],
            "or_95ci_high": [ci_high],
            "or_10pp": [or_10pp],
            "or_10pp_ci_low": [or_10pp_ci_low],
            "or_10pp_ci_high": [or_10pp_ci_high],
        })
        return summary_df

    # ── Binned agreement summary  ──
    def agreement_bin_summary(patient_df):
        out = patient_df.copy()
        out["agreement_bin"] = pd.cut(
            out["agreement_rate"],
            bins=[0.0, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
            include_lowest=True,
        )
        summary = (
            out.groupby("agreement_bin", observed=False)
            .agg(
                n_hosp=("hospitalization_id", "size"),
                mean_agreement=("agreement_rate", "mean"),
                mean_cpc_ord_good=("cpc_ord_good", "mean"),
            )
            .reset_index()
        )
        return summary

    # ── Run full evaluation ──
    patient_df = build_patient_agreement_dataset(
        transition_test, test_pred, df_test_raw,
    )

    result = fit_ordinal_agreement_model(patient_df)
    coef_summary = summarize_ordinal_result(result, predictor_name="agreement_rate")
    bin_summary = agreement_bin_summary(patient_df)

    # Save results
    coef_summary.to_csv(training_dir / "coef_summary.csv", index=False)
    bin_summary.to_csv(training_dir / "bin_summary.csv", index=False)

    print(result.summary())

    mo.md(
        f"**Agreement-Outcome Evaluation:**\n\n"
        f"**Ordinal Logistic Regression** (higher cpc_ord_good = better outcome):\n\n"
        f"{coef_summary.to_markdown(index=False)}\n\n"
        f"**Agreement Bin Summary:**\n\n"
        f"{bin_summary.to_markdown(index=False)}"
    )
    return


@app.cell
def _(
    ACTION_REMAP,
    checkpoint_dir,
    json,
    mo,
    shared_dir,
    shutil,
    training_dir,
):
    # Copy model artifacts to shared/ for multi-site exchange
    _files_to_copy = {
        checkpoint_dir / "best_model.pt": shared_dir / "best_model.pt",
        training_dir / "preprocessor.json": shared_dir / "preprocessor.json",
        training_dir / "state_features.json": shared_dir / "state_features.json",
        training_dir / "training_config.json": shared_dir / "training_config.json",
    }

    for _src, _dst in _files_to_copy.items():
        if _src.exists():
            shutil.copy2(_src, _dst)

    # Save action remap for reference
    with open(training_dir / "action_remap.json", "w") as _f:
        json.dump(
            {
                "pipeline_to_new": {str(k): int(v) for k, v in ACTION_REMAP.items()},
                "new_encoding": {
                    "0": "increase",
                    "1": "decrease",
                    "2": "stop",
                    "3": "stay",
                },
                "pipeline_encoding": {
                    "0": "stay",
                    "1": "increase",
                    "2": "decrease",
                    "3": "stop",
                },
            },
            _f,
            indent=2,
        )

    shutil.copy2(training_dir / "action_remap.json", shared_dir / "action_remap.json")

    mo.md(
        f"**Artifacts copied to `shared/` for multi-site exchange.**\n\n"
        f"Files:\n"
        + "\n".join(f"- `{dst.name}`" for dst in _files_to_copy.values())
        + "\n- `action_remap.json`"
    )
    return


# ── Cell: Export to upload_to_box/ + copy results to output/final/ ────
@app.cell
def _(
    Path,
    checkpoint_dir,
    df_train_raw,
    df_test_raw,
    final_dir,
    json,
    mo,
    os,
    pd,
    shared_dir,
    shutil,
    site_name,
    torch,
    training_dir,
):
    # ── 1. Copy training results to output/final/ ──
    _results_to_copy = [
        "coef_summary.csv",
        "bin_summary.csv",
        "test_action_summary.csv",
        "training_history.csv",
        "training_config.json",
    ]
    for _fname in _results_to_copy:
        _src = training_dir / _fname
        if _src.exists():
            shutil.copy2(_src, final_dir / _fname)

    # ── 2. Build upload_to_box/ directory ──
    _upload_dir = Path("output/upload_to_box")
    _std_dir = _upload_dir / "standardization"
    _eval_dir = _upload_dir / "evaluation"
    os.makedirs(_std_dir, exist_ok=True)
    os.makedirs(_eval_dir, exist_ok=True)

    # Checkpoint with both online + target state dicts
    _ckpt_path = checkpoint_dir / "best_model.pt"
    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
    _n_train = int(df_train_raw["hospitalization_id"].nunique())
    _n_test = int(df_test_raw["hospitalization_id"].nunique())
    _best_epoch = int(_ckpt.get("epoch", 0))

    _site_ckpt = {
        "model_state_dict": _ckpt["model_state_dict"],
        "target_state_dict": _ckpt.get("target_state_dict", _ckpt["model_state_dict"]),
        "site_id": site_name,
        "round": 0,
        "num_samples": _n_train,
        "epoch": _best_epoch,
    }
    torch.save(_site_ckpt, _upload_dir / f"{site_name}_weights.pt")

    # Metadata JSON
    _history = pd.read_csv(training_dir / "training_history.csv")
    _coef = pd.read_csv(training_dir / "coef_summary.csv")
    _metadata = {
        "site_id": site_name,
        "n_train_patients": _n_train,
        "n_test_patients": _n_test,
        "n_total_vaso": _n_train + _n_test,
        "best_epoch": _best_epoch,
        "best_test_loss": float(_history["test_loss"].min()),
        "ordinal_coef": float(_coef["coef"].iloc[0]),
        "ordinal_or_10pp": float(_coef["or_10pp"].iloc[0]),
        "ordinal_p_value": float(_coef["p_value"].iloc[0]),
    }
    with open(_upload_dir / f"{site_name}_metadata.json", "w") as _f:
        json.dump(_metadata, _f, indent=2)

    # Standardization artifacts (for other sites to download)
    for _fname in ["best_model.pt", "preprocessor.json", "state_features.json",
                    "training_config.json", "action_remap.json"]:
        _src = shared_dir / _fname
        if _src.exists():
            shutil.copy2(_src, _std_dir / _fname)

    # Evaluation results (aggregate only — no patient-level data)
    _eval_files = [
        (final_dir / "table1_ohca.csv", "table1_ohca.csv"),
        (final_dir / "table1_ohca_vaso.csv", "table1_ohca_vaso.csv"),
        (final_dir / "strobe_counts.csv", "strobe_counts.csv"),
        (training_dir / "coef_summary.csv", "coef_summary.csv"),
        (training_dir / "bin_summary.csv", "bin_summary.csv"),
        (training_dir / "test_action_summary.csv", "test_action_summary.csv"),
    ]
    for _src, _dst_name in _eval_files:
        if _src.exists():
            shutil.copy2(_src, _eval_dir / _dst_name)

    mo.md(
        f"**Export complete.**\n\n"
        f"**Training results → `output/final/`:**\n"
        + "\n".join(f"- `{f}`" for f in _results_to_copy)
        + f"\n\n**Upload package → `output/upload_to_box/`:**\n"
        f"- `{site_name}_weights.pt` ({_n_train} train patients)\n"
        f"- `{site_name}_metadata.json`\n"
        f"- `standardization/` (5 files for external sites)\n"
        f"- `evaluation/` ({len([s for s, _ in _eval_files if s.exists()])} aggregate CSVs)"
    )
    return


if __name__ == "__main__":
    app.run()
