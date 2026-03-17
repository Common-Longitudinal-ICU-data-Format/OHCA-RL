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
#     "matplotlib",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", app_title="OHCA-RL External Validation")


# ── Cell 0: Imports & Config ─────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import os
    import copy
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
    shared_dir = Path("shared")
    eval_dir = final_dir / "external_validation"

    os.makedirs(eval_dir, exist_ok=True)

    mo.md(
        f"# OHCA-RL External Validation\n\n"
        f"**Local site:** {site_name}\n\n"
        f"**Model source:** `shared/` directory\n\n"
        f"This script applies a trained model from another site to local data."
    )

    return (
        mo, json, os, copy, np, pd, torch, nn, F, Dataset, DataLoader,
        Path, config, site_name, out_dir, final_dir, shared_dir, eval_dir,
    )


# ── Cell 1: Load External Model Artifacts ────────────────────────────
@app.cell
def _(mo, json, shared_dir):
    # ── Load preprocessor (mean/std from training site) ──
    with open(shared_dir / "preprocessor.json") as _f:
        ext_preprocessor = json.load(_f)

    # ── Load state features list ──
    with open(shared_dir / "state_features.json") as _f:
        ext_state_features = json.load(_f)

    # ── Load training config (architecture params) ──
    with open(shared_dir / "training_config.json") as _f:
        ext_training_config = json.load(_f)

    # ── Load action remap info ──
    with open(shared_dir / "action_remap.json") as _f:
        ext_action_remap = json.load(_f)

    mo.md(
        f"**External model artifacts loaded from `shared/`:**\n\n"
        f"- Training site: {ext_training_config.get('site_name', 'unknown')}\n"
        f"- State dim: {ext_training_config['state_dim']}\n"
        f"- Hidden dims: {ext_training_config['hidden_dims']}\n"
        f"- Binary features: {len(ext_preprocessor['binary_features'])}\n"
        f"- Continuous features: {len(ext_preprocessor['continuous_features'])}"
    )

    return ext_preprocessor, ext_state_features, ext_training_config, ext_action_remap


# ── Cell 2: Load Local Data & Feature Engineering ────────────────────
@app.cell
def _(mo, np, pd, out_dir, ext_action_remap):
    # Load local bucketed data and hospitalization summary
    _bucketed_df = pd.read_parquet(out_dir / "wide_df_bucketed.parquet")
    hosp_summary = pd.read_parquet(out_dir / "hospitalization_summary.parquet")

    # ── Build demographic features (same as 06_training.py) ──
    def build_demo_features(pa):
        pa_new = pa.copy()
        pa_new["demo_race_white"] = (
            pa_new["race_category"].str.lower() == "white"
        ).astype(int)
        pa_new["demo_race_black"] = (
            pa_new["race_category"].str.lower() == "black or african american"
        ).astype(int)
        pa_new["demo_race_asian"] = (
            pa_new["race_category"].str.lower() == "asian"
        ).astype(int)
        pa_new["demo_hispanic"] = (
            pa_new["ethnicity_category"].str.lower() == "hispanic"
        ).astype(int)
        pa_new["demo_female"] = (
            pa_new["sex_category"].str.lower() == "female"
        ).astype(int)
        pa_new["demo_age"] = pa_new["age_at_admission"]
        return pa_new

    _pa = build_demo_features(hosp_summary)
    _pa = _pa[["hospitalization_id"] + [c for c in _pa.columns if c.startswith("demo_")]]

    local_df = _bucketed_df.merge(_pa, on="hospitalization_id", how="left")

    # Filter to vasopressor patients
    local_df = local_df.query("ever_vaso == 1").copy()

    # Remap actions to PI's encoding
    _remap = {int(k): int(v) for k, v in ext_action_remap["pipeline_to_new"].items()}
    local_df["action"] = local_df["action"].map(_remap)

    # ── Respiratory mode dummies ──
    def build_resp_mode_dummies(df_in):
        df_new = df_in.copy()
        mode = df_new["resp_mode_category"].str.lower()
        df_new["resp_mode_group"] = "unknown_or_other"
        df_new.loc[mode.isin([
            "pressure support/cpap", "volume support", "blow by"
        ]), "resp_mode_group"] = "spontaneous_or_lighter_support"
        df_new.loc[mode.isin([
            "pressure-regulated volume control", "assist control-volume control",
            "pressure control", "simv"
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
            "nasal cannula", "room air"
        ]), "resp_device_group"] = "low_intensity"
        df_new.loc[device.isin([
            "face mask", "high flow nc", "nippv", "cpap"
        ]), "resp_device_group"] = "moderate_intensity"
        df_new.loc[device.isin([
            "imv", "vent", "trach collar"
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

    local_df = build_resp_mode_dummies(local_df)
    local_df = build_resp_device_dummies(local_df)

    # Ensure all expected dummy columns exist (training site may have
    # categories absent in the validation site)
    for _prefix, _levels in [
        ("resp_mode", ["controlled_or_mandatory_ventilation",
                       "spontaneous_or_lighter_support", "unknown_or_other"]),
        ("resp_device", ["high_intensity", "low_intensity",
                         "moderate_intensity", "unknown"]),
    ]:
        for _lvl in _levels:
            _cname = f"{_prefix}_{_lvl}"
            if _cname not in local_df.columns:
                local_df[_cname] = 0

    local_df = add_delta_features(local_df)

    # ── Compute rewards (same as training) ──
    def compute_rewards(df_in, cpc_map=None):
        if cpc_map is None:
            cpc_map = {"CPC1_2": 100.0, "CPC3": 40.0, "CPC4": -40.0, "CPC5": -100.0}
        df_out = df_in.sort_values(["hospitalization_id", "time_bucket"]).copy()
        lactate_prev = df_out.groupby("hospitalization_id")["lab_lactate"].shift(1)
        lactate_prev = lactate_prev.fillna(df_out["lab_lactate"])
        map_clipped = np.clip(df_out["vital_map"], 55.0, 75.0)
        df_out["reward_intermediate"] = (
            1.0 * np.tanh((map_clipped - 65.0) / 10.0)
            + 0.5 * np.tanh((lactate_prev - df_out["lab_lactate"]) / 2.0)
            - 0.2 * np.tanh(df_out["med_cont_nee"] / 0.2)
        )
        df_out["reward_intermediate"] = (
            df_out["reward_intermediate"] - df_out["reward_intermediate"].mean()
        )
        df_out["reward_terminal"] = 0.0
        is_terminal = (
            df_out.groupby("hospitalization_id")["time_bucket"].transform("max")
            == df_out["time_bucket"]
        )
        df_out.loc[is_terminal, "reward_terminal"] = (
            df_out.loc[is_terminal, "cpc"].map(cpc_map).fillna(0.0)
        )
        df_out["reward"] = df_out["reward_intermediate"] + df_out["reward_terminal"]
        return df_out

    local_df = compute_rewards(local_df)

    _n_patients = local_df.hospitalization_id.nunique()
    mo.md(
        f"**Local data loaded:** {_n_patients:,} patients, "
        f"{len(local_df):,} hourly rows (vasopressor patients only)"
    )

    return local_df, build_demo_features, hosp_summary


# ── Cell 3: Apply External Standardization ───────────────────────────
@app.cell
def _(mo, np, local_df, ext_state_features, ext_preprocessor):
    # Apply the TRAINING site's standardization (not local)
    _df_ext = local_df.copy()

    # Check for missing columns
    _missing = [c for c in ext_state_features if c not in _df_ext.columns]
    if _missing:
        raise ValueError(f"Local data missing state features: {_missing}")

    # Standardize continuous features using external preprocessor
    for _col in ext_preprocessor["continuous_features"]:
        _mu = ext_preprocessor["mean"][_col]
        _sd = ext_preprocessor["std"][_col]
        _df_ext[_col] = ((_df_ext[_col] - _mu) / _sd).astype(np.float32)

    # Cast binary features to float32
    for _col in ext_preprocessor["binary_features"]:
        _df_ext[_col] = _df_ext[_col].astype(np.float32)

    _df_ext[ext_state_features] = _df_ext[ext_state_features].astype(np.float32)

    # Keep raw copy for mask construction
    local_df_raw = local_df.copy()
    local_df_proc = _df_ext.copy()

    mo.md(
        f"**External standardization applied.**\n\n"
        f"Using mean/std from training site "
        f"({len(ext_preprocessor['continuous_features'])} continuous, "
        f"{len(ext_preprocessor['binary_features'])} binary features)"
    )

    return local_df_raw, local_df_proc


# ── Cell 4: Build Transitions ────────────────────────────────────────
@app.cell
def _(
    mo, np, pd, torch, Dataset, DataLoader,
    local_df_proc, local_df_raw, ext_state_features,
):
    # Action constants (PI's encoding)
    ACTION_INCREASE = 0
    ACTION_DECREASE = 1
    ACTION_STOP = 2
    ACTION_STAY = 3

    def build_action_mask(map_t, nee_t, nee_min=0.0, nee_max=1.1):
        if nee_t <= nee_min:
            return np.array([1, 0, 0, 1], dtype=np.int8)
        if map_t < 55:
            return np.array([1, 0, 0, 1], dtype=np.int8)
        if map_t > 90:
            return np.array([0, 1, 1, 1], dtype=np.int8)
        if nee_t >= nee_max:
            return np.array([0, 1, 0, 1], dtype=np.int8)
        return np.array([1, 1, 1, 1], dtype=np.int8)

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
            df_state.groupby(id_col)[feats].shift(-1).fillna(0.0).astype(np.float32)
        )

        raw_map = df_mask_raw[raw_map_col].to_numpy()
        raw_nee = df_mask_raw[raw_nee_col].to_numpy()
        next_raw_map = df_mask_raw.groupby(id_col)[raw_map_col].shift(-1).fillna(0.0).to_numpy()
        next_raw_nee = df_mask_raw.groupby(id_col)[raw_nee_col].shift(-1).fillna(0.0).to_numpy()

        masks = np.stack([
            build_action_mask(map_t=m, nee_t=n, nee_min=nee_min, nee_max=nee_max)
            for m, n in zip(raw_map, raw_nee)
        ]).astype(np.int8)

        next_masks = np.stack([
            build_action_mask(map_t=m, nee_t=n, nee_min=nee_min, nee_max=nee_max)
            for m, n in zip(next_raw_map, next_raw_nee)
        ]).astype(np.int8)

        next_masks[done.to_numpy() == 1] = np.array([1, 1, 1, 1], dtype=np.int8)

        base_df = pd.DataFrame({
            id_col: df_state[id_col].to_numpy(),
            time_col: df_state[time_col].to_numpy(),
            action_col: df_state[action_col].to_numpy(),
            reward_col: df_state[reward_col].to_numpy(),
            "done": done.to_numpy(),
        }, index=df_state.index)

        state_block = df_state[feats].astype(np.float32).copy()
        state_block.columns = [f"s_{c}" for c in feats]

        next_state_block = next_state_df.copy()
        next_state_block.columns = [f"ns_{c}" for c in feats]

        mask_block = pd.DataFrame({
            "mask_increase": masks[:, ACTION_INCREASE],
            "mask_decrease": masks[:, ACTION_DECREASE],
            "mask_stop": masks[:, ACTION_STOP],
            "mask_stay": masks[:, ACTION_STAY],
            "next_mask_increase": next_masks[:, ACTION_INCREASE],
            "next_mask_decrease": next_masks[:, ACTION_DECREASE],
            "next_mask_stop": next_masks[:, ACTION_STOP],
            "next_mask_stay": next_masks[:, ACTION_STAY],
        }, index=df_state.index)

        out = pd.concat([base_df, state_block, next_state_block, mask_block], axis=1)
        return out.copy()

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

    # Build transitions on full local data (no train/test split)
    transition_all = build_transition_dataframe(
        df_state=local_df_proc, df_mask_raw=local_df_raw,
        feats=ext_state_features, reward_col="reward", action_col="action",
        nee_min=0.0, nee_max=1.1,
    )

    all_batch = extract_numpy_batches(
        transition_df=transition_all, feats=ext_state_features,
        action_col="action", reward_col="reward",
    )

    all_dataset = OfflineRLDataset(all_batch)
    all_loader = DataLoader(
        all_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False,
    )

    mo.md(f"**Transitions built:** {len(transition_all):,} total")

    return transition_all, all_loader, local_df_raw


# ── Cell 5: Load Model & Predict ─────────────────────────────────────
@app.cell
def _(mo, np, pd, torch, nn, all_loader, ext_training_config, shared_dir):
    # ── QNetwork (same architecture as training) ──
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

    def apply_action_mask(q_values, action_masks, invalid_penalty=-1e9):
        masked_q = q_values.clone()
        masked_q[action_masks <= 0] = invalid_penalty
        return masked_q

    # Initialize model with training config
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _hidden = tuple(ext_training_config["hidden_dims"])
    _n_actions = ext_training_config["n_actions"]
    _state_dim = ext_training_config["state_dim"]

    ext_model = QNetwork(
        state_dim_val=_state_dim, n_actions=_n_actions,
        hidden_dims=_hidden, dropout=0.0,
    ).to(_device)

    # Load trained weights
    _ckpt = torch.load(shared_dir / "best_model.pt", map_location=_device, weights_only=False)
    ext_model.load_state_dict(_ckpt["model_state_dict"])
    ext_model.eval()

    print(f"Loaded model from epoch {_ckpt.get('epoch', '?')}")

    # ── Predict actions ──
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
        return {
            "pred_actions": np.concatenate(pred_actions),
            "pred_q_values": np.concatenate(pred_q_values),
            "observed_actions": np.concatenate(observed_actions),
            "rewards": np.concatenate(rewards),
            "dones": np.concatenate(dones),
        }

    def summarize_predictions(pred_dict):
        pred_actions = pred_dict["pred_actions"]
        observed_actions = pred_dict["observed_actions"]
        pred_counts = pd.Series(pred_actions).value_counts().sort_index()
        obs_counts = pd.Series(observed_actions).value_counts().sort_index()
        summary_df = pd.DataFrame({
            "pred_count": pred_counts,
            "obs_count": obs_counts,
        }).fillna(0).astype(int)
        action_name_map = {0: "increase", 1: "decrease", 2: "stop", 3: "stay"}
        summary_df["action_name"] = summary_df.index.map(action_name_map)
        agreement = float((pred_actions == observed_actions).mean())
        return summary_df, agreement

    ext_pred = predict_actions(loader=all_loader, online_net=ext_model, device=_device)
    ext_summary_df, ext_agreement = summarize_predictions(ext_pred)

    mo.md(
        f"**External model predictions:**\n\n"
        f"- Action agreement: {ext_agreement:.1%}\n\n"
        f"{ext_summary_df.to_markdown()}"
    )

    return ext_pred, ext_summary_df, ext_agreement, _device


# ── Cell 6: Agreement-Outcome Evaluation ─────────────────────────────
@app.cell
def _(
    mo, np, pd, json,
    transition_all, ext_pred, local_df_raw,
    ext_agreement, ext_summary_df,
    eval_dir, site_name, ext_training_config,
    hosp_summary,
):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    def build_patient_agreement_dataset(
        trans_df, t_pred, raw_df,
        id_col="hospitalization_id", time_col="time_bucket",
    ):
        df_eval = trans_df.copy().reset_index(drop=True)
        if len(df_eval) != len(t_pred["pred_actions"]):
            raise ValueError("transition_df and pred lengths do not match.")
        df_eval["pred_action"] = t_pred["pred_actions"]
        df_eval["agree"] = (df_eval["pred_action"] == df_eval["action"]).astype(int)
        outcome_df = (
            raw_df.sort_values([id_col, time_col])
            .groupby(id_col).tail(1)[[id_col, "cpc"]].copy()
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

    def fit_ordinal_agreement_model(patient_df):
        X = patient_df[["agreement_rate"]].copy()
        y = patient_df["cpc_ord_good"].copy()
        model = OrderedModel(endog=y, exog=X, distr="logit")
        result = model.fit(method="bfgs", disp=False)
        return result

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
        return pd.DataFrame({
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

    def agreement_bin_summary(patient_df):
        out = patient_df.copy()
        out["agreement_bin"] = pd.cut(
            out["agreement_rate"],
            bins=[0.0, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
            include_lowest=True,
        )
        return (
            out.groupby("agreement_bin", observed=False)
            .agg(
                n_hosp=("hospitalization_id", "size"),
                mean_agreement=("agreement_rate", "mean"),
                mean_cpc_ord_good=("cpc_ord_good", "mean"),
            )
            .reset_index()
        )

    # Run evaluation
    ext_patient_df = build_patient_agreement_dataset(
        transition_all, ext_pred, local_df_raw,
    )

    ext_result = fit_ordinal_agreement_model(ext_patient_df)
    ext_coef_summary = summarize_ordinal_result(ext_result)
    ext_bin_summary = agreement_bin_summary(ext_patient_df)

    # Save unadjusted results (unchanged)
    ext_summary_df.to_csv(eval_dir / "action_summary.csv")
    ext_coef_summary.to_csv(eval_dir / "coef_summary.csv", index=False)
    ext_bin_summary.to_csv(eval_dir / "bin_summary.csv", index=False)

    # ── Adjusted Concordance Models ──────────────────────────────────
    def summarize_model(result, predictor_names):
        """Extract coef, SE, p-value, OR [95% CI] for each predictor."""
        rows = []
        for name in predictor_names:
            _coef = result.params[name]
            _se = result.bse[name]
            _pval = result.pvalues[name]
            _or = np.exp(_coef)
            _ci_lo = np.exp(_coef - 1.96 * _se)
            _ci_hi = np.exp(_coef + 1.96 * _se)
            if name == "agreement_rate":
                _or_10pp = np.exp(_coef * 0.10)
                _or_10pp_lo = np.exp((_coef - 1.96 * _se) * 0.10)
                _or_10pp_hi = np.exp((_coef + 1.96 * _se) * 0.10)
            else:
                _or_10pp = np.nan
                _or_10pp_lo = np.nan
                _or_10pp_hi = np.nan
            rows.append({
                "term": name,
                "coef": round(_coef, 4),
                "std_err": round(_se, 4),
                "p_value": _pval,
                "odds_ratio": round(_or, 4),
                "or_95ci_low": round(_ci_lo, 4),
                "or_95ci_high": round(_ci_hi, 4),
                "or_10pp": round(_or_10pp, 4) if not np.isnan(_or_10pp) else np.nan,
                "or_10pp_ci_low": round(_or_10pp_lo, 4) if not np.isnan(_or_10pp_lo) else np.nan,
                "or_10pp_ci_high": round(_or_10pp_hi, 4) if not np.isnan(_or_10pp_hi) else np.nan,
            })
        return pd.DataFrame(rows)

    # Merge baseline covariates for adjusted analysis
    _adj_covars = [
        "hospitalization_id", "age_at_admission", "sex_category",
        "sofa_0_24", "max_nee", "max_lactate",
    ]
    _adj_df = ext_patient_df.merge(
        hosp_summary[_adj_covars], on="hospitalization_id", how="left",
    )
    _adj_df["female"] = (_adj_df["sex_category"].str.lower() == "female").astype(int)
    for _col in ["age_at_admission", "sofa_0_24", "max_nee", "max_lactate"]:
        _adj_df[_col] = pd.to_numeric(_adj_df[_col], errors="coerce").astype("float64")
    _adj_df["log_max_nee"] = np.log(_adj_df["max_nee"] + 0.01)
    _adj_df["log_max_lactate"] = np.log(_adj_df["max_lactate"] + 0.01)
    _adj_df = _adj_df.dropna(subset=["cpc_ord_good", "age_at_admission", "sofa_0_24"])
    _y_adj = _adj_df["cpc_ord_good"]

    print(f"\nAdjusted analysis sample: {len(_adj_df)} patients "
          f"(dropped {len(ext_patient_df) - len(_adj_df)} with missing covariates)")

    # M1: Unadjusted (re-fit on adjusted sample for fair comparison)
    _X1 = _adj_df[["agreement_rate"]]
    _res1 = OrderedModel(endog=_y_adj, exog=_X1, distr="logit").fit(method="bfgs", disp=False)
    _s1 = summarize_model(_res1, ["agreement_rate"])
    _s1["model"] = "M1: Unadjusted"

    # M2: Adjusted — core confounders (age, sex, admission SOFA)
    _preds2 = ["agreement_rate", "age_at_admission", "female", "sofa_0_24"]
    _X2 = _adj_df[_preds2]
    _res2 = OrderedModel(endog=_y_adj, exog=_X2, distr="logit").fit(method="bfgs", disp=False)
    _s2 = summarize_model(_res2, _preds2)
    _s2["model"] = "M2: Adjusted (core)"

    # M3: Adjusted — extended (+ log lactate, log NEE)
    _preds3 = _preds2 + ["log_max_lactate", "log_max_nee"]
    _X3 = _adj_df[_preds3]
    _res3 = OrderedModel(endog=_y_adj, exog=_X3, distr="logit").fit(method="bfgs", disp=False)
    _s3 = summarize_model(_res3, _preds3)
    _s3["model"] = "M3: Adjusted (extended)"

    _adj_all = pd.concat([_s1, _s2, _s3], ignore_index=True)
    _adj_all.to_csv(eval_dir / "adjusted_coef_summary.csv", index=False)

    # Extract adjusted agreement_rate ORs for display
    _adj_agree = _adj_all[_adj_all["term"] == "agreement_rate"].copy()

    print("\n── Adjusted Concordance Results ──")
    for _, _r in _adj_agree.iterrows():
        _p = "p < 0.001" if _r["p_value"] < 0.001 else f"p = {_r['p_value']:.4f}"
        print(f"  {_r['model']}: OR/10pp = {_r['or_10pp']:.3f} "
              f"[{_r['or_10pp_ci_low']:.3f}, {_r['or_10pp_ci_high']:.3f}], {_p}")

    # Save metadata file (includes adjusted results)
    _m2_agree = _adj_agree[_adj_agree["model"] == "M2: Adjusted (core)"].iloc[0]
    _eval_meta = {
        "local_site": site_name,
        "training_site": ext_training_config.get("site_name", "unknown"),
        "n_patients": int(ext_patient_df.hospitalization_id.nunique()),
        "n_patients_adjusted": int(len(_adj_df)),
        "n_transitions": len(transition_all),
        "overall_agreement": float(ext_agreement),
        "ordinal_coef": float(ext_coef_summary["coef"].iloc[0]),
        "ordinal_or": float(ext_coef_summary["odds_ratio"].iloc[0]),
        "ordinal_pvalue": float(ext_coef_summary["p_value"].iloc[0]),
        "adjusted_core_or_10pp": float(_m2_agree["or_10pp"]),
        "adjusted_core_pvalue": float(_m2_agree["p_value"]),
    }
    with open(eval_dir / "evaluation_metadata.json", "w") as _f:
        json.dump(_eval_meta, _f, indent=2)

    print(ext_result.summary())

    mo.md(
        f"**External Validation Results:**\n\n"
        f"Training site: {ext_training_config.get('site_name', 'unknown')} → "
        f"Validation site: {site_name}\n\n"
        f"**Overall agreement:** {ext_agreement:.1%}\n\n"
        f"**Ordinal Logistic Regression (Unadjusted):**\n\n"
        f"{ext_coef_summary.to_markdown(index=False)}\n\n"
        f"**Adjusted Concordance (agreement_rate OR/10pp):**\n\n"
        f"{_adj_agree[['model', 'or_10pp', 'or_10pp_ci_low', 'or_10pp_ci_high', 'p_value']].to_markdown(index=False)}\n\n"
        f"**Agreement Bin Summary:**\n\n"
        f"{ext_bin_summary.to_markdown(index=False)}\n\n"
        f"Results saved to `{eval_dir}`"
    )

    return ext_patient_df, ext_coef_summary, ext_bin_summary


# ── Cell 7: External Validation Figures ───────────────────────────────
@app.cell
def _(
    mo, np, pd, os,
    transition_all, ext_pred, ext_patient_df, ext_coef_summary, ext_bin_summary,
    ext_summary_df, eval_dir, site_name,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import seaborn as sns

    _fig_dir = eval_dir / "figures"
    os.makedirs(_fig_dir, exist_ok=True)

    # ── Style + constants ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.2,
    })

    _ACTION_LABELS = {0: "Increase", 1: "Decrease", 2: "Stop", 3: "Stay"}
    _ACTION_COLORS = {
        0: "#E53935",   # red — escalation
        1: "#42A5F5",   # blue — de-escalation
        2: "#66BB6A",   # green — liberation
        3: "#F8BBD0",   # light pink — maintenance
    }
    _CPC_COLORS = {
        "CPC1_2": "#22c55e",
        "CPC3": "#facc15",
        "CPC4": "#f97316",
        "CPC5": "#ef4444",
    }

    def _save_fig(fig, name):
        fig.savefig(_fig_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(_fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Fig 1: Action Distribution (diverging bar chart) ──
    _actions = sorted(ext_summary_df.index.tolist())
    _pred_counts = [int(ext_summary_df.loc[_a, "pred_count"]) for _a in _actions]
    _obs_counts = [int(ext_summary_df.loc[_a, "obs_count"]) for _a in _actions]
    _labels = [_ACTION_LABELS[_a] for _a in _actions]
    _colors = [_ACTION_COLORS[_a] for _a in _actions]

    _fig1, _ax1 = plt.subplots(figsize=(10, 5))
    _y = np.arange(len(_actions))
    _ax1.barh(_y + 0.15, _pred_counts, height=0.3, color=_colors, edgecolor="black",
              linewidth=0.5, label="RL Policy")
    _ax1.barh(_y - 0.15, _obs_counts, height=0.3, color=_colors, edgecolor="black",
              linewidth=0.5, alpha=0.5, label="Clinician")
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels(_labels)
    _ax1.set_xlabel("Number of Transitions")
    _ax1.set_title(f"Action Distribution: RL vs Clinician — {site_name}", fontweight="bold")
    _ax1.legend(loc="lower right")
    _save_fig(_fig1, "fig_action_distribution")

    # ── Fig 2: Patient Timeline Heatmaps ──
    _df_eval = transition_all.copy().reset_index(drop=True)
    _df_eval["pred_action"] = ext_pred["pred_actions"]

    # Merge CPC outcomes
    _cpc_map = ext_patient_df[["hospitalization_id", "cpc"]].drop_duplicates()
    _df_eval = _df_eval.merge(_cpc_map, on="hospitalization_id", how="left")

    # Sample patients with at least 12h trajectories
    _traj_len = _df_eval.groupby("hospitalization_id").size().reset_index(name="n_steps")
    _traj_len = _traj_len[_traj_len["n_steps"] >= 12]
    _sample_ids = _traj_len.sample(n=min(100, len(_traj_len)), random_state=42)
    _sample_ids = _sample_ids.sort_values("n_steps", ascending=False)["hospitalization_id"].tolist()

    if len(_sample_ids) > 0:
        _max_t = int(_df_eval["time_bucket"].max()) + 1
        _n_patients = len(_sample_ids)

        _physician_matrix = np.full((_n_patients, _max_t), np.nan)
        _rl_matrix = np.full((_n_patients, _max_t), np.nan)
        _cpc_list = []
        _last_bucket = []

        for _i, _pid in enumerate(_sample_ids):
            _patient = _df_eval[_df_eval["hospitalization_id"] == _pid].sort_values("time_bucket")
            for _, _row in _patient.iterrows():
                _t = int(_row["time_bucket"])
                if _t < _max_t:
                    _physician_matrix[_i, _t] = _row["action"]
                    _rl_matrix[_i, _t] = _row["pred_action"]
            _cpc_list.append(_patient["cpc"].iloc[0] if len(_patient) > 0 else "unknown")
            _last_bucket.append(int(_patient["time_bucket"].max()))

        _cmap = ListedColormap([_ACTION_COLORS[_a] for _a in range(4)])
        _norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

        _fig2, (_ax2a, _ax2b) = plt.subplots(2, 1, figsize=(14, 10))

        for _ax, _matrix, _title in [
            (_ax2a, _physician_matrix, "Clinician Actions"),
            (_ax2b, _rl_matrix, "RL Recommended Actions"),
        ]:
            _ax.imshow(_matrix, aspect="auto", cmap=_cmap, norm=_norm,
                       interpolation="nearest", origin="upper")
            for _i, (_cpc, _lb) in enumerate(zip(_cpc_list, _last_bucket)):
                _color = _CPC_COLORS.get(_cpc, "#888888")
                _ax.plot(_lb, _i, marker="D", color=_color, markersize=3,
                         markeredgecolor="black", markeredgewidth=0.3)
            _ax.set_ylabel("Patients (sorted by trajectory length)")
            _ax.set_title(_title, fontsize=12, fontweight="bold")
            _ax.set_xlim(-0.5, min(_max_t, 120) - 0.5)
            for _h in range(24, 121, 24):
                _ax.axvline(_h - 0.5, color="white", linewidth=0.5, alpha=0.5)

        _ax2b.set_xlabel("Hours Since First Event")

        # Legend at the bottom
        _action_patches = [mpatches.Patch(color=_ACTION_COLORS[_a], label=_ACTION_LABELS[_a]) for _a in range(4)]
        _cpc_markers = [plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=_CPC_COLORS[c],
                                    markersize=6, label=c, markeredgecolor="black", markeredgewidth=0.5)
                        for c in ["CPC1_2", "CPC3", "CPC4", "CPC5"]]
        _all_handles = _action_patches + [mpatches.Patch(color="none", label="")] + _cpc_markers
        _fig2.legend(handles=_all_handles, loc="lower center",
                     ncol=len(_all_handles), frameon=True, fontsize=8,
                     bbox_to_anchor=(0.5, -0.04))

        _fig2.suptitle(f"Patient Action Timelines — {site_name}", fontsize=14, y=1.01)
        _fig2.tight_layout()
        _save_fig(_fig2, "fig_patient_timelines")

    # ── Fig 3: Agreement-Outcome Bar Chart ──
    _n_bins = len(ext_bin_summary)
    _bin_labels = ext_bin_summary["agreement_bin"].tolist()
    _cpc_means = ext_bin_summary["mean_cpc_ord_good"].tolist()
    _n_hosps = ext_bin_summary["n_hosp"].tolist()

    _fig3, _ax3 = plt.subplots(figsize=(8, 5))
    _bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, _n_bins))
    _bars = _ax3.bar(range(_n_bins), _cpc_means, color=_bar_colors, edgecolor="black", linewidth=0.5)
    for _j, (_bar, _n) in enumerate(zip(_bars, _n_hosps)):
        _ax3.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.02,
                  f"n={_n}", ha="center", fontsize=9)
    _ax3.set_xticks(range(_n_bins))
    _ax3.set_xticklabels(_bin_labels)
    _ax3.set_xlabel("RL-Clinician Agreement Bin")
    _ax3.set_ylabel("Mean CPC Ordinal (higher = better)")
    _ax3.set_title(f"Agreement vs Outcome — {site_name}", fontweight="bold")
    _ax3.set_ylim(0, max(_cpc_means) * 1.2)
    _save_fig(_fig3, "fig_agreement_outcome")

    # ── Fig 4: Concordance OR Forest Plot ──
    _or_10pp = float(ext_coef_summary["or_10pp"].iloc[0])
    _or_low = float(ext_coef_summary["or_10pp_ci_low"].iloc[0])
    _or_high = float(ext_coef_summary["or_10pp_ci_high"].iloc[0])
    _pval = float(ext_coef_summary["p_value"].iloc[0])

    _fig4, _ax4 = plt.subplots(figsize=(8, 3))
    _ax4.errorbar(_or_10pp, 0, xerr=[[_or_10pp - _or_low], [_or_high - _or_10pp]],
                  fmt="D", color="#1976D2", markersize=10, capsize=6, linewidth=2, capthick=2)
    _ax4.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    _ax4.set_yticks([0])
    _ax4.set_yticklabels([site_name])
    _ax4.set_xlabel("Odds Ratio per 10pp Agreement Increase")
    _ax4.set_title(f"Concordance OR — {site_name} (p={_pval:.4f})", fontweight="bold")
    _ax4.set_xlim(max(0.5, _or_low - 0.2), _or_high + 0.3)
    _ax4.text(_or_10pp, 0.15, f"{_or_10pp:.2f} [{_or_low:.2f}, {_or_high:.2f}]",
              ha="center", fontsize=9, fontweight="bold")
    _save_fig(_fig4, "fig_concordance_or")

    # ── Fig 5: Action Confusion Matrix ──
    _observed = ext_pred["observed_actions"]
    _predicted = ext_pred["pred_actions"]
    _n_actions = 4
    _conf = np.zeros((_n_actions, _n_actions), dtype=int)
    for _o, _p in zip(_observed, _predicted):
        _conf[int(_o), int(_p)] += 1

    # Normalize by row for percentages
    _conf_pct = _conf / _conf.sum(axis=1, keepdims=True) * 100

    _fig5, _ax5 = plt.subplots(figsize=(7, 6))
    sns.heatmap(_conf_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=[_ACTION_LABELS[_a] for _a in range(_n_actions)],
                yticklabels=[_ACTION_LABELS[_a] for _a in range(_n_actions)],
                ax=_ax5, vmin=0, vmax=100, cbar_kws={"label": "% of Clinician Action"})
    # Add raw counts in smaller text
    for _r in range(_n_actions):
        for _c in range(_n_actions):
            _ax5.text(_c + 0.5, _r + 0.72, f"(n={_conf[_r, _c]})",
                      ha="center", va="center", fontsize=7, color="gray")
    _ax5.set_xlabel("RL Recommended Action")
    _ax5.set_ylabel("Clinician Action")
    _ax5.set_title(f"Action Confusion Matrix — {site_name}", fontweight="bold")
    _save_fig(_fig5, "fig_action_confusion_matrix")

    mo.md(
        f"### External Validation Figures\n\n"
        f"Saved 5 figures to `{_fig_dir}/`:\n\n"
        f"1. Action distribution (RL vs clinician)\n"
        f"2. Patient timeline heatmaps\n"
        f"3. Agreement-outcome bar chart\n"
        f"4. Concordance OR forest plot\n"
        f"5. Action confusion matrix"
    )
    return


if __name__ == "__main__":
    app.run()
