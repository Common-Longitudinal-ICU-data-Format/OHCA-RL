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
    _hosp_summary = pd.read_parquet(out_dir / "hospitalization_summary.parquet")

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

    _pa = build_demo_features(_hosp_summary)
    _pa = _pa[["hospitalization_id"] + [c for c in _pa.columns if c.startswith("demo_")]]

    local_df = _bucketed_df.merge(_pa, on="hospitalization_id", how="left")

    # Filter to vasopressor patients
    local_df = local_df.query("ever_vaso == 1").copy()

    # Remap actions to PI's encoding
    _remap = {int(k): int(v) for k, v in ext_action_remap["pipeline_to_pi"].items()}
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

    return local_df, build_demo_features


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
        return pd.DataFrame({
            "term": [predictor_name],
            "coef": [coef],
            "std_err": [se],
            "p_value": [pval],
            "odds_ratio": [or_value],
            "or_95ci_low": [ci_low],
            "or_95ci_high": [ci_high],
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

    # Save all results
    ext_summary_df.to_csv(eval_dir / "action_summary.csv")
    ext_coef_summary.to_csv(eval_dir / "coef_summary.csv", index=False)
    ext_bin_summary.to_csv(eval_dir / "bin_summary.csv", index=False)

    # Save a metadata file
    _eval_meta = {
        "local_site": site_name,
        "training_site": ext_training_config.get("site_name", "unknown"),
        "n_patients": int(ext_patient_df.hospitalization_id.nunique()),
        "n_transitions": len(transition_all),
        "overall_agreement": float(ext_agreement),
        "ordinal_coef": float(ext_coef_summary["coef"].iloc[0]),
        "ordinal_or": float(ext_coef_summary["odds_ratio"].iloc[0]),
        "ordinal_pvalue": float(ext_coef_summary["p_value"].iloc[0]),
    }
    with open(eval_dir / "evaluation_metadata.json", "w") as _f:
        json.dump(_eval_meta, _f, indent=2)

    print(ext_result.summary())

    mo.md(
        f"**External Validation Results:**\n\n"
        f"Training site: {ext_training_config.get('site_name', 'unknown')} → "
        f"Validation site: {site_name}\n\n"
        f"**Overall agreement:** {ext_agreement:.1%}\n\n"
        f"**Ordinal Logistic Regression:**\n\n"
        f"{ext_coef_summary.to_markdown(index=False)}\n\n"
        f"**Agreement Bin Summary:**\n\n"
        f"{ext_bin_summary.to_markdown(index=False)}\n\n"
        f"Results saved to `{eval_dir}`"
    )

    return


if __name__ == "__main__":
    app.run()
