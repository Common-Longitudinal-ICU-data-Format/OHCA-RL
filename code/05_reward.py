# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ohca-rl (3.11.11)
#     language: python
#     name: python3
# ---

# %%
import sys, json, logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# ── Single source of truth: config/config.json ──
# Resolves relative to current working dir; sites override paths in their own config.json.
_config_path = Path("config/config.json")
if not _config_path.exists():
    # Fallback: search upward from current notebook location
    for p in Path.cwd().parents:
        if (p / "config" / "config.json").exists():
            _config_path = p / "config" / "config.json"
            break

with open(_config_path) as f:
    site_config = json.load(f)

SITE_NAME    = site_config["site_name"]
TABLES_PATH  = site_config["tables_path"]
FILE_TYPE    = site_config["file_type"]
TIMEZONE     = site_config["timezone"]
PROJECT_ROOT = Path(site_config["project_root"])

CODE_DIR     = PROJECT_ROOT / "code"
CONFIG_DIR   = PROJECT_ROOT / "config"
OUT_DIR      = PROJECT_ROOT / "output" / "intermediate"
MODEL_DIR    = PROJECT_ROOT / "output" / "model"
FINAL_DIR    = PROJECT_ROOT / "output" / "final"

for _d in (OUT_DIR, MODEL_DIR, FINAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(f"Site: {SITE_NAME}  |  tables: {TABLES_PATH}")

from utils import init_log_capture
init_log_capture(__file__, PROJECT_ROOT)


# Variable config
def deep_merge(base, override):
    out = dict(base) if base else {}
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if (k in out and isinstance(out[k], dict)
                                            and isinstance(v, dict)) else v
    return out

with open(CONFIG_DIR / "ohca_rl_config.yaml") as f:
    _base = yaml.safe_load(f)
_local_path = CONFIG_DIR / "ohca_rl_config_local.yaml"
_local = (yaml.safe_load(open(_local_path)) or {}) if _local_path.exists() else {}
ohca_config = deep_merge(_base, _local)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("04_reward")

# Load notebook 03 artifacts
bucketed_dp    = pd.read_parquet(OUT_DIR / "bucketed_decision_points_reviewed.parquet")
bucketed_full  = pd.read_parquet(OUT_DIR / "bucketed_train_reviewed.parquet")
patient_static = pd.read_parquet(OUT_DIR / "patient_static.parquet")
rl_cohort      = pd.read_parquet(OUT_DIR / "rl_cohort_reviewed.parquet")
cohort_df      = pd.read_parquet(OUT_DIR / "cohort_ohca_icu.parquet")

for _df in (bucketed_dp, bucketed_full, patient_static, rl_cohort, cohort_df):
    if "hospitalization_id" in _df.columns:
        _df["hospitalization_id"] = _df["hospitalization_id"].astype(str)

print(f"Loaded artifacts:")
print(f"  bucketed_decision_points : {len(bucketed_dp):,} rows × {bucketed_dp.shape[1]} cols")
print(f"  bucketed_train (full)    : {len(bucketed_full):,} rows × {bucketed_full.shape[1]} cols")
print(f"  rl_cohort                : {len(rl_cohort):,} patients")
print(f"  patient_static           : {len(patient_static):,} patients")
print(f"  cohort_df                : {len(cohort_df):,} OHCA-ICU patients")

# Verify we have the columns we need
_required = ["hospitalization_id", "hour", "action_tier", "in_decision_window",
             "vital_map", "lab_lactate", "window_close_reason"]
_missing = [c for c in _required if c not in bucketed_dp.columns]
if _missing:
    print(f"\n⚠️  MISSING columns in bucketed_dp: {_missing}")
else:
    print(f"\n✓ All required columns present")

# Quick cohort summary
print(f"\nDecision point cohort:")
print(f"  Patients : {bucketed_dp['hospitalization_id'].nunique():,}")
print(f"  Total DPs: {len(bucketed_dp):,}")
print(f"  Median DPs/patient: {rl_cohort['n_decision_points'].median():.0f}")
print(f"  IQR: [{rl_cohort['n_decision_points'].quantile(.25):.0f}, "
      f"{rl_cohort['n_decision_points'].quantile(.75):.0f}]")

# %%
# ── 4-tier disposition mapping ──
# Mapping MIMIC discharge_category strings to CPC-equivalent tiers.
# Substring matching for site portability (MIMIC has parenthetical suffixes like
# "skilled nursing facility (snf)" — other CLIF sites may use plain "snf").
# Order matters: more specific tiers should match BEFORE more general ones.

DISPOSITION_MAPPING = {
    "CPC5": [  # Death (worst) — match first to avoid hospice being missed
        "expired",
        "hospice",
        "deceased",
        "died",
    ],
    "CPC4": [  # Severe disability, long-term care
        "skilled nursing",
        "snf",
        "long term care",
        "ltach",
        "long-term care",
    ],
    "CPC3": [  # Mild–moderate disability, rehab/dependent care
        "acute inpatient rehab",
        "rehabilitation",
        "rehab facility",
        "psychiatric",
        "acute care hospital",  # transfer to peer hospital — uncertain but functional
        "group home",
    ],
    "CPC1_2": [  # Good outcome, independent
        "home",
        "against medical advice",
        "ama",
        "jail"
    ],
}

def map_disposition_to_cpc(discharge_str):
    """Map a discharge_category string to its CPC tier via substring matching.
    
    Returns the first matching tier (priority order: CPC5 > CPC4 > CPC3 > CPC1_2).
    Returns None if no match.
    """
    if pd.isna(discharge_str):
        return None
    s = str(discharge_str).lower().strip()
    # Check tiers in priority order (worst-to-best so CPC5 matches before CPC1_2)
    for tier in ["CPC5", "CPC4", "CPC3", "CPC1_2"]:
        for pattern in DISPOSITION_MAPPING[tier]:
            if pattern.lower() in s:
                return tier
    return None

# Apply to all 1,457 OHCA-ICU patients
patient_static["cpc_tier"] = patient_static["discharge_category"].apply(map_disposition_to_cpc)

# ── Cohort-wide disposition distribution ──
print(f"Full OHCA-ICU cohort (n={len(patient_static):,}):")
_dist_full = patient_static["cpc_tier"].value_counts(dropna=False).reindex(
    ["CPC1_2", "CPC3", "CPC4", "CPC5", np.nan], fill_value=0
)
for tier, n in _dist_full.items():
    label = tier if pd.notna(tier) else "UNMAPPED"
    print(f"  {label:10s}  {n:>6,}  ({n/len(patient_static)*100:5.1f}%)")

# ── RL cohort disposition distribution ──
_rl_static = patient_static[patient_static["hospitalization_id"].isin(rl_cohort["hospitalization_id"])].copy()
print(f"\nRL cohort (n={len(_rl_static):,}):")
_dist_rl = _rl_static["cpc_tier"].value_counts(dropna=False).reindex(
    ["CPC1_2", "CPC3", "CPC4", "CPC5", np.nan], fill_value=0
)
for tier, n in _dist_rl.items():
    label = tier if pd.notna(tier) else "UNMAPPED"
    print(f"  {label:10s}  {n:>6,}  ({n/len(_rl_static)*100:5.1f}%)")

# ── Sanity check: show which raw discharge strings mapped to which tier ──
print(f"\nDischarge string → CPC tier (for verification):")
_check = (patient_static.groupby(["discharge_category", "cpc_tier"], dropna=False)
          .size().reset_index(name="n").sort_values("n", ascending=False))
print(_check.to_string(index=False))

# ── Exclude patients with unmapped discharge_category ──
# A NaN cpc_tier means we can't compute the terminal reward; one such patient is
# enough to poison FQE training to NaN downstream. We drop them here at the source
# (rather than guarding every downstream consumer) and emit a STROBE row so the
# exclusion is auditable.
_unmapped = patient_static[patient_static["cpc_tier"].isna()]
if len(_unmapped):
    print(f"\n⚠️  {len(_unmapped)} patient(s) have UNMAPPED discharge_category — dropping:")
    print(_unmapped["discharge_category"].value_counts().to_string())
    print("(Add the discharge categories to DISPOSITION_MAPPING above if they should be kept.)")

    _unmapped_ids = set(_unmapped["hospitalization_id"].astype(str))
    _n_dp_before = bucketed_dp["hospitalization_id"].astype(str).nunique()
    _n_full_before = bucketed_full["hospitalization_id"].astype(str).nunique()

    patient_static = patient_static[patient_static["cpc_tier"].notna()].copy()
    bucketed_dp    = bucketed_dp[~bucketed_dp["hospitalization_id"].astype(str).isin(_unmapped_ids)].copy()
    bucketed_full  = bucketed_full[~bucketed_full["hospitalization_id"].astype(str).isin(_unmapped_ids)].copy()
    rl_cohort      = rl_cohort[~rl_cohort["hospitalization_id"].astype(str).isin(_unmapped_ids)].copy()

    _n_dp_after = bucketed_dp["hospitalization_id"].astype(str).nunique()
    print(f"  RL cohort (decision-point rows): {_n_dp_before:,} → {_n_dp_after:,} patients")
    print(f"  RL cohort (full bucketed rows) : {_n_full_before:,} → {bucketed_full['hospitalization_id'].astype(str).nunique():,} patients")

# ── Append STROBE row reflecting the post-CPC-mapping cohort ──
_strobe_path = FINAL_DIR / "strobe_counts.csv"
if _strobe_path.exists():
    _strobe = pd.read_csv(_strobe_path)
    _site = _strobe["site"].iloc[0] if "site" in _strobe.columns and len(_strobe) else SITE_NAME
    _strobe = _strobe[~_strobe["counter"].str.startswith("6_")]  # idempotent
    _final_n = rl_cohort["hospitalization_id"].astype(str).nunique()
    _rl_n_pre = int(_strobe.loc[_strobe["counter"] == "5_rl_cohort_patients", "value"].iloc[0]) \
                if (_strobe["counter"] == "5_rl_cohort_patients").any() else _final_n
    _new_rows = pd.DataFrame([
        {"counter": "6_modeling_cohort_patients",   "value": _final_n,             "site": _site},
        {"counter": "6_excluded_unmapped_cpc",      "value": _rl_n_pre - _final_n, "site": _site},
    ])
    _strobe = pd.concat([_strobe, _new_rows], ignore_index=True)
    _strobe.to_csv(_strobe_path, index=False)
    print(f"Appended modeling-cohort STROBE rows → {_strobe_path}")

# Save the per-patient CPC tier for downstream use (no NaNs now)
patient_static[["hospitalization_id", "discharge_category", "cpc_tier"]].to_parquet(
    OUT_DIR / "patient_disposition_reviewed.parquet", index=False
)
print(f"\nSaved → {OUT_DIR / 'patient_disposition.parquet'}")

# %%
import math

# ── Step 1: Compute raw per-step intermediate reward components ──
# Using the spec formulation:
#   MAP component  = tanh((clip(MAP, 55, 75) - 65) / 10)
#   Lact component = tanh((Lactate_{t-1} - Lactate_t) / 2)
# Both are in [-1, +1] range BEFORE the weights are applied.
# We compute the raw signal here, then explore weights.

bucketed_dp = bucketed_dp.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)

# MAP component: capped to [55, 75], smoothly rewarded around target 65
MAP_TARGET    = 65.0
MAP_CLIP_LOW  = 55.0
MAP_CLIP_HIGH = 75.0
MAP_SLOPE     = 10.0

_map_clipped = bucketed_dp["vital_map"].clip(lower=MAP_CLIP_LOW, upper=MAP_CLIP_HIGH)
bucketed_dp["raw_map_component"] = np.tanh((_map_clipped - MAP_TARGET) / MAP_SLOPE)

# Lactate clearance component: delta from prior decision point
# (NaN where there's no prior — fill with 0 = no signal)
LACTATE_SCALE = 2.0
_prev_lactate = (bucketed_dp.groupby("hospitalization_id")["lab_lactate"]
                 .shift(1))
_lactate_delta = _prev_lactate - bucketed_dp["lab_lactate"]   # positive = clearance
bucketed_dp["raw_lactate_component"] = np.tanh(_lactate_delta / LACTATE_SCALE).fillna(0)

# ── Step 2: Summary stats on raw components ──
print(f"Raw component distributions (before weight scaling):")
print(f"\n  MAP component (target 65, smoothly rewarded between 55-75):")
print(bucketed_dp["raw_map_component"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
print(f"  ↑ Bounded in [-1, +1]. Positive = MAP above 65, negative = below.")

print(f"\n  Lactate component (cleared = positive):")
print(bucketed_dp["raw_lactate_component"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
print(f"  ↑ Bounded in [-1, +1]. Positive = lactate dropped, negative = rose.")

# ── Step 3: Per-trajectory undiscounted cumulative raw intermediate ──
# With weights w_map and w_lact, cumulative intermediate = sum over DPs.
# For now, use weights from the spec draft (w_map=0.15, w_lact=0.075).
W_MAP_INITIAL  = 0.15
W_LACT_INITIAL = 0.075

bucketed_dp["raw_intermediate"] = (
    W_MAP_INITIAL  * bucketed_dp["raw_map_component"] +
    W_LACT_INITIAL * bucketed_dp["raw_lactate_component"]
)

_traj_sum = (bucketed_dp.groupby("hospitalization_id")["raw_intermediate"]
             .sum().rename("traj_cum_intermediate"))
print(f"\nTrajectory cumulative intermediate (with initial weights w_map={W_MAP_INITIAL}, "
      f"w_lact={W_LACT_INITIAL}):")
print(_traj_sum.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(2).to_string())

# ── Step 4: Per-trajectory DISCOUNTED cumulative intermediate ──
# With γ=0.99 backwards from terminal step, the geometric series has effective horizon
# 1/(1-γ) = 100. With trajectory length L, the effective coefficient sum is
# sum_{t=0}^{L-1} γ^t = (1 - γ^L) / (1 - γ).
# A uniform reward r over L steps gives discounted return r·(1-γ^L)/(1-γ).
GAMMA = 0.99

def _discounted_sum(group):
    """Compute discounted intermediate sum from t=0 forward (not backwards from terminal)."""
    r = group["raw_intermediate"].values
    L = len(r)
    discounts = GAMMA ** np.arange(L)
    return (r * discounts).sum()

_traj_disc = bucketed_dp.groupby("hospitalization_id").apply(
    _discounted_sum, include_groups=False
).rename("traj_disc_intermediate")

print(f"\nTrajectory DISCOUNTED cumulative intermediate (γ={GAMMA}):")
print(_traj_disc.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(2).to_string())

# ── Step 5: Pair with terminal reward to see balance ──
patient_disp = patient_static[["hospitalization_id", "cpc_tier"]].copy()
TERMINAL = {"CPC1_2": 100.0, "CPC3": 40.0, "CPC4": -40.0, "CPC5": -100.0}

_traj_balance = (_traj_disc.reset_index()
                 .merge(patient_disp, on="hospitalization_id", how="left"))
_traj_balance["terminal"] = _traj_balance["cpc_tier"].map(TERMINAL)
# Discounted terminal: γ^L applied to terminal
_traj_lengths = bucketed_dp.groupby("hospitalization_id").size().rename("n_dp")
_traj_balance = _traj_balance.merge(_traj_lengths.reset_index(), on="hospitalization_id")
_traj_balance["discounted_terminal"] = (
    _traj_balance["terminal"] * (GAMMA ** _traj_balance["n_dp"])
)

print(f"\n{'='*70}")
print(f"TERMINAL vs INTERMEDIATE BALANCE per CPC tier (with initial weights)")
print(f"{'='*70}")
print(_traj_balance.groupby("cpc_tier").agg(
    n_patients=("hospitalization_id", "count"),
    mean_terminal=("terminal", "mean"),
    mean_disc_terminal=("discounted_terminal", "mean"),
    mean_intermediate=("traj_disc_intermediate", "mean"),
    median_intermediate=("traj_disc_intermediate", "median"),
).round(2).to_string())

# Ratio: |terminal| / |intermediate|
_traj_balance["abs_term"] = _traj_balance["discounted_terminal"].abs()
_traj_balance["abs_inter"] = _traj_balance["traj_disc_intermediate"].abs()
print(f"\n  Median |discounted terminal| : {_traj_balance['abs_term'].median():.1f}")
print(f"  Median |discounted intermediate|: {_traj_balance['abs_inter'].median():.2f}")
print(f"  Ratio (term/inter)             : "
      f"{_traj_balance['abs_term'].median() / _traj_balance['abs_inter'].median():.1f}x")

# ── Step 6: Coefficient sweep — show what different scalings give us ──
print(f"\n{'='*70}")
print(f"COEFFICIENT SWEEP — terminal/intermediate ratio at different weights")
print(f"(Target: terminal should be 5–10× larger than intermediate on average)")
print(f"{'='*70}")

# Raw normalized components (weights=1) for the sweep
bucketed_dp["raw_intermediate_unit"] = (
    bucketed_dp["raw_map_component"] +
    0.5 * bucketed_dp["raw_lactate_component"]  # keep 2:1 MAP-to-lactate ratio
)

def _unit_disc_sum(group):
    r = group["raw_intermediate_unit"].values
    L = len(r)
    return (r * (GAMMA ** np.arange(L))).sum()

_unit_disc = bucketed_dp.groupby("hospitalization_id").apply(
    _unit_disc_sum, include_groups=False
).rename("unit_disc_inter")

_median_unit_disc = abs(_unit_disc).median()
_median_disc_terminal = _traj_balance["abs_term"].median()

print(f"\nWith MAP/lactate weights set to (w_map, w_map/2) and γ={GAMMA}:")
print(f"Median |trajectory unit intermediate| (per unit weight): {_median_unit_disc:.3f}")
print(f"\n{'w_map':>8}  {'w_lact':>8}  {'med_inter':>10}  {'med_term':>10}  {'ratio':>8}")
print("-" * 56)
for w_map in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
    w_lact = w_map / 2
    med_inter = w_map * _median_unit_disc
    ratio = _median_disc_terminal / med_inter
    print(f"  {w_map:>6.2f}  {w_lact:>8.3f}  {med_inter:>10.2f}  "
          f"{_median_disc_terminal:>10.1f}  {ratio:>7.1f}x")

# %%
# ============================================================
# REWARD FUNCTION DESIGN  (REVIEWED: Fix B + Fix C applied)
# ============================================================
# REVIEWED CHANGES:
#   Fix B — lactate component: compute delta from RAW event-level measurements
#           (loaded from wide_df.parquet), not from the ffilled+normal-imputed
#           lab_lactate column in bucketed_dp. The old version produced FAKE
#           clearance rewards at the boundary where 12h ffill expired and the
#           value fell back to the "normal" 1.0 imputation.
#   Fix C — MAP component: replaced one-sided clip(55,75)+tanh (which gave +0.76
#           reward to MAP=120 just like MAP=75) with a two-sided plateau:
#           reward = min(tanh((MAP-55)/5), tanh((95-MAP)/5)).
#           Peak ≈+1 in [65, 90], 0 at MAP≈55 or 95, drops to ≈-1 at extremes.
# ============================================================
# Per-step intermediate (bounded, smooth):
#   MAP   :  min(tanh((MAP-55)/5), tanh((95-MAP)/5))                    # in [-1, +1]
#   Lact  :  tanh((Lactate_prev - Lactate_now) / 2)  on RAW measurements # in [-1, +1]
#
# Terminal (at last decision point, by disposition):
#   r_terminal = {CPC1_2: +100, CPC3: +40, CPC4: -40, CPC5: -100}
#
# Discount: γ = 0.99

# ── Reward parameters ──
W_MAP        = 0.5
W_LACT       = 0.25
MAP_LO       = 55.0   # below this: hypotension penalty
MAP_HI       = 95.0   # above this: hypertension penalty
MAP_SLOPE    = 5.0    # sigmoid steepness on each side
LACT_SCALE   = 2.0
GAMMA        = 0.99
TERMINAL_REWARDS = {"CPC1_2": 100.0, "CPC3": 40.0, "CPC4": -40.0, "CPC5": -100.0}

# ── Step 1: Compute per-step components ──
bucketed_dp["hospitalization_id"] = bucketed_dp["hospitalization_id"].astype(str)
bucketed_dp = bucketed_dp.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)

# ── Fix C: MAP component (two-sided plateau) ──
_map = bucketed_dp["vital_map"]
_map_lower = np.tanh((_map - MAP_LO) / MAP_SLOPE)
_map_upper = np.tanh((MAP_HI - _map) / MAP_SLOPE)
bucketed_dp["raw_map_component"] = np.minimum(_map_lower, _map_upper)
# Keep NaN MAP → 0 (no signal), since min(NaN, NaN) is NaN
bucketed_dp["raw_map_component"] = bucketed_dp["raw_map_component"].fillna(0)

# ── Fix B: Lactate component on RAW measurements ──
# Load event-level lactate (pre-ffill, pre-imputation) from wide_df.parquet
print("Loading raw lactate events from wide_df.parquet for Fix B...")
_wide = pd.read_parquet(OUT_DIR / "wide_df.parquet")
_wide["hospitalization_id"] = _wide["hospitalization_id"].astype(str)
_anchor = pd.read_parquet(OUT_DIR / "anchor_mapping.parquet")
_anchor["hospitalization_id"] = _anchor["hospitalization_id"].astype(str)

_lact = _wide.loc[_wide["lab_lactate"].notna(),
                   ["hospitalization_id", "event_dttm", "lab_lactate"]].copy()
_lact = _lact.merge(_anchor[["hospitalization_id", "anchor_dttm"]],
                    on="hospitalization_id", how="left")

# Normalize tz for subtraction
if pd.api.types.is_datetime64_any_dtype(_lact["event_dttm"]):
    if _lact["event_dttm"].dt.tz is None:
        _lact["event_dttm"] = _lact["event_dttm"].dt.tz_localize("UTC")
if pd.api.types.is_datetime64_any_dtype(_lact["anchor_dttm"]):
    if _lact["anchor_dttm"].dt.tz is None:
        _lact["anchor_dttm"] = _lact["anchor_dttm"].dt.tz_localize("UTC")
    if _lact["anchor_dttm"].dt.tz != _lact["event_dttm"].dt.tz:
        _lact["anchor_dttm"] = _lact["anchor_dttm"].dt.tz_convert(_lact["event_dttm"].dt.tz)

# Go through float64 first: subtraction can produce object-dtype on tz-mismatch / NaT;
# Int64 cast then fails. Float drop-NaN cast-int is robust.
_lact["hours_since_anchor"] = pd.to_numeric(
    (_lact["event_dttm"] - _lact["anchor_dttm"]).dt.total_seconds() / 3600.0,
    errors="coerce",
)
_lact = _lact.dropna(subset=["hours_since_anchor"]).copy()
_lact = _lact[(_lact["hours_since_anchor"] >= 0) & (_lact["hours_since_anchor"] < 120)].copy()
_lact["hour"] = _lact["hours_since_anchor"].astype(int)

# Per patient, compute delta between consecutive ACTUAL measurements
_lact = _lact.sort_values(["hospitalization_id", "event_dttm"]).reset_index(drop=True)
_lact["prev_lactate"] = _lact.groupby("hospitalization_id")["lab_lactate"].shift(1)
_lact["lactate_delta_raw"] = _lact["prev_lactate"] - _lact["lab_lactate"]

# One delta per (hosp, hour) — take the last measurement in that hour
_lact_delta_per_hour = (_lact.dropna(subset=["lactate_delta_raw"])
                              .groupby(["hospitalization_id", "hour"])["lactate_delta_raw"]
                              .last().reset_index())
print(f"  Real lactate measurements: {len(_lact):,} events across {_lact['hospitalization_id'].nunique():,} patients")
print(f"  Hours with computable delta: {len(_lact_delta_per_hour):,}")

# Merge into bucketed_dp; non-measurement hours get NaN → 0 (no signal)
bucketed_dp = bucketed_dp.merge(_lact_delta_per_hour,
                                 on=["hospitalization_id", "hour"], how="left")
bucketed_dp["raw_lactate_component"] = np.tanh(
    bucketed_dp["lactate_delta_raw"].fillna(0) / LACT_SCALE)

# Sanity print
_n_signal = (bucketed_dp["raw_lactate_component"].abs() > 1e-3).sum()
print(f"  Decision points with non-zero lactate signal: {_n_signal:,} / {len(bucketed_dp):,}")

# Weighted intermediate reward
bucketed_dp["r_intermediate"] = (
    W_MAP  * bucketed_dp["raw_map_component"] +
    W_LACT * bucketed_dp["raw_lactate_component"]
)

print(f"\nPer-step intermediate reward (w_map={W_MAP}, w_lact={W_LACT}):")
print(bucketed_dp["r_intermediate"]
      .describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(3).to_string())

print(f"\nRaw MAP component distribution (two-sided plateau):")
print(bucketed_dp["raw_map_component"]
      .describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

print(f"\nRaw lactate component distribution (from raw measurements only):")
print(bucketed_dp["raw_lactate_component"]
      .describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

# ── Step 2: Verify balance with target ratio ──
def _discounted_sum(group):
    r = group["r_intermediate"].values
    L = len(r)
    return (r * (GAMMA ** np.arange(L))).sum()

_traj_disc_inter = bucketed_dp.groupby("hospitalization_id").apply(
    _discounted_sum, include_groups=False
).rename("disc_intermediate")

_traj_lengths = bucketed_dp.groupby("hospitalization_id").size().rename("n_dp")

_balance = (_traj_disc_inter.reset_index()
            .merge(patient_static[["hospitalization_id", "cpc_tier"]],
                   on="hospitalization_id", how="left")
            .merge(_traj_lengths.reset_index(), on="hospitalization_id"))
_balance["terminal"] = _balance["cpc_tier"].map(TERMINAL_REWARDS)
_balance["disc_terminal"] = _balance["terminal"] * (GAMMA ** _balance["n_dp"])

print(f"\nFinal balance per CPC tier:")
print(_balance.groupby("cpc_tier").agg(
    n=("hospitalization_id", "count"),
    mean_disc_term=("disc_terminal", "mean"),
    median_disc_inter=("disc_intermediate", "median"),
    mean_disc_inter=("disc_intermediate", "mean"),
).round(2).to_string())

_med_term  = _balance["disc_terminal"].abs().median()
_med_inter = _balance["disc_intermediate"].abs().median()
_ratio = _med_term / max(_med_inter, 1e-6)
print(f"\n  Median |discounted terminal|     : {_med_term:.1f}")
print(f"  Median |discounted intermediate| : {_med_inter:.2f}")
print(f"  Ratio (target: 10-15x)           : {_ratio:.1f}x")

# ── Step 3: Assign terminal reward ──
patient_disp = patient_static[["hospitalization_id", "cpc_tier"]].copy()
bucketed_dp = bucketed_dp.merge(patient_disp, on="hospitalization_id", how="left")
bucketed_dp["terminal_reward_value"] = bucketed_dp["cpc_tier"].map(TERMINAL_REWARDS)

bucketed_dp["is_last_dp"] = (
    bucketed_dp.groupby("hospitalization_id")["hour"].transform("max")
    == bucketed_dp["hour"]
)

bucketed_dp["r_terminal"] = np.where(
    bucketed_dp["is_last_dp"],
    bucketed_dp["terminal_reward_value"],
    0.0,
)

# ── Step 4: Total reward per step ──
bucketed_dp["r_intermediate"] = bucketed_dp["r_intermediate"].fillna(0)
bucketed_dp["reward"] = bucketed_dp["r_intermediate"] + bucketed_dp["r_terminal"]


print(f"\nTotal reward distribution (all decision points):")
print(bucketed_dp["reward"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(2).to_string())

# At terminal step only
_term_rows = bucketed_dp[bucketed_dp["is_last_dp"]]
print(f"\nTerminal-step reward distribution ({len(_term_rows)} patients):")
print(_term_rows["reward"].value_counts().sort_index().to_string())

# ── Step 5: Save updated decision-points artifact ──
_keep_cols = ["hospitalization_id", "hour",
              "raw_map_component", "raw_lactate_component",
              "r_intermediate", "r_terminal", "reward",
              "is_last_dp", "cpc_tier"]
bucketed_dp[_keep_cols].to_parquet(OUT_DIR / "reward_components_reviewed.parquet", index=False)


print(f"\nSaved reward components → {OUT_DIR / 'reward_components_reviewed.parquet'}")

# %% [markdown]
# **Reward**: per-step `r_t = 0.5·tanh((clip(MAP_t,55,75)−65)/10) + 0.25·tanh((Lact_{t−1}−Lact_t)/2)` (bounded ~±0.75, no dose penalty); terminal `r_T ∈ {CPC1_2: +100, CPC3: +40, CPC4: −40, CPC5: −100}` by discharge disposition; γ = 0.99; weights calibrated to give a median |discounted terminal| / |discounted intermediate| ≈ 13× ("outcome-anchored" band).

# %%
# ── Cell 4: Compute discounted return G_t for each decision point ──
# G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(T-t)·r_T
# Computed via backward recursion: G_t = r_t + γ·G_{t+1}
# This is the value the DDQN's value head will be trained to predict.

print(f"Computing discounted returns (γ={GAMMA}) via backward recursion...")

bucketed_dp = bucketed_dp.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)

# Per-patient backward pass using numpy slicing
pid_arr    = bucketed_dp["hospitalization_id"].values
reward_arr = bucketed_dp["reward"].values
n = len(bucketed_dp)
G = np.zeros(n, dtype=float)

_, group_starts = np.unique(pid_arr, return_index=True)
group_starts = np.sort(group_starts)
group_ends = np.append(group_starts[1:], n)

for gs, ge in zip(group_starts, group_ends):
    g = 0.0
    for i in range(ge - 1, gs - 1, -1):
        g = reward_arr[i] + GAMMA * g
        G[i] = g

bucketed_dp["return"] = G

# ── Sanity check 1: return distribution ──
print(f"\nDiscounted return distribution (all {len(bucketed_dp):,} decision points):")
print(bucketed_dp["return"]
      .describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(2).to_string())

# ── Sanity check 2: returns at last step should equal rewards at last step ──
_last_dp = bucketed_dp[bucketed_dp["is_last_dp"]]
_check_diff = (_last_dp["return"] - _last_dp["reward"]).abs().max()
print(f"\nSanity: max |return - reward| at terminal step: {_check_diff:.6f}")
print(f"  (should be ~0 — at terminal, future contribution is 0)")

# ── Sanity check 3: per-CPC-tier separation ──
print(f"\nReturn at FIRST decision point per patient (= V*(s_0), expected return from window start):")
_first_dp = bucketed_dp.groupby("hospitalization_id").head(1)
print(_first_dp.groupby("cpc_tier")["return"]
      .describe(percentiles=[.25, .5, .75]).round(2).to_string())

# Visual separation between tiers
_means = _first_dp.groupby("cpc_tier")["return"].mean().to_dict()
print(f"\nMean V*(s_0) by tier:")
for tier in ["CPC1_2", "CPC3", "CPC4", "CPC5"]:
    print(f"  {tier:6s}  {_means.get(tier, float('nan')):+8.2f}")
print(f"  Spread (CPC1_2 - CPC5): {_means['CPC1_2'] - _means['CPC5']:+.2f}")

# ── Save the final RL-ready table ──
bucketed_dp = bucketed_dp.drop(columns=["raw_map_component", "raw_lactate_component",
                                          "terminal_reward_value", "is_last_dp"],
                                errors="ignore")

_out_path = OUT_DIR / "bucketed_with_reward_reviewed.parquet"
bucketed_dp.to_parquet(_out_path, index=False)
print(f"\n{'='*60}")
print(f"Saved RL-ready training data → {_out_path}")
print(f"  Rows           : {len(bucketed_dp):,}")
print(f"  Cols           : {bucketed_dp.shape[1]}")
print(f"  Patients       : {bucketed_dp['hospitalization_id'].nunique():,}")
print(f"  Size           : {_out_path.stat().st_size / 1024**2:.1f} MB")
