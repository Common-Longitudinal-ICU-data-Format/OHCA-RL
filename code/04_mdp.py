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

sys.path.insert(0, str(CODE_DIR))
for _stale in ("utils",):
    if _stale in sys.modules:
        del sys.modules[_stale]
import utils

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("03_ffill_bucket")

# Load artifacts from notebook 01 and 02
wide_df       = pd.read_parquet(OUT_DIR / "wide_df.parquet")
anchor_map    = pd.read_parquet(OUT_DIR / "anchor_mapping.parquet")
first_vaso    = pd.read_parquet(OUT_DIR / "first_vaso_mapping.parquet")
sofa_0_24     = pd.read_parquet(OUT_DIR / "sofa_0_24_reviewed.parquet")
meds_cont_raw = pd.read_parquet(OUT_DIR / "meds_cont_df.parquet")
cohort_df     = pd.read_parquet(OUT_DIR / "cohort_ohca_icu.parquet")
patient_static = pd.read_parquet(OUT_DIR / "patient_static.parquet")

# Force tz-aware on all date columns (notebook 01 produced tz-aware ones,
# but parquet roundtrips can drop tz on some columns)
for _df in (wide_df, anchor_map, first_vaso, meds_cont_raw):
    for _col in _df.columns:
        if _col.endswith(("_dttm", "_time")) and pd.api.types.is_datetime64_any_dtype(_df[_col]):
            if _df[_col].dt.tz is None:
                _df[_col] = _df[_col].dt.tz_localize(TIMEZONE)

print(f"wide_df        : {len(wide_df):,} rows × {wide_df.shape[1]} cols")
print(f"anchor_map     : {len(anchor_map):,} patients")
print(f"first_vaso     : {len(first_vaso):,} patients ever on vasopressor")
print(f"sofa_0_24      : {len(sofa_0_24):,} patients with SOFA")
print(f"meds_cont_raw  : {len(meds_cont_raw):,} med admin events (long-form)")
print(f"cohort         : {len(cohort_df):,} OHCA-ICU patients")
print(f"patient_static : {len(patient_static):,} patient demographics")

# %%
# ── Merge anchor_dttm into wide_df ──
wide_df["hospitalization_id"] = wide_df["hospitalization_id"].astype(str)
anchor_map["hospitalization_id"] = anchor_map["hospitalization_id"].astype(str)

_anchor_keep = anchor_map[["hospitalization_id", "anchor_dttm", "anchor_source"]].copy()
wide_df = wide_df.merge(_anchor_keep, on="hospitalization_id", how="left")

# ── Drop the 1 patient with no anchor ──
_no_anchor = wide_df["anchor_dttm"].isna()
print(f"Dropping {_no_anchor.sum():,} rows from patient(s) with no anchor")
wide_df = wide_df[~_no_anchor].copy()

# ── Compute hours since anchor ──
# Force tz match between event_dttm and anchor_dttm before subtraction
if wide_df["event_dttm"].dt.tz != wide_df["anchor_dttm"].dt.tz:
    wide_df["anchor_dttm"] = wide_df["anchor_dttm"].dt.tz_convert(wide_df["event_dttm"].dt.tz)

wide_df["hours_since_anchor"] = (
    (wide_df["event_dttm"] - wide_df["anchor_dttm"]).dt.total_seconds() / 3600.0
)

# ── Clip to 0-120h post-anchor (the context window) ──
TIME_WINDOW_HOURS = 120
_before = len(wide_df)
wide_df = wide_df[
    (wide_df["hours_since_anchor"] >= 0) &
    (wide_df["hours_since_anchor"] < TIME_WINDOW_HOURS)
].copy()
_after = len(wide_df)
print(f"Clipped to [0, {TIME_WINDOW_HOURS}h) post-anchor: "
      f"{_before:,} → {_after:,} rows ({_before-_after:,} dropped)")

# ── Assign hour bucket ──
wide_df["hour"] = wide_df["hours_since_anchor"].astype(int)
print(f"\nHour bucket range: {wide_df['hour'].min()}–{wide_df['hour'].max()}")
print(f"Unique hosps remaining: {wide_df['hospitalization_id'].nunique():,}")

# Quick distribution
print(f"\nEvents per hour-bucket (cohort-wide):")
print(wide_df.groupby("hour").size().describe(percentiles=[.5, .9]).round(0).to_string())

# %%
# Column categories
_id_cols    = ["hospitalization_id", "hour"]
_meta_cols  = ["event_dttm", "anchor_dttm", "anchor_source", "hours_since_anchor"]

_vital_cols     = sorted(c for c in wide_df.columns if c.startswith("vital_"))
_lab_cols       = sorted(c for c in wide_df.columns if c.startswith("lab_"))
_med_cont_cols  = sorted(c for c in wide_df.columns if c.startswith("med_cont_"))
_med_int_cols   = sorted(c for c in wide_df.columns if c.startswith("med_int_"))
_resp_num_cols  = sorted(c for c in wide_df.columns
                         if c.startswith("resp_") and pd.api.types.is_numeric_dtype(wide_df[c]))
_resp_cat_cols  = sorted(c for c in wide_df.columns
                         if c.startswith("resp_") and not pd.api.types.is_numeric_dtype(wide_df[c]))
_assess_cols    = sorted(c for c in wide_df.columns if c.startswith("assess_"))
_adt_cols       = sorted(c for c in wide_df.columns if c.startswith("adt_"))

print(f"Column groups:")
print(f"  vitals (num)    : {_vital_cols}")
print(f"  labs (num)      : {_lab_cols}")
print(f"  med_cont        : {_med_cont_cols}")
print(f"  med_int         : {_med_int_cols}")
print(f"  resp num        : {_resp_num_cols}")
print(f"  resp cat        : {_resp_cat_cols}")
print(f"  assess          : {_assess_cols}")
print(f"  adt             : {_adt_cols}")
print(f"  on_crrt         : {'on_crrt' in wide_df.columns}")

# ── Build aggregation dict per spec ──
# Spec: action = last NEE dose within hour → last() for ALL med_cont columns
# Vitals/labs/resp numeric → mean (preserves the upstream convention)
# Intermittent meds (boluses) → max
# Categorical → last (end-of-bucket state)
# on_crrt → max (any CRRT in the hour means "on CRRT")
agg_dict = {}
for c in _vital_cols:    agg_dict[c] = "mean"
for c in _lab_cols:      agg_dict[c] = "mean"
for c in _resp_num_cols: agg_dict[c] = "mean"
for c in _assess_cols:   agg_dict[c] = "last"   # GCS/RASS: end-of-hour state
for c in _med_cont_cols: agg_dict[c] = "last"   # SPEC: last dose in hour
for c in _med_int_cols:  agg_dict[c] = "max"    # boluses: largest dose
for c in _resp_cat_cols: agg_dict[c] = "last"
for c in _adt_cols:      agg_dict[c] = "last"
if "on_crrt" in wide_df.columns:
    agg_dict["on_crrt"] = "max"

# Also keep anchor metadata (constant per patient)
agg_dict["anchor_dttm"]   = "first"
agg_dict["anchor_source"] = "first"

print(f"\nAggregating {len(agg_dict)} columns over (hospitalization_id, hour)...")

bucketed = (wide_df
            .sort_values(["hospitalization_id", "hours_since_anchor"])
            .groupby(_id_cols, as_index=False)
            .agg(agg_dict))

print(f"Bucketed: {len(bucketed):,} rows × {bucketed.shape[1]} cols")
print(f"  Hospitalizations: {bucketed['hospitalization_id'].nunique():,}")
print(f"  Hour range: {bucketed['hour'].min()}–{bucketed['hour'].max()}")
print(f"  Rows/patient: mean={len(bucketed) / bucketed['hospitalization_id'].nunique():.1f}, "
      f"max={bucketed.groupby('hospitalization_id').size().max()}")

# %%
# For each patient, find their last bucket (= min of last event hour, 119)
_max_per_pt = bucketed.groupby("hospitalization_id")["hour"].max().reset_index().rename(columns={"hour": "max_hour"})
print(f"Last-bucket distribution across patients:")
print(_max_per_pt["max_hour"].describe(percentiles=[.25, .5, .75, .9, .99]).round(0).to_string())

# Build dense (hospitalization_id, hour) skeleton
_dense_frames = []
for _hid, _mh in zip(_max_per_pt["hospitalization_id"], _max_per_pt["max_hour"]):
    _dense_frames.append(pd.DataFrame({
        "hospitalization_id": _hid,
        "hour": np.arange(0, int(_mh) + 1),
    }))
dense_index = pd.concat(_dense_frames, ignore_index=True)

# Outer-merge sparse buckets onto the dense skeleton
_before = len(bucketed)
bucketed = dense_index.merge(bucketed, on=["hospitalization_id", "hour"], how="left")
print(f"\nDensified: {_before:,} → {len(bucketed):,} rows "
      f"(added {len(bucketed) - _before:,} empty buckets)")

# Forward-fill anchor metadata (it's patient-level constant)
for c in ["anchor_dttm", "anchor_source"]:
    bucketed[c] = bucketed.groupby("hospitalization_id")[c].transform(
        lambda s: s.ffill().bfill()
    )

# Mark "scaffold" rows that were added by densification — these have no
# original data and need to be filled by the next cell's ffill logic
_feat_cols_for_scaffold = [c for c in bucketed.columns
                           if c not in ("hospitalization_id", "hour",
                                        "anchor_dttm", "anchor_source")]
bucketed["is_scaffold"] = bucketed[_feat_cols_for_scaffold].isna().all(axis=1)

print(f"\nScaffold rows (no original data): {bucketed['is_scaffold'].sum():,} / {len(bucketed):,} "
      f"({bucketed['is_scaffold'].mean()*100:.1f}%)")
print(f"\nRows/patient now: mean={len(bucketed) / bucketed['hospitalization_id'].nunique():.1f}")

# %%
bucketed = bucketed.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)
_grp = bucketed.groupby("hospitalization_id", sort=False)

# ── Vitals: unlimited ffill + bfill for slow-changing ones ──
# Vitals are charted at least q1-q2h in the ICU. Forward-filling across hours
# is appropriate — the patient was at that value until the next reading.
print("Step 1: Vitals (unlimited ffill, bfill for height/weight/temp)...")
for c in _vital_cols:
    bucketed[c] = _grp[c].ffill()
# Leading NA bfill for slow-changing values (you can have an MAP recorded at
# hour 6 but no rows before; the height/weight from hour 6 applies to hours 0-5)
_bfill_vitals = [c for c in _vital_cols
                 if any(k in c for k in ("height", "weight", "temp"))]
for c in _bfill_vitals:
    bucketed[c] = _grp[c].bfill()

# ── Labs: 12h time-limited ffill, then normal-value fallback, then cohort median ──
# Labs aren't continuously monitored. A lactate from 18h ago is no longer
# representative. Cap ffill at 12h.
print("Step 2: Labs (12h time-limited ffill → normal-value imputation)...")
_LAB_FFILL_MAX_H = 12

for c in _lab_cols:
    # Build per-row "hours since last actual observation" for this lab
    _notna = bucketed[c].notna()
    _last_obs_hour = bucketed["hour"].where(_notna)
    _last_obs_hour = _last_obs_hour.groupby(bucketed["hospitalization_id"]).ffill()
    _hours_since_obs = bucketed["hour"] - _last_obs_hour

    # Unlimited ffill, but mask anything > 12h since last observation
    _ffilled = _grp[c].ffill()
    bucketed[c] = _ffilled.where(_hours_since_obs <= _LAB_FFILL_MAX_H)

# Step 2b: fill remaining lab NaN with normal values, then cohort median
print("Step 2b: Lab remaining NaN → normal value → cohort median...")
_lab_normals = ohca_config.get("labs_normal_values", {})
for c in _lab_cols:
    _stub = c.replace("lab_", "")
    if _stub in _lab_normals:
        bucketed[c] = bucketed[c].fillna(_lab_normals[_stub])
    else:
        # Fallback: cohort median for labs without a configured normal value
        bucketed[c] = bucketed[c].fillna(bucketed[c].median())

# ── Assessments (GCS, RASS): 8h time-limited ffill, NO default imputation ──
# Spec/upstream insight: GCS=15 default is wrong for sedated/comatose patients.
# Leave NaN where ffill expires.
print("Step 3: Assessments (8h time-limited ffill, NO default imputation)...")
_ASSESS_FFILL_MAX_H = ohca_config.get("assessments_ffill_hours", 8)
for c in _assess_cols:
    _notna = bucketed[c].notna()
    _last_obs_hour = bucketed["hour"].where(_notna)
    _last_obs_hour = _last_obs_hour.groupby(bucketed["hospitalization_id"]).ffill()
    _hours_since_obs = bucketed["hour"] - _last_obs_hour
    _ffilled = _grp[c].ffill()
    bucketed[c] = _ffilled.where(_hours_since_obs <= _ASSESS_FFILL_MAX_H)

# ── Continuous meds: ffill between observations, time-limited after last obs ──
# Logic:
#   - Between recorded values: unlimited ffill (infusion is running)
#   - After LAST recorded value > 0: ffill for up to 4h, then 0
#     (drug likely stopped but nurse forgot to chart the stop)
#   - After LAST recorded value = 0: fillna(0) (drug stopped)
#   - Before first observation: fillna(0) (drug not yet started)
print("Step 4: Continuous meds (ffill between obs, 4h trailing limit)...")
_MED_TRAILING_H = ohca_config.get("action_inference", {}).get("med_ffill_trailing_hours", 4)

for c in _med_cont_cols:
    # 1. Unlimited ffill across the patient's whole stay
    _filled = _grp[c].ffill()

    # 2. Find last observation hour AND last observation value per patient
    _notna = bucketed[c].notna()
    _last_obs_hour  = bucketed["hour"].where(_notna).groupby(bucketed["hospitalization_id"]).transform("max")
    _last_obs_value = bucketed[c].where(_notna).groupby(bucketed["hospitalization_id"]).transform("last")

    # 3. Mask rows that are AFTER last obs AND last obs was > 0 AND time gap > 4h
    _after_last = bucketed["hour"] > _last_obs_hour
    _time_since_last = bucketed["hour"] - _last_obs_hour
    _beyond_limit = _after_last & (_last_obs_value > 0) & (_time_since_last > _MED_TRAILING_H)
    _filled = _filled.where(~_beyond_limit)

    # 4. Anything still NaN (before first obs, or after time limit, or last obs = 0): set to 0
    bucketed[c] = _filled.fillna(0)

# ── Intermittent meds (boluses): no ffill, NaN → 0 ──
# Boluses are discrete events. Between events, the dose is genuinely 0.
print("Step 5: Intermittent meds (NaN → 0, no ffill)...")
for c in _med_int_cols:
    bucketed[c] = bucketed[c].fillna(0)

# ── Respiratory categorical: unlimited ffill + bfill leading NAs ──
print("Step 6: Respiratory categoricals (unlimited ffill + bfill leading NAs)...")
for c in _resp_cat_cols:
    bucketed[c] = _grp[c].ffill()
    bucketed[c] = _grp[c].bfill()

# ── Respiratory numeric: unlimited ffill ──
print("Step 7: Respiratory numerics (unlimited ffill)...")
for c in _resp_num_cols:
    bucketed[c] = _grp[c].ffill()

# ── on_crrt: ffill (once started, assume on until stopped) ──
if "on_crrt" in bucketed.columns:
    bucketed["on_crrt"] = _grp["on_crrt"].ffill().fillna(0).astype(int)

# ── ADT: ffill location ──
print("Step 8: ADT location ffill...")
if "adt_location_category" in bucketed.columns:
    bucketed["adt_location_category"] = _grp["adt_location_category"].ffill()

print(f"\n{'='*60}")
print(f"Post-ffill missingness:")
for grp_label, cols in [("vitals", _vital_cols), ("labs", _lab_cols),
                        ("med_cont", _med_cont_cols), ("med_int", _med_int_cols),
                        ("resp_num", _resp_num_cols), ("resp_cat", _resp_cat_cols),
                        ("assess", _assess_cols)]:
    if cols:
        _pct = bucketed[cols].isna().mean().mean() * 100
        print(f"  {grp_label:12s}  {_pct:5.1f}%  ({len(cols)} cols)")

# %%
# Per-patient missingness: how many patients have ANY data vs. zero data per group?
print("Patients with at least one non-NaN value in each group (post-ffill):")
for grp_label, cols in [("vitals", _vital_cols), ("labs", _lab_cols),
                        ("med_cont", _med_cont_cols),
                        ("resp_num", _resp_num_cols), ("resp_cat", _resp_cat_cols),
                        ("assess", _assess_cols)]:
    _any = bucketed[cols].notna().any(axis=1)
    _pts = bucketed.loc[_any, "hospitalization_id"].nunique()
    print(f"  {grp_label:12s}  {_pts:>5,} / 1,456 patients with any data")

# Row-level: what hour ranges drive missingness?
print(f"\nVital MAP missingness by hour:")
_map_miss = bucketed.groupby("hour")["vital_map"].apply(lambda s: s.isna().mean() * 100).round(1)
print(_map_miss.describe(percentiles=[.5, .9]).round(1).to_string())

# Sample 3 patients with high missing vitals
_pt_miss = (bucketed.groupby("hospitalization_id")["vital_map"]
            .apply(lambda s: s.isna().mean())
            .sort_values(ascending=False).head(5))
print(f"\nTop 5 patients by vital_map missingness:")
print(_pt_miss.round(2).to_string())

# %%
# NEE coefficients (from config)
_nee_coefs = ohca_config["nee_coefficients"]
print(f"NEE coefficients: {_nee_coefs}")

# Compute NEE per row from ffilled doses
# All component doses are already in mcg/kg/min (or u/min for vasopressin)
# from the unit conversion in notebook 01.
_nee_components = []
for med, coef in _nee_coefs.items():
    col = f"med_cont_{med}"
    if col in bucketed.columns:
        _nee_components.append(bucketed[col].fillna(0) * coef)
        print(f"  {col:30s} × {coef:>5.2f} = component")
    else:
        print(f"  {col:30s} NOT IN DATA — skipping")

bucketed["med_cont_nee"] = sum(_nee_components)

# Sanity check
_nee_nonzero = (bucketed["med_cont_nee"] > 0)
print(f"\nNEE > 0: {_nee_nonzero.sum():,} hour-rows ({_nee_nonzero.mean()*100:.1f}%)")
print(f"NEE distribution where > 0:")
print(bucketed.loc[_nee_nonzero, "med_cont_nee"]
      .describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(3).to_string())
print(f"\nMax NEE observed: {bucketed['med_cont_nee'].max():.3f} mcg/kg/min")

# Patients ever with NEE > 0 (should match first_vaso count of 838)
_ever_nee = bucketed.groupby("hospitalization_id")["med_cont_nee"].max() > 0
print(f"\nPatients with NEE > 0 at any hour: {_ever_nee.sum():,} / 1,456")
print(f"  (cross-check with first_vaso: {len(first_vaso):,})")

# %%
# Confirm: how many patients have first_vaso_dttm WITHIN the 120h post-anchor window?
_fv = first_vaso.merge(anchor_map[["hospitalization_id", "anchor_dttm"]],
                        on="hospitalization_id", how="left")
_fv["lag_h"] = (_fv["first_vaso_dttm"] - _fv["anchor_dttm"]).dt.total_seconds() / 3600
_within_120h = (_fv["lag_h"] >= 0) & (_fv["lag_h"] < 120)
print(f"first_vaso patients with first dose in [0, 120h): {_within_120h.sum():,}")
print(f"first_vaso patients with first dose outside [0, 120h): {(~_within_120h).sum():,}")
print(f"  → vs. bucketed NEE>0 patients ({_ever_nee.sum():,})")

# %%
# ── Action tier definitions per new spec ──
# Off:       NEE == 0
# Low:       0 <  NEE ≤ 0.05
# Medium:    0.05 < NEE ≤ 0.15
# High:      0.15 < NEE ≤ 0.30
# Very high: NEE > 0.30
#
# Spec wording is "0 to 0.05" for Low and "0.05 to 0.15" for Medium, which is
# ambiguous at the boundaries. We use right-inclusive bins (≤) consistent with
# the upstream YAML's encoding comments.

ACTION_LABELS = {0: "Off", 1: "Low", 2: "Medium", 3: "High", 4: "VeryHigh"}
ACTION_NEE_UPPER = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.30, 4: float("inf")}

def nee_to_tier(nee):
    """Map a NEE dose to the 5 absolute-dose tier."""
    if pd.isna(nee) or nee <= 0:
        return 0
    if nee <= 0.05:
        return 1
    if nee <= 0.15:
        return 2
    if nee <= 0.30:
        return 3
    return 4

bucketed["action_tier"] = bucketed["med_cont_nee"].apply(nee_to_tier).astype(int)
bucketed["action_label"] = bucketed["action_tier"].map(ACTION_LABELS)

# ── Distribution check ──
print(f"Action tier distribution (all {len(bucketed):,} hour-buckets):")
_dist = bucketed["action_tier"].value_counts().sort_index()
for tier, n in _dist.items():
    print(f"  {tier} {ACTION_LABELS[tier]:8s}  {n:>8,}  ({n/len(bucketed)*100:5.1f}%)")

# Among hours where the patient was on pressors (NEE > 0):
_on_vaso = bucketed[bucketed["action_tier"] > 0]
print(f"\nAmong on-pressor hours ({len(_on_vaso):,}):")
_dist_on = _on_vaso["action_tier"].value_counts().sort_index()
for tier, n in _dist_on.items():
    print(f"  {tier} {ACTION_LABELS[tier]:8s}  {n:>8,}  ({n/len(_on_vaso)*100:5.1f}%)")

# %%
# ── Load raw long-form med events (already unit-converted in notebook 01) ──
meds_raw = pd.read_parquet(OUT_DIR / "meds_cont_df.parquet")
meds_raw["hospitalization_id"] = meds_raw["hospitalization_id"].astype(str)

# Restrict to NEE-contributing meds
_nee_coefs = ohca_config["nee_coefficients"]
_nee_meds = list(_nee_coefs.keys())
print(f"NEE-contributing meds: {_nee_meds}")

meds_nee = meds_raw[meds_raw["med_category"].isin(_nee_meds)].copy()
print(f"NEE-contributing events: {len(meds_nee):,} (of {len(meds_raw):,} total)")

# Ensure tz-aware
if meds_nee["admin_dttm"].dt.tz is None:
    meds_nee["admin_dttm"] = meds_nee["admin_dttm"].dt.tz_localize(TIMEZONE)

# Merge anchor and clip to [0, 120h)
meds_nee = meds_nee.merge(anchor_map[["hospitalization_id", "anchor_dttm"]],
                          on="hospitalization_id", how="left")
if meds_nee["anchor_dttm"].dt.tz != meds_nee["admin_dttm"].dt.tz:
    meds_nee["anchor_dttm"] = meds_nee["anchor_dttm"].dt.tz_convert(meds_nee["admin_dttm"].dt.tz)

meds_nee["hours_since_anchor"] = (
    (meds_nee["admin_dttm"] - meds_nee["anchor_dttm"]).dt.total_seconds() / 3600.0
)
meds_nee = meds_nee[(meds_nee["hours_since_anchor"] >= 0) &
                    (meds_nee["hours_since_anchor"] < TIME_WINDOW_HOURS)].copy()
meds_nee["hour"] = meds_nee["hours_since_anchor"].astype(int)
print(f"After [0, 120h) clip: {len(meds_nee):,} events")

# ── Pivot wide on med_category so each event has the current dose of every NEE drug ──
# Then forward-fill within patient so each event row shows the running dose state
meds_nee = meds_nee.sort_values(["hospitalization_id", "admin_dttm"]).reset_index(drop=True)

# One row per (hosp, admin_dttm) with one column per drug (NaN where drug wasn't charted)
event_wide = meds_nee.pivot_table(
    index=["hospitalization_id", "admin_dttm", "hour"],
    columns="med_category", values="med_dose", aggfunc="last",
).reset_index()

# Make sure all 6 NEE drugs are present as columns (some may be missing in MIMIC)
for _m in _nee_meds:
    if _m not in event_wide.columns:
        event_wide[_m] = np.nan

# Forward-fill each drug within patient, then fillna(0) (before-first-event = 0)
event_wide = event_wide.sort_values(["hospitalization_id", "admin_dttm"]).reset_index(drop=True)
event_wide[_nee_meds] = (event_wide.groupby("hospitalization_id")[_nee_meds]
                                   .ffill().fillna(0))

# Compute event-level NEE
event_wide["event_nee"] = sum(event_wide[m] * c for m, c in _nee_coefs.items())

# ── Detect changes per event ──
EPS = 1e-6
event_wide["prev_nee"] = event_wide.groupby("hospitalization_id")["event_nee"].shift(1).fillna(0)
event_wide["delta_nee"] = event_wide["event_nee"] - event_wide["prev_nee"]
event_wide["is_change"]   = event_wide["delta_nee"].abs() > EPS
event_wide["is_esc"]      = event_wide["delta_nee"] >  EPS
event_wide["is_desc"]     = event_wide["delta_nee"] < -EPS

print(f"\nEvent-level change summary:")
print(f"  Total events           : {len(event_wide):,}")
print(f"  Changes (|Δ| > {EPS}): {event_wide['is_change'].sum():,}")
print(f"  Escalations            : {event_wide['is_esc'].sum():,}")
print(f"  De-escalations         : {event_wide['is_desc'].sum():,}")
print(f"  Unchanged (re-verify)  : {(~event_wide['is_change']).sum():,}")

# ── Aggregate per (hospitalization_id, hour) ──
hour_agg = (event_wide.groupby(["hospitalization_id", "hour"], as_index=False)
            .agg(nee_changes_in_hour=("is_change", "sum"),
                 _has_esc =("is_esc",  "any"),
                 _has_desc=("is_desc", "any")))

# Direction one-hot
def _direction(row):
    if row["nee_changes_in_hour"] == 0:
        return "none"
    if row["_has_esc"] and not row["_has_desc"]:
        return "escalation"
    if row["_has_desc"] and not row["_has_esc"]:
        return "de-escalation"
    return "mixed"

hour_agg["nee_direction_in_hour"] = hour_agg.apply(_direction, axis=1)
hour_agg = hour_agg.drop(columns=["_has_esc", "_has_desc"])

# One-hot encode direction
for _d, _col in [("none",           "nee_dir_none"),
                 ("escalation",     "nee_dir_esc"),
                 ("de-escalation",  "nee_dir_desc"),
                 ("mixed",          "nee_dir_mixed")]:
    hour_agg[_col] = (hour_agg["nee_direction_in_hour"] == _d).astype(int)

print(f"\nHour-aggregate distribution (only hours with NEE events: {len(hour_agg):,}):")
print(f"  Changes per hour:")
print(hour_agg["nee_changes_in_hour"].describe(percentiles=[.5, .9, .99]).round(1).to_string())
print(f"  Direction distribution:")
print(hour_agg["nee_direction_in_hour"].value_counts().to_string())

# ── Merge into bucketed ──
hour_agg["hospitalization_id"] = hour_agg["hospitalization_id"].astype(str)
bucketed = bucketed.merge(
    hour_agg[["hospitalization_id", "hour",
              "nee_changes_in_hour",
              "nee_dir_none", "nee_dir_esc", "nee_dir_desc", "nee_dir_mixed"]],
    on=["hospitalization_id", "hour"], how="left",
)

# ── Fill scaffold / no-event hours with "none, 0 changes" ──
bucketed["nee_changes_in_hour"] = bucketed["nee_changes_in_hour"].fillna(0).astype(int)
for _col in ["nee_dir_none", "nee_dir_esc", "nee_dir_desc", "nee_dir_mixed"]:
    bucketed[_col] = bucketed[_col].fillna(0).astype(int)
# Set "none" = 1 for filled-zero rows (no events = no titration)
_no_event = (bucketed["nee_changes_in_hour"] == 0) & (bucketed[
    ["nee_dir_esc", "nee_dir_desc", "nee_dir_mixed"]].sum(axis=1) == 0)
bucketed.loc[_no_event, "nee_dir_none"] = 1

# ── Sanity check ──
print(f"\nbucketed now has {bucketed.shape[1]} cols (+5 from cell 7)")
print(f"\nFinal nee_changes_in_hour distribution across all {len(bucketed):,} hour-buckets:")
print(bucketed["nee_changes_in_hour"].value_counts().sort_index().head(10).to_string())

print(f"\nDirection one-hot distribution:")
for _col in ["nee_dir_none", "nee_dir_esc", "nee_dir_desc", "nee_dir_mixed"]:
    _n = bucketed[_col].sum()
    print(f"  {_col:18s}  {_n:>8,}  ({_n/len(bucketed)*100:5.1f}%)")

# Cross-check: on-pressor hours with no changes should be rare in actively-titrating periods
_on_press_no_change = ((bucketed["action_tier"] > 0) &
                       (bucketed["nee_changes_in_hour"] == 0)).sum()
_on_press = (bucketed["action_tier"] > 0).sum()
print(f"\nOn-pressor hours with 0 NEE changes: {_on_press_no_change:,} / {_on_press:,} "
      f"({_on_press_no_change/_on_press*100:.1f}%)")
print("  (these are 'stable maintenance' hours — should be a meaningful chunk)")

# %%
# ── hours_since_last_on_pressor ──
# Definition:
#   = 0 while patient is on pressors (action_tier > 0)
#   = duration of current off-bout while off (resets to 0 the moment pressors restart)
#   For pre-first-pressor rows (before any pressor ever), set to hours_since_anchor
#     so the policy gets a continuous "time without vasopressor support" signal.

bucketed = bucketed.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)

# Map first vaso time onto each row
first_vaso["hospitalization_id"] = first_vaso["hospitalization_id"].astype(str)
first_vaso_with_hour = first_vaso.merge(
    anchor_map[["hospitalization_id", "anchor_dttm"]],
    on="hospitalization_id", how="left",
)
first_vaso_with_hour["first_vaso_hour"] = (
    (first_vaso_with_hour["first_vaso_dttm"] -
     first_vaso_with_hour["anchor_dttm"]).dt.total_seconds() / 3600.0
).astype(int)

bucketed = bucketed.merge(
    first_vaso_with_hour[["hospitalization_id", "first_vaso_hour"]],
    on="hospitalization_id", how="left",
)

# ── Compute hours_since_last_on_pressor per patient ──
# Vectorized over the whole frame using numpy slicing — avoids the
# groupby-apply pitfall where include_groups=False drops hospitalization_id.

print("Computing hours_since_last_on_pressor (per-patient pass)...")

bucketed = bucketed.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)

pid_arr = bucketed["hospitalization_id"].values
on_arr  = (bucketed["action_tier"] > 0).values
n = len(bucketed)
out = np.zeros(n, dtype=float)

# Find group boundaries
_, group_starts = np.unique(pid_arr, return_index=True)
group_starts = np.sort(group_starts)
group_ends = np.append(group_starts[1:], n)

for gs, ge in zip(group_starts, group_ends):
    counter = 0
    for i in range(gs, ge):
        if on_arr[i]:
            counter = 0
        else:
            counter += 1
        out[i] = counter

bucketed["hours_since_last_on_pressor"] = out

# Wait — the above counts ALL off hours, including pre-first-pressor ones.
# That's what we want for patients before their first dose (the counter rises
# from hour 0 until the first pressor is given).
# For patients who never get pressors, the counter just equals hours_since_anchor.

# ── Sanity check ──
print(f"\nhours_since_last_on_pressor distribution (all {len(bucketed):,} rows):")
print(bucketed["hours_since_last_on_pressor"].describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(1).to_string())

# Among on-pressor hours, value should be 0 (just reset)
_on = bucketed["action_tier"] > 0
print(f"\nAmong on-pressor hours ({_on.sum():,}):")
print(bucketed.loc[_on, "hours_since_last_on_pressor"].describe().round(1).to_string())

# Among off-pressor hours within a course (between first vaso and never-restarting):
# Pre-first-pressor: counter rises monotonically
# Post-last-pressor: counter rises monotonically
# In-course off-bouts: counter rises then resets
_off = bucketed["action_tier"] == 0
print(f"\nAmong off-pressor hours ({_off.sum():,}):")
print(bucketed.loc[_off, "hours_since_last_on_pressor"].describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(1).to_string())

# ── Diagnostic: show one patient's trajectory ──
_sample_pid = first_vaso_with_hour.loc[
    (first_vaso_with_hour["first_vaso_hour"] > 5) &
    (first_vaso_with_hour["first_vaso_hour"] < 50),
    "hospitalization_id"
].iloc[0]

print(f"\nExample trajectory for patient {_sample_pid} (first vaso at hour "
      f"{int(first_vaso_with_hour.loc[first_vaso_with_hour['hospitalization_id']==_sample_pid, 'first_vaso_hour'].iloc[0])}):")
_sample = bucketed[bucketed["hospitalization_id"] == _sample_pid][
    ["hour", "action_tier", "med_cont_nee", "hours_since_last_on_pressor"]
].head(60)
print(_sample.to_string(index=False))

# %%
# ── Snapshot features captured at first_vaso_hour, carried forward as constants ──

sofa_0_24["hospitalization_id"] = sofa_0_24["hospitalization_id"].astype(str)

# Patients with first_vaso in the 120h window
_snap_eligible = first_vaso_with_hour[
    (first_vaso_with_hour["first_vaso_hour"] >= 0) &
    (first_vaso_with_hour["first_vaso_hour"] < TIME_WINDOW_HOURS)
].copy()
print(f"Patients eligible for snapshot (first_vaso in [0, 120h)): {len(_snap_eligible):,}")

# ── Extract the bucketed row at hour == first_vaso_hour for each patient ──
# bucketed already has `first_vaso_hour` (added in cell 9), so we just filter:
_at_first_vaso = bucketed[
    bucketed["hour"] == bucketed["first_vaso_hour"]
][["hospitalization_id", "first_vaso_hour", "vital_map", "lab_lactate"]].copy()

_snap_rows = _at_first_vaso.rename(columns={
    "first_vaso_hour": "snapshot_hours_anchor_to_first_pressor",
    "vital_map":       "snapshot_map_at_first_pressor",
    "lab_lactate":     "snapshot_lactate_at_first_pressor",
})

print(f"\nSnapshot rows extracted: {len(_snap_rows):,}")
print(f"  Missing MAP at first vaso    : "
      f"{_snap_rows['snapshot_map_at_first_pressor'].isna().sum()}")
print(f"  Missing lactate at first vaso: "
      f"{_snap_rows['snapshot_lactate_at_first_pressor'].isna().sum()}")

# ── Add SOFA 0-24h ──
_snap_rows = _snap_rows.merge(
    sofa_0_24[["hospitalization_id", "sofa_total_0_24"]],
    on="hospitalization_id", how="left",
)
_snap_rows = _snap_rows.rename(columns={"sofa_total_0_24": "snapshot_sofa_24h"})

# Availability flag
_snap_rows["snapshot_sofa_24h_available"] = _snap_rows["snapshot_sofa_24h"].notna().astype(int)

# Impute missing SOFA with cohort median
_sofa_median = sofa_0_24["sofa_total_0_24"].median()
_snap_rows["snapshot_sofa_24h"] = _snap_rows["snapshot_sofa_24h"].fillna(_sofa_median)
print(f"\nSOFA snapshot:")
print(f"  Available: {_snap_rows['snapshot_sofa_24h_available'].sum()} / {len(_snap_rows)}")
print(f"  Median (used for imputation): {_sofa_median:.1f}")

# ── Distributions ──
print(f"\nSnapshot feature distributions:")
for _col in ["snapshot_hours_anchor_to_first_pressor",
             "snapshot_map_at_first_pressor",
             "snapshot_lactate_at_first_pressor",
             "snapshot_sofa_24h"]:
    print(f"\n  {_col}")
    print(_snap_rows[_col].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())

# ── Merge into bucketed (carry forward as constants per patient) ──
_snap_cols = ["hospitalization_id",
              "snapshot_hours_anchor_to_first_pressor",
              "snapshot_map_at_first_pressor",
              "snapshot_lactate_at_first_pressor",
              "snapshot_sofa_24h",
              "snapshot_sofa_24h_available"]
bucketed = bucketed.merge(_snap_rows[_snap_cols], on="hospitalization_id", how="left")

print(f"\nbucketed now has {bucketed.shape[1]} cols")
print(f"Patients with NaN snapshot (will be filtered later): "
      f"{bucketed.loc[bucketed['snapshot_map_at_first_pressor'].isna(), 'hospitalization_id'].nunique():,}")
print(f"Patients with snapshot values: "
      f"{bucketed.loc[bucketed['snapshot_map_at_first_pressor'].notna(), 'hospitalization_id'].nunique():,}")

# %%
# ── hours_since_anchor (already present from cell 2 — verify) ──
if "hours_since_anchor" not in bucketed.columns:
    bucketed["hours_since_anchor"] = bucketed["hour"].astype(float)
print(f"hours_since_anchor: range {bucketed['hours_since_anchor'].min():.0f}–"
      f"{bucketed['hours_since_anchor'].max():.0f}")

# ── hours_since_first_pressor ──
# - Negative before first pressor (the policy sees "first pressor will come in X hours")
# - 0 at the first pressor hour
# - Positive after
# - NaN for patients with no first_vaso_hour (never on pressors in window)
bucketed["hours_since_first_pressor"] = (
    bucketed["hour"] - bucketed["first_vaso_hour"]
)

print(f"\nhours_since_first_pressor distribution:")
print(bucketed["hours_since_first_pressor"]
      .describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

# Sanity check breakdown
_neg  = (bucketed["hours_since_first_pressor"] < 0).sum()
_zero = (bucketed["hours_since_first_pressor"] == 0).sum()
_pos  = (bucketed["hours_since_first_pressor"] > 0).sum()
_nan  = bucketed["hours_since_first_pressor"].isna().sum()
print(f"\n  Pre-first-pressor (negative): {_neg:,} rows")
print(f"  At first pressor (0)        : {_zero:,} rows")
print(f"  Post-first-pressor (positive): {_pos:,} rows")
print(f"  No first pressor in window (NaN): {_nan:,} rows  "
      f"({bucketed.loc[bucketed['hours_since_first_pressor'].isna(), 'hospitalization_id'].nunique():,} patients)")

print(f"\nbucketed now has {bucketed.shape[1]} cols")

# %%
# Re-do hours_since_first_pressor using only in-window first vaso
# Mark first_vaso_hour as NaN for patients whose first vaso was outside [0, 120h)
_out_of_window = (bucketed["first_vaso_hour"] < 0) | (bucketed["first_vaso_hour"] >= TIME_WINDOW_HOURS)
print(f"Setting first_vaso_hour=NaN for {bucketed.loc[_out_of_window, 'hospitalization_id'].nunique():,} "
      f"patients with first vaso outside [0, 120h)")
bucketed.loc[_out_of_window, "first_vaso_hour"] = np.nan

# Re-compute hours_since_first_pressor
bucketed["hours_since_first_pressor"] = bucketed["hour"] - bucketed["first_vaso_hour"]

print(f"\nhours_since_first_pressor distribution (corrected):")
print(bucketed["hours_since_first_pressor"]
      .describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

_neg  = (bucketed["hours_since_first_pressor"] < 0).sum()
_zero = (bucketed["hours_since_first_pressor"] == 0).sum()
_pos  = (bucketed["hours_since_first_pressor"] > 0).sum()
_nan  = bucketed["hours_since_first_pressor"].isna().sum()
print(f"\n  Pre-first-pressor (negative): {_neg:,} rows")
print(f"  At first pressor (0)        : {_zero:,} rows")
print(f"  Post-first-pressor (positive): {_pos:,} rows")
print(f"  No first pressor in window (NaN): {_nan:,} rows  "
      f"({bucketed.loc[bucketed['hours_since_first_pressor'].isna(), 'hospitalization_id'].nunique():,} patients)")

# %%
# ── Build exit_hour per patient (discharge or death, whichever defines end of stay) ──
patient_static["hospitalization_id"] = patient_static["hospitalization_id"].astype(str)

_exit = patient_static[["hospitalization_id", "discharge_dttm", "death_dttm",
                        "survival_status"]].copy()
_exit["exit_dttm"] = np.where(
    _exit["survival_status"] == "non-survivor",
    _exit["death_dttm"].fillna(_exit["discharge_dttm"]),
    _exit["discharge_dttm"],
)
_exit["exit_dttm"] = pd.to_datetime(_exit["exit_dttm"], utc=True, errors="coerce")

_exit = _exit.merge(anchor_map[["hospitalization_id", "anchor_dttm"]],
                    on="hospitalization_id", how="left")
if _exit["anchor_dttm"].dt.tz != _exit["exit_dttm"].dt.tz:
    _exit["anchor_dttm"] = _exit["anchor_dttm"].dt.tz_convert(_exit["exit_dttm"].dt.tz)

_exit["exit_hour"] = (
    (_exit["exit_dttm"] - _exit["anchor_dttm"]).dt.total_seconds() / 3600.0
)
print(f"Exit hour distribution:")
print(_exit["exit_hour"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

# Merge exit_hour into bucketed
bucketed = bucketed.merge(
    _exit[["hospitalization_id", "exit_hour", "survival_status"]],
    on="hospitalization_id", how="left",
)

# ── Per-patient pass: vectorized version that preserves hospitalization_id ──
def _build_window_flags_vectorized(df, time_window_hours):
    """Vectorized over the whole frame: builds in_decision_window + close_reason."""
    df = df.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)
    n = len(df)
    in_window     = np.zeros(n, dtype=int)
    close_reason  = np.array(["pre_first_vaso"] * n, dtype=object)

    # Process patient-by-patient via numpy slicing (avoids groupby-apply pitfalls)
    pid_arr      = df["hospitalization_id"].values
    hour_arr     = df["hour"].values
    tier_arr     = df["action_tier"].values
    fv_arr       = df["first_vaso_hour"].values
    exit_arr     = df["exit_hour"].values

    # Find group boundaries
    _, group_starts = np.unique(pid_arr, return_index=True)
    group_starts = np.sort(group_starts)
    group_ends = np.append(group_starts[1:], n)

    for gs, ge in zip(group_starts, group_ends):
        fv = fv_arr[gs]
        ex = exit_arr[gs]

        # Patients with no in-window first vaso: no decision points
        if pd.isna(fv):
            for i in range(gs, ge):
                close_reason[i] = "no_first_vaso"
            continue

        fv_int = int(fv)
        ex_bucket = int(ex) if pd.notna(ex) else time_window_hours
        closed = False

        for i in range(gs, ge):
            h = hour_arr[i]

            if h < fv_int:
                close_reason[i] = "pre_first_vaso"
                continue

            if closed:
                continue

            # (b) 120h cap
            if h >= time_window_hours:
                close_reason[i] = "120h_cap"
                closed = True
                continue

            # (c) exit
            if h > ex_bucket:
                close_reason[i] = "exit"
                closed = True
                continue

            # (a) 24h-off rule — look forward
            if tier_arr[i] == 0:
                fwd_end = i + 24
                if fwd_end <= ge:
                    window_tiers = tier_arr[i:fwd_end]
                    window_hours = hour_arr[i:fwd_end]
                    consecutive = (window_hours[-1] - window_hours[0] == 23)
                    if consecutive and (window_tiers == 0).all():
                        close_reason[i] = "24h_off"
                        closed = True
                        continue

            # No closure: valid decision point
            in_window[i] = 1
            close_reason[i] = "open"

    df["in_decision_window"]  = in_window
    df["window_close_reason"] = close_reason
    return df

print("\nComputing decision-point windows (per-patient pass)...")
bucketed = _build_window_flags_vectorized(bucketed, TIME_WINDOW_HOURS)

# ── Summary ──
print(f"\nWindow closure reason distribution (per row):")
print(bucketed["window_close_reason"].value_counts().to_string())

print(f"\nDecision-point rows: {bucketed['in_decision_window'].sum():,} / {len(bucketed):,}")

# Patient-level closure reason
_close_per_pt = (bucketed[bucketed["window_close_reason"].isin(["24h_off", "120h_cap", "exit"])]
                 .groupby("hospitalization_id")["window_close_reason"].first())
print(f"\nWindow closure reason (per patient):")
print(_close_per_pt.value_counts().to_string())

# Decision points per patient
_dp_per_pt = (bucketed[bucketed["in_decision_window"] == 1]
              .groupby("hospitalization_id").size())
print(f"\nDecision points per patient (n={len(_dp_per_pt):,} patients):")
print(_dp_per_pt.describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_string())

# Action tier distribution AMONG DECISION POINTS
_dp = bucketed[bucketed["in_decision_window"] == 1]
print(f"\nAction tier distribution among decision points ({len(_dp):,} rows):")
for tier in range(5):
    _n = (_dp["action_tier"] == tier).sum()
    print(f"  {tier} {ACTION_LABELS[tier]:8s}  {_n:>7,}  ({_n/len(_dp)*100:5.1f}%)")

# %%
# ── State-based mask (MAP-driven) ──
# Spec primary thresholds:
#   MAP < 55         → forbid Off (hypotensive, can't withdraw pressors)
#   55 ≤ MAP ≤ 90    → all 5 tiers allowed
#   MAP > 90         → forbid High and VeryHigh (over-pressorized)
# NaN MAP → treat as middle zone (allow all)
MAP_LOWER = 55.0
MAP_UPPER = 90.0

def _state_mask(map_val):
    """Returns boolean array [Off, Low, Med, High, VHigh] of allowed tiers."""
    mask = np.ones(5, dtype=bool)
    if pd.isna(map_val):
        return mask
    if map_val < MAP_LOWER:
        mask[0] = False  # forbid Off
    elif map_val > MAP_UPPER:
        mask[3] = False  # forbid High
        mask[4] = False  # forbid VeryHigh
    return mask

# Build state-mask columns
_state_masks = np.array([_state_mask(m) for m in bucketed["vital_map"].values])
bucketed["state_mask_off"]   = _state_masks[:, 0].astype(int)
bucketed["state_mask_low"]   = _state_masks[:, 1].astype(int)
bucketed["state_mask_med"]   = _state_masks[:, 2].astype(int)
bucketed["state_mask_high"]  = _state_masks[:, 3].astype(int)
bucketed["state_mask_vhigh"] = _state_masks[:, 4].astype(int)

# Sanity check
print("State mask: how often is each tier forbidden by MAP?")
for tier_idx, tier_name in ACTION_LABELS.items():
    forbidden = (bucketed[f"state_mask_{['off','low','med','high','vhigh'][tier_idx]}"] == 0).sum()
    print(f"  {tier_idx} {tier_name:8s}  forbidden in {forbidden:>6,} rows ({forbidden/len(bucketed)*100:5.1f}%)")

# ── Transition-based mask (current-tier-driven) ──
# 5x5 matrix: rows = previous tier, cols = next allowed tier
# Row i, col j: True if going from tier i → tier j is allowed
TRANSITION_MASK = np.array([
    # to:   Off  Low  Med  High VHigh
    [True,  True,  True,  False, False],  # from Off
    [True,  True,  True,  True,  False],  # from Low
    [True,  True,  True,  True,  True ],  # from Med
    [False, True,  True,  True,  True ],  # from High
    [False, False, True,  True,  True ],  # from VHigh
], dtype=bool)

# Previous tier per patient (shift 1). For the first row of each patient,
# use Off (tier 0) as the implicit pre-window tier.
bucketed["prev_action_tier"] = (
    bucketed.groupby("hospitalization_id")["action_tier"]
            .shift(1)
            .fillna(0)
            .astype(int)
)

# Build transition-mask columns
_trans_masks = TRANSITION_MASK[bucketed["prev_action_tier"].values]  # shape (n, 5)
bucketed["trans_mask_off"]   = _trans_masks[:, 0].astype(int)
bucketed["trans_mask_low"]   = _trans_masks[:, 1].astype(int)
bucketed["trans_mask_med"]   = _trans_masks[:, 2].astype(int)
bucketed["trans_mask_high"]  = _trans_masks[:, 3].astype(int)
bucketed["trans_mask_vhigh"] = _trans_masks[:, 4].astype(int)

print("\nTransition mask: how often is each tier forbidden by prev-tier rule?")
for tier_idx, tier_name in ACTION_LABELS.items():
    forbidden = (bucketed[f"trans_mask_{['off','low','med','high','vhigh'][tier_idx]}"] == 0).sum()
    print(f"  {tier_idx} {tier_name:8s}  forbidden in {forbidden:>6,} rows ({forbidden/len(bucketed)*100:5.1f}%)")

# ── Final mask: intersection of state-mask AND transition-mask ──
for _t in ["off", "low", "med", "high", "vhigh"]:
    bucketed[f"mask_{_t}"] = (
        bucketed[f"state_mask_{_t}"] & bucketed[f"trans_mask_{_t}"]
    )

# ── Soft constraint: always allow the observed action ──
# Critical for offline RL — if a clinician took action k, we cannot mask it out,
# or the policy learns from no transitions from that (state, action) pair.
_tier_to_mask = {0: "mask_off", 1: "mask_low", 2: "mask_med",
                 3: "mask_high", 4: "mask_vhigh"}
_overrides = 0
for tier, mask_col in _tier_to_mask.items():
    _was_masked = (bucketed["action_tier"] == tier) & (bucketed[mask_col] == 0)
    _overrides += _was_masked.sum()
    bucketed.loc[_was_masked, mask_col] = 1

print(f"\nObserved-action override: re-enabled {_overrides:,} mask entries")
print(f"  (these were violated by clinicians — preserved for offline RL stability)")

# ── Final mask sanity check ──
print("\nFinal mask: tier-availability per decision point (decision-point rows only):")
_dp = bucketed[bucketed["in_decision_window"] == 1]
for tier_idx, tier_name in ACTION_LABELS.items():
    _mask_col = _tier_to_mask[tier_idx]
    allowed = (_dp[_mask_col] == 1).sum()
    print(f"  {tier_idx} {tier_name:8s}  allowed in {allowed:>6,} / {len(_dp):,} ({allowed/len(_dp)*100:5.1f}%)")

# How many tiers are typically available per decision point?
_dp = bucketed[bucketed["in_decision_window"] == 1]
_n_allowed = _dp[[f"mask_{t}" for t in ["off", "low", "med", "high", "vhigh"]]].sum(axis=1)
print(f"\nNumber of allowed actions per decision point:")
print(_n_allowed.value_counts().sort_index().to_string())

# Drop intermediate state_mask_* and trans_mask_* columns (keep only final mask_*)
_intermediate_cols = [c for c in bucketed.columns
                      if c.startswith("state_mask_") or c.startswith("trans_mask_")]
# REVIEWED: keep prev_action_tier (lagged action) — it is a clean state feature
# that does NOT leak the current action, and replaces the leaky med_cont_*_prev usage.
bucketed = bucketed.drop(columns=_intermediate_cols)
print(f"\nbucketed now has {bucketed.shape[1]} cols (dropped intermediates; kept prev_action_tier)")

# %%
_dp = bucketed[bucketed["in_decision_window"] == 1].copy()
_dp["n_allowed"] = _dp[["mask_off","mask_low","mask_med","mask_high","mask_vhigh"]].sum(axis=1)

_single = _dp[_dp["n_allowed"] == 1]
print(f"Single-allowed-tier decision points: {len(_single)}")
print(_single[["hospitalization_id", "hour", "vital_map", "action_tier",
               "mask_off", "mask_low", "mask_med", "mask_high", "mask_vhigh"]]
      .head(15).to_string(index=False))

# %%
# ── REVIEWED: Lag action-leakage-prone state features by 1 hour ──
# The bucket at hour H aggregates events in [H, H+1), so columns aggregated over
# hour H represent the action ITSELF (current pressor dose, within-hour titration
# pattern, off-pressor counter that resets at H if action_tier[H] > 0). These
# cannot be in the state at decision point H — the model would just decode the
# action. Fix: shift these columns by 1 hour per patient; the first hour gets 0.
# prev_action_tier is already created in the mask cell — we keep that.

_NEE_CONT_MEDS = [f"med_cont_{m}" for m in ohca_config["nee_coefficients"].keys()]
_NEE_DIR_COLS  = ["nee_changes_in_hour", "nee_dir_none", "nee_dir_esc",
                  "nee_dir_desc", "nee_dir_mixed"]
_LAG_COLS = _NEE_CONT_MEDS + _NEE_DIR_COLS + ["hours_since_last_on_pressor"]

bucketed = bucketed.sort_values(["hospitalization_id", "hour"]).reset_index(drop=True)
_grp_lag = bucketed.groupby("hospitalization_id", sort=False)

_added = []
for c in _LAG_COLS:
    if c in bucketed.columns:
        bucketed[f"{c}_prev"] = _grp_lag[c].shift(1).fillna(0)
        _added.append(f"{c}_prev")
    else:
        print(f"  skipped (not in df): {c}")

print(f"Added {len(_added)} lagged columns:")
for c in _added:
    print(f"  - {c}")
print(f"\nprev_action_tier (already kept from mask cell): "
      f"present = {('prev_action_tier' in bucketed.columns)}")
print(f"\nbucketed now has {bucketed.shape[1]} cols")

# Sanity: show one patient's trajectory of action_tier vs prev_action_tier
_sample_pid = bucketed.loc[bucketed["action_tier"] > 0, "hospitalization_id"].iloc[0]
print(f"\nSample patient {_sample_pid} — confirm lag is correct:")
print(bucketed[bucketed["hospitalization_id"] == _sample_pid][[
    "hour", "action_tier", "prev_action_tier",
    "med_cont_norepinephrine", "med_cont_norepinephrine_prev",
    "hours_since_last_on_pressor", "hours_since_last_on_pressor_prev",
]].head(10).to_string(index=False))

# %%
# ── Identify RL cohort: patients with ≥1 decision point ──
_rl_cohort = (bucketed[bucketed["in_decision_window"] == 1]
              .groupby("hospitalization_id").size()
              .reset_index(name="n_decision_points"))
_rl_hosp_ids = set(_rl_cohort["hospitalization_id"])

print(f"RL cohort: {len(_rl_hosp_ids):,} patients with ≥1 decision point")
print(f"  median decision points/patient: {_rl_cohort['n_decision_points'].median():.0f}")
print(f"  total decision points: {_rl_cohort['n_decision_points'].sum():,}")

# Save the RL cohort identifier list for downstream notebooks
_rl_cohort.to_parquet(OUT_DIR / "rl_cohort_reviewed.parquet", index=False)
print(f"Saved RL cohort identifiers → {OUT_DIR / 'rl_cohort.parquet'}")

# ── Save full bucketed (all 1,456 patients, all hours within 120h window) ──
# Useful for future analyses, sensitivity studies, alternative cohort definitions
bucketed_full = bucketed[bucketed["hospitalization_id"].isin(_rl_hosp_ids)].copy()

# Order columns logically
_id_cols       = ["hospitalization_id", "hour", "in_decision_window",
                  "window_close_reason", "exit_hour", "survival_status",
                  "anchor_dttm", "anchor_source", "is_scaffold"]
_action_cols   = ["action_tier", "action_label", "med_cont_nee"]
_mask_cols     = ["mask_off", "mask_low", "mask_med", "mask_high", "mask_vhigh"]
_first_cols    = [c for c in _id_cols + _action_cols + _mask_cols if c in bucketed_full.columns]
_other_cols    = [c for c in bucketed_full.columns if c not in _first_cols]
bucketed_full  = bucketed_full[_first_cols + _other_cols]

# Save
_full_path = OUT_DIR / "bucketed_train_reviewed.parquet"
bucketed_full.to_parquet(_full_path, index=False)
_size_mb = _full_path.stat().st_size / 1024**2
print(f"\nSaved full RL-cohort bucketed → {_full_path}")
print(f"  Rows           : {len(bucketed_full):,}")
print(f"  Cols           : {bucketed_full.shape[1]}")
print(f"  Patients       : {bucketed_full['hospitalization_id'].nunique():,}")
print(f"  Size           : {_size_mb:.1f} MB")

# ── Save decision-points-only view ──
bucketed_dp = bucketed_full[bucketed_full["in_decision_window"] == 1].copy().reset_index(drop=True)
_dp_path = OUT_DIR / "bucketed_decision_points_reviewed.parquet"
bucketed_dp.to_parquet(_dp_path, index=False)
_size_mb_dp = _dp_path.stat().st_size / 1024**2
print(f"\nSaved decision-points-only view → {_dp_path}")
print(f"  Rows           : {len(bucketed_dp):,}")
print(f"  Patients       : {bucketed_dp['hospitalization_id'].nunique():,}")
print(f"  Size           : {_size_mb_dp:.1f} MB")

# ── Final summary ──
print(f"\n{'='*60}")
print(f"NOTEBOOK 03 — DONE (REVIEWED)")
print(f"{'='*60}")
print(f"Outputs in {OUT_DIR}:")
print(f"  bucketed_train_reviewed.parquet            (all 120h × {bucketed_full['hospitalization_id'].nunique():,} pts, {len(bucketed_full):,} rows)")
print(f"  bucketed_decision_points_reviewed.parquet  (DP rows × {bucketed_dp['hospitalization_id'].nunique():,} pts, {len(bucketed_dp):,} rows)")
print(f"  rl_cohort_reviewed.parquet                 ({len(_rl_cohort):,} pts × n_decision_points)")

print(f"\nState feature inventory:")
print(f"  Time-varying clinical (~51): vital_*, lab_*, resp_*, assess_*, med_cont_*, med_int_*, on_crrt")
print(f"  Within-hour titration: nee_changes_in_hour, nee_dir_{{none,esc,desc,mixed}}")
print(f"  Course context: hours_since_last_on_pressor")
print(f"  Snapshot (at first pressor, carried forward):")
print(f"    snapshot_hours_anchor_to_first_pressor, snapshot_map_at_first_pressor,")
print(f"    snapshot_lactate_at_first_pressor, snapshot_sofa_24h, snapshot_sofa_24h_available")
print(f"  Temporal: hours_since_anchor, hours_since_first_pressor")
print(f"  Action+masks: action_tier (0-4), mask_off/low/med/high/vhigh")
print(f"  Decision filter: in_decision_window (1 = train on this row)")

print(f"\nNext step → 05_reward_reviewed.ipynb (disposition mapping + intermediate + terminal reward)")

print(f"\nREVIEWED notes:")
print(f"  - prev_action_tier kept in state (lagged action, no leak)")
print(f"  - *_prev columns added for NEE-contributing meds, nee_dir_*, hours_since_last_on_pressor")
print(f"  - downstream notebooks (reward_reviewed, model_reviewed) drop the current-hour leaky cols from state")

# %%
