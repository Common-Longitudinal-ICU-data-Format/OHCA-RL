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
from pathlib import Path
import json
from pathlib import Path

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


LOCAL_OVERRIDE = CONFIG_DIR / "ohca_rl_config_local.yaml"

local_yaml = """# Local overrides for ohca_rl_config.yaml
# Only put keys that differ from upstream here.

# Add meds listed in the upstream README but missing from the upstream YAML.
# These are NOT used in NEE calculation (so don't affect actions), but matter
# as state features (sedation depth, hemodynamic context).
meds_continuous_of_interest:
  # Vasoactives
  - norepinephrine
  - epinephrine
  - phenylephrine
  - vasopressin
  - dopamine
  - angiotensin
  - dobutamine
  - milrinone
  - isoproterenol
  # Antihypertensives (added)
  - nicardipine
  - clevidipine
  # Sedatives / analgesics
  - propofol
  - midazolam
  - lorazepam
  - dexmedetomidine
  - ketamine          # added
  - fentanyl_drip     # added
  # NMB
  - vecuronium
  - rocuronium
  - cisatracurium

meds_continuous_preferred_units:
  nicardipine: mg/hr
  clevidipine: mg/hr
  ketamine: mcg/kg/min
  fentanyl_drip: mcg/kg/hr
"""

LOCAL_OVERRIDE.write_text(local_yaml)
print(f"Wrote {LOCAL_OVERRIDE}")
print(f"\n{LOCAL_OVERRIDE.read_text()}")

# %%
import sys, json, logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# ── Variable config (deep-merge upstream + optional local overrides) ──
def deep_merge(base, override):
    out = dict(base) if base else {}
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

with open(CONFIG_DIR / "ohca_rl_config.yaml") as f:
    _base = yaml.safe_load(f)

_local_path = CONFIG_DIR / "ohca_rl_config_local.yaml"
_local = {}
if _local_path.exists():
    with open(_local_path) as f:
        _local = yaml.safe_load(f) or {}
    print(f"Applied local overrides from {_local_path.name}")

ohca_config = deep_merge(_base, _local)

# ── Make code/utils.py importable ──
sys.path.insert(0, str(CODE_DIR))
for _stale in ("utils",):
    if _stale in sys.modules:
        del sys.modules[_stale]
import utils
print(f"utils loaded from: {utils.__file__}")

# ── clifpy + project utils ──
import clifpy
from clifpy import (Vitals, Labs, MedicationAdminContinuous,
                    MedicationAdminIntermittent, RespiratorySupport,
                    CrrtTherapy, PatientAssessments, Adt)
from clifpy.utils.outlier_handler import apply_outlier_handling
from utils import (build_weight_table, convert_med_doses,
                   categorize_device_from_tracheostomy, categorize_device,
                   impute_fio2)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("01_wide_df")

# ── Load cohort from notebook 00 ──
cohort_df = pd.read_parquet(OUT_DIR / "cohort_ohca_icu.parquet")
cohort_hosp_ids = cohort_df["hospitalization_id"].astype(str).unique().tolist()

print(f"\nSite           : {SITE_NAME}")
print(f"Tables path    : {TABLES_PATH}")
print(f"Cohort         : {len(cohort_hosp_ids):,} hospitalizations")
print(f"\nVariables to pull from CLIF:")
print(f"  labs       : {len(ohca_config['labs_of_interest'])}")
print(f"  vitals     : {len(ohca_config['vitals_of_interest'])}")
print(f"  med_cont   : {len(ohca_config['meds_continuous_of_interest'])}")
print(f"  med_int    : {len(ohca_config['meds_intermittent_of_interest'])}")
print(f"  assess     : {len(ohca_config['assessments_of_interest'])}")

# %%
print("med_cont list:")
for m in ohca_config["meds_continuous_of_interest"]:
    print(f"  - {m}")

# %%
# ── Load vitals ──
vitals_tbl = Vitals.from_file(
    data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
    filters={"hospitalization_id": cohort_hosp_ids,
             "vital_category": ohca_config["vitals_of_interest"]},
)
apply_outlier_handling(vitals_tbl)

vitals_df = vitals_tbl.df.copy()
vitals_df["hospitalization_id"] = vitals_df["hospitalization_id"].astype(str)
vitals_df["vital_category"]     = vitals_df["vital_category"].str.lower()

# ── Build weight lookup (needed by continuous-med unit conversion below) ──
weight_df = build_weight_table(vitals_df)
weight_df.to_parquet(OUT_DIR / "weight_lookup.parquet", index=False)

# ── Pivot long → wide ──
vitals_wide = vitals_df.pivot_table(
    index=["hospitalization_id", "recorded_dttm"],
    columns="vital_category",
    values="vital_value",
    aggfunc="first",
).reset_index()
vitals_wide.columns = [f"vital_{c}" if c not in ("hospitalization_id", "recorded_dttm") else c
                       for c in vitals_wide.columns]
vitals_wide = vitals_wide.rename(columns={"recorded_dttm": "event_dttm"})

# ── Recompute MAP from SBP + DBP where both are available ──
# Recorded MAP often contains transducer artifacts; derived MAP is cleaner.
if {"vital_sbp", "vital_dbp"}.issubset(vitals_wide.columns):
    _has_bp = vitals_wide["vital_sbp"].notna() & vitals_wide["vital_dbp"].notna()
    vitals_wide["vital_map"] = np.where(
        _has_bp,
        (vitals_wide["vital_sbp"] + 2 * vitals_wide["vital_dbp"]) / 3,
        vitals_wide.get("vital_map", np.nan),
    )
    print(f"Recomputed MAP for {_has_bp.sum():,} rows where SBP+DBP both present")

# ── Summary ──
_vital_cols = sorted(c for c in vitals_wide.columns if c.startswith("vital_"))
print(f"\nVitals wide: {len(vitals_wide):,} rows, "
      f"{vitals_wide['hospitalization_id'].nunique():,} hosps")
print(f"  columns ({len(_vital_cols)}): {_vital_cols}")
print(f"\nWeight lookup: {len(weight_df):,} measurements, "
      f"{weight_df['hospitalization_id'].nunique():,}/{len(cohort_hosp_ids):,} patients with any weight")

print(f"\nValue ranges (post outlier handling):")
print(vitals_wide[_vital_cols].describe(percentiles=[.5]).T[["count", "mean", "50%", "min", "max"]].round(1).to_string())

# %%
# ── Load labs ──
labs_tbl = Labs.from_file(
    data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
    filters={"hospitalization_id": cohort_hosp_ids,
             "lab_category": ohca_config["labs_of_interest"]},
)
apply_outlier_handling(labs_tbl)

labs_df = labs_tbl.df.copy()
labs_df["hospitalization_id"] = labs_df["hospitalization_id"].astype(str)
labs_df["lab_category"]       = labs_df["lab_category"].str.lower()

# ── Pivot long → wide on lab_result_dttm ──
labs_wide = labs_df.pivot_table(
    index=["hospitalization_id", "lab_result_dttm"],
    columns="lab_category",
    values="lab_value_numeric",
    aggfunc="first",
).reset_index()
labs_wide.columns = [f"lab_{c}" if c not in ("hospitalization_id", "lab_result_dttm") else c
                     for c in labs_wide.columns]
labs_wide = labs_wide.rename(columns={"lab_result_dttm": "event_dttm"})

# ── Keep only labs of interest ──
_lab_keep = ["hospitalization_id", "event_dttm"] + \
            [f"lab_{c}" for c in ohca_config["labs_of_interest"]]
labs_wide = labs_wide[[c for c in _lab_keep if c in labs_wide.columns]]

_lab_cols = sorted(c for c in labs_wide.columns if c.startswith("lab_"))
print(f"Labs wide: {len(labs_wide):,} rows, "
      f"{labs_wide['hospitalization_id'].nunique():,} hosps")
print(f"  columns ({len(_lab_cols)}): {_lab_cols}")

print(f"\nValue ranges (post outlier handling):")
print(labs_wide[_lab_cols].describe(percentiles=[.5]).T[["count", "mean", "50%", "min", "max"]].round(2).to_string())

# Sanity check: how many patients have any lactate (critical for the spec's
# snapshot-at-first-pressor feature)
if "lab_lactate" in labs_wide.columns:
    _lac_pts = labs_wide.loc[labs_wide["lab_lactate"].notna(), "hospitalization_id"].nunique()
    print(f"\nPatients with any lactate: {_lac_pts:,} / {len(cohort_hosp_ids):,}")

# %%
# ── Load continuous meds ──
meds_cont_tbl = MedicationAdminContinuous.from_file(
    data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
    filters={"hospitalization_id": cohort_hosp_ids,
             "med_category": ohca_config["meds_continuous_of_interest"]},
)
apply_outlier_handling(meds_cont_tbl)

meds_cont_df = meds_cont_tbl.df.copy()
meds_cont_df["hospitalization_id"] = meds_cont_df["hospitalization_id"].astype(str)
meds_cont_df["med_category"]       = meds_cont_df["med_category"].str.lower()

# Drop NaN doses and explicitly negative doses
meds_cont_df = meds_cont_df[
    meds_cont_df["med_dose"].notna() & (meds_cont_df["med_dose"] >= 0)
].copy()

# ── Inspect unit diversity BEFORE conversion ──
print("Unit diversity BEFORE conversion:")
_pre = meds_cont_df.groupby(["med_category", "med_dose_unit"]).size().reset_index(name="n")
print(_pre.sort_values(["med_category", "n"], ascending=[True, False]).to_string(index=False))

# ── Convert all doses to canonical units (mcg/kg/min for vasoactives, etc.) ──
print("\nRunning unit conversion ...")
meds_cont_df, _conv_counts = convert_med_doses(
    meds_cont_df, weight_df,
    ohca_config["meds_continuous_preferred_units"],
)
if "med_dose_converted" in meds_cont_df.columns:
    meds_cont_df["med_dose"] = meds_cont_df["med_dose_converted"]

# ── Save raw long-form (needed in step 02 for within-hour NEE change counts) ──
# Spec calls for `nee_changes_in_hour` and `nee_direction_in_hour` features,
# which need the original event-level dose stream (not just hourly aggregates).
meds_cont_df.to_parquet(OUT_DIR / "meds_cont_df.parquet", index=False)
print(f"\nSaved raw long-form continuous meds → {OUT_DIR / 'meds_cont_df.parquet'}")
print(f"  {len(meds_cont_df):,} rows post-conversion")

# ── Pivot to wide ──
meds_cont_wide = meds_cont_df.pivot_table(
    index=["hospitalization_id", "admin_dttm"],
    columns="med_category", values="med_dose", aggfunc="first",
).reset_index()
meds_cont_wide.columns = [f"med_cont_{c}" if c not in ("hospitalization_id", "admin_dttm") else c
                          for c in meds_cont_wide.columns]
meds_cont_wide = meds_cont_wide.rename(columns={"admin_dttm": "event_dttm"})
if meds_cont_wide["event_dttm"].dtype == object:
    meds_cont_wide["event_dttm"] = pd.to_datetime(meds_cont_wide["event_dttm"], utc=True)

# Keep only the med columns we configured
_keep = ["hospitalization_id", "event_dttm"] + \
        [f"med_cont_{m}" for m in ohca_config["meds_continuous_of_interest"]]
meds_cont_wide = meds_cont_wide[[c for c in _keep if c in meds_cont_wide.columns]]

# ── Summary ──
print(f"\nContinuous meds wide: {len(meds_cont_wide):,} rows, "
      f"{meds_cont_wide['hospitalization_id'].nunique():,} hosps")
print(f"\nPer-med charting counts (post-conversion, dose > 0):")
for _c in sorted(c for c in meds_cont_wide.columns if c.startswith("med_cont_")):
    _n_rows = (meds_cont_wide[_c] > 0).sum()
    _n_pts  = meds_cont_wide.loc[meds_cont_wide[_c] > 0, "hospitalization_id"].nunique()
    if _n_rows:
        print(f"  {_c:30s} {_n_rows:>10,} rows  {_n_pts:>5,} patients")

# Patients who received ANY vasopressor (spec inclusion criterion preview)
_vaso_meds = ["med_cont_norepinephrine", "med_cont_epinephrine",
              "med_cont_phenylephrine", "med_cont_vasopressin",
              "med_cont_dopamine", "med_cont_angiotensin"]
_vaso_cols = [c for c in _vaso_meds if c in meds_cont_wide.columns]
if _vaso_cols:
    _on_vaso = (meds_cont_wide[_vaso_cols].fillna(0) > 0).any(axis=1)
    _n_ever_vaso = meds_cont_wide.loc[_on_vaso, "hospitalization_id"].nunique()
    print(f"\nPatients who ever received any vasopressor: {_n_ever_vaso:,} / {len(cohort_hosp_ids):,}")

# %%
try:
    meds_int_tbl = MedicationAdminIntermittent.from_file(
        data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_hosp_ids,
                 "med_category": ohca_config["meds_intermittent_of_interest"]},
    )
    apply_outlier_handling(meds_int_tbl)
    _int_df = meds_int_tbl.df.copy()
    _int_df["hospitalization_id"] = _int_df["hospitalization_id"].astype(str)
    _int_df["med_category"]       = _int_df["med_category"].str.lower()
    _int_df = _int_df[_int_df["med_dose"].notna()].copy()

    # Inspect unit diversity before conversion
    if len(_int_df):
        print("Intermittent unit diversity BEFORE conversion:")
        _pre = _int_df.groupby(["med_category", "med_dose_unit"]).size().reset_index(name="n")
        print(_pre.sort_values(["med_category", "n"], ascending=[True, False]).to_string(index=False))

    # Unit-convert (intermittents are bolus mg, no rate)
    _int_pref = ohca_config.get("meds_intermittent_preferred_units", {})
    if len(_int_df) and _int_pref:
        _int_df, _ = convert_med_doses(_int_df, weight_df, _int_pref)
        if "med_dose_converted" in _int_df.columns:
            _int_df["med_dose"] = _int_df["med_dose_converted"]

    # Pivot
    meds_int_wide = _int_df.pivot_table(
        index=["hospitalization_id", "admin_dttm"],
        columns="med_category", values="med_dose", aggfunc="first",
    ).reset_index()
    meds_int_wide.columns = [f"med_int_{c}" if c not in ("hospitalization_id", "admin_dttm") else c
                             for c in meds_int_wide.columns]
    meds_int_wide = meds_int_wide.rename(columns={"admin_dttm": "event_dttm"})
    if meds_int_wide["event_dttm"].dtype == object:
        meds_int_wide["event_dttm"] = pd.to_datetime(meds_int_wide["event_dttm"], utc=True)

    _keep = ["hospitalization_id", "event_dttm"] + \
            [f"med_int_{m}" for m in ohca_config["meds_intermittent_of_interest"]]
    meds_int_wide = meds_int_wide[[c for c in _keep if c in meds_int_wide.columns]]

    print(f"\nIntermittent meds wide: {len(meds_int_wide):,} rows, "
          f"{meds_int_wide['hospitalization_id'].nunique():,} hosps")
    print(f"Per-med charting counts:")
    for _c in sorted(c for c in meds_int_wide.columns if c.startswith("med_int_")):
        _n_rows = (meds_int_wide[_c] > 0).sum()
        _n_pts  = meds_int_wide.loc[meds_int_wide[_c] > 0, "hospitalization_id"].nunique()
        if _n_rows:
            print(f"  {_c:30s} {_n_rows:>6,} rows  {_n_pts:>5,} patients")
except Exception as e:
    meds_int_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
    print(f"Intermittent meds: skipped ({type(e).__name__}: {e})")

# %%
print("Current med_int list in loaded config:")
for m in ohca_config["meds_intermittent_of_interest"]:
    print(f"  - {m}")

# %%
# Load respiratory support
resp_tbl = RespiratorySupport.from_file(
    data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
    filters={"hospitalization_id": cohort_hosp_ids},
)
apply_outlier_handling(resp_tbl)

# Pre-waterfall device categorization
resp_tbl.df = categorize_device_from_tracheostomy(resp_tbl.df)
resp_tbl.df = categorize_device(resp_tbl.df)

# clifpy waterfall: ffill device/mode/params + infer IMV/NIPPV
print("Running clifpy respiratory waterfall (may take 30–90s)...")
resp_tbl = resp_tbl.waterfall()

# Post-waterfall FiO2 imputation for residual gaps (NC LPM → FiO2 lookup)
resp_tbl.df = impute_fio2(resp_tbl.df)

resp_df = resp_tbl.df.copy()
resp_df["hospitalization_id"] = resp_df["hospitalization_id"].astype(str)

# Prefix columns
_rename = {c: f"resp_{c}" for c in resp_df.columns
           if c not in ("hospitalization_id", "recorded_dttm")}
resp_wide = resp_df.rename(columns=_rename).rename(columns={"recorded_dttm": "event_dttm"})

# Keep columns of interest
_keep = ["hospitalization_id", "event_dttm",
         "resp_device_name", "resp_device_category", "resp_mode_name",
         "resp_mode_category", "resp_vent_brand_name", "resp_artificial_airway",
         "resp_tracheostomy", "resp_fio2_set", "resp_lpm_set",
         "resp_tidal_volume_set", "resp_resp_rate_set", "resp_peep_set",
         "resp_tidal_volume_obs", "resp_resp_rate_obs"]
resp_wide = resp_wide[[c for c in _keep if c in resp_wide.columns]]

print(f"\nRespiratory wide: {len(resp_wide):,} rows, "
      f"{resp_wide['hospitalization_id'].nunique():,} hosps, "
      f"{sum(c.startswith('resp_') for c in resp_wide.columns)} cols")

# Coverage by device category
if "resp_device_category" in resp_wide.columns:
    print(f"\nDevice category distribution:")
    print(resp_wide["resp_device_category"].value_counts(dropna=False).head(10).to_string())

# %%
try:
    crrt_tbl = CrrtTherapy.from_file(
        data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_hosp_ids},
    )
    crrt_df = crrt_tbl.df.copy()
    crrt_df["hospitalization_id"] = crrt_df["hospitalization_id"].astype(str)
    crrt_df["on_crrt"] = 1
    _rename = {c: f"crrt_{c}" for c in crrt_df.columns
               if c not in ("hospitalization_id", "recorded_dttm", "on_crrt")}
    crrt_wide = crrt_df.rename(columns=_rename).rename(columns={"recorded_dttm": "event_dttm"})
    _keep = ["hospitalization_id", "event_dttm", "on_crrt",
             "crrt_crrt_mode_name", "crrt_crrt_mode_category"]
    crrt_wide = crrt_wide[[c for c in _keep if c in crrt_wide.columns]]
    print(f"CRRT wide: {len(crrt_wide):,} rows, "
          f"{crrt_wide['hospitalization_id'].nunique()} hosps on CRRT")
    if "crrt_crrt_mode_category" in crrt_wide.columns:
        print(f"Mode distribution:")
        print(crrt_wide["crrt_crrt_mode_category"].value_counts(dropna=False).to_string())
except Exception as e:
    crrt_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
    print(f"CRRT skipped ({type(e).__name__}: {e})")
crrt_wide.head()

# %%
try:
    assess_tbl = PatientAssessments.from_file(
        data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
        filters={"hospitalization_id": cohort_hosp_ids,
                 "assessment_category": ohca_config["assessments_of_interest"]},
    )
    apply_outlier_handling(assess_tbl)

    assess_df = assess_tbl.df.copy()
    assess_df["hospitalization_id"]  = assess_df["hospitalization_id"].astype(str)
    assess_df["assessment_category"] = assess_df["assessment_category"].str.lower()

    # Pre-pivot diagnostic — see what categories MIMIC actually returned
    print("Assessment category counts (raw):")
    print(assess_df["assessment_category"].value_counts(dropna=False).to_string())

    assess_wide = assess_df.pivot_table(
        index=["hospitalization_id", "recorded_dttm"],
        columns="assessment_category", values="numerical_value", aggfunc="first",
    ).reset_index()
    assess_wide.columns = [f"assess_{c}" if c not in ("hospitalization_id", "recorded_dttm") else c
                           for c in assess_wide.columns]
    assess_wide = assess_wide.rename(columns={"recorded_dttm": "event_dttm"})
    if assess_wide["event_dttm"].dtype == object:
        assess_wide["event_dttm"] = pd.to_datetime(assess_wide["event_dttm"], utc=True)

    _keep = ["hospitalization_id", "event_dttm", "assess_gcs_total", "assess_rass"]
    assess_wide = assess_wide[[c for c in _keep if c in assess_wide.columns]]

    print(f"\nAssessments wide: {len(assess_wide):,} rows, "
          f"{assess_wide['hospitalization_id'].nunique():,} hosps")

    for _c in [c for c in assess_wide.columns if c.startswith("assess_")]:
        _n = assess_wide[_c].notna().sum()
        _pts = assess_wide.loc[assess_wide[_c].notna(), "hospitalization_id"].nunique()
        print(f"  {_c:25s} {_n:>8,} values  {_pts:>5,} patients")
        print(f"    {assess_wide[_c].describe(percentiles=[.5]).round(1).to_dict()}")
except Exception as e:
    assess_wide = pd.DataFrame(columns=["hospitalization_id", "event_dttm"])
    print(f"Assessments skipped ({type(e).__name__}: {e})")

# %%
adt_tbl = Adt.from_file(
    data_directory=TABLES_PATH, filetype=FILE_TYPE, timezone=TIMEZONE,
    filters={"hospitalization_id": cohort_hosp_ids},
)
adt_df = adt_tbl.df.copy()
adt_df["hospitalization_id"] = adt_df["hospitalization_id"].astype(str)
if "location_category" in adt_df.columns:
    adt_df["location_category"] = adt_df["location_category"].str.lower()

# ── Extract earliest ED arrival per hospitalization ──
# New spec defines anchor time as the earliest of: ED arrival, first vital, first lab.
# We compute the ED arrival component here; first-vital and first-lab will be
# derived in cell 10 from the wide_df, then we combine them into the anchor.
ed_arrival = (adt_df.loc[adt_df["location_category"] == "ed",
                         ["hospitalization_id", "in_dttm"]]
              .groupby("hospitalization_id")["in_dttm"].min()
              .reset_index().rename(columns={"in_dttm": "ed_first_dttm"}))
print(f"Patients with ED arrival recorded: {len(ed_arrival):,} / {len(cohort_hosp_ids):,}")

# Build adt_wide for the merge
adt_wide = adt_df.rename(columns={"in_dttm": "event_dttm",
                                  "location_category": "adt_location_category"})
if "out_dttm" in adt_wide.columns:
    adt_wide = adt_wide.rename(columns={"out_dttm": "adt_out_dttm"})

_keep = ["hospitalization_id", "event_dttm", "adt_location_category"]
if "adt_out_dttm" in adt_wide.columns:
    _keep.append("adt_out_dttm")
adt_wide = adt_wide[[c for c in _keep if c in adt_wide.columns]]

print(f"\nADT wide: {len(adt_wide):,} rows, "
      f"{adt_wide['hospitalization_id'].nunique():,} hosps")

# Location distribution (each row is a location movement, so this tells us
# what locations patients passed through, not where they spent time)
print(f"\nLocation transitions:")
print(adt_wide["adt_location_category"].value_counts(dropna=False).to_string())

# %%
# ── Outer-merge all event sources on (hospitalization_id, event_dttm) ──
wide_df = vitals_wide.copy()
for _name, _df in [("labs",     labs_wide),
                   ("meds_cont", meds_cont_wide),
                   ("meds_int",  meds_int_wide),
                   ("resp",      resp_wide),
                   ("crrt",      crrt_wide),
                   ("assess",    assess_wide),
                   ("adt",       adt_wide)]:
    if len(_df) and "event_dttm" in _df.columns:
        # Ensure tz consistency before merge
        if pd.api.types.is_datetime64_any_dtype(_df["event_dttm"]):
            if _df["event_dttm"].dt.tz is None:
                _df = _df.copy()
                _df["event_dttm"] = pd.to_datetime(_df["event_dttm"], utc=True)
        wide_df = wide_df.merge(_df, on=["hospitalization_id", "event_dttm"],
                                how="outer", suffixes=("", f"_{_name}_dup"))

wide_df = wide_df.sort_values(["hospitalization_id", "event_dttm"]).reset_index(drop=True)
print(f"Wide df merged: {len(wide_df):,} rows × {wide_df.shape[1]} cols")

# ── ANCHOR TIME (per new spec) ──
# Anchor = earliest of ED arrival, first vital, first lab.
first_vital = (vitals_wide.groupby("hospitalization_id")["event_dttm"].min()
               .reset_index().rename(columns={"event_dttm": "first_vital_dttm"}))
first_lab   = (labs_wide.groupby("hospitalization_id")["event_dttm"].min()
               .reset_index().rename(columns={"event_dttm": "first_lab_dttm"}))

anchor = (pd.DataFrame({"hospitalization_id": cohort_hosp_ids})
          .merge(ed_arrival,  on="hospitalization_id", how="left")
          .merge(first_vital, on="hospitalization_id", how="left")
          .merge(first_lab,   on="hospitalization_id", how="left"))
anchor["anchor_dttm"] = anchor[["ed_first_dttm", "first_vital_dttm", "first_lab_dttm"]].min(axis=1)

# Tag which source contributed the anchor (for QC)
def _which_anchor(row):
    cands = {"ed":    row["ed_first_dttm"],
             "vital": row["first_vital_dttm"],
             "lab":   row["first_lab_dttm"]}
    valid = {k: v for k, v in cands.items() if pd.notna(v)}
    if not valid:
        return None
    return min(valid, key=valid.get)
anchor["anchor_source"] = anchor.apply(_which_anchor, axis=1)

print(f"\nAnchor source distribution:")
print(anchor["anchor_source"].value_counts(dropna=False).to_string())
print(f"\nPatients with NO anchor (no ED + no vital + no lab): "
      f"{anchor['anchor_dttm'].isna().sum()} / {len(anchor)}")

anchor[["hospitalization_id", "anchor_dttm", "anchor_source",
        "ed_first_dttm", "first_vital_dttm", "first_lab_dttm"]] \
    .to_parquet(OUT_DIR / "anchor_mapping.parquet", index=False)
print(f"Saved → {OUT_DIR / 'anchor_mapping.parquet'}")

# ── FIRST VASOPRESSOR DOSE TIME (per new spec) ──
# Decision-point window starts here.
_vaso_meds = ["med_cont_norepinephrine", "med_cont_epinephrine",
              "med_cont_phenylephrine", "med_cont_vasopressin",
              "med_cont_dopamine", "med_cont_angiotensin"]
_vaso_cols = [c for c in _vaso_meds if c in wide_df.columns]

if _vaso_cols:
    _on_vaso = (wide_df[_vaso_cols].fillna(0) > 0).any(axis=1)
    first_vaso = (wide_df.loc[_on_vaso, ["hospitalization_id", "event_dttm"]]
                  .groupby("hospitalization_id")["event_dttm"].min()
                  .reset_index().rename(columns={"event_dttm": "first_vaso_dttm"}))
    first_vaso.to_parquet(OUT_DIR / "first_vaso_mapping.parquet", index=False)
    print(f"\nPatients ever on a vasopressor: {len(first_vaso):,} / {len(cohort_hosp_ids):,}")
    print(f"Saved → {OUT_DIR / 'first_vaso_mapping.parquet'}")
else:
    print("\nWARNING: no vasopressor columns found in wide_df")
    first_vaso = pd.DataFrame(columns=["hospitalization_id", "first_vaso_dttm"])

# ── Save the merged wide df ──
wide_df.to_parquet(OUT_DIR / "wide_df.parquet", index=False)
size_mb = (OUT_DIR / "wide_df.parquet").stat().st_size / 1024**2

print(f"\n{'='*60}")
print(f"wide_df: {len(wide_df):,} rows × {wide_df.shape[1]} cols ({size_mb:.1f} MB)")
print(f"  Hospitalizations : {wide_df['hospitalization_id'].nunique():,}")
print(f"  Vitals cols      : {sum(c.startswith('vital_')    for c in wide_df.columns)}")
print(f"  Labs cols        : {sum(c.startswith('lab_')      for c in wide_df.columns)}")
print(f"  Med-cont cols    : {sum(c.startswith('med_cont_') for c in wide_df.columns)}")
print(f"  Med-int cols     : {sum(c.startswith('med_int_')  for c in wide_df.columns)}")
print(f"  Resp cols        : {sum(c.startswith('resp_')     for c in wide_df.columns)}")
print(f"  Assess cols      : {sum(c.startswith('assess_')   for c in wide_df.columns)}")
print(f"  ADT cols         : {sum(c.startswith('adt_')      for c in wide_df.columns)}")
print(f"  on_crrt          : {'on_crrt' in wide_df.columns}")
print(f"  Saved to         : {OUT_DIR / 'wide_df.parquet'}")

# %%
configured = set(ohca_config["meds_continuous_of_interest"])
present    = {c.replace("med_cont_", "") for c in wide_df.columns if c.startswith("med_cont_")}
missing    = configured - present
print(f"Configured: {len(configured)}, present: {len(present)}, missing: {sorted(missing)}")

configured_int = set(ohca_config["meds_intermittent_of_interest"])
present_int    = {c.replace("med_int_", "") for c in wide_df.columns if c.startswith("med_int_")}
print(f"Intermittent configured: {sorted(configured_int)}, present: {sorted(present_int)}")


# %%
# Column-group missingness (how much of each group is NaN on average across all rows)
def _miss(prefix):
    cols = [c for c in wide_df.columns if c.startswith(prefix)]
    return f"{wide_df[cols].isna().mean().mean() * 100:.1f}%" if cols else "—"

print("Average missingness by column group (event-level, before bucketing):")
for _p in ["vital_", "lab_", "med_cont_", "med_int_", "resp_", "assess_"]:
    cols_n = sum(c.startswith(_p) for c in wide_df.columns)
    print(f"  {_p:12s} ({cols_n:>2} cols)  {_miss(_p)}")

# Per-source coverage by patient
print(f"\nPer-source patient coverage:")
for _p, _label in [("vital_", "vitals"), ("lab_", "labs"),
                   ("med_cont_", "any continuous med"),
                   ("resp_", "respiratory"), ("assess_", "assess (gcs/rass)")]:
    cols = [c for c in wide_df.columns if c.startswith(_p)]
    if not cols: continue
    _has = wide_df[cols].notna().any(axis=1)
    _pts = wide_df.loc[_has, "hospitalization_id"].nunique()
    print(f"  {_label:25s} {_pts:>5,} / {len(cohort_hosp_ids):,}")

# Anchor → first-vaso lag (early vs late initiator distribution)
if len(first_vaso):
    _lag = (first_vaso.merge(anchor[["hospitalization_id", "anchor_dttm"]],
                             on="hospitalization_id")
            .assign(lag_h=lambda d: (d["first_vaso_dttm"] - d["anchor_dttm"])
                                    .dt.total_seconds() / 3600))
    print(f"\nHours from anchor → first vasopressor (n={_lag['lag_h'].notna().sum():,}):")
    print(_lag["lag_h"].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    print(f"  Immediate initiators (≤0.5h): {(_lag['lag_h'] <= 0.5).sum():,}")
    print(f"  Within 24h                  : {(_lag['lag_h'] <= 24).sum():,}")
    print(f"  After 24h                   : {(_lag['lag_h'] > 24).sum():,}")
    print(f"  >120h or never              : {(_lag['lag_h'] > 120).sum():,}")

# %%
