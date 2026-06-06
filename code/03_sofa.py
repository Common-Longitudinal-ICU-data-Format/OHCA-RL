# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import json
from pathlib import Path
import pandas as pd

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


anchor_map = pd.read_parquet(OUT_DIR / "anchor_mapping.parquet")
anchor_map["hospitalization_id"] = anchor_map["hospitalization_id"].astype(str)
print(f"Site: {SITE_NAME}  |  anchor_map: {len(anchor_map):,} rows")


# %%
# ── Compute 0-24h SOFA via clifpy.utils.sofa_polars.compute_sofa_polars ──
# Replaces buggy pandas compute_sofa (which scored 992 UCMC patients with
# P/F < 100 as sofa_resp = 0). Polars version correctly handles respiratory.

import polars as pl
from clifpy.utils.sofa_polars import compute_sofa_polars

anchor_with_window = anchor_map[["hospitalization_id", "anchor_dttm"]].copy()
anchor_with_window["start_dttm"] = anchor_with_window["anchor_dttm"]
anchor_with_window["end_dttm"]   = anchor_with_window["anchor_dttm"] + pd.Timedelta(hours=24)
anchor_with_window["hospitalization_id"] = anchor_with_window["hospitalization_id"].astype(str)
anchor_with_window = anchor_with_window.dropna(subset=["anchor_dttm"]).copy()

cohort_window_pl = pl.from_pandas(
    anchor_with_window[["hospitalization_id", "start_dttm", "end_dttm"]]
)
print(f"Built 0-24h cohort window: {len(cohort_window_pl):,} patients")

print("Computing SOFA via compute_sofa_polars (loads raw CLIF tables internally)...")
sofa_pl = compute_sofa_polars(
    data_directory=TABLES_PATH,
    cohort_df=cohort_window_pl,
    filetype=FILE_TYPE,
    id_name="hospitalization_id",
    extremal_type="worst",
    fill_na_scores_with_zero=True,
    remove_outliers=False,
    timezone=TIMEZONE,
)
sofa_0_24 = sofa_pl.to_pandas()
sofa_0_24["hospitalization_id"] = sofa_0_24["hospitalization_id"].astype(str)
print(f"compute_sofa_polars returned: {len(sofa_0_24):,} patients × {sofa_0_24.shape[1]} cols")


# %%
# ── Restrict to schema downstream code expects + save ──
# Original pandas version output these 10 columns (hospitalization_id + 9 SOFA).
# Drop the raw clinical data polars also returns (creatinine, weight_kg, etc.)
# so downstream merges + regression code don't see extra noise.

KEEP = ["hospitalization_id"] + ['p_f', 'p_f_imputed', 'sofa_cv_97', 'sofa_coag', 'sofa_liver', 'sofa_resp', 'sofa_cns', 'sofa_renal', 'sofa_total']

# Apply _0_24 suffix to the score columns
sofa_0_24 = sofa_0_24[[c for c in KEEP if c in sofa_0_24.columns]].copy()
sofa_0_24 = sofa_0_24.rename(columns={c: f"{c}_0_24" for c in sofa_0_24.columns if c != "hospitalization_id"})

print(f"Final schema: {len(sofa_0_24.columns)} cols = {list(sofa_0_24.columns)}")
print(f"\nSOFA total distribution:")
print(sofa_0_24["sofa_total_0_24"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
print(f"\nSubscore medians:")
for c in sorted(c for c in sofa_0_24.columns if c.startswith("sofa_")):
    print(f"  {c:30s} median={sofa_0_24[c].median():.1f}")

sofa_0_24.to_parquet(OUT_DIR / "sofa_0_24_reviewed.parquet", index=False)
print(f"\n✓ Saved → {OUT_DIR / 'sofa_0_24_reviewed.parquet'}  ({sofa_0_24.shape})")

