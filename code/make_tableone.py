# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Table 1 — OHCA-RL cohort baseline characteristics
#
# Generates `output/final/tableone.csv` (and `.html` for readability) summarizing
# demographics, baseline severity, and outcomes of the OHCA-RL cohort.
# Stratified by survival status.

# %%
import json
from pathlib import Path

import pandas as pd
from tableone import TableOne

# %%
_config_path = Path("config/config.json")
if not _config_path.exists():
    for p in Path.cwd().parents:
        if (p / "config" / "config.json").exists():
            _config_path = p / "config" / "config.json"
            break

with open(_config_path) as f:
    site_config = json.load(f)

PROJECT_ROOT = Path(site_config["project_root"])
SITE_NAME    = site_config["site_name"]
OUT_DIR      = PROJECT_ROOT / "output" / "intermediate"
FINAL_DIR    = PROJECT_ROOT / "output" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Site: {SITE_NAME}")

from utils import init_log_capture
init_log_capture(__file__, PROJECT_ROOT)

# %%
cohort = pd.read_parquet(OUT_DIR / "cohort_ohca_icu.parquet")
static = pd.read_parquet(OUT_DIR / "patient_static.parquet")
sofa   = pd.read_parquet(OUT_DIR / "sofa_0_24_reviewed.parquet")
disp   = pd.read_parquet(OUT_DIR / "patient_disposition_reviewed.parquet")

df = (
    static
    .merge(sofa[["hospitalization_id", "sofa_total_0_24",
                 "sofa_cv_97_0_24", "sofa_resp_0_24", "sofa_renal_0_24",
                 "sofa_cns_0_24", "sofa_liver_0_24", "sofa_coag_0_24"]],
           on="hospitalization_id", how="left")
    .merge(disp[["hospitalization_id", "cpc_tier"]],
           on="hospitalization_id", how="left")
)

df["length_of_stay_days"] = (
    (df["discharge_dttm"] - df["admission_dttm"]).dt.total_seconds() / 86400.0
)
df["survived"] = (df["survival_status"].str.lower() == "alive").astype(int)
df["good_cpc"] = df["cpc_tier"].isin(["CPC1_2"]).astype(int)

print(f"Cohort n = {len(df)}")
print(f"Survived: {df['survived'].sum()} ({100*df['survived'].mean():.1f}%)")
print(f"CPC tier counts:")
print(df["cpc_tier"].value_counts(dropna=False).to_string())

# %%
continuous = ["age_at_admission", "length_of_stay_days",
              "sofa_total_0_24", "sofa_cv_97_0_24", "sofa_resp_0_24",
              "sofa_renal_0_24", "sofa_cns_0_24", "sofa_liver_0_24",
              "sofa_coag_0_24"]
categorical = ["sex_category", "race_category", "ethnicity_category",
               "arrest_type", "cpc_tier"]
nonnormal   = ["length_of_stay_days", "sofa_total_0_24"]

columns = continuous + categorical
labels  = {
    "age_at_admission": "Age (years)",
    "length_of_stay_days": "Length of stay (days)",
    "sofa_total_0_24": "SOFA total (0–24h)",
    "sofa_cv_97_0_24": "SOFA cardiovascular",
    "sofa_resp_0_24": "SOFA respiratory",
    "sofa_renal_0_24": "SOFA renal",
    "sofa_cns_0_24": "SOFA neurologic",
    "sofa_liver_0_24": "SOFA hepatic",
    "sofa_coag_0_24": "SOFA coagulation",
    "sex_category": "Sex",
    "race_category": "Race",
    "ethnicity_category": "Ethnicity",
    "arrest_type": "Arrest type",
    "cpc_tier": "CPC tier (discharge)",
}

table1_overall = TableOne(
    df, columns=columns, categorical=categorical,
    nonnormal=nonnormal, labels=labels,
    pval=False, missing=True,
)

table1_strat = TableOne(
    df, columns=columns, categorical=categorical,
    nonnormal=nonnormal, labels=labels,
    groupby="survival_status",
    pval=True, missing=True,
)

# %%
out_overall = FINAL_DIR / "tableone_overall.csv"
out_strat   = FINAL_DIR / "tableone_by_survival.csv"
out_html    = FINAL_DIR / "tableone_by_survival.html"

table1_overall.to_csv(out_overall)
table1_strat.to_csv(out_strat)
with open(out_html, "w") as f:
    f.write(table1_strat.tabulate(tablefmt="html"))

print(f"Saved → {out_overall}")
print(f"Saved → {out_strat}")
print(f"Saved → {out_html}")

# %%
# ─────────────────────────────────────────────────────────────────────
# Table 1 on the RL cohort — the patients actually fed to the model.
# This is the cohort the abstract should report demographics for.
# Drops patients with no decision point (no vasopressor in [0, 120h) window).
# ─────────────────────────────────────────────────────────────────────
_rl_path = OUT_DIR / "rl_cohort_reviewed.parquet"
if _rl_path.exists():
    rl_cohort = pd.read_parquet(_rl_path)
    rl_ids = set(rl_cohort["hospitalization_id"].astype(str))
    df_rl = df[df["hospitalization_id"].astype(str).isin(rl_ids)].copy()

    print(f"\nRL cohort n = {len(df_rl)} "
          f"(of {len(df)} OHCA-ICU; "
          f"dropped {len(df) - len(df_rl)} with no decision point)")
    print(f"Survived: {df_rl['survived'].sum()} ({100*df_rl['survived'].mean():.1f}%)")

    table1_rl_overall = TableOne(
        df_rl, columns=columns, categorical=categorical,
        nonnormal=nonnormal, labels=labels,
        pval=False, missing=True,
    )
    table1_rl_strat = TableOne(
        df_rl, columns=columns, categorical=categorical,
        nonnormal=nonnormal, labels=labels,
        groupby="survival_status",
        pval=True, missing=True,
    )

    out_rl_overall = FINAL_DIR / "tableone_rl_cohort_overall.csv"
    out_rl_strat   = FINAL_DIR / "tableone_rl_cohort_by_survival.csv"
    out_rl_html    = FINAL_DIR / "tableone_rl_cohort_by_survival.html"
    table1_rl_overall.to_csv(out_rl_overall)
    table1_rl_strat.to_csv(out_rl_strat)
    with open(out_rl_html, "w") as f:
        f.write(table1_rl_strat.tabulate(tablefmt="html"))

    print(f"Saved → {out_rl_overall}")
    print(f"Saved → {out_rl_strat}")
    print(f"Saved → {out_rl_html}")
else:
    print(f"\n⚠️  {_rl_path} not found — skipping RL-cohort Table 1. "
          "Run 04_mdp.py first.")

# %%
print("\n=== Table 1: OHCA-ICU cohort (stratified by survival_status) ===")
print(table1_strat.tabulate(tablefmt="grid"))

if _rl_path.exists():
    print("\n=== Table 1: RL cohort (stratified by survival_status) ===")
    print(table1_rl_strat.tabulate(tablefmt="grid"))
