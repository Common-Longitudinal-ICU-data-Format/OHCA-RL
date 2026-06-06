# OHCA-RL External Validation Guide

This document is for **external validation sites** running the
pre-trained OHCA-RL model against their own CLIF-formatted data.

## What you receive from the coordinating center

A zipped copy of this repository plus a `shared/` folder containing:

| File | Purpose |
|---|---|
| `shared/ddqn_cql_reviewed.pt` | Trained Dueling DQN + CQL model weights |
| `shared/normalization_stats.parquet` | Per-feature mean/SD used during training |
| `shared/feature_metadata.json` | State feature list, dims, training fingerprint |

If `shared/` is empty, contact the coordinating center.

## What you need on your end

1. A CLIF-formatted CLIF Spec 2.x database for your site, accessible via
   the [`clifpy`](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy)
   reader (DuckDB or pandas backend).
2. Python 3.11+ with the packages in `requirements.txt` (or your existing
   `ohca-rl` conda environment).
3. Read access to:
   - `clif_hospitalization`
   - `clif_hospital_diagnosis`
   - `clif_adt`
   - `clif_patient`
   - `clif_vitals`
   - `clif_labs`
   - `clif_medication_admin_continuous`
   - `clif_respiratory_support`
   - `clif_patient_assessments`

## Setup

1. **Edit `config/config.json`** with your site's values:

   ```json
   {
     "site_name": "Your Site Name",
     "site_id": "your_site_short_id",
     "tables_path": "/path/to/your/CLIF/parquets",
     "file_type": "parquet",
     "timezone": "America/Chicago",
     "project_root": "/path/to/this/OHCA-RL/clone"
   }
   ```

   `site_id` should be a short slug (e.g., `umich`, `rush`) — it will be
   appended to every shared artifact filename, so the coordinating center
   can disambiguate sites.

2. **Confirm `shared/` is populated.** Run:

   ```bash
   ls shared/
   ```

   You should see three files (model `.pt`, normalization `.parquet`, metadata
   `.json`). If not, request them from the coordinating center.

3. **Install dependencies** (skip if you already have the `ohca-rl` env):

   ```bash
   pip install -r requirements.txt
   ```

## Run

**Linux / macOS:**
```bash
bash run_validation.sh
```

**Windows:**
```cmd
run_validation.bat
```

The script runs, in order:
1. `code/01_cohort.py` — identifies OHCA + ICU cohort from your CLIF tables
2. `code/02_wide.py` — builds the hourly wide-format dataset
3. `code/03_sofa.py` — computes 0–24h SOFA scores
4. `code/04_mdp.py` — discretizes into the MDP state/action format
5. `code/05_reward.py` — computes per-step rewards
6. `code/external_validation.py` — loads `shared/ddqn_cql_reviewed.pt`
   and runs the frozen policy against your data, producing concordance
   tables, FQE estimates, and concordance–outcome regressions
7. `code/make_tableone.py` — generates a Table 1 of your cohort

Expected runtime: ~30 min – 2 hours depending on your cohort size and disk speed.

## What to send back

After the pipeline completes, the artifacts you share back with the
coordinating center are in:

```
output/final/<your_site_id>/
```

This folder will contain (all CSVs, no PHI):

| File | Contents |
|---|---|
| `<site_id>_cohort_summary.csv` | Aggregate cohort counts |
| `<site_id>_action_distribution.csv` | Clinician vs RL action distribution |
| `<site_id>_disagreement_summary.csv` | Decision-level + patient-level concordance |
| `<site_id>_all_model_results_scaled_per10pp.csv` | Concordance–outcome regression (primary) |
| `<site_id>_early24_results_scaled_per10pp.csv` | Early-window sensitivity analysis |
| `<site_id>_sofa_stratified_results_scaled_per10pp.csv` | Severity-stratified analysis |
| `<site_id>_behavior_by_cpc.csv` | Behavioral patterns by CPC outcome tier |
| `<site_id>_behavior_by_survival.csv` | Behavioral patterns by survival |
| `<site_id>_behavior_by_good_outcome.csv` | Behavioral patterns by good outcome |
| `<site_id>_behavior_support_summary.csv` | Behavior policy support of RL recommendations |
| `<site_id>_patient_level_policy_features.csv` | Per-patient policy summary (deidentified IDs) |
| `tableone_overall.csv`, `tableone_by_survival.csv` | Cohort Table 1 |
| `fqe_only_*.parquet` | FQE point estimates, bootstrap CIs, support diagnostics |

**Please zip and upload `output/final/<your_site_id>/` plus the two
`tableone_*.csv` files to the coordinating center via the shared Box
folder.**

## Quality checks before sending

Before uploading, please verify:

1. `output/final/<your_site_id>/<site_id>_cohort_summary.csv` shows a
   non-empty cohort (n > 50 patients).
2. `<site_id>_action_distribution.csv` shows clinician actions distributed
   across multiple tiers (not concentrated in one).
3. `tableone_by_survival.csv` shows plausible age and SOFA distributions.
4. No errors in the pipeline log.

If any of these look off, please pause the upload and contact the
coordinating center.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `shared/ddqn_cql_reviewed.pt not found` | Coordinating center bundle missing | Request from coordinating center |
| `KeyError: 'site_id'` | Old config.json | Add `site_id` field per setup step 1 |
| `clifpy` table read fails | `tables_path` or `file_type` wrong | Check config.json against your data |
| FQE NaN values | Insufficient patients (<30) | Note in upload; coordinating center will assess |

## Contact

Coordinating center: Kaveri Chhikara (kaveri.chhikara@gmail.com),
PI: David Beiser. Please email if you hit any issue not covered above.
