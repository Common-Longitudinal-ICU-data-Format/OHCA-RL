# OHCA-RL External Validation: Site Instructions

## Overview

Validate the coordinating center's trained DDQN vasopressor management model against your local OHCA cohort. This produces concordance metrics (agreement rate, ordinal OR, binned outcomes) that measure how well the RL policy aligns with your clinicians' decisions and patient outcomes.

**What you will run**: Steps 00-05 (data preparation) + step 07 (external validation).
**What you skip**: Step 06 (model training) — the trained model is provided.
**What you upload**: Aggregate summary statistics only. No patient-level data.

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **CLIF-formatted data tables** (see `README.md` for required tables and fields):
  - patient, hospitalization, hospital_diagnosis, adt
  - vitals, labs, medication_admin_continuous, medication_admin_intermittent
  - respiratory_support, crrt_therapy

## Step 1: Clone the Repository

```bash
git clone <repo-url>
cd OHCA-RL
```

## Step 2: Install Dependencies

```bash
uv sync
```

## Step 3: Configure Your Site

Create `config/config.json` (copy from template if available):

```json
{
  "site_name": "your_site_name",
  "tables_path": "/absolute/path/to/your/clif/tables",
  "file_type": "parquet",
  "timezone": "US/Central"
}
```

- `site_name`: Short identifier for your site (e.g., "rush", "nu")
- `tables_path`: Absolute path to directory containing CLIF parquet files
- `file_type`: Usually "parquet" (also supports "csv")
- `timezone`: Your local timezone (e.g., "US/Eastern", "US/Central")

## Step 4: Download Model Artifacts

Download all files from Box (`OHCA-RL-Federation/standardization/`) into your local `shared/` directory:

```
shared/
├── best_model.pt           # Trained DDQN model weights
├── preprocessor.json       # Standardization mean/std from training site
├── state_features.json     # Feature ordering (51 features)
├── training_config.json    # Model architecture (state_dim, hidden_dims)
└── action_remap.json       # Action encoding mapping
```

Verify all files are present:

```bash
ls shared/
# Should show: best_model.pt  preprocessor.json  state_features.json
#              training_config.json  action_remap.json
```

## Step 5: Run the Pipeline

**Mac/Linux:**

```bash
bash run_external_validation.sh
```

**Windows:**

```cmd
run_external_validation.bat
```

The script will:
1. Verify `shared/` artifacts exist
2. Run steps 00-05 (cohort identification through figures)
3. Run step 07 (external validation using the trained model)
4. Save all results to `output/final/`

**Expected runtime**: 10-30 minutes depending on dataset size.

If a step fails, check the log file printed at startup. You can also run individual steps to debug:

```bash
uv run code/00_cohort_identification.py
```

Or open interactively:

```bash
uv run marimo edit code/07_external_validation.py
```

## Step 6: Upload Results to Box

### Files to Upload

**External validation results** (`output/final/external_validation/`):
- `coef_summary.csv` — Concordance OR (includes per-10pp OR)
- `bin_summary.csv` — Agreement bins with outcome means
- `action_summary.csv` — RL vs clinician action distribution
- `evaluation_metadata.json` — Summary metrics

**Cohort characteristics** (`output/final/`):
- `table1_ohca.csv` — Full cohort Table One
- `table1_ohca_vaso.csv` — Vasopressor cohort Table One
- `strobe_counts.csv` — CONSORT flow counts
- `feature_summary.csv` — Feature-level statistics
- `figures/` — All figures (PNG files only)

### Do NOT Upload

- **Any `.parquet` files** — these contain patient-level data with hospitalization IDs
- **`output/intermediate/` directory** — contains raw patient data and transitions
- **`.log` files** — may contain processing details
- **`table1_*_long.csv`** — verbose format (unnecessary)

## Troubleshooting

### Missing state features
If step 07 raises `ValueError: Local data missing state features: [...]`, your CLIF data may not have certain medication or lab categories. Check that your `medication_admin_continuous` table includes the required `med_category` values listed in `README.md`.

### Different CLIF version
This pipeline requires CLIF 2.1 tables. If your site uses an older schema, some column names may differ. Check the error messages for specific column mismatches.

### Marimo notebook errors
If a script fails with import errors, try running `uv sync` again to ensure all dependencies are installed. Each script specifies its own inline dependencies.

### Action remap errors
If you see `KeyError` related to action remapping, verify that `shared/action_remap.json` contains the correct `pipeline_to_pi` mapping for your action encoding.
