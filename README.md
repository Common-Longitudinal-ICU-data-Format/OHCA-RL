# OHCA-RL: Federated Reinforcement Learning for Out-of-Hospital Cardiac Arrest

## CLIF Version

2.1

## Objective

Develop a federated reinforcement learning framework for optimizing ICU management of out-of-hospital cardiac arrest (OHCA) patients using the Common Longitudinal ICU Format (CLIF).

## Required CLIF Tables and Fields

Please refer to the [CLIF data dictionary](https://clif-icu.com/data-dictionary), [CLIF Tools](https://clif-icu.com/tools), [ETL Guide](https://clif-icu.com/etl-guide), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information.

1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`, `discharge_category`
3. **hospital_diagnosis**: `hospitalization_id`, `diagnosis_code`, `present_on_admission`
4. **adt**: `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category`
5. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category`: heart_rate, resp_rate, sbp, dbp, map, temp, weight, height, spo2
6. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value`
   - `lab_category`: po2_arterial, pco2_arterial, ph_arterial, bicarbonate, so2_arterial, sodium, potassium, chloride, calcium_total, magnesium, creatinine, bun, glucose_serum, lactate, hemoglobin
7. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_category`
   - `med_category`: norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, angiotensin, propofol, dexmedetomidine, midazolam, ketamine, fentanyl_drip, cisatracurium, rocuronium, vecuronium, nicardipine, clevidipine, milrinone
8. **medication_admin_intermittent**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_category`
   - `med_category`: levetiracetam, phenytoin, valproic_acid
9. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `tracheostomy`, `fio2_set`, `lpm_set`, `peep_set`, `tidal_volume_set`, `resp_rate_set`, `resp_rate_obs`
10. **crrt_therapy**: `hospitalization_id`, `recorded_dttm`

For Python users, the [clifpy](https://common-longitudinal-icu-data-format.github.io/clifpy/) package provides utilities for outlier handling, respiratory support waterfall, medication unit conversion, SOFA scoring, and more. See the [clifpy user guide](https://common-longitudinal-icu-data-format.github.io/clifpy/user-guide/).

## Cohort Identification

1. All cardiac arrest patients identified by ICD codes (I46.x, I49.0x)
2. Filter to OHCA only (present on admission = 1)
3. First encounter per patient (deduplicated by earliest admission)
4. ICU-admitted only (exclude ED-only patients via ADT location data)

## Pipeline

There are two modes: **Training Site** (UCMC) runs the full pipeline including model training; **Validation Sites** skip training and run external validation using the trained model.

| Step | Script | Mode | Description |
|------|--------|------|-------------|
| 00 | `00_cohort_identification.py` | Both | OHCA cohort with CONSORT diagram |
| 01 | `01_create_wide_df.py` | Both | CLIF tables, unit conversion, pivot to wide DataFrame |
| 02 | `02_sofa_calculator.py` | Both | SOFA score calculation |
| 03 | `03_ffill_and_bucketing.py` | Both | Forward-fill imputation, 1h bucketing, action inference |
| 04 | `04_create_tableone.py` | Both | Baseline characteristics (full cohort + vasopressor-only) |
| 05 | `05_figures.py` | Both | Pre-training figures (PDFs + PNGs) |
| 06 | `06_training.py` | Training only | DDQN local training + concordance evaluation |
| 07 | `07_external_validation.py` | Validation only | Apply trained model to local data |
| 08 | `08_visualize_results.py` | Training only | Post-training concordance figures |
| 09 | `09_combined_dashboard.py` | Training only | Standalone HTML dashboard |

Scripts are [marimo](https://marimo.io/) notebooks and can also be run interactively:

```bash
uv run marimo edit code/00_cohort_identification.py
```

## Setup

### 1. Update `config/config.json`

Follow instructions in [config/README.md](config/README.md). Required fields:

```json
{
  "site_name": "your_site",
  "tables_path": "/path/to/clif/tables",
  "file_type": "parquet",
  "timezone": "US/Central"
}
```

### 2. Set Up the Environment

Requires [uv](https://docs.astral.sh/uv/). Install dependencies and create the virtual environment:

```bash
uv sync
```

## Running the Pipeline

### Training Site (Coordinating Center)

Runs all steps including DDQN training, visualization, and dashboard generation.

**Mac/Linux:**

```bash
bash run_pipeline.sh
```

**Windows:**

```cmd
run_pipeline.bat
```

### Validation Site (External Validation)

Runs data preparation (steps 00-05) and external validation (step 07). Requires trained model artifacts in `shared/` — download from Box first. See [SITE_INSTRUCTIONS.md](SITE_INSTRUCTIONS.md) for the full setup guide.

**Mac/Linux:**

```bash
bash run_external_validation.sh
```

**Windows:**

```cmd
run_external_validation.bat
```

Both scripts run steps sequentially, halt on first failure, and save a timestamped log to `output/final/`.

Or run individual steps:

```bash
uv run code/00_cohort_identification.py
```

