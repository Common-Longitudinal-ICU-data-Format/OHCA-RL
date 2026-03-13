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

## Expected Results

- CONSORT flow diagram and cohort summary statistics (`output/final/`)
- Trained DDQN+CQL policy for OHCA ICU management
- Federated model weights (Phase 1: local, Phase 2: FedAvg aggregated)
- Evaluation: concordance with clinician actions, mortality odds ratios, action distributions

## Detailed Instructions

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

### 3. Run the Pipeline

Run all steps sequentially:

```bash
bash run_pipeline.sh
```

Or run individual steps:

```bash
uv run code/00_cohort_identification.py
```

| Step | Script | Description |
|------|--------|-------------|
| 00 | `00_cohort_identification.py` | OHCA cohort identification with CONSORT diagram |
| 01 | `01_create_wide_df.py` | Load CLIF tables, unit conversion, pivot to wide DataFrame |
| 02 | `02_sofa_calculator.py` | SOFA score calculation |
| 03 | `03_ffill_and_bucketing.py` | Forward-fill imputation, 1h time bucketing, action inference |
| 04 | `04_create_tableone.py` | Baseline characteristics table |

Scripts are [marimo](https://marimo.io/) notebooks and can also be run interactively:

```bash
uv run marimo edit code/00_cohort_identification.py
```
