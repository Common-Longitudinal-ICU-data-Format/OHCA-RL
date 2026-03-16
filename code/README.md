## Code Directory

All scripts are [marimo](https://marimo.io/) reactive Python notebooks that can be run headlessly or interactively.

### Pipeline Steps

| Step | Script | Mode | Description |
|------|--------|------|-------------|
| 00 | `00_cohort_identification.py` | Both | Apply ICD inclusion/exclusion criteria, identify OHCA cohort, generate CONSORT diagram |
| 01 | `01_create_wide_df.py` | Both | Load CLIF tables, unit conversion via clifpy, pivot to wide hourly DataFrame |
| 02 | `02_sofa_calculator.py` | Both | Calculate SOFA component and total scores |
| 03 | `03_ffill_and_bucketing.py` | Both | Forward-fill imputation, 1-hour time bucketing, vasopressor action inference |
| 04 | `04_create_tableone.py` | Both | Baseline characteristics table (full cohort + vasopressor-only subset) |
| 05 | `05_figures.py` | Both | Pre-training descriptive figures (missingness, vitals, labs, SOFA, treatments) |
| 06 | `06_training.py` | Training | Double DQN training, concordance evaluation, export to `shared/` and `upload_to_box/` |
| 07 | `07_external_validation.py` | Validation | Apply trained model from `shared/` to local data, compute concordance metrics |
| 08 | `08_visualize_results.py` | Training | Post-training figures (action distributions, patient timelines, OR forest plot) |
| 09 | `09_combined_dashboard.py` | Training | Standalone HTML dashboard aggregating all results |

**Training mode** = coordinating center (runs steps 00-06, 08-09)
**Validation mode** = external sites (runs steps 00-05, 07)

### Supporting Files

| File | Purpose |
|------|---------|
| `ohca_training.py` | Shared utilities: QNetwork, dataset classes, training helpers |
| `utils.py` | General utility functions |

### Running

Headless (batch):
```bash
uv run code/00_cohort_identification.py
```

Interactive (browser):
```bash
uv run marimo edit code/00_cohort_identification.py
```

Full pipeline:
```bash
bash run_pipeline.sh                  # training site
bash run_external_validation.sh       # validation site
```
