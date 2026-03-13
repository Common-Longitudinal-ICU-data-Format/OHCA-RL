# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "duckdb",
#     "pyarrow",
#     "matplotlib",
#     "tabulate",
#     "pytz",
#     "clifpy==0.3.8",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", app_title="OHCA-RL Cohort Identification")


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import duckdb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from pathlib import Path

    import clifpy
    from clifpy import Hospitalization, HospitalDiagnosis, Adt
    return (
        Adt,
        FancyBboxPatch,
        HospitalDiagnosis,
        Hospitalization,
        Path,
        clifpy,
        duckdb,
        json,
        mo,
        pd,
        plt,
    )


@app.cell
def _(Path, clifpy, json, mo):
    # Load configuration
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    tables_path = config["tables_path"]
    file_type = config["file_type"]
    site_name = config["site_name"]
    timezone = config["timezone"]

    data_exists = Path(tables_path).exists()

    mo.md(f"""
    ## Setup & Configuration

    | Setting | Value |
    |---------|-------|
    | **Site** | `{site_name}` |
    | **Data path** | `{tables_path}` |
    | **File type** | `{file_type}` |
    | **Timezone** | `{timezone}` |
    | **Data directory exists** | {'Yes' if data_exists else '**NO — check config**'} |
    | **clifpy version** | `{clifpy.__version__}` |
    """)
    return file_type, project_root, site_name, tables_path, timezone


@app.cell
def _():
    # ── Constants ──────────────────────────────────────────────────────
    # ICD codes for cardiac arrest (lowercase, periods removed)
    ICD_PREFIXES = ["i460", "i461", "i462", "i468", "i469", "i4900", "i4901"]
    ICD_DESCRIPTIONS = {
        "i460": "Cardiac arrest with successful resuscitation",
        "i461": "Sudden cardiac death, so described",
        "i462": "Cardiac arrest due to underlying cardiac condition",
        "i468": "Cardiac arrest due to other underlying condition",
        "i469": "Cardiac arrest, cause unspecified",
        "i4900": "Ventricular fibrillation",
        "i4901": "Ventricular flutter",
    }

    def build_icd_filter(column="dx_clean"):
        """Build SQL WHERE clause for ICD prefix matching on cleaned codes."""
        conditions = " OR ".join(f"{column} LIKE '{code}%'" for code in ICD_PREFIXES)
        return f"({conditions})"

    # ── STROBE counts tracker ──────────────────────────────────────────
    strobe_counts = {}
    return ICD_DESCRIPTIONS, build_icd_filter, strobe_counts


@app.cell
def _(
    Adt,
    HospitalDiagnosis,
    Hospitalization,
    duckdb,
    file_type,
    mo,
    tables_path,
    timezone,
):
    # ── Load CLIF Tables via clifpy ────────────────────────────────────
    hosp_obj = Hospitalization.from_file(data_directory=tables_path, filetype=file_type, timezone=timezone)
    dx_obj = HospitalDiagnosis.from_file(data_directory=tables_path, filetype=file_type, timezone=timezone)
    adt_obj = Adt.from_file(data_directory=tables_path, filetype=file_type, timezone=timezone)

    hosp_df = hosp_obj.df
    dx_df = dx_obj.df
    adt_df = adt_obj.df

    # Register in DuckDB for SQL queries
    con = duckdb.connect()
    con.register("hosp_tbl", hosp_df)
    con.register("dx_tbl", dx_df)
    con.register("adt_tbl", adt_df)

    n_hosp = len(hosp_df)
    n_dx = len(dx_df)
    n_adt = len(adt_df)

    mo.md(f"""
    ## Checkpoint: CLIF Tables Loaded (via clifpy)

    | Table | Rows | Columns |
    |-------|------|---------|
    | `clif_hospitalization` | {n_hosp:,} | {', '.join(hosp_df.columns.tolist())} |
    | `clif_hospital_diagnosis` | {n_dx:,} | {', '.join(dx_df.columns.tolist())} |
    | `clif_adt` | {n_adt:,} | {', '.join(adt_df.columns.tolist())} |
    """)
    return (con,)


@app.cell
def _(ICD_DESCRIPTIONS, build_icd_filter, con, mo, strobe_counts):
    # ── Step 1: All Cardiac Arrest Encounters ──────────────────────────
    icd_filter = build_icd_filter("dx_clean")

    cohort_v2 = con.execute(f"""
        WITH dx_cleaned AS (
            SELECT
                d.hospitalization_id,
                d.diagnosis_code,
                LOWER(REPLACE(d.diagnosis_code, '.', '')) AS dx_clean,
                COALESCE(CAST(d.poa_present AS INT), 0) AS poa_present
            FROM dx_tbl d
        )
        SELECT DISTINCT
            h.patient_id,
            dc.hospitalization_id,
            dc.diagnosis_code,
            dc.dx_clean,
            dc.poa_present,
            -- Arrest type based on POA
            CASE
                WHEN dc.poa_present = 1 THEN 'ohca'
                WHEN dc.poa_present = 0 THEN 'ihca'
                ELSE 'unknown'
            END AS arrest_type,
            -- Survival status based on lowercase discharge_category
            LOWER(hc.discharge_category) AS discharge_category,
            CASE WHEN LOWER(hc.discharge_category) = 'expired' THEN 'non-survivor' ELSE 'survivor' END AS survival_status
        FROM dx_cleaned dc
        INNER JOIN hosp_tbl hc ON dc.hospitalization_id = hc.hospitalization_id
        INNER JOIN (SELECT DISTINCT patient_id, hospitalization_id FROM hosp_tbl) h
            ON dc.hospitalization_id = h.hospitalization_id
        WHERE {icd_filter}
    """).fetchdf()

    # Map ICD descriptions
    cohort_v2["icd_description"] = cohort_v2["dx_clean"].apply(
        lambda code: next(
            (desc for prefix, desc in ICD_DESCRIPTIONS.items() if code.startswith(prefix)),
            "Unknown"
        )
    )

    # Record strobe counts
    n_all_patients = cohort_v2["patient_id"].nunique()
    n_all_encounters = cohort_v2["hospitalization_id"].nunique()
    strobe_counts["1_all_cardiac_arrest_patients"] = n_all_patients
    strobe_counts["1_all_cardiac_arrest_encounters"] = n_all_encounters

    # Summary by arrest_type × survival_status
    summary = (
        cohort_v2.groupby(["arrest_type", "survival_status"])
        .agg(patients=("patient_id", "nunique"), encounters=("hospitalization_id", "nunique"))
        .reset_index()
    )

    # ICD breakdown
    icd_breakdown = (
        cohort_v2.groupby(["dx_clean", "icd_description"])["hospitalization_id"]
        .nunique()
        .reset_index()
        .rename(columns={"hospitalization_id": "encounters"})
        .sort_values("encounters", ascending=False)
    )

    # Mortality by type
    mort_lines = []
    for _atype in ["ohca", "ihca"]:
        _sub = cohort_v2[cohort_v2["arrest_type"] == _atype]
        _t = _sub["hospitalization_id"].nunique()
        _d = _sub[_sub["survival_status"] == "non-survivor"]["hospitalization_id"].nunique()
        if _t > 0:
            mort_lines.append(f"- **{_atype.upper()}** mortality: {_d/_t*100:.1f}% ({_d:,}/{_t:,})")

    summary_table = summary.to_markdown(index=False)
    icd_table = icd_breakdown.to_markdown(index=False)
    mort_text = "\n".join(mort_lines)

    mo.md(f"""
    ## Checkpoint: Step 1 — All Cardiac Arrest Encounters

    **Total**: {n_all_patients:,} patients, {n_all_encounters:,} encounters

    ### By Arrest Type × Survival Status
    {summary_table}

    ### Mortality
    {mort_text}

    ### ICD Code Breakdown
    {icd_table}
    """)
    return cohort_v2, n_all_encounters


@app.cell
def _(cohort_v2, mo, site_name, strobe_counts):
    # ── Step 2: Filter to OHCA ─────────────────────────────────────────
    # MIMIC does not have POA data — treat all cardiac arrest as OHCA
    if site_name.lower() == "mimic":
        ohca_all = cohort_v2.copy()
        ohca_all["arrest_type"] = "ohca"
        _poa_note = "**Note:** MIMIC has no POA data — all cardiac arrest encounters treated as OHCA."
    else:
        ohca_all = cohort_v2[cohort_v2["arrest_type"] == "ohca"].copy()
        _poa_note = ""

    n_ohca_patients = ohca_all["patient_id"].nunique()
    n_ohca_encounters = ohca_all["hospitalization_id"].nunique()
    n_excluded_ihca = cohort_v2["hospitalization_id"].nunique() - n_ohca_encounters

    strobe_counts["2_ohca_patients"] = n_ohca_patients
    strobe_counts["2_ohca_encounters"] = n_ohca_encounters
    strobe_counts["2_excluded_ihca_unknown"] = n_excluded_ihca

    mo.md(f"""
    ## Checkpoint: Step 2 — OHCA Only

    {_poa_note}

    | Metric | Count |
    |--------|-------|
    | OHCA patients | {n_ohca_patients:,} |
    | OHCA encounters | {n_ohca_encounters:,} |
    | Excluded (IHCA/Unknown) | {n_excluded_ihca:,} |
    """)
    return n_excluded_ihca, n_ohca_encounters, ohca_all


@app.cell
def _(con, mo, ohca_all, strobe_counts):
    # ── Step 3: First Encounter per Patient ────────────────────────────
    con.register("ohca_all_df", ohca_all)

    cohort_ohca_first = con.execute(f"""
        WITH ranked AS (
            SELECT c.*, h.admission_dttm,
                ROW_NUMBER() OVER (PARTITION BY c.patient_id ORDER BY h.admission_dttm ASC) AS rn
            FROM ohca_all_df c
            INNER JOIN hosp_tbl h ON CAST(c.hospitalization_id AS VARCHAR) = CAST(h.hospitalization_id AS VARCHAR)
            WHERE c.arrest_type = 'ohca'
        )
        SELECT * FROM ranked WHERE rn = 1
    """).fetchdf()

    n_first_patients = cohort_ohca_first["patient_id"].nunique()
    n_first_encounters = cohort_ohca_first["hospitalization_id"].nunique()
    n_removed_repeat = ohca_all["hospitalization_id"].nunique() - n_first_encounters

    strobe_counts["3_first_encounter_patients"] = n_first_patients
    strobe_counts["3_first_encounter_encounters"] = n_first_encounters
    strobe_counts["3_excluded_repeat_encounters"] = n_removed_repeat

    mo.md(f"""
    ## Checkpoint: Step 3 — First Encounter per Patient

    | Metric | Count |
    |--------|-------|
    | Patients (first encounter) | {n_first_patients:,} |
    | Encounters | {n_first_encounters:,} |
    | Removed (repeat encounters) | {n_removed_repeat:,} |
    """)
    return cohort_ohca_first, n_first_encounters, n_removed_repeat


@app.cell
def _(cohort_ohca_first, con, mo, pd, strobe_counts):
    # ── Step 4: ICU Admitted ───────────────────────────────────────────
    con.register("ohca_first_df", cohort_ohca_first)

    # Admission path breakdown
    admission_paths = con.execute(f"""
        WITH patient_locations AS (
            SELECT a.hospitalization_id,
                MAX(CASE WHEN LOWER(a.location_category)='ed' THEN 1 ELSE 0 END) AS has_ed,
                MAX(CASE WHEN LOWER(a.location_category)='icu' THEN 1 ELSE 0 END) AS has_icu,
                MAX(CASE WHEN LOWER(a.location_category)='ward' THEN 1 ELSE 0 END) AS has_ward,
                MAX(CASE WHEN LOWER(a.location_category)='stepdown' THEN 1 ELSE 0 END) AS has_stepdown
            FROM adt_tbl a
            WHERE CAST(a.hospitalization_id AS VARCHAR) IN (SELECT CAST(hospitalization_id AS VARCHAR) FROM ohca_first_df)
            GROUP BY a.hospitalization_id
        )
        SELECT CASE
                WHEN has_ed=1 AND has_icu=0 AND has_ward=0 AND has_stepdown=0 THEN 'ED only'
                WHEN has_icu=1 THEN 'ICU admitted'
                WHEN has_ward=1 AND has_icu=0 THEN 'Ward only (no ICU)'
                ELSE 'Other'
            END AS admission_path, COUNT(*) AS encounters
        FROM patient_locations GROUP BY admission_path ORDER BY encounters DESC
    """).fetchdf()

    # Mortality by admission path
    mort_by_path = con.execute(f"""
        WITH patient_locations AS (
            SELECT a.hospitalization_id,
                MAX(CASE WHEN LOWER(a.location_category)='icu' THEN 1 ELSE 0 END) AS has_icu,
                MAX(CASE WHEN LOWER(a.location_category)='ward' THEN 1 ELSE 0 END) AS has_ward,
                MAX(CASE WHEN LOWER(a.location_category)='ed' THEN 1 ELSE 0 END) AS has_ed,
                MAX(CASE WHEN LOWER(a.location_category)='stepdown' THEN 1 ELSE 0 END) AS has_stepdown
            FROM adt_tbl a
            WHERE CAST(a.hospitalization_id AS VARCHAR) IN (SELECT CAST(hospitalization_id AS VARCHAR) FROM ohca_first_df)
            GROUP BY a.hospitalization_id
        ),
        paths AS (
            SELECT hospitalization_id,
                CASE WHEN has_ed=1 AND has_icu=0 AND has_ward=0 AND has_stepdown=0 THEN 'ED only'
                     WHEN has_icu=1 THEN 'ICU admitted'
                     WHEN has_ward=1 AND has_icu=0 THEN 'Ward only'
                     ELSE 'Other' END AS admission_path
            FROM patient_locations
        )
        SELECT p.admission_path, c.survival_status, COUNT(*) AS n
        FROM paths p INNER JOIN ohca_first_df c ON CAST(p.hospitalization_id AS VARCHAR)=CAST(c.hospitalization_id AS VARCHAR)
        GROUP BY p.admission_path, c.survival_status ORDER BY p.admission_path
    """).fetchdf()

    # Build mortality summary table
    path_rows = []
    for _path in mort_by_path["admission_path"].unique():
        _sub = mort_by_path[mort_by_path["admission_path"] == _path]
        _s = _sub[_sub["survival_status"] == "survivor"]["n"].sum()
        _d = _sub[_sub["survival_status"] == "non-survivor"]["n"].sum()
        _t = _s + _d
        path_rows.append({"Path": _path, "Survivor": _s, "Non-Survivor": _d, "Total": _t, "Mortality %": f"{_d/_t*100:.1f}" if _t else "N/A"})
    path_summary_df = pd.DataFrame(path_rows)

    # Filter to ICU-admitted
    icu_ids = con.execute(f"""
        SELECT DISTINCT hospitalization_id FROM adt_tbl
        WHERE LOWER(location_category)='icu'
            AND CAST(hospitalization_id AS VARCHAR) IN (SELECT CAST(hospitalization_id AS VARCHAR) FROM ohca_first_df)
    """).fetchdf()

    # Cast to string for type-safe comparison
    icu_id_set = set(icu_ids["hospitalization_id"].astype(str))
    cohort_ohca_icu = cohort_ohca_first[
        cohort_ohca_first["hospitalization_id"].astype(str).isin(icu_id_set)
    ].copy()

    # Drop patients whose discharge_category maps to CPC "exclude" (no valid reward)
    _exclude_dc = ["still admitted", "missing", "other"]
    _invalid_dc = cohort_ohca_icu["discharge_category"].str.lower().str.strip().isin(_exclude_dc)
    _n_invalid_dc = _invalid_dc.sum()
    if _n_invalid_dc > 0:
        cohort_ohca_icu = cohort_ohca_icu[~_invalid_dc].copy()
    strobe_counts["4_excluded_unclassifiable_dc"] = int(_n_invalid_dc)

    n_icu_patients = cohort_ohca_icu["patient_id"].nunique()
    n_icu_encounters = cohort_ohca_icu["hospitalization_id"].nunique()
    n_excluded_no_icu = cohort_ohca_first["hospitalization_id"].nunique() - n_icu_encounters - int(_n_invalid_dc)
    n_surv = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "survivor"]["patient_id"].nunique()
    n_died = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "non-survivor"]["patient_id"].nunique()
    mortality_pct = n_died / n_icu_patients * 100 if n_icu_patients > 0 else 0

    strobe_counts["4_icu_admitted_patients"] = n_icu_patients
    strobe_counts["4_icu_admitted_encounters"] = n_icu_encounters
    strobe_counts["4_excluded_no_icu"] = n_excluded_no_icu
    strobe_counts["4_survivors"] = n_surv
    strobe_counts["4_non_survivors"] = n_died

    # ICU type breakdown
    icu_types = con.execute(f"""
        SELECT LOWER(a.location_type) AS location_type, COUNT(DISTINCT a.hospitalization_id) AS encounters
        FROM adt_tbl a
        WHERE CAST(a.hospitalization_id AS VARCHAR) IN (SELECT CAST(hospitalization_id AS VARCHAR) FROM ohca_first_df)
            AND LOWER(a.location_category)='icu'
        GROUP BY LOWER(a.location_type) ORDER BY encounters DESC
    """).fetchdf()

    # Date range
    dates = con.execute(f"""
        SELECT MIN(h.admission_dttm) AS first_date, MAX(h.admission_dttm) AS last_date
        FROM ohca_first_df c INNER JOIN hosp_tbl h ON CAST(c.hospitalization_id AS VARCHAR)=CAST(h.hospitalization_id AS VARCHAR)
        WHERE CAST(c.hospitalization_id AS VARCHAR) IN (SELECT CAST(hospitalization_id AS VARCHAR) FROM icu_ids)
    """).fetchone()
    con.register("icu_ids", icu_ids)

    path_table = path_summary_df.to_markdown(index=False)
    admission_path_table = admission_paths.to_markdown(index=False)
    icu_types_table = icu_types.to_markdown(index=False)

    mo.md(f"""
    ## Checkpoint: Step 4 — ICU Admitted

    ### Admission Path Breakdown
    {admission_path_table}

    ### Mortality by Admission Path
    {path_table}

    ### Final OHCA ICU Cohort
    | Metric | Count |
    |--------|-------|
    | **Patients** | **{n_icu_patients:,}** |
    | Encounters | {n_icu_encounters:,} |
    | Excluded (no ICU) | {n_excluded_no_icu:,} |
    | Excluded (unclassifiable discharge) | {_n_invalid_dc:,} |
    | Survivors | {n_surv:,} |
    | Non-Survivors | {n_died:,} |
    | **Mortality** | **{mortality_pct:.1f}%** |
    | Date range | {dates[0]} to {dates[1]} |

    ### ICU Type Breakdown
    {icu_types_table}
    """)
    return cohort_ohca_icu, n_excluded_no_icu, n_icu_encounters


@app.cell
def _(
    FancyBboxPatch,
    cohort_ohca_icu,
    mo,
    n_all_encounters,
    n_excluded_ihca,
    n_excluded_no_icu,
    n_first_encounters,
    n_icu_encounters,
    n_ohca_encounters,
    n_removed_repeat,
    pd,
    plt,
    project_root,
    site_name,
    strobe_counts,
):
    # ── STROBE / CONSORT Diagram ───────────────────────────────────────
    output_dir = project_root / "output" / "intermediate"
    final_dir = project_root / "output" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save strobe counts CSV
    strobe_df = pd.DataFrame(list(strobe_counts.items()), columns=["counter", "value"])
    strobe_df["site"] = site_name
    strobe_csv_path = final_dir / "strobe_counts.csv"
    strobe_df.to_csv(strobe_csv_path, index=False)

    # ── Build CONSORT flow diagram ─────────────────────────────────────
    n_all = n_all_encounters
    n_ohca = n_ohca_encounters
    n_first = n_first_encounters
    n_icu = n_icu_encounters
    n_surv_final = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "survivor"]["hospitalization_id"].nunique()
    n_died_final = cohort_ohca_icu[cohort_ohca_icu["survival_status"] == "non-survivor"]["hospitalization_id"].nunique()

    stages = [
        f"All Cardiac Arrest\n(ICD I46.x / I49.0x)\nn = {n_all:,}",
        f"OHCA\n(Present on Admission = 1)\nn = {n_ohca:,}",
        f"First Encounter\nper Patient\nn = {n_first:,}",
        f"ICU Admitted\nn = {n_icu:,}",
    ]
    drops = [
        f"Excluded: IHCA / Unknown\nn = {n_excluded_ihca:,}",
        f"Excluded: Repeat encounters\nn = {n_removed_repeat:,}",
        f"Excluded: No ICU admission\nn = {n_excluded_no_icu:,}",
    ]

    # ── Figure layout (CRRT-style straight flow with right exclusions) ─
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_h = 0.08
    box_w = 0.40
    x_main_start = 0.05
    x_main_center = x_main_start + box_w / 2
    x_excl_start = 0.55
    excl_arrow_gap = 0.015
    v_spacing = 0.16

    def draw_box(x, y, w, h, text, fontsize=11, weight="normal"):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01",
            linewidth=2, edgecolor="black", facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="black")
        return x + w / 2, y

    arrow_main = dict(arrowstyle="->", lw=2, color="black")

    # Title
    ax.text(0.5, 0.98, "OHCA Cohort Selection", ha="center", va="center",
            fontsize=16, fontweight="bold", color="black")

    # Top box
    top_y = 0.88 - box_h
    draw_box(x_main_start, top_y, box_w, box_h, stages[0], fontsize=11)

    # Draw each step
    for _i in range(len(drops)):
        current_y = top_y - ((_i + 1) * v_spacing)
        y_parent = top_y - (_i * v_spacing)

        box_center_y_top = y_parent + box_h / 2
        box_center_y_bottom = current_y + box_h / 2
        arrow_vertical_center = (box_center_y_top + box_center_y_bottom) / 2

        # Remaining box (main column)
        draw_box(x_main_start, current_y, box_w, box_h, stages[_i + 1], fontsize=11)

        # Exclusion box (right column)
        draw_box(x_excl_start, arrow_vertical_center - box_h / 2, box_w, box_h,
                 drops[_i], fontsize=11)

        # Main vertical arrow
        ax.annotate("", xy=(x_main_center, current_y + box_h),
                    xytext=(x_main_center, y_parent), arrowprops=arrow_main)

        # Exclusion horizontal arrow
        p1 = (x_main_center, arrow_vertical_center)
        p2 = (x_excl_start - excl_arrow_gap, arrow_vertical_center)
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"),
                    annotation_clip=False)

    # Final outcomes box at the bottom
    final_y = top_y - (len(drops) * v_spacing) - v_spacing
    _mort_pct = n_died_final / n_icu * 100 if n_icu > 0 else 0
    final_text = f"Final Cohort\nSurvivors: {n_surv_final:,} | Non-Survivors: {n_died_final:,}\nMortality: {_mort_pct:.1f}%"
    draw_box(x_main_start, final_y, box_w, box_h, final_text, fontsize=10, weight="bold")

    # Arrow from ICU box to final
    icu_box_y = top_y - (len(drops) * v_spacing)
    ax.annotate("", xy=(x_main_center, final_y + box_h),
                xytext=(x_main_center, icu_box_y), arrowprops=arrow_main)

    # Save
    consort_path = final_dir / "consort_diagram.png"
    fig.savefig(consort_path, dpi=300, bbox_inches="tight", facecolor="white")

    # Display: combine markdown + figure in a single output
    _strobe_table = strobe_df.to_markdown(index=False)
    _md_text = f"## STROBE / CONSORT Diagram\n\n**Saved:**\n- STROBE counts CSV: `{strobe_csv_path}`\n- CONSORT diagram: `{consort_path}`\n\n### STROBE Counts\n{_strobe_table}"
    mo.vstack([mo.md(_md_text), fig])
    return (output_dir,)


@app.cell
def _(cohort_ohca_icu, mo, output_dir):
    # ── Save Final Cohort ──────────────────────────────────────────────
    # Drop helper columns before saving
    save_cols = [
        "patient_id", "hospitalization_id", "diagnosis_code", "dx_clean",
        "poa_present", "arrest_type", "discharge_category", "survival_status",
        "icd_description", "admission_dttm",
    ]
    existing_cols = [c for c in save_cols if c in cohort_ohca_icu.columns]
    cohort_save = cohort_ohca_icu[existing_cols].copy()

    parquet_path = output_dir / "cohort_ohca_icu.parquet"
    cohort_save.to_parquet(parquet_path, index=False)
    parquet_size = parquet_path.stat().st_size / 1024

    mo.md(f"""
    ## Checkpoint: Cohort Saved

    | File | Path | Size |
    |------|------|------|
    | Parquet | `{parquet_path}` | {parquet_size:.1f} KB |

    **Rows**: {len(cohort_save):,} | **Columns**: {', '.join(existing_cols)}
    """)
    return


@app.cell
def _(cohort_ohca_icu, con, mo, output_dir, pd, tables_path):
    # ── Save Static Patient-Level DataFrame ───────────────────────────
    # One row per hospitalization with demographics, discharge info, and death timing.
    # Downstream scripts (03+) can load this instead of re-reading CLIF tables.

    # Load clif_patient for demographics + death_dttm
    _pat_df = pd.read_parquet(f"{tables_path}/clif_patient.parquet")

    # Get discharge_dttm and age from hospitalization table (already loaded as hosp_tbl in DuckDB)
    _hosp_extra = con.execute("""
        SELECT hospitalization_id, patient_id, discharge_dttm, age_at_admission
        FROM hosp_tbl
    """).fetchdf()

    # Build static df
    _cohort_ids = cohort_ohca_icu[["patient_id", "hospitalization_id", "admission_dttm",
                                    "discharge_category", "survival_status", "arrest_type"]].drop_duplicates(
        subset=["hospitalization_id"]
    )

    _static = (
        _cohort_ids
        .merge(
            _hosp_extra[["hospitalization_id", "discharge_dttm", "age_at_admission"]],
            on="hospitalization_id", how="left",
        )
        .merge(
            _pat_df[["patient_id", "sex_category", "race_category", "ethnicity_category", "death_dttm"]],
            on="patient_id", how="left",
        )
    )

    # Prefer death_dttm from clif_patient, fallback to discharge_dttm for expired patients
    _expired_mask = _static["discharge_category"].str.lower().str.contains("expired|hospice", na=False)
    _static.loc[_expired_mask & _static["death_dttm"].isna(), "death_dttm"] = (
        _static.loc[_expired_mask & _static["death_dttm"].isna(), "discharge_dttm"]
    )
    # Clear death_dttm for non-expired patients (may have died in a later hospitalization)
    _static.loc[~_expired_mask, "death_dttm"] = pd.NaT

    # Standardize category columns to lowercase
    for _col in ["sex_category", "race_category", "ethnicity_category", "discharge_category"]:
        if _col in _static.columns:
            _static[_col] = _static[_col].str.lower().str.strip()

    _output_path = output_dir / "patient_static.parquet"
    _static.to_parquet(_output_path, index=False)
    _file_size = _output_path.stat().st_size / 1024

    # Summary
    _n = len(_static)
    _n_with_death = _static["death_dttm"].notna().sum()

    mo.md(f"""
    ## Checkpoint: Static Patient-Level DataFrame

    | Metric | Value |
    |--------|-------|
    | **Patients** | {_n:,} |
    | **Columns** | {len(_static.columns)} |
    | **With death_dttm** | {_n_with_death:,} ({_n_with_death / _n * 100:.1f}%) |
    | **Saved to** | `{_output_path}` |
    | **Size** | {_file_size:.1f} KB |

    **Columns**: {', '.join(_static.columns.tolist())}

    **Demographics**:

    | | Values |
    |-|--------|
    | **Age** | median {_static['age_at_admission'].median():.0f}, IQR {_static['age_at_admission'].quantile(0.25):.0f}–{_static['age_at_admission'].quantile(0.75):.0f} |
    | **Sex** | {_static['sex_category'].value_counts().to_dict()} |
    | **Race** | {_static['race_category'].value_counts().head(5).to_dict()} |
    """)
    return


if __name__ == "__main__":
    app.run()
