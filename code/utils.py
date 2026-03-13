"""
OHCA-RL Utility Functions

Derived feature calculations and medication dose unit conversion.
Dose conversion logic adapted from clifpy.utils.unit_converter for local iteration.

For table loading, outlier handling, and waterfall — use clifpy directly.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Set

try:
    import duckdb
except ImportError:
    duckdb = None  # Only needed for dose conversion functions

logger = logging.getLogger(__name__)


def setup_logging(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Configure logging to both console and a log file.

    Parameters
    ----------
    name : str
        Logger name (e.g., "03_ffill_bucketing"). Also used for the log filename.
    log_dir : str or Path, optional
        Directory for log files. Defaults to ``<project_root>/output/final/``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "output" / "final"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Get or create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-import
    if not _logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        _logger.addHandler(ch)

        # File handler (overwrite each run)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        _logger.addHandler(fh)

    _logger.info("Logging to %s", log_file)
    return _logger


# =============================================================================
# RESPIRATORY SUPPORT: DEVICE CATEGORIZATION & FiO2 IMPUTATION
# =============================================================================

# Standard clinical nasal cannula LPM → FiO2 lookup
# Each L/min adds ~4% above room air (21%)
NC_LPM_TO_FIO2 = {
    1: 0.24, 2: 0.28, 3: 0.32, 4: 0.36, 5: 0.40,
    6: 0.44, 7: 0.48, 8: 0.52, 9: 0.56, 10: 0.60,
}


def categorize_device_from_tracheostomy(df: pd.DataFrame) -> pd.DataFrame:
    """Infer device_category for rows with tracheostomy or LPM but no device.

    Applied BEFORE waterfall to give it better device info.

    Logic (from clinical heuristics):
    - Has tracheostomy (ever) → 'trach collar'
    - No tracheostomy AND lpm_set < 30 → 'face mask'
    - No tracheostomy AND lpm_set >= 30 → 'high flow nc'

    Only fills rows where device_category is missing and lpm_set is present.
    """
    df = df.copy()

    # Build per-patient tracheostomy flag (ever had trach = 1)
    if "tracheostomy" in df.columns:
        _trach_ever = (
            df.groupby("hospitalization_id")["tracheostomy"]
            .transform("max")
            .fillna(0)
        )
    else:
        _trach_ever = pd.Series(0, index=df.index)

    _missing_device = df["device_category"].isna()
    _has_lpm = df["lpm_set"].notna()
    _needs_fill = _missing_device & _has_lpm

    # Tracheostomy → trach collar
    _trach_mask = _needs_fill & (_trach_ever == 1)
    df.loc[_trach_mask, "device_category"] = "trach collar"

    # No trach, lpm < 30 → face mask
    _fm_mask = _needs_fill & (_trach_ever != 1) & (df["lpm_set"] < 30)
    df.loc[_fm_mask, "device_category"] = "face mask"

    # No trach, lpm >= 30 → high flow nc
    _hfnc_mask = _needs_fill & (_trach_ever != 1) & (df["lpm_set"] >= 30)
    df.loc[_hfnc_mask, "device_category"] = "high flow nc"

    _n_filled = _trach_mask.sum() + _fm_mask.sum() + _hfnc_mask.sum()
    if _n_filled > 0:
        logger.info(
            "categorize_device_from_tracheostomy: filled %d rows "
            "(trach collar: %d, face mask: %d, HFNC: %d)",
            _n_filled, _trach_mask.sum(), _fm_mask.sum(), _hfnc_mask.sum(),
        )

    return df


def categorize_device(df: pd.DataFrame) -> pd.DataFrame:
    """Infer device_category from mode, fio2, lpm, peep, tidal_volume.

    Applied BEFORE waterfall. Fills missing device_category using heuristics:
    - Known vent modes → 'vent'
    - FiO2=0.21, no LPM/PEEP/TV → 'room air'
    - LPM=0, no FiO2/PEEP/TV → 'room air'
    - 0 < LPM <= 20, no FiO2/PEEP/TV → 'nasal cannula'
    - LPM > 20, no FiO2/PEEP/TV → 'high flow nc'
    - device='nasal cannula' but LPM > 20 → 'high flow nc'
    """
    df = df.copy()

    _vent_modes = {
        "simv", "pressure-regulated volume control",
        "assist control-volume control",
    }

    _dc = df["device_category"]
    _mc = df.get("mode_category", pd.Series(dtype=object))
    _fio2 = df.get("fio2_set", pd.Series(dtype=float))
    _lpm = df.get("lpm_set", pd.Series(dtype=float))
    _peep = df.get("peep_set", pd.Series(dtype=float))
    _tv = df.get("tidal_volume_set", pd.Series(dtype=float))

    _missing = _dc.isna()

    # Known vent mode → vent
    _vent_mask = _missing & _mc.str.lower().isin(_vent_modes)
    df.loc[_vent_mask, "device_category"] = "vent"

    # Room air: fio2=0.21, no lpm/peep/tv
    _ra_mask1 = (
        _missing & ~_vent_mask
        & (_fio2 == 0.21)
        & _lpm.isna() & _peep.isna() & _tv.isna()
    )
    df.loc[_ra_mask1, "device_category"] = "room air"

    # Room air: lpm=0, no fio2/peep/tv
    _ra_mask2 = (
        _missing & ~_vent_mask & ~_ra_mask1
        & _fio2.isna()
        & (_lpm == 0)
        & _peep.isna() & _tv.isna()
    )
    df.loc[_ra_mask2, "device_category"] = "room air"

    # Nasal cannula: 0 < lpm <= 20, no fio2/peep/tv
    _nc_mask = (
        _missing & ~_vent_mask & ~_ra_mask1 & ~_ra_mask2
        & _fio2.isna()
        & (_lpm > 0) & (_lpm <= 20)
        & _peep.isna() & _tv.isna()
    )
    df.loc[_nc_mask, "device_category"] = "nasal cannula"

    # HFNC: lpm > 20, no fio2/peep/tv
    _hfnc_mask = (
        _missing & ~_vent_mask & ~_ra_mask1 & ~_ra_mask2 & ~_nc_mask
        & _fio2.isna()
        & (_lpm > 20)
        & _peep.isna() & _tv.isna()
    )
    df.loc[_hfnc_mask, "device_category"] = "high flow nc"

    # Reclassify: labeled nasal cannula but lpm > 20 → HFNC
    _reclass = (df["device_category"] == "nasal cannula") & (_lpm > 20)
    df.loc[_reclass, "device_category"] = "high flow nc"

    _n_filled = sum(m.sum() for m in [
        _vent_mask, _ra_mask1, _ra_mask2, _nc_mask, _hfnc_mask, _reclass,
    ])
    if _n_filled > 0:
        logger.info("categorize_device: filled/reclassified %d rows", _n_filled)

    return df


def impute_fio2(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing FiO2 values after waterfall.

    Applied AFTER waterfall. Uses:
    1. Room air → FiO2 = 0.21
    2. Nasal cannula + LPM → clinical lookup (0.24 + 0.04 * LPM)
    """
    df = df.copy()

    _fio2_missing = df["fio2_set"].isna()

    # Room air → 0.21
    _ra_mask = _fio2_missing & (df["device_category"] == "room air")
    df.loc[_ra_mask, "fio2_set"] = 0.21

    # Nasal cannula + LPM → lookup
    _nc_mask = (
        _fio2_missing & ~_ra_mask
        & (df["device_category"] == "nasal cannula")
        & df["lpm_set"].notna()
    )
    if _nc_mask.any():
        # Use formula: 0.20 + 0.04 * LPM (matches NC_LPM_TO_FIO2 lookup table)
        df.loc[_nc_mask, "fio2_set"] = (
            0.20 + 0.04 * df.loc[_nc_mask, "lpm_set"]
        ).clip(upper=1.0)

    _n_ra = _ra_mask.sum()
    _n_nc = _nc_mask.sum()
    if _n_ra + _n_nc > 0:
        logger.info(
            "impute_fio2: filled %d rows (room air: %d, nasal cannula+LPM: %d)",
            _n_ra + _n_nc, _n_ra, _n_nc,
        )

    return df


# =============================================================================
# DERIVED FEATURE CALCULATIONS
# =============================================================================

def calculate_pf_ratio(po2: pd.Series, fio2: pd.Series) -> pd.Series:
    """PaO2/FiO2 ratio. Handles percentage vs fraction FiO2."""
    fio2 = fio2.copy()
    mask_pct = fio2 > 1
    fio2[mask_pct] = fio2[mask_pct] / 100
    fio2 = fio2.clip(lower=0.21)
    return po2 / fio2


def calculate_base_excess(ph: pd.Series, hco3: pd.Series) -> pd.Series:
    """Base excess: BE = 0.929 * (HCO3 - 24.4 + 14.8 * (pH - 7.4))"""
    return 0.929 * (hco3 - 24.4 + 14.8 * (ph - 7.4))


def calculate_ibw(height_cm: pd.Series, sex: pd.Series) -> pd.Series:
    """
    Ideal Body Weight using Devine formula.
    Male: 50 + 2.3 * (height_inches - 60)
    Female: 45.5 + 2.3 * (height_inches - 60)
    """
    height_inches = height_cm / 2.54
    sex_lower = sex.astype(str).str.lower()
    ibw = np.where(
        sex_lower.isin(["male", "m"]),
        50 + 2.3 * (height_inches - 60),
        np.where(
            sex_lower.isin(["female", "f"]),
            45.5 + 2.3 * (height_inches - 60),
            np.nan,
        ),
    )
    result = pd.Series(ibw, index=height_cm.index)
    result[height_cm.isna() | sex.isna()] = np.nan
    return result


def compute_nee(wide_df: pd.DataFrame, nee_config: dict) -> pd.Series:
    """
    Compute Norepinephrine Equivalent (NEE) from vasopressor doses.

    nee_config: {med_name: coefficient, ...}
    Expects columns named med_cont_{med_name} in wide_df.
    """
    nee = pd.Series(0.0, index=wide_df.index)
    any_present = pd.Series(False, index=wide_df.index)

    for med, coeff in nee_config.items():
        col = f"med_cont_{med}"
        if col in wide_df.columns:
            vals = wide_df[col].fillna(0)
            nee += coeff * vals
            any_present |= wide_df[col].notna()

    # Set NEE to NaN where no vasopressor data exists
    nee[~any_present] = np.nan
    return nee


# =============================================================================
# MEDICATION DOSE UNIT CONVERSION
# Adapted from clifpy.utils.unit_converter (v0.3.8) for local iteration.
#
# Pipeline:
#   1. Clean unit strings (whitespace, casing, name variants)
#   2. Standardize to base units (mcg/min, ml/min, u/min for rates)
#   3. Convert base → preferred units per med_category
#
# All heavy lifting uses DuckDB SQL for performance.
# =============================================================================

# ── Unit naming variants: regex pattern → standard abbreviation ──────────────
UNIT_NAMING_VARIANTS = {
    "/hr": r'/h(r|our)?$',
    "/min": r'/m(in|inute)?$',
    "u": r'u(nits|nit)?',
    "m": r'milli-?',
    "l": r'l(iters|itres|itre|iter)?',
    "mcg": r'^(u|µ|μ)g',
    "g": r'^g(rams|ram)?',
}

# ── Regex building blocks ────────────────────────────────────────────────────
_END = r"($|/*)"
MASS_REGEX = rf"^(mcg|mg|ng|g){_END}"
VOLUME_REGEX = rf"^(l|ml){_END}"
UNIT_REGEX = rf"^(u|mu){_END}"
HR_REGEX = r"/hr$"
MU_REGEX = rf"^(mu){_END}"
MG_REGEX = rf"^(mg){_END}"
NG_REGEX = rf"^(ng){_END}"
G_REGEX = rf"^(g){_END}"
L_REGEX = rf"^l{_END}"
LB_REGEX = r"/lb/"
KG_REGEX = r"/kg/"
WEIGHT_REGEX = r"/(lb|kg)/"

# ── Conversion factors: regex → SQL expression to multiply dose by ───────────
# Goal: convert everything to base units (mcg for mass, ml for volume, u for
# units, /min for time, raw dose for weight-based after multiplying by weight)
REGEX_TO_FACTOR = {
    HR_REGEX: "1.0/60",           # /hr → /min
    L_REGEX: "1000",              # L → mL
    MU_REGEX: "1.0/1000",         # milli-units → units
    MG_REGEX: "1000",             # mg → mcg
    NG_REGEX: "1.0/1000",         # ng → mcg
    G_REGEX: "1000000",           # g → mcg
    KG_REGEX: "weight_kg",        # /kg → multiply by patient weight
    LB_REGEX: "weight_kg * 2.20462",  # /lb → multiply by weight×conversion
}

# ── Acceptable unit sets ─────────────────────────────────────────────────────
ACCEPTABLE_AMOUNT_UNITS: Set[str] = {"ml", "l", "mu", "u", "mcg", "mg", "ng", "g"}
ACCEPTABLE_RATE_UNITS: Set[str] = {
    f"{a}{w}{t}"
    for a in ACCEPTABLE_AMOUNT_UNITS
    for w in ("/kg", "/lb", "")
    for t in ("/hr", "/min")
}
ALL_ACCEPTABLE_UNITS: Set[str] = ACCEPTABLE_RATE_UNITS | ACCEPTABLE_AMOUNT_UNITS

_RATE_STR = "','".join(ACCEPTABLE_RATE_UNITS)
_AMOUNT_STR = "','".join(ACCEPTABLE_AMOUNT_UNITS)


def _build_case(patterns: list, col: str, inverse: bool = False, else_val: str = "1") -> str:
    """Build a SQL CASE expression applying conversion factors for matched regex patterns.

    Parameters
    ----------
    patterns : list of str
        Regex patterns to check against `col`.
    col : str
        Column name to match against ('_clean_unit' or '_preferred_unit').
    inverse : bool
        If True, apply 1/(factor) — used for base→preferred conversion.
    else_val : str
        Default value when no pattern matches.
    """
    clauses = []
    for p in patterns:
        factor = REGEX_TO_FACTOR[p]
        expr = f"1.0/({factor})" if inverse else factor
        clauses.append(f"WHEN regexp_matches({col}, '{p}') THEN {expr}")
    return f"CASE {' '.join(clauses)} ELSE {else_val} END"


# ── Step 1: Clean unit strings ───────────────────────────────────────────────

def clean_dose_units(med_df: pd.DataFrame, unit_col: str = "med_dose_unit") -> pd.DataFrame:
    """Clean and standardize dose unit strings.

    Adds a `_clean_unit` column: lowercased, whitespace removed, name variants
    normalized (e.g. 'MCG / KG / Hour' → 'mcg/kg/hr').
    """
    # Format: strip whitespace, lowercase
    expr = f"NULLIF(lower(regexp_replace({unit_col}, '\\s+', '', 'g')), '')"
    # Names: apply variant replacements
    for repl, pattern in UNIT_NAMING_VARIANTS.items():
        expr = f"regexp_replace({expr}, '{pattern}', '{repl}', 'g')"

    result = duckdb.sql(f"""
        SELECT *, {expr} AS _clean_unit
        FROM med_df
    """).fetchdf()
    return result


# ── Step 2: Patient weight lookup ────────────────────────────────────────────

def build_weight_table(vitals_df: pd.DataFrame) -> pd.DataFrame:
    """Extract weight_kg measurements from vitals into a standalone lookup table.

    Returns a DataFrame with columns: hospitalization_id, recorded_dttm, weight_kg.
    Save this as an intermediate parquet so it only needs to be computed once.
    """
    return duckdb.sql("""
        SELECT hospitalization_id, recorded_dttm, vital_value AS weight_kg
        FROM vitals_df
        WHERE vital_category = 'weight_kg' AND vital_value IS NOT NULL
        ORDER BY hospitalization_id, recorded_dttm
    """).fetchdf()


def attach_weight(
    med_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """Attach patient weight to each med administration row.

    Strategy (in priority order):
      1. ASOF backward — most recent weight_kg recorded BEFORE admin_dttm
      2. Forward lookup — patient's earliest recorded weight (for med rows
         that occur before the patient was first weighed)
      3. Cohort median — only for patients with NO weight ever recorded
         (when impute_missing=True)

    A `_weight_imputed` column flags how weight was obtained:
      0 = observed (backward ASOF match)
      1 = forward-filled (patient's own weight, recorded later)
      2 = cohort median (no weight ever recorded for this patient)

    Parameters
    ----------
    med_df : pd.DataFrame
        Medication data with hospitalization_id and admin_dttm.
    weight_df : pd.DataFrame
        Pre-computed weight lookup (from build_weight_table or loaded from parquet).
        Columns: hospitalization_id, recorded_dttm, weight_kg.
    impute_missing : bool
        If True, fill remaining NULLs with cohort median and log the count.
    """
    # Build per-patient first weight (for forward fill)
    first_weight = duckdb.sql("""
        SELECT hospitalization_id,
               FIRST(weight_kg ORDER BY recorded_dttm) AS first_weight_kg
        FROM weight_df
        GROUP BY hospitalization_id
    """).fetchdf()

    # Backward ASOF + forward fallback in one query
    result = duckdb.sql("""
        SELECT m.*,
            COALESCE(wb.weight_kg, fw.first_weight_kg) AS weight_kg,
            CASE
                WHEN wb.weight_kg IS NOT NULL THEN 0   -- backward match
                WHEN fw.first_weight_kg IS NOT NULL THEN 1  -- forward fill
                ELSE 2                                  -- no weight at all
            END AS _weight_imputed
        FROM med_df m
        ASOF LEFT JOIN weight_df wb
            ON m.hospitalization_id = wb.hospitalization_id
            AND wb.recorded_dttm <= m.admin_dttm
        LEFT JOIN first_weight fw
            ON m.hospitalization_id = fw.hospitalization_id
        ORDER BY m.hospitalization_id, m.admin_dttm, m.med_category
    """).fetchdf()

    # Log forward-fill stats
    n_forward = int((result["_weight_imputed"] == 1).sum())
    if n_forward > 0:
        n_pts_fwd = result.loc[result["_weight_imputed"] == 1, "hospitalization_id"].nunique()
        logger.info(
            "Forward-filled weight_kg for %d rows across %d patients "
            "using patient's own weight recorded after med administration",
            n_forward, n_pts_fwd,
        )

    # Cohort median for patients with NO weight ever recorded
    n_still_missing = int(result["weight_kg"].isna().sum())
    n_patients_missing = result.loc[result["weight_kg"].isna(), "hospitalization_id"].nunique()

    if n_still_missing > 0 and impute_missing:
        median_wt = weight_df["weight_kg"].median()
        result["weight_kg"] = result["weight_kg"].fillna(median_wt)
        logger.warning(
            "Imputed weight_kg with cohort median (%.1f kg) for %d rows "
            "across %d patients with NO weight ever recorded",
            median_wt, n_still_missing, n_patients_missing,
        )
    elif n_still_missing > 0:
        logger.warning(
            "%d rows across %d patients have no weight_kg (imputation disabled)",
            n_still_missing, n_patients_missing,
        )

    return result


# ── Step 3: Standardize to base units ────────────────────────────────────────

def standardize_to_base_units(med_df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned dose units to base units (mcg/min, ml/min, u/min).

    Expects columns: _clean_unit, med_dose, weight_kg.
    Adds: _unit_class, _base_dose, _base_unit.
    """
    amount_case = _build_case([L_REGEX, MU_REGEX, MG_REGEX, NG_REGEX, G_REGEX], "_clean_unit")
    time_case = _build_case([HR_REGEX], "_clean_unit")
    weight_case = _build_case([KG_REGEX, LB_REGEX], "_clean_unit")

    return duckdb.sql(f"""
        SELECT *
            , CASE
                WHEN _clean_unit IN ('{_RATE_STR}') THEN 'rate'
                WHEN _clean_unit IN ('{_AMOUNT_STR}') THEN 'amount'
                ELSE 'unrecognized'
              END AS _unit_class
            , CASE
                WHEN regexp_matches(_clean_unit, '{WEIGHT_REGEX}') THEN 1 ELSE 0
              END AS _weighted
            , CASE
                WHEN _unit_class = 'unrecognized' THEN med_dose
                WHEN _weighted = 1 AND weight_kg IS NULL THEN med_dose
                ELSE med_dose * ({amount_case}) * ({time_case}) * ({weight_case})
              END AS _base_dose
            , CASE
                WHEN _weighted = 1 AND weight_kg IS NULL THEN _clean_unit
                WHEN _unit_class = 'unrecognized' THEN _clean_unit
                WHEN _unit_class = 'rate' AND regexp_matches(_clean_unit, '{MASS_REGEX}') THEN 'mcg/min'
                WHEN _unit_class = 'rate' AND regexp_matches(_clean_unit, '{VOLUME_REGEX}') THEN 'ml/min'
                WHEN _unit_class = 'rate' AND regexp_matches(_clean_unit, '{UNIT_REGEX}') THEN 'u/min'
                WHEN _unit_class = 'amount' AND regexp_matches(_clean_unit, '{MASS_REGEX}') THEN 'mcg'
                WHEN _unit_class = 'amount' AND regexp_matches(_clean_unit, '{VOLUME_REGEX}') THEN 'ml'
                WHEN _unit_class = 'amount' AND regexp_matches(_clean_unit, '{UNIT_REGEX}') THEN 'u'
              END AS _base_unit
        FROM med_df
    """).fetchdf()


# ── Step 4: Convert base → preferred units ───────────────────────────────────

def convert_to_preferred_units(
    med_df: pd.DataFrame,
    preferred_units: dict,
) -> pd.DataFrame:
    """Convert from base units to per-medication preferred units.

    Parameters
    ----------
    med_df : pd.DataFrame
        Output of standardize_to_base_units (must have _base_dose, _base_unit, weight_kg).
    preferred_units : dict
        {med_category: preferred_unit_string}, e.g. {'norepinephrine': 'mcg/kg/min'}.

    Returns
    -------
    pd.DataFrame
        With columns: med_dose_converted, med_dose_unit_converted, _convert_status.
    """
    # Validate preferred units
    bad = set(preferred_units.values()) - ALL_ACCEPTABLE_UNITS
    if bad:
        raise ValueError(f"Unrecognized preferred units: {bad}. Must be in ALL_ACCEPTABLE_UNITS.")

    # Join preferred units onto med_df
    pref_df = pd.DataFrame(
        list(preferred_units.items()),
        columns=["med_category", "_preferred_unit"],
    )
    med_df = duckdb.sql("""
        SELECT m.*, COALESCE(p._preferred_unit, m._base_unit) AS _preferred_unit
        FROM med_df m
        LEFT JOIN pref_df p USING (med_category)
    """).fetchdf()

    # Build inverse conversion factors (base → preferred)
    amount_case = _build_case([L_REGEX, MU_REGEX, MG_REGEX, NG_REGEX, G_REGEX], "_preferred_unit", inverse=True)
    time_case = _build_case([HR_REGEX], "_preferred_unit", inverse=True)
    weight_case = _build_case([KG_REGEX, LB_REGEX], "_preferred_unit", inverse=True)

    return duckdb.sql(f"""
        SELECT *
            -- classify preferred unit
            , CASE
                WHEN _preferred_unit IN ('{_RATE_STR}') THEN 'rate'
                WHEN _preferred_unit IN ('{_AMOUNT_STR}') THEN 'amount'
                ELSE 'unrecognized'
              END AS _unit_class_preferred
            , CASE
                WHEN regexp_matches(_base_unit, '{MASS_REGEX}') THEN 'mass'
                WHEN regexp_matches(_base_unit, '{VOLUME_REGEX}') THEN 'volume'
                WHEN regexp_matches(_base_unit, '{UNIT_REGEX}') THEN 'unit'
                ELSE 'unrecognized'
              END AS _unit_subclass
            , CASE
                WHEN regexp_matches(_preferred_unit, '{MASS_REGEX}') THEN 'mass'
                WHEN regexp_matches(_preferred_unit, '{VOLUME_REGEX}') THEN 'volume'
                WHEN regexp_matches(_preferred_unit, '{UNIT_REGEX}') THEN 'unit'
                ELSE 'unrecognized'
              END AS _unit_subclass_preferred
            , CASE
                WHEN regexp_matches(_preferred_unit, '{WEIGHT_REGEX}') THEN 1 ELSE 0
              END AS _weighted_preferred
            -- status
            , CASE
                WHEN _base_unit IS NULL THEN 'original unit is missing'
                WHEN _unit_class = 'unrecognized' THEN 'original unit ' || _clean_unit || ' is not recognized'
                WHEN _unit_class_preferred = 'unrecognized' THEN 'preferred unit ' || _preferred_unit || ' is not recognized'
                WHEN _unit_class != _unit_class_preferred THEN 'cannot convert ' || _unit_class || ' to ' || _unit_class_preferred
                WHEN _unit_subclass != _unit_subclass_preferred THEN 'cannot convert ' || _unit_subclass || ' to ' || _unit_subclass_preferred
                WHEN _weighted_preferred = 1 AND weight_kg IS NULL THEN 'missing weight_kg for weighted preferred unit'
                ELSE 'success'
              END AS _convert_status
            -- converted dose
            , CASE
                WHEN _convert_status = 'success'
                    THEN _base_dose * ({amount_case}) * ({time_case}) * ({weight_case})
                ELSE med_dose
              END AS med_dose_converted
            , CASE
                WHEN _convert_status = 'success' THEN _preferred_unit
                ELSE _clean_unit
              END AS med_dose_unit_converted
        FROM med_df
    """).fetchdf()


# ── Logging ──────────────────────────────────────────────────────────────────

def _log_conversion_results(df: pd.DataFrame, counts: pd.DataFrame) -> None:
    """Log summary of dose conversion successes and failures."""
    total = len(df)
    n_success = int((df["_convert_status"] == "success").sum())
    n_fail = total - n_success

    logger.info(
        "Dose conversion complete: %d/%d rows successful (%.1f%%)",
        n_success, total, 100 * n_success / total if total else 0,
    )

    if n_fail > 0:
        # Summarize failures by med_category and reason
        failures = counts[counts["_convert_status"] != "success"]
        for _, row in failures.iterrows():
            logger.warning(
                "  %s: %d rows failed — %s (unit: %s → %s)",
                row["med_category"],
                row["count"],
                row["_convert_status"],
                row["med_dose_unit"],
                row.get("_preferred_unit", "N/A"),
            )

    # Log per-med success summary
    success_by_med = counts[counts["_convert_status"] == "success"]
    for _, row in success_by_med.iterrows():
        logger.info(
            "  %s: %d rows converted %s → %s",
            row["med_category"],
            row["count"],
            row["med_dose_unit"],
            row["med_dose_unit_converted"],
        )


def _clean_preferred_units(preferred_units: dict) -> dict:
    """Clean preferred unit strings through the same naming pipeline as dose units.

    Applies: lowercase, whitespace removal, UNIT_NAMING_VARIANTS replacements.
    E.g. 'units/min' → 'u/min', 'MCG / KG / Hour' → 'mcg/kg/hr'.
    """
    import re
    cleaned = {}
    for med, unit in preferred_units.items():
        u = re.sub(r'\s+', '', unit.lower())
        for repl, pattern in UNIT_NAMING_VARIANTS.items():
            u = re.sub(pattern, repl, u)
        cleaned[med] = u
    return cleaned


# ── Public API: full pipeline ────────────────────────────────────────────────

def convert_med_doses(
    med_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    preferred_units: dict,
    keep_intermediate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full dose conversion pipeline: clean → base units → preferred units.

    Parameters
    ----------
    med_df : pd.DataFrame
        Medication admin data with columns: hospitalization_id, admin_dttm,
        med_category, med_dose, med_dose_unit.
    weight_df : pd.DataFrame
        Pre-computed weight lookup table (from build_weight_table or loaded
        from parquet). Columns: hospitalization_id, recorded_dttm, weight_kg.
    preferred_units : dict
        {med_category: target_unit}, e.g. {'norepinephrine': 'mcg/kg/min'}.
    keep_intermediate : bool
        If True, keep all intermediate columns (_base_dose, multipliers, etc.).

    Returns
    -------
    (converted_df, counts_df)
        converted_df: med_df with med_dose_converted, med_dose_unit_converted, _convert_status.
        counts_df: summary of conversion outcomes per med_category.
    """
    # 0. Clean preferred unit strings through the same pipeline as dose units
    #    (e.g. 'units/min' → 'u/min', 'MCG/KG/Hour' → 'mcg/kg/hr')
    preferred_units = _clean_preferred_units(preferred_units)

    # 1. Clean unit strings
    df = clean_dose_units(med_df)

    # 2. Attach weight
    if "weight_kg" not in df.columns:
        df = attach_weight(df, weight_df)

    # 3. Standardize to base units
    df = standardize_to_base_units(df)

    # 4. Convert to preferred units
    df = convert_to_preferred_units(df, preferred_units)

    # 5. Build summary counts
    counts = duckdb.sql("""
        SELECT med_category, med_dose_unit, _clean_unit, _base_unit,
               _preferred_unit, med_dose_unit_converted, _convert_status,
               COUNT(*) AS count
        FROM df
        GROUP BY ALL
        ORDER BY med_category, count DESC
    """).fetchdf()

    # 6. Log conversion results
    _log_conversion_results(df, counts)

    if not keep_intermediate:
        drop_cols = [c for c in df.columns if c.startswith("_") and c not in
                     ("_convert_status", "_clean_unit")]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, counts
