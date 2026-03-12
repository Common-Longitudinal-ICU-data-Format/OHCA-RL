#!/usr/bin/env bash
# OHCA-RL Pipeline Runner
# Runs all pipeline steps sequentially using marimo in headless mode.
# Usage: bash run_pipeline.sh

set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo "  OHCA-RL Pipeline"
echo "=========================================="

STEPS=(
    "code/00_cohort_identification.py"
    "code/01_sofa_calculator.py"
    "code/02_create_wide_df.py"
    "code/03_ffill_and_bucketing.py"
    "code/04_create_tableone.py"
)

for step in "${STEPS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Running: ${step}"
    echo "──────────────────────────────────────────"
    uv run "$step"
    echo "  ✓ Done: ${step}"
done

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "=========================================="
