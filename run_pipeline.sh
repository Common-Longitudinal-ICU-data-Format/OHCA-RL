#!/usr/bin/env bash
# OHCA-RL Pipeline Runner
# Runs all pipeline steps sequentially.
# Usage: bash run_pipeline.sh
#
# Logs: Each script writes its own log to output/final/<script_name>.log
#        Combined pipeline output is saved to output/final/pipeline_<timestamp>.log

set -euo pipefail

cd "$(dirname "$0")"

# Combined log file with timestamp
mkdir -p output/final
LOG_FILE="output/final/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "  OHCA-RL Pipeline"
echo "  Log: ${LOG_FILE}"
echo "=========================================="

STEPS=(
    "code/00_cohort_identification.py"
    "code/01_sofa_calculator.py"
    "code/02_create_wide_df.py"
    "code/03_ffill_and_bucketing.py"
    "code/04_create_tableone.py"
)

# Run pipeline, tee all stdout+stderr to log file
{
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
} 2>&1 | tee -a "$LOG_FILE"
