#!/usr/bin/env bash
#
# OHCA-RL external validation pipeline (site use)
#
# Runs the site's data through cohort → state → reward construction,
# then loads the pre-trained model from shared/ddqn_cql_reviewed.pt and
# runs the same evaluation that the coordinating center ran on its own data.
#
# Steps:
#   01_cohort → 02_wide → 03_sofa → 04_mdp → 05_reward
#   → external_validation (uses shared model, no training)
#   → make_tableone
#
# Full stdout/stderr is captured to output/final/validation_log_<timestamp>.txt.
#
# Before running:
#   1. Edit config/config.json with your site's name, tables path, etc.
#   2. Confirm shared/ddqn_cql_reviewed.pt and shared/feature_metadata.json
#      were placed in this repo by the coordinating center.
#
# Usage:
#   bash run_validation.sh

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"

if [ ! -f "shared/ddqn_cql_reviewed.pt" ]; then
    echo "ERROR: shared/ddqn_cql_reviewed.pt not found."
    echo "       Obtain this file from the coordinating center."
    exit 1
fi

LOG_DIR="output/final"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/validation_log_${TS}.txt"

{
    echo "================================================================"
    echo "OHCA-RL EXTERNAL VALIDATION PIPELINE — $(date)"
    echo "Log: $LOG_FILE"
    echo "================================================================"

    for step in 01_cohort 02_wide 03_sofa 04_mdp 05_reward external_validation make_tableone; do
        echo ""
        echo "──── Running code/${step}.py ────"
        "$PY" "code/${step}.py"
    done

    echo ""
    echo "================================================================"
    echo "Validation pipeline complete — $(date)"
    echo "  Site-specific shareable artifacts → output/final/<site_id>/"
    echo "  Log                               → $LOG_FILE"
    echo "  Upload that folder + this log to the coordinating center"
    echo "  per EXTERNAL_VALIDATION.md"
    echo "================================================================"
} 2>&1 | tee "$LOG_FILE"
