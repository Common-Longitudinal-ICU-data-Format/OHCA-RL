#!/usr/bin/env bash
#
# OHCA-RL training pipeline (coordinating-center / training-site use)
#
# Runs the full pipeline:
#   01_cohort → 02_wide → 03_sofa → 04_mdp → 05_reward → 06_model_cql_fqe
#   → make_tableone
#
# Trained model checkpoint and feature metadata are copied to shared/
# at the end of step 06, ready for distribution to external sites.
#
# Full stdout/stderr is captured to output/final/training_log_<timestamp>.txt.
#
# Usage:
#   bash run_training.sh
#
# Requires: config/config.json with site_name, tables_path, file_type,
# timezone, project_root set for this site.

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"

LOG_DIR="output/final"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/training_log_${TS}.txt"

{
    echo "================================================================"
    echo "OHCA-RL TRAINING PIPELINE — $(date)"
    echo "Log: $LOG_FILE"
    echo "================================================================"

    for step in 01_cohort 02_wide 03_sofa 04_mdp 05_reward 06_model_cql_fqe make_tableone; do
        echo ""
        echo "──── Running code/${step}.py ────"
        "$PY" "code/${step}.py"
    done

    echo ""
    echo "================================================================"
    echo "Training pipeline complete — $(date)"
    echo "  Shareable artifacts → output/final/"
    echo "  Shareable model     → shared/ddqn_cql_reviewed.pt"
    echo "  Log                 → $LOG_FILE"
    echo "================================================================"
} 2>&1 | tee "$LOG_FILE"
