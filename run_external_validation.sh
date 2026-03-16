#!/usr/bin/env bash
# OHCA-RL External Validation Pipeline
# For participating sites validating the trained model on local data.
# Skips step 06 (training) -- uses pre-trained model from shared/.
#
# Usage: bash run_external_validation.sh
# See SITE_INSTRUCTIONS.md for full setup guide.

set -euo pipefail

cd "$(dirname "$0")"

# Combined log file with timestamp
mkdir -p output/final
LOG_FILE="output/final/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "  OHCA-RL External Validation Pipeline"
echo "  Log: ${LOG_FILE}"
echo "=========================================="

# Verify shared/ artifacts exist before starting
REQUIRED_FILES=(
    "shared/best_model.pt"
    "shared/preprocessor.json"
    "shared/state_features.json"
    "shared/training_config.json"
    "shared/action_remap.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing ${f}"
        echo "Download standardization artifacts from Box first."
        echo "See SITE_INSTRUCTIONS.md for details."
        exit 1
    fi
done

echo "  Shared artifacts verified."

STEPS=(
    "code/00_cohort_identification.py"
    "code/01_create_wide_df.py"
    "code/02_sofa_calculator.py"
    "code/03_ffill_and_bucketing.py"
    "code/04_create_tableone.py"
    "code/05_figures.py"
    "code/07_external_validation.py"
)

# Run pipeline, tee all stdout+stderr to log file
{
    for step in "${STEPS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Running: ${step}"
        echo "------------------------------------------"
        uv run "$step"
        echo "  Done: ${step}"
    done

    echo ""
    echo "=========================================="
    echo "  External validation complete!"
    echo "  Results: output/final/external_validation/"
    echo "=========================================="
} 2>&1 | tee -a "$LOG_FILE"
