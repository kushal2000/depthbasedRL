#!/bin/bash
# Submits 4 SLURM jobs for beam assembly: parts 6 and 2, each from scratch and finetuned.
# Usage: bash fabrica/train_scripts/submit_beam.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

CHECKPOINT="$REPO_ROOT/pretrained_policy/model.pth"

for PART in 6 2; do
    for CKPT in "" "$CHECKPOINT"; do
        if [[ -n "$CKPT" ]]; then
            MODE="finetune"
            EXPORT_VARS="ALL,REPO_ROOT=$REPO_ROOT,ASSEMBLY=beam,PART_ID=$PART,CHECKPOINT=$CKPT"
        else
            MODE="scratch"
            EXPORT_VARS="ALL,REPO_ROOT=$REPO_ROOT,ASSEMBLY=beam,PART_ID=$PART"
        fi

        RUN_LOG_DIR="$LOG_DIR/beam_${PART}_${MODE}"
        mkdir -p "$RUN_LOG_DIR"

        sbatch \
            -J "beam_${PART}_${MODE}" \
            -o "$RUN_LOG_DIR/%j.log" \
            -e "$RUN_LOG_DIR/%j.err" \
            --export="$EXPORT_VARS" \
            "$SCRIPT_DIR/train_fabrica.sub"

        echo "Submitted beam_${PART} (${MODE})"
    done
done
