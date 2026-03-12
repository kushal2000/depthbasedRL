#!/bin/bash
# Submits 4 SLURM jobs for beam assembly: parts 6 and 2, each from scratch and finetuned.
# Usage: bash fabrica/train_scripts/submit_beam.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

        sbatch \
            -o /dev/null \
            -e /dev/null \
            --export="$EXPORT_VARS" \
            "$SCRIPT_DIR/train_fabrica.sub"

        echo "Submitted beam_${PART} (${MODE})"
    done
done
