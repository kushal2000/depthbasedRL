#!/bin/bash
# Submits beam part 2 finetuning with tight final goal tolerance.
# Usage: bash fabrica/train_scripts/submit_beam_tight_tolerance.sh
set -euo pipefail
sbatch fabrica/train_scripts/beam_2_tight_tolerance.sub
