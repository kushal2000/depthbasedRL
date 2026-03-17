#!/bin/bash
# Submits beam part 2 finetuning with CoACD collision and tight final goal tolerance (0.005).
# Usage: bash fabrica/train_scripts/submit_beam_coacd_tight_tolerance.sh
set -euo pipefail
sbatch fabrica/train_scripts/beam_2_coacd_tight_tolerance.sub
