#!/bin/bash
# Submit fig-3 ablation finetuning jobs, optionally filtered by task / axis / seed.
#
# The 108 .sub files live under runs/<task>/<axis>/<level>/seed<N>.sub (play2win
# under runs/<task>/play2win/seed<N>.sub). Run generate_subs.py first to create
# them. Filters are substring matches on the .sub path, so they compose.
#
# Usage:
#   ./submit.sh --task furniture_bench            # start here: 27 jobs
#   ./submit.sh --task furniture_bench --seed 0   # one seed: 9 jobs
#   ./submit.sh --axis object_diversity           # one axis across all tasks
#   ./submit.sh --task peg --axis play2win        # peg reference, 3 seeds
#   ./submit.sh                                   # everything: 108 jobs
#   ./submit.sh --task furniture_bench --dry-run  # list selection, submit nothing
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$DIR/runs"

TASK=""
AXIS=""
SEED=""
DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --task)    TASK="$2"; shift 2 ;;
        --axis)    AXIS="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$RUNS_DIR" ]; then
    echo "No runs/ dir — run 'python generate_subs.py' first." >&2
    exit 1
fi

# Build the filtered, sorted file list.
mapfile -t FILES < <(find "$RUNS_DIR" -type f -name '*.sub' | sort | while read -r f; do
    rel="${f#"$RUNS_DIR"/}"
    [ -n "$TASK" ] && [ "${rel%%/*}" != "$TASK" ] && continue
    [ -n "$AXIS" ] && [[ "/$rel/" != *"/$AXIS/"* ]] && continue
    [ -n "$SEED" ] && [[ "$(basename "$rel")" != "seed${SEED}.sub" ]] && continue
    echo "$f"
done)

N=${#FILES[@]}
echo "Matched $N job(s)  [task='${TASK:-*}' axis='${AXIS:-*}' seed='${SEED:-*}']"
if [ "$N" -eq 0 ]; then exit 0; fi

if [ "$DRY" -eq 1 ]; then
    for f in "${FILES[@]}"; do echo "  ${f#"$RUNS_DIR"/}"; done
    echo "(dry-run — nothing submitted)"
    exit 0
fi

for f in "${FILES[@]}"; do
    sbatch "$f"
    echo "Submitted ${f#"$RUNS_DIR"/}"
done
