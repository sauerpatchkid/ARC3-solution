#!/usr/bin/env bash
# sweep_night2.sh — Broader characterization: 4 more games × 3 seeds × 50K actions
#
# Shorter runs on games with interesting probe characteristics:
#   ka59 — mixed, 91% change rate, 19 cells (possibly learnable)
#   tn36 — click-only, 100% change rate, 1-cell changes (may show rapid progress)
#   r11l — click-only, click_changes_clicked_cell=0.68 (spatial learning matters)
#   wa30 — keyboard-only, 92% change rate, high spatial complexity (3196 cells)
#
# If any of these hit a level-up, they should be promoted to a full 200K sweep.
#
# Estimated runtime: ~1.5 hours at ~100 act/s.
#
# Usage:  nohup bash sweep_night2.sh > sweep_night2.log 2>&1 &
set -u

SEEDS=(0 1 2)
CAP=50000
SUITE="local_suite.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="sweep_night2_${STAMP}.manifest"
: > "$MANIFEST"

echo "=================================================================="
echo " Sweep Night 2: Broader Characterization"
echo " Games: ka59, tn36, r11l, wa30"
echo " Seeds: ${SEEDS[*]}   Cap: $CAP"
echo " Manifest: $MANIFEST    Suite: $SUITE"
echo "=================================================================="

run_one () {
  local game="$1" seed="$2" arm="$3" label out expdir rundir
  echo ""
  echo ">>> game=$game seed=$seed arm=reset_$arm   started $(date +%H:%M:%S)"
  if [ "$arm" = "off" ]; then
    label="goose_persist"
    out=$(EVAL_RESET_ON_LEVEL=0 EVAL_SEED="$seed" EVAL_MAX_ACTIONS="$CAP" PYTHONHASHSEED=0 \
          uv run python run_local.py --game "$game" 2>&1)
  else
    label="goose"
    out=$(EVAL_SEED="$seed" EVAL_MAX_ACTIONS="$CAP" PYTHONHASHSEED=0 \
          uv run python run_local.py --game "$game" 2>&1)
  fi
  echo "$out" | grep -E 'Score changed|\[run_local\]' || true
  expdir=$(echo "$out" | grep -oE 'runs/[0-9_]+' | head -1)
  if [ -z "$expdir" ]; then
    echo "!! could not locate run dir for game=$game seed=$seed arm=$arm - skipping metrics"
    echo "$out" | tail -20
    return 1
  fi
  rundir="$expdir/$game"
  uv run python compute_metrics.py "$rundir/transitions" \
      --game "$game" --agent "$label" --seed "$seed" --suite "$SUITE"
  printf '%s\t%s\t%s\t%s\n' "$rundir" "$game" "$seed" "$arm" >> "$MANIFEST"
}

for s in "${SEEDS[@]}"; do
  run_one ka59 "$s" on
  run_one tn36 "$s" on
  run_one r11l "$s" on
  run_one wa30 "$s" on
done

echo ""
echo "=================================================================="
echo " Sweep Night 2 complete - aggregating"
echo "=================================================================="
uv run python summarize_overnight.py "$MANIFEST" --out "sweep_night2_${STAMP}_summary"
echo ""
echo "Done. Per-run rows in $SUITE ; aggregate in sweep_night2_${STAMP}_summary.{md,csv}"
