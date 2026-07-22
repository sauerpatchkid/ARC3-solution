#!/usr/bin/env bash
# sweep_night1.sh — Diverse baseline: 5 games × 5 seeds × 200K actions
#
# Covers click-only (ft09), keyboard-only (tu93, g50t, ls20), mixed (dc22).
# Both reset arms on ft09 (persistence ablation); reset-on only elsewhere
# (they likely won't complete levels, making the flag a no-op).
#
# Estimated runtime: ~15 hours at ~100 act/s (the new optimized speed).
#
# Usage:  nohup bash sweep_night1.sh > sweep_night1.log 2>&1 &
set -u

SEEDS=(0 1 2 3 4)
CAP=200000
SUITE="local_suite.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="sweep_night1_${STAMP}.manifest"
: > "$MANIFEST"

echo "=================================================================="
echo " Sweep Night 1: Diverse Baseline"
echo " Games: ft09 (both arms), tu93, g50t, dc22, ls20"
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
  # ft09: both reset arms (persistence ablation — it actually completes levels)
  run_one ft09 "$s" on
  run_one ft09 "$s" off
  # Keyboard-only games (reset-on only — they don't complete levels)
  run_one tu93 "$s" on
  run_one g50t "$s" on
  run_one ls20 "$s" on
  # Mixed game
  run_one dc22 "$s" on
done

echo ""
echo "=================================================================="
echo " Sweep Night 1 complete - aggregating"
echo "=================================================================="
uv run python summarize_overnight.py "$MANIFEST" --out "sweep_night1_${STAMP}_summary"
echo ""
echo "Done. Per-run rows in $SUITE ; aggregate in sweep_night1_${STAMP}_summary.{md,csv}"
