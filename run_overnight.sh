#!/usr/bin/env bash
# run_overnight.sh - StochasticGoose overnight sweep (Team B, ARC-AGI-3).
#
# Runs the local engine (run_local.py) across games x seeds x reset-arms,
# scores each run with compute_metrics.py (which appends to suite_summary.csv),
# records a manifest, then aggregates everything with summarize_overnight.py.
#
# Usage:  bash run_overnight.sh
# Background it for a real overnight:  nohup bash run_overnight.sh > overnight.log 2>&1 &
#
# Edit the CONFIG block below to change the sweep.
set -u

# ------------------------------- CONFIG ------------------------------------
SEEDS=(0 1 2 3 4)
CAP=200000
# ft09 reaches level boundaries -> test BOTH reset arms (persistence ablation).
# ls20 never completes -> the reset flag is a no-op there, so reset-on only
# (it serves as the exploration contrast game).
FT09_ARMS=("on" "off")
LS20_ARMS=("on")
SUITE="local_suite.csv"
# ---------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="overnight_${STAMP}.manifest"
: > "$MANIFEST"

n_runs=$(( ${#SEEDS[@]} * (${#FT09_ARMS[@]} + ${#LS20_ARMS[@]}) ))
per_run_min=$(python3 -c "print(round($CAP/60/60,1))")   # ~60 act/s assumption
total_h=$(python3 -c "print(round($n_runs*$CAP/60/60/60,1))")
echo "=================================================================="
echo " Overnight sweep   cap=$CAP   seeds=${SEEDS[*]}"
echo " ft09 arms: ${FT09_ARMS[*]}    ls20 arms: ${LS20_ARMS[*]}"
echo " $n_runs runs  ~${per_run_min} min/run  ->  up to ~${total_h} h total"
echo " manifest: $MANIFEST     suite: $SUITE"
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
  # show the interesting lines (level-ups, final summary); full stdout captured above
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
  for a in "${FT09_ARMS[@]}"; do run_one ft09 "$s" "$a"; done
  for a in "${LS20_ARMS[@]}"; do run_one ls20 "$s" "$a"; done
done

echo ""
echo "=================================================================="
echo " Sweep complete - aggregating"
echo "=================================================================="
uv run python summarize_overnight.py "$MANIFEST" --out "overnight_${STAMP}_summary"
echo ""
echo "Done. Per-run rows in $SUITE ; aggregate in overnight_${STAMP}_summary.{md,csv}"
