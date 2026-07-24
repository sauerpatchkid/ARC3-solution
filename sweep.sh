#!/usr/bin/env bash
# sweep.sh — unified overnight sweep (StochasticGoose, Team B, ARC-AGI-3).
#
# Replaces run_overnight.sh + sweep_night1.sh + sweep_night2.sh with ONE
# config-driven script. Runs games × seeds × reset-arms on the LOCAL engine,
# scores each run with compute_metrics.py (appending to local_suite.csv),
# records a manifest, then aggregates with summarize_overnight.py.
#
# Configure via the CONFIG block below or override on the command line:
#   GAMES  space-separated game ids. Tag a game ':both' to run BOTH reset arms
#          (the persistence ablation), ':off' for persist-only, else reset-on.
#   SEEDS  space-separated seeds.
#   CAP    per-game action cap.
#
# Presets (copy-paste):
#   # diverse baseline (was sweep_night1.sh):
#   GAMES="ft09:both tu93 g50t dc22 ls20" SEEDS="0 1 2 3 4" CAP=200000 bash sweep.sh
#   # broader characterization (was sweep_night2.sh):
#   GAMES="ka59 tn36 r11l wa30" SEEDS="0 1 2" CAP=50000 bash sweep.sh
#
# Background it for a real overnight:  nohup bash sweep.sh > sweep.log 2>&1 &
set -u

# ------------------------------- CONFIG ------------------------------------
GAMES="${GAMES:-ft09:both tu93 g50t dc22 ls20}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CAP="${CAP:-200000}"
SUITE="${SUITE:-local_suite.csv}"
# ---------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="sweep_${STAMP}.manifest"
: > "$MANIFEST"

echo "=================================================================="
echo " Unified sweep"
echo " Games: $GAMES"
echo " Seeds: $SEEDS   Cap: $CAP"
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

for s in $SEEDS; do
  for tok in $GAMES; do
    game="${tok%%:*}"
    arm="on"
    [ "$tok" != "$game" ] && arm="${tok#*:}"   # suffix after ':' if present
    case "$arm" in
      both) run_one "$game" "$s" on ; run_one "$game" "$s" off ;;
      off)  run_one "$game" "$s" off ;;
      *)    run_one "$game" "$s" on ;;
    esac
  done
done

echo ""
echo "=================================================================="
echo " Sweep complete - aggregating"
echo "=================================================================="
uv run python summarize_overnight.py "$MANIFEST" --out "sweep_${STAMP}_summary"
echo ""
echo "Done. Per-run rows in $SUITE ; aggregate in sweep_${STAMP}_summary.{md,csv}"
