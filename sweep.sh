#!/usr/bin/env bash
# sweep.sh — unified overnight sweep (StochasticGoose, Team B, ARC-AGI-3).
#
# Replaces run_overnight.sh + sweep_night1.sh + sweep_night2.sh with ONE
# config-driven script. Runs games × seeds × reset-arms on the LOCAL engine,
# scores each run with compute_metrics.py (appending to results/local_suite.csv),
# records a manifest, then aggregates with summarize_overnight.py.
#
# All output lands under results/ (override root with EVAL_RESULTS_DIR):
#   results/runs/<ts>/<game>/     per-run trees
#   results/sweeps/               manifest + aggregate summary
#   results/local_suite.csv       per-run metric rows
#
# Configure via the CONFIG block below or override on the command line:
#   BENCH  name of a FROZEN suite from benchmark.py (smoke/quick/standard/full).
#          Setting it overrides GAMES/SEEDS/CAP so results are comparable
#          between people. This is what `make bench` uses.
#   AGENT  any name registered in custom_agents/__init__.py.
#   GAMES  space-separated game ids, for ad-hoc sweeps only. Tag a game ':both'
#          to run BOTH reset arms (the persistence ablation), ':off' for
#          persist-only, else reset-on.
#   SEEDS  space-separated seeds.
#   CAP    per-game action cap.
#
# Presets (copy-paste):
#   # diverse baseline (was sweep_night1.sh):
#   GAMES="ft09:both tu93 g50t dc22 ls20" SEEDS="0 1 2 3 4" CAP=200000 bash sweep.sh
#   # broader characterization (was sweep_night2.sh):
#   GAMES="ka59 tn36 r11l wa30" SEEDS="0 1 2" CAP=50000 bash sweep.sh
#
#   # long run on the two games that complete levels (see Makefile: make long):
#   GAMES="ft09 tu93" SEEDS="0" CAP=1000000 bash sweep.sh
#
# Background it for a real overnight:
#   mkdir -p results/sweeps && nohup bash sweep.sh > results/sweeps/sweep.log 2>&1 &
set -u

# ------------------------------- CONFIG ------------------------------------
# BENCH names a frozen suite from benchmark.py. When set, it OVERRIDES
# GAMES/SEEDS/CAP so that everyone comparing results ran the same thing:
#     BENCH=standard AGENT=goose bash sweep.sh
# Leave BENCH unset for an ad-hoc exploratory sweep with your own GAMES/SEEDS.
BENCH="${BENCH:-}"
AGENT="${AGENT:-goose}"          # any name in custom_agents/__init__.py REGISTRY
RESULTS="${EVAL_RESULTS_DIR:-results}"
SUITE_CSV="${SUITE_CSV:-$RESULTS/local_suite.csv}"

if [ -n "$BENCH" ]; then
  BENCH_ENV="$(uv run python benchmark.py --suite "$BENCH" --env)" || exit 1
  eval "$BENCH_ENV"              # sets GAMES, SEEDS, CAP, SUITE_VERSION
  TAG="bench_${BENCH}_v${SUITE_VERSION}_${AGENT}"
else
  GAMES="${GAMES:-ft09:both tu93 g50t dc22 ls20}"
  SEEDS="${SEEDS:-0 1 2 3 4}"
  CAP="${CAP:-200000}"
  TAG="sweep"
fi
# ---------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEPDIR="$RESULTS/sweeps"
mkdir -p "$SWEEPDIR"
MANIFEST="$SWEEPDIR/${TAG}_${STAMP}.manifest"
: > "$MANIFEST"

echo "=================================================================="
if [ -n "$BENCH" ]; then
  echo " FROZEN BENCHMARK: $BENCH (v$SUITE_VERSION)"
else
  echo " Ad-hoc sweep (not comparable across people - use BENCH= for that)"
fi
echo " Agent: $AGENT"
echo " Games: $GAMES"
echo " Seeds: $SEEDS   Cap: $CAP"
echo " Manifest: $MANIFEST    Metrics CSV: $SUITE_CSV"
echo "=================================================================="

# Count the runs and estimate the cost before committing hours to it.
n_runs=0
for _s in $SEEDS; do for _t in $GAMES; do
  case "${_t#*:}" in both) n_runs=$((n_runs+2));; *) n_runs=$((n_runs+1));; esac
done; done
total_actions=$((n_runs * CAP))
echo " Runs: $n_runs   Total actions: $total_actions"
awk -v a=$total_actions 'BEGIN{printf " ETA : %.1f h @140 act/s | %.1f h @130 | %.1f h @120\n", a/140/3600, a/130/3600, a/120/3600}'
echo "=================================================================="
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 - plan printed above, nothing executed."
  exit 0
fi

run_one () {
  local game="$1" seed="$2" arm="$3" label out expdir rundir
  echo ""
  echo ">>> game=$game seed=$seed arm=reset_$arm   started $(date +%H:%M:%S)"
  if [ "$arm" = "off" ]; then
    label="${AGENT}_persist"
    out=$(EVAL_RESET_ON_LEVEL=0 EVAL_SEED="$seed" EVAL_MAX_ACTIONS="$CAP" PYTHONHASHSEED=0 \
          uv run python run_local.py --game "$game" --agent "$AGENT" 2>&1)
  else
    label="$AGENT"
    out=$(EVAL_SEED="$seed" EVAL_MAX_ACTIONS="$CAP" PYTHONHASHSEED=0 \
          uv run python run_local.py --game "$game" --agent "$AGENT" 2>&1)
  fi
  echo "$out" | grep -E 'Score changed|\[run_local\]' || true
  # run_local.py's last line is "[run_local] transitions: <path>/transitions" -
  # parse that rather than pattern-matching the run tree layout.
  corpus=$(echo "$out" | sed -n 's/^\[run_local\] transitions: //p' | tail -1)
  if [ -z "$corpus" ]; then
    echo "!! could not locate run dir for game=$game seed=$seed arm=$arm - skipping metrics"
    echo "$out" | tail -20
    return 1
  fi
  rundir=$(dirname "$corpus")
  uv run python compute_metrics.py "$rundir/transitions" \
      --game "$game" --agent "$label" --seed "$seed" --suite "$SUITE_CSV"
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
uv run python summarize_overnight.py "$MANIFEST" --out "$SWEEPDIR/${TAG}_${STAMP}_summary"
uv run python analyze_curves.py "$MANIFEST" --out "$SWEEPDIR/curves_${TAG}_${STAMP}" \
  || echo "!! curve analysis failed (metrics + summary above are unaffected)"
echo ""
echo "Done."
echo "  per-run rows : $SUITE_CSV"
echo "  aggregate    : $SWEEPDIR/${TAG}_${STAMP}_summary.{md,csv}"
echo "  curves/AULC  : $SWEEPDIR/curves_${TAG}_${STAMP}.{md,csv,png}"
echo "  manifest     : $MANIFEST"
if [ -n "$BENCH" ]; then
  echo ""
  echo "Compare against another agent's run of the SAME suite:"
  echo "  make compare M1=$MANIFEST M2=<their manifest>"
fi
