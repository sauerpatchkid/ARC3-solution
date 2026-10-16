#!/usr/bin/env bash
# ARCHIVED 2026-09-04 - superseded by the frozen benchmark:
#
#     make bench SUITE=full AGENT=goose
#
# which runs the same 25 games x 1 seed x 100k actions through sweep.sh, with the
# same metrics and a manifest that `make compare` can read. Kept here only for
# its game-availability preflight, which the benchmark path does not have.
# sweep_all25.sh — one-shot overnight sweep: ALL 25 public games, 1 seed each.
#
# Drop-in and self-contained: it does not modify anything in the repo, it just
# drives the existing, already-tested sweep.sh with a 25-game list. Output goes
# where every other run's output goes (results/), so compute_metrics.py,
# summarize_overnight.py and analyze_curves.py all apply unchanged.
#
#   bash sweep_all25.sh                  # runs it, logs to results/sweeps/
#   nohup bash sweep_all25.sh >/dev/null 2>&1 &     # detached, for overnight
#   CAP=50000 bash sweep_all25.sh        # shorter budget
#   SKIP_PREFLIGHT=1 bash sweep_all25.sh # skip the game-availability check
#   DRY_RUN=1 bash sweep_all25.sh        # print the plan and exit
#
# Estimated wall time at 120-140 act/s: about 5.3-6.1 h for 25 x 100k, including
# per-game engine/model startup. See the ETA the script prints at launch.
set -u

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# The 25 public games.
#
# Ordered UNCHARACTERIZED FIRST, deliberately. The last five already have
# 200k x 4-5 seeds of data plus (for ft09/tu93) a 2M-action long run, so if the
# sweep is cut short overnight you lose only duplicate coverage. The first
# twenty are the ones the report has nothing on.
# ---------------------------------------------------------------------------
GAMES_NEW="ar25 bp35 cd82 cn04 ka59 lf52 lp85 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 vc33 wa30"
GAMES_KNOWN="ft09 tu93 g50t dc22 ls20"
GAMES="${GAMES:-$GAMES_NEW $GAMES_KNOWN}"

SEED="${SEED:-0}"
CAP="${CAP:-100000}"
RESULTS="${EVAL_RESULTS_DIR:-results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS/sweeps/all25_${STAMP}.log"

mkdir -p "$RESULTS/sweeps"
exec > >(tee -a "$LOG") 2>&1      # log even when run without a redirect

n_games=$(echo "$GAMES" | wc -w)
total_actions=$((n_games * CAP))

echo "=================================================================="
echo " All-25 overnight sweep"
echo "=================================================================="
echo " games   : $n_games x ${CAP} actions, seed $SEED, reset-on"
echo " total   : $total_actions actions"
echo " ETA     : $(awk -v a=$total_actions 'BEGIN{printf "%.1f h @140 act/s  |  %.1f h @130  |  %.1f h @120", a/140/3600, a/130/3600, a/120/3600}')"
echo "           (+ roughly 20 min of per-game engine/model startup)"
echo " log     : $LOG"
echo " started : $(date)"
echo "=================================================================="
echo " order   : $GAMES"
echo ""

# ---------------------------------------------------------------------------
# Preflight: confirm every game can actually be built before committing hours
# to the sweep. 20 of these 25 have never run on the local engine, so their
# game files still need a one-time download — doing it now means the long run
# is not exposed to a network hiccup at 3am, and a bad id fails in seconds
# instead of silently costing a slot.
# ---------------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo ">>> preflight: checking/caching $n_games games ..."
  PREFLIGHT_OUT="$(mktemp)"
  GAMES="$GAMES" uv run python - <<'PY' > "$PREFLIGHT_OUT" 2>/dev/null
import os, sys, contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    import run_local
    arc = run_local.make_arcade(False)
    ok, bad = [], []
    for g in os.environ["GAMES"].split():
        try:
            env = arc.make(g, seed=0)
            obs = env.reset()
            f = run_local.ShimFrame(obs, getattr(env, "action_space", None))
            ok.append((g, [a.name for a in f.available_actions]))
        except Exception as e:
            bad.append((g, f"{type(e).__name__}: {e}"))
for g, acts in ok:
    print(f"OK\t{g}\t{','.join(acts)}")
for g, err in bad:
    print(f"BAD\t{g}\t{err}")
PY
  awk -F'\t' '$1=="OK"  {printf "    ok   %s  [%s]\n", $2, $3}' "$PREFLIGHT_OUT"
  awk -F'\t' '$1=="BAD" {printf "    FAIL %s  %s\n", $2, $3}' "$PREFLIGHT_OUT"
  GOOD="$(awk -F'\t' '$1=="OK"{printf "%s ", $2}' "$PREFLIGHT_OUT")"
  NBAD=$(awk -F'\t' '$1=="BAD"' "$PREFLIGHT_OUT" | wc -l)
  rm -f "$PREFLIGHT_OUT"
  if [ -z "${GOOD// /}" ]; then
    echo "!! preflight: no games could be built - aborting (check the API key in"
    echo "   ARC-AGI-3-Agents/.env and that the network is up)."
    exit 1
  fi
  if [ "$NBAD" -gt 0 ]; then
    echo "!! $NBAD game(s) failed preflight and are EXCLUDED from this sweep."
  fi
  GAMES="$GOOD"
  n_games=$(echo "$GAMES" | wc -w)
  echo ">>> preflight done: $n_games game(s) will run"
  echo ""
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 - plan only, exiting without running."
  exit 0
fi

# ---------------------------------------------------------------------------
# Hand off to the existing sweep. It scores each run with compute_metrics.py,
# appends to results/local_suite.csv, writes a manifest, then aggregates with
# summarize_overnight.py and analyze_curves.py. A game that crashes is skipped
# and the sweep continues.
# ---------------------------------------------------------------------------
GAMES="$GAMES" SEEDS="$SEED" CAP="$CAP" AGENT=goose bash sweep.sh

echo ""
echo "=================================================================="
echo " All-25 sweep finished at $(date)"
echo " full log: $LOG"
echo "=================================================================="
echo ""
echo "For the report, read in this order:"
echo "  1. results/sweeps/curves_<stamp>.png / .md   levels vs budget, AULC"
echo "  2. results/sweeps/sweep_<stamp>_summary.md   per-game levels + metrics"
echo "  3. results/local_suite.csv                   one row per run"
echo ""
echo "Caveats to carry into the write-up:"
echo "  * 1 seed per game. tu93 needed 4 seeds to show ANY level completion,"
echo "    so treat a 0 here as 'not observed', never as 'cannot solve'."
echo "  * 100k actions is a CENSORING budget: ft09 reached L2 at ~23k but"
echo "    nothing more by 2M. Report the budget next to every number."
echo "  * Exploration metrics are not comparable across games without the"
echo "    matched random floor (make random). Until then, read"
echo "    unique_states_per_action, novelty_late_per_1k and redundancy"
echo "    together, never a change rate on its own."
