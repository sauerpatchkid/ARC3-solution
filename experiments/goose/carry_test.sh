#!/usr/bin/env bash
# carry_test.sh - the carry test (docs/1-goose-upgrades/carry-inside-goose.md), runnable in parts.
#
# The problem it fixes: the test is about 11-13 GPU-hours of one run after another, and the PC is
# not always free for that long. This starts it detached, says how far it is, pauses it, stops it
# and picks it up again, and at the end prints the tables its registration names.
#
# The run set is fixed here, as registered: arm A5 (the adopted Goose's three settings by flag) and
# arm A6 (A5 + carry_noops,carry_hints), 7 games x seeds 7-10 x 100,000 actions = 56 runs, one at a
# time through sweep.sh, the two arms back to back for each game and seed.
#
#   bash experiments/goose/carry_test.sh start     begin (detached: it outlives the terminal)
#   bash experiments/goose/carry_test.sh status    runs done, the run in progress, time left
#   bash experiments/goose/carry_test.sh pause     stop when the run in progress finishes (up to ~15 min)
#   bash experiments/goose/carry_test.sh stop      stop now; the run in progress is rerun whole on resume
#   bash experiments/goose/carry_test.sh resume    continue after a pause, a stop, a crash or a reboot
#   bash experiments/goose/carry_test.sh verdict   when all 56 runs are in: the registered tables
#
# A run enters the manifest only after it has been played to the end and scored, so a pause loses
# nothing and a stop (or a crash, or a reboot) loses only the run in progress.
#
# SMOKE=1 does the same on a tiny set (2 games x 1 seed x 3,000 actions, under results/carry_smoke/),
# to try start, pause, stop and resume before the real run.
set -u
cd "$(dirname "$0")/../.." || exit 1

if [ "${SMOKE:-0}" = "1" ]; then
  RESULTS=results/carry_smoke; GAMES="ft09 tu93"; SEEDS="7"; CAP=3000; CLICK_GAMES="ft09"
else
  RESULTS=results; GAMES="vc33 ft09 cd82 dc22 m0r0 lp85 tu93"; SEEDS="7 8 9 10"; CAP=100000
  CLICK_GAMES="vc33,ft09,cd82,dc22,m0r0,lp85"
fi
PLACEBO="tu93"; ARMS="A5 A6"
SW="$RESULTS/sweeps"; LOG="$SW/carry_test.log"; PIDFILE="$SW/carry_test.pid"
N_RUNS=$(( $(echo $GAMES | wc -w) * $(echo $SEEDS | wc -w) * $(echo $ARMS | wc -w) ))
N_PAIRS=$(( N_RUNS / 2 ))

manifest () {           # the sweep's manifest, as its log header names it (the last one, after a resume)
  [ -f "$LOG" ] && sed -n 's/^ Manifest: \([^ ]*\) .*/\1/p' "$LOG" | tail -1
}

runs_done () {
  local m; m=$(manifest)
  if [ -n "$m" ] && [ -f "$m" ]; then wc -l < "$m" | tr -d ' '; else echo 0; fi
}

running () {            # prints the sweep's PID if it is alive (and is this sweep, not a reused PID)
  [ -f "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "sweep.sh" && echo "$pid"
}

launch () {             # $1: the manifest to resume, or nothing
  mkdir -p "$SW"
  # no setting of the caller's shell reaches a run: each arm sets its own EVAL_* flags
  for v in $(env | sed -n 's/^\(EVAL_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
  EVAL_RESULTS_DIR="$RESULTS" RESUME="${1:-}" GAMES="$GAMES" SEEDS="$SEEDS" CAP="$CAP" ARMS="$ARMS" \
    AGENT=goose BENCH= setsid nohup bash sweep.sh >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "started detached (pid $(cat "$PIDFILE")); log: $LOG"
  echo "  status: bash experiments/goose/carry_test.sh status     pause: ... pause     stop now: ... stop"
}

case "${1:-}" in
  start)
    running > /dev/null && { echo "the carry test is already running (pid $(running)): see status"; exit 1; }
    [ -n "$(manifest)" ] && { echo "a carry test was already started ($(manifest), $(runs_done) of $N_RUNS runs): use resume"; exit 1; }
    mkdir -p "$SW"; : > "$LOG"
    launch ""
    ;;
  resume)
    running > /dev/null && { echo "the carry test is running (pid $(running)): nothing to resume"; exit 1; }
    m=$(manifest)
    [ -n "$m" ] && [ -f "$m" ] || { echo "no carry test to resume: use start"; exit 1; }
    [ "$(runs_done)" -ge "$N_RUNS" ] && { echo "all $N_RUNS runs are in: use verdict"; exit 0; }
    echo "resuming $m: $(runs_done) of $N_RUNS runs done"
    launch "$m"
    ;;
  pause)
    running > /dev/null || { echo "the carry test is not running: nothing to pause"; exit 0; }
    touch "$SW/STOP"
    echo "pausing: no new run starts, and the sweep stops when the run in progress finishes (up to ~15 min)."
    echo "  see it with: status      continue with: resume"
    ;;
  stop)
    pid=$(running) || { echo "the carry test is not running"; exit 0; }
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [ "$pgid" = "$pid" ] || { echo "the sweep does not lead its own process group (pid $pid, group $pgid): use pause"; exit 1; }
    kill -TERM -- "-$pgid" 2>/dev/null
    for _ in $(seq 1 40); do pgrep -g "$pgid" > /dev/null || break; sleep 0.5; done
    pgrep -g "$pgid" > /dev/null && kill -KILL -- "-$pgid" 2>/dev/null
    rm -f "$SW/STOP"
    echo "=== STOPPED NOW by carry_test.sh stop ($(date +%H:%M:%S)); the run in progress is rerun whole on resume" >> "$LOG"
    echo "stopped now: $(runs_done) of $N_RUNS runs are in the manifest; the run that was in progress is in"
    echo "no manifest and is rerun whole. continue with: bash experiments/goose/carry_test.sh resume"
    ;;
  status)
    d=$(runs_done)
    if pid=$(running); then state="RUNNING (pid $pid)"; else state="NOT running"; fi
    echo "carry test: $d of $N_RUNS runs done; the sweep is $state"
    if [ -n "$(manifest)" ]; then echo "  manifest: $(manifest)"; fi
    if running > /dev/null; then
      if [ -f "$SW/STOP" ]; then echo "  pausing: it stops when the run in progress finishes"; fi
      echo "  in progress: $(grep '^>>> game=' "$LOG" | tail -1 | sed 's/^>>> //')"
    fi
    secs=$(sed -n 's/^\[run_local\] done: [0-9]* actions in \([0-9.]*\)s.*/\1/p' "$LOG" 2>/dev/null \
           | awk '{s+=$1; n++} END{if (n) printf "%.0f", s/n}')
    if [ -n "$secs" ] && [ "$d" -lt "$N_RUNS" ]; then
      awk -v left=$((N_RUNS - d)) -v s="$secs" 'BEGIN{printf "  about %.1f h of running left (finished runs took %.1f min each, plus scoring)\n", left*(s+25)/3600, s/60}'
    fi
    if [ "$d" -ge "$N_RUNS" ]; then echo "  all runs are in: bash experiments/goose/carry_test.sh verdict"; fi
    exit 0
    ;;
  verdict)
    m=$(manifest); d=$(runs_done)
    [ "$d" -ge "$N_RUNS" ] || { echo "only $d of $N_RUNS runs are in: no verdict yet"; exit 1; }
    echo "=== safety: levels at $CAP actions, the dev rule, all games"
    uv run python tools/paired_compare.py --base "$m:A5" --new "$m:A6" --rule dev --expect "$N_PAIRS" \
        --out "$SW/carry_test_levels.md" | tail -9
    echo; echo "=== primary: level races on the click games ($CLICK_GAMES)"
    uv run python tools/levels_by_budget.py --base "$m:A5" --new "$m:A6" --games "$CLICK_GAMES" --by-game \
        --out "$SW/carry_test_click_games.md" | tail -n +4
    echo; echo "=== placebo: $PLACEBO (buttons only: the carry cannot act)"
    uv run python tools/levels_by_budget.py --base "$m:A5" --new "$m:A6" --games "$PLACEBO" \
        --out "$SW/carry_test_placebo.md" | grep -E '^\| (Which|level)'
    echo
    uv run python - "$m" "$CLICK_GAMES" <<'PY'
import sys
sys.path.insert(0, "tools")
import levels_by_budget as L
m, games = sys.argv[1], set(sys.argv[2].split(","))
base, new = L.load(m + ":A5", games), L.load(m + ":A6", games)
later = [r for r in L.level_races(base, new) if r[2] >= 2]
won, lost = sum(r[3] == "new" for r in later), sum(r[3] == "base" for r in later)
per_game = {}
for game, _, _, winner, _ in later:
    per_game[game] = per_game.get(game, 0) + (winner == "new") - (winner == "base")
lo, hi = L.bootstrap_delta(list(per_game.values())) if per_game else (0, 0)
ok = won - lost > 0 and lo > 0
print(f"PRIMARY (levels 2 and up, the click games): A6 faster on {won}, A5 faster on {lost}; "
      f"A6 - A5 = {won - lost:+d}, 95% interval [{lo:+.0f}, {hi:+.0f}] (games resampled): "
      f"{'PASS' if ok else 'FAIL'}")
PY
    echo "tables: $SW/carry_test_levels.md, carry_test_click_games.md, carry_test_placebo.md"
    ;;
  *)
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
