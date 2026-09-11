#!/usr/bin/env bash
# harvest.sh - overnight data collection for the LLM track.
#
# WHY: the heuristics' referee grades Qwen's code against recorded level
# completions. ft09 has plenty (31 level-1, 25 level-2); every other game has
# one or two, too few to grade anything. This runs plain Goose - the unmodified
# baseline agent - with NEW seeds on the games that complete levels, so the LLM
# track has more completions to learn from and grade against.
#
# These are ordinary baseline runs through sweep.sh: same agent, same contract,
# same scoring. Only the choice of games, seeds and budgets is LLM-specific,
# which is why this file lives in llm_track/.
#
# Budgets are ~2-3x the action count at which each game's last level was
# completed in existing runs. Those estimates rest on 1-2 runs each, so yields
# are uncertain. Seeds 100+ keep these runs apart from benchmark seeds (0-2)
# and earlier sweeps (0-4). ft09 is skipped (enough data already); tu93 is
# skipped (its first level took 126k-400k actions). Click games go first:
# they are the heuristics' strongest case, so the most valuable data.
#
#   make -C llm_track harvest-plan     # print the plan and ETA only
#   make -C llm_track harvest          # run it in the background
set -u
cd "$(dirname "$0")/.."

# Priority order. Each row: cap | seeds | games
PLAN=(
  "40000|100 101 102 103 104 105 106 107 108 109|lp85 r11l"
  "20000|100 101 102 103 104 105 106 107 108 109|ar25"
  "60000|100 101 102 103 104 105 106 107 108 109|vc33"
  "100000|100 101 102 103 104 105|cd82"
  "30000|100 101 102 103 104 105|sp80 m0r0"
  "50000|100 101 102 103 104 105|cn04"
  "100000|100 101 102 103|tr87"
  "150000|100 101|sk48"
)

total=0; runs=0
echo "LLM-track data harvest (plain Goose, new seeds):"
for row in "${PLAN[@]}"; do
  IFS='|' read -r cap seeds games <<< "$row"
  ns=$(wc -w <<< "$seeds"); ng=$(wc -w <<< "$games")
  runs=$((runs + ns * ng)); total=$((total + ns * ng * cap))
  printf '  %-10s x %2d seeds @ %7d actions\n' "$games" "$ns" "$cap"
done
awk -v a=$total -v r=$runs 'BEGIN{printf "  => %d runs, %.2fM actions: ~%.1f h at 125 act/s, plus ~%d min scoring\n", r, a/1e6, a/125/3600, r}'
if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN=1 - nothing executed."; exit 0; fi

for row in "${PLAN[@]}"; do
  IFS='|' read -r cap seeds games <<< "$row"
  echo; echo "######## $(date +%H:%M) group: [$games] seeds [$seeds] cap $cap"
  GAMES="$games" SEEDS="$seeds" CAP="$cap" bash sweep.sh || echo "!! group failed, continuing"
done
echo; echo "######## $(date +%H:%M) harvest done"
