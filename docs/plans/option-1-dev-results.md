# Option 1 dev test — results (2026-09-24)

Sweep: `make map-dev`, 8 games × 3 seeds × 100k actions × 2 arms = 48 runs,
2026-09-23 21:46 to ~07:15, no crashes. Manifest
`results/sweeps/sweep_20260923_214629.manifest`; summary, curves and
`map_dev_paired.md` beside it. Verdict reproduced by
`uv run python tools/paired_compare.py --base <manifest>:A1 --new <manifest>:A4`.

- **A1** — the adopted agent: novelty label.
- **A4** — A1 plus the return map (`EVAL_RETURN_MAP=1`, default settings).

## Pre-registered rule: PASS

| rule (written before the run) | result |
|---|---|
| 1. Levels summed: A4 ≥ A1, and paired wins > losses | 45 vs 32; 6 wins, 16 ties, 2 losses: **PASS** |
| 2. No game where A4 is worse on all 3 seeds | none: **PASS** |

Throughput 139.7 vs 140.1 act/s.

## Per game

| game | A1 levels | A4 levels | W/T/L | unique states A1 → A4 | map share of actions | routes finished / abandoned |
|---|---|---|---|---|---:|---|
| ls20 | 1,1,1 | 1,1,1 | 0/3/0 | 7,392 → 7,227 | 41% | 2,713 / 769 |
| dc22 | 0,0,0 | **1**,0,0 | 1/2/0 | 3,949 → 6,350 | 18% | 562 / 2,558 |
| g50t | 1,1,1 | 1,1,1 | 0/3/0 | 36,234 → 22,677 | 28% | 333 / 2,076 |
| tu93 | 2,0,2 | **5,5,5** | 3/0/0 | 1,259 → 3,356 | 13% | 2,713 / 219 |
| ft09 | 2,2,2 | 2,2,2 | 0/3/0 | 89,445 → 47,623 | 48% | 3,380 / 0 |
| lp85 | 5,5,5 | 7,3,7 | 2/0/1 | 84,469 → 38,583 | 45% | 2,997 / 44 |
| re86 | 0,0,1 | 0,0,0 | 0/2/1 | 87,437 → 41,815 | 51% | 2,967 / 0 |
| wa30 | 0,0,0 | 0,0,0 | 0/3/0 | 54,348 → 22,080 | 47% | 629 / 862 |

## What it means

- **tu93 carries most of the gain.** Level 5 on every seed; the best tu93
  result in any earlier run was level 2. Level 1 arrives at ~1.9k actions
  instead of ~20–31k. Both uses of the map are active there: ~350 stall routes
  and ~620 short walks back (about 11 moves) per run, almost all completed.
  Without tu93 the totals are 30 vs 28, roughly even.
- **dc22 produced its first level ever** (seed 0), in any arm of any sweep.
- **Mixed elsewhere.** lp85 won two seeds and lost one; re86 lost the one level
  the novelty-only agent found (itself the first re86 level ever recorded).
  First levels came later with the map on ls20 (all 3 seeds) and g50t (2 of 3),
  earlier on ft09 (2 of 3) and much earlier on tu93.
- **Coverage roughly halves where the map is busiest.** On ft09, lp85, re86 and
  wa30 the map chooses 45–51% of actions, mostly walking back after game overs
  (re86: every attempt, exactly the 50-move limit), and unique states drop by
  about half. Level counts did not drop with them except on re86's single seed.
- **Walks back still drift on dc22 and g50t** (most routes abandoned). The step
  check handles it; the cost is small.

## Decision

Per the pre-registered rule, the map goes to the 25-game Confirm tier. Because
the gain is concentrated in one game, Confirm is where it has to show it is not
a tu93-only effect.

Confirm plan (`make map-confirm`): A4 on all 25 games × 3 seeds × 100k = 75
runs, ~15 h, compared with the Plan B Confirm sweep's A1 runs (same seeds; the
novelty-only code path is unchanged, verified action-for-action). Same rule.

Parameters stay as they are for Confirm. Changing them now, after seeing dev
results, would need a new dev round. Two questions for later, as separate
ablations: whether the walk back (`EVAL_MAP_RETURN=0`) or the stall route
drives tu93, and whether a shorter walk-back limit keeps the gains while giving
coverage back on the click games.
