# Upgrade screen, round 2 — combinations of round 1's best

Status: RUN 2026-09-26 (176 runs, no failures). Results: `upgrade-screen2-results.md`.
Winner: mb_gated_att (41 levels vs map_bars 38, novelty 23). Arms were confirmed from
round 1's full results (`upgrade-screen-results.md`, tag `upgrade-screen-round1`).

## Why a round 2

Matt, away until 2026-09-27, asked: if round 1 finishes and the results can be
improved on, run a separate sweep of improved or different ideas, keeping
round 1's best untouched.

Round 1 (8 games, 2 seeds, 50k) found the map to be the strongest family
once either of its two weaknesses is fixed:

| round 1 arm | levels | vs novelty (better/same/worse) | note |
|---|---:|---|---|
| map_bars (map + progress bars masked) | 30 | 7/9/0 | dc22 levels on both seeds: masking the bar cut abandoned walks 881 → 67 |
| map_gated (walk back only while it pays) | 30 | 6/10/0 | stopped walking back on ar25 and ft09, kept it on tu93 and vc33; never worse than the map |
| map_diverse, map_objects | 28 | | |
| map (as in Option 1) | 26 | 5/9/2 | |
| attempt (bars + half credit within an attempt) | 20 | 6/10/0 | positive, no losses |
| deadclick | 18 | 3/13/0 | positive, no losses |
| novelty (adopted agent) | 14 | | |
| map_stuck, graded | 15, 11 | | dropped: turning off the walk back or grading the reward hurt |

The two leaders fix different failures and win on different games (map_bars:
dc22; map_gated: tu93, vc33), so combining them is the obvious next test, with
the two cheap ideas that were positive with no losses layered on top.

## Arms (all existing switches; no new agent code)

| arm | switches (all `EVAL_LABEL=novel`, `EVAL_RETURN_MAP=1` except novelty) |
|---|---|
| novelty | reference: the adopted agent |
| map_bars | reference: round 1's best, unchanged (`EVAL_UPGRADES=bars`) |
| map_gated | round 1's runner-up, unchanged |
| mb_gated | `bars,map_gated` |
| mb_gated_att | `bars,map_gated,attempt` |
| mb_gated_dead | `bars,map_gated,deadclick` |
| mb_all | `bars,map_gated,attempt,deadclick` |
| mb_att | `bars,attempt` |

Games: round 1's 8 (tu93, su15, ar25, tr87, dc22, g50t, vc33, ft09) plus 3 it
never used (sp80, cn04, bp35) as a check against tuning to round 1's games.
2 seeds × 50k actions, 4 at a time: 176 runs, about 17 hours.

## Pre-registered rule (written before round 2 runs)

Implemented in `experiments/upgrade_screen2/rank2.py`.

- Rank by levels gained over **map_bars** (round 1's best, run fresh here),
  then paired wins minus losses, then AULC gain.
- An arm **improves on round 1** if, against map_bars: more levels, more paired
  wins than losses, and no game worse on both seeds.
- The three held-out games are reported separately; a winner that only wins
  on round 1's games is flagged as possibly tuned to them.
- Whatever wins, the next step is a proper test (3 seeds × 100k, all 25 games
  eventually), not adoption straight from a screen.

## Separation

Round 2 lives in `experiments/upgrade_screen2/` and `results/screen2/`, runs
through round 1's runner imported as a library (no round-1 file edited), and
adds one Makefile line tagged `# [upgrades2]`. Round 1's best is preserved at
git tag `upgrade-screen-round1` and in `results/screen/20260925_121259/`.
