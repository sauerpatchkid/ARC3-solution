# Upgrade screen, round 2 — results (2026-09-26)

176 runs (8 arms × 11 games × 2 seeds × 50k actions), 2026-09-26 06:49 to
~23:40, four at a time, no failed runs. Output and leaderboard:
`results/screen2/20260926_064927/`. Plan and pre-registered rule:
`upgrade-screen2.md`. Round 1's best is untouched at tag
`upgrade-screen-round1`.

## Leaderboard (reference: round 1's best, map_bars, run fresh)

| rank | arm | levels | vs map_bars: gain, W/T/L | vs novelty: gain, W/T/L | held-out games vs map_bars | improves on round 1 |
|---:|---|---:|---|---|---|---|
| 1 | **mb_gated_att** (bars + gate + attempt) | **41** | +3, 5/14/3 | +18, 11/10/1 | +0, 1/4/1 | **yes** |
| 2 | mb_all (+ dead clicks) | 40 | +2, 4/16/2 | +17, 11/10/1 | +1, 1/5/0 | yes |
| 3 | mb_att (bars + attempt) | 38 | +0, 3/16/3 | +15, 9/11/2 | −1, 0/5/1 | no |
| 4 | mb_gated (bars + gate) | 38 | +0, 2/17/3 | +15, 11/8/3 | +0, 0/6/0 | no |
| 5 | map_gated (round 1 runner-up) | 37 | −1, 5/14/3 | +14, 8/12/2 | −1, 0/5/1 | no |
| 6 | mb_gated_dead | 36 | −2, 4/13/5 | +13, 13/5/4 | +2, 1/5/0 | no |
| — | map_bars (round 1 best) | 38 | — | +15 | — | — |
| — | novelty (adopted agent) | 23 | −15 | — | — | — |

## Levels per game and seed (seed 0, seed 1)

| arm | ar25 | bp35 | cn04 | dc22 | ft09 | g50t | sp80 | su15 | tr87 | tu93 | vc33 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| novelty | 2,2 | 0,0 | 1,1 | 0,0 | 3,2 | 1,1 | 2,1 | 0,0 | 1,0 | 0,2 | 2,2 |
| map_bars | 1,2 | 1,1 | 1,1 | 3,3 | 2,2 | 1,1 | 1,2 | 1,1 | 0,0 | 4,4 | 4,2 |
| map_gated | 2,2 | 1,1 | 1,1 | 0,0 | 2,2 | 1,1 | 1,1 | 1,1 | 1,0 | 5,5 | 4,4 |
| **mb_gated_att** | 2,2 | 1,1 | 1,1 | 2,3 | 2,2 | 1,1 | 2,1 | 1,1 | 1,1 | 4,4 | 3,4 |
| mb_all | 2,2 | 1,1 | 1,1 | 2,3 | 2,2 | 1,1 | 2,2 | 1,1 | 1,0 | 4,4 | 3,3 |

bp35, cn04 and sp80 were never used in round 1 (held out).

## What it means

- **By the pre-registered rule, mb_gated_att improves on round 1** (+3 levels
  over map_bars, 5 wins to 3 losses, no game worse on both seeds), and so
  does mb_all. **The margin over map_bars is small** (3 levels over 22
  pairs), and on the three held-out games it only ties map_bars. Treat it as
  "at least as good, and safer", not as a clear jump.
- **The real gain is robustness against the adopted agent.** mb_gated_att
  is worse than the novelty agent on only 1 of 22 game-seeds (ft09, 2 vs 3
  levels), against 4 for map_bars (ar25, ft09, sp80, tr87 — exactly the
  "already solved" games that sank the map in Option 1's 25-game test). It
  keeps the map's big gains (tu93 4,4; dc22 2,3; vc33 3,4; su15 and bp35
  first levels) while the gate stops the walk back from costing the solved
  games. On the held-out games it has 7 levels to the novelty agent's 5.
- **Across both screens the map family, fixed, beats the adopted agent by a
  wide, consistent margin:** round 1, 30 vs 14 levels on 8 games; round 2,
  41 vs 23 on 11 games.
- The combination is not simply additive: adding bars to the gate gave back
  some of the gate's tu93/vc33 edge (tu93 4,4 vs 5,5), and dead-click
  skipping on its own on top of the gate hurt g50t and dc22. The per-attempt
  credit is what made the combination work.

## Recommended next step (not started)

A proper test of **mb_gated_att** against the novelty agent on all 25 games ×
3 seeds × 100k actions, with the Confirm rule used before (more levels, more
paired wins than losses, no game worse on every seed). The novelty agent's
runs from the Plan B Confirm sweep can be reused, so only 75 runs are needed:
about 14 hours. Switches: `EVAL_LABEL=novel EVAL_RETURN_MAP=1
EVAL_UPGRADES=bars,map_gated,attempt`. If it passes, it would replace the
novelty agent as the adopted Goose.
