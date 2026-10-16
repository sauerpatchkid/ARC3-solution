# Upgrade screen, round 1 — results (2026-09-26)

192 runs (12 arms × 8 games × 2 seeds × 50k actions), 2026-09-25 12:13 to
2026-09-26 ~06:40, four at a time, no failed runs. Output and leaderboard:
`results/screen/20260925_121259/`. Saved untouched at git tag
`upgrade-screen-round1`. Design and pre-registered rule: `upgrade-screen.md`.

## Leaderboard

| rank | arm | levels | vs novelty: gain, better/same/worse | vs map: gain, W/T/L | promising |
|---:|---|---:|---|---|---|
| 1 | **map_bars** | 30 | +16, 7/9/0 | +4, 4/10/2 | yes |
| 2 | **map_gated** | 30 | +16, 6/10/0 | +4, 3/13/0 | yes |
| 3 | map_diverse | 28 | +14, 7/8/1 | +2, 5/8/3 | yes |
| 4 | map_objects | 28 | +14, 7/7/2 | +2, 3/11/2 | yes |
| 5 | map (reference) | 26 | +12, 5/9/2 | — | yes |
| 6 | attempt | 20 | +6, 6/10/0 | — | yes |
| 7 | deadclick | 18 | +4, 3/13/0 | — | yes |
| 8 | persist | 18 | +4, 3/13/0 | — | yes |
| 9 | bars | 16 | +2, 3/12/1 | — | yes |
| 10 | map_stuck | 15 | +1, 3/9/4 | −11, 1/9/6 | no |
| 11 | graded | 11 | −3, 2/9/5 | — | no |
| — | novelty (reference) | 14 | — | — | — |

Ranking rule as pre-registered: levels gained over novelty, then wins minus
losses, then AULC gain. map_bars and map_gated tie on levels; map_bars ranks
first on wins minus losses (7 vs 6) and AULC (+4.75 vs +3.98). With only 2
seeds and 50k actions the "promising" rule is lenient (9 of 11 pass); the
order is the useful part.

## Levels per game and seed (seed 0, seed 1)

| arm | ar25 | dc22 | ft09 | g50t | su15 | tr87 | tu93 | vc33 |
|---|---|---|---|---|---|---|---|---|
| novelty | 2,2 | 0,0 | 2,2 | 1,1 | 0,0 | 0,0 | 0,0 | 2,2 |
| map | 1,2 | 0,0 | 2,2 | 1,0 | 1,1 | 0,0 | 5,5 | 4,2 |
| map_bars | 2,2 | **3,1** | 2,2 | 1,1 | 1,1 | 0,0 | 4,4 | 4,2 |
| map_gated | 2,2 | 0,0 | 2,2 | 1,1 | 1,1 | 0,0 | **5,5** | **4,4** |
| attempt | 2,2 | 1,1 | 2,2 | 1,1 | 1,1 | 1,0 | 0,0 | 3,2 |

## Why the two winners won (mechanism, from the map's own counters)

| game | map: walks finished / abandoned | map_bars | map_gated: walks finished / abandoned; walk vs stay decisions |
|---|---|---|---|
| dc22 | 145 / 881 | **392 / 67** | 174 / 513; 432 vs 349 |
| g50t | 20 / 766 | 184 / 740 | 0 / 281; 281 vs 479 |
| ar25 | 1,493 / 0 (map chose 48% of moves) | 1,069 / 11 (47%) | 26 / 0; **26 vs 1,110** (map chose 1%) |
| ft09 | 2,634 / 0 (45%) | 1,671 / 2 (47%) | 15 / 0; **15 vs 1,432** (0%) |
| tu93 | 1,298 / 65 | 746 / 277 | 1,255 / 41; **5,459 vs 271** |
| vc33 | 508 / 103 | 716 / 24 | 1,141 / 88; **1,256 vs 167** |

- **Bars fixed the map's drift on dc22**, as the analysis before the screen
  predicted: abandoned walks fell from 881 to 67, and dc22 produced levels on
  both seeds, where every other arm but attempt got none. On g50t bars helped
  walks finish (20 → 184) but most still drift: g50t hides something else.
- **The gate learned per game what Option 1 got wrong:** it nearly stopped
  walking back on ar25 and ft09 (where walking back cost ~half of all moves)
  and kept walking back on tu93 and vc33 (where it paid). No game got worse
  than the plain map.
- The two are complementary: map_bars wins dc22, map_gated wins tu93 and vc33.
- Walking back matters: turning it off (map_stuck) lost 11 levels against the
  map. The graded reward hurt.

## Decision

Round 1's best is **map_bars**, with **map_gated** tied on levels. Both are
saved unchanged at tag `upgrade-screen-round1`. Because they fix different
failures, round 2 (separate: `experiments/upgrade_screen2/`,
`docs/plans/upgrade-screen2.md`) tests them combined, with the two cheap ideas
that were positive with no losses (attempt, deadclick) layered on top, and
adds three games round 1 never used.
