# Option 1 — give Goose a map and a way back

Status: dev test passed (2026-09-24); 25-game Confirm stopped after two seeds
(2026-09-25) and did NOT meet the rule: 58 vs 50 levels, but ar25 and tr87 worse
on both seeds. Helps stuck games, hurts solved ones. See `option-1-confirm-results.md`.
Builds on Plan B's adopted agent (novelty label, A1).
Plain-language summary for the advisor: `docs/reports/Goose_Semester2_Progress.docx`.

## What is still wrong after Plan B

- Goose wants new screens but only looks one move ahead. Unexplored parts of a
  level that are several moves away are reached only by chance.
- Goose restarts levels constantly. In the Confirm-tier corpora (novelty label,
  seed 0) the number of game-over restarts per 100k actions is 497–5,438; on
  most games an attempt lasts 20–200 moves. Every restart throws away the
  position Goose had reached.
- 7 of 25 games still produce no level for any arm.

## The feature

`custom_agents/return_map.py`, class `ReturnMap`, switched on by
`EVAL_RETURN_MAP=1`. It keeps a map of the current level: which move led from
which screen to which. Two uses:

1. **Stall.** After `EVAL_MAP_STALL` (200) decisions without a never-seen
   screen, walk the shortest known route to the nearest screen with something
   untried (a button never pressed there, or a clickable screen clicked fewer
   than `EVAL_MAP_CLICK_TRIES` (20) times), then press an untried button or
   let Goose pick a click.
2. **Game over.** Walk the shortest known route to the most recently
   discovered screen that still has something untried, using at most half a
   typical attempt (median moves between game overs, last 20 attempts), capped
   at `EVAL_MAP_MAX_ROUTE` (100). `EVAL_MAP_RETURN=0` turns this use off.

Safety rules: every route step is checked against the map and the route is
dropped the moment the game disagrees or a move is unavailable; moves followed
by a game over are never routed through; the map is cleared at each new level.

Screen identity uses a *sticky* copy of Plan B's decoration mask (every cell
ever flagged stays masked). Measured on the Confirm corpora: the live mask
flips 44–87 times per 100k actions on tu93, vc33 and lf52, which would scramble
every key; the sticky union settles within ~3k actions (≤ 9k) and only ever
covers row 0 or row 63, the progress-bar rows.

## Why this option (sources)

| Source | What it shows |
|---|---|
| [ARC Prize, ARC-AGI-3 Preview: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) | 2nd place (Blind Squirrel) built a state graph from frames and pruned loops |
| [Graph-Based Exploration for ARC-AGI-3 (arXiv 2512.24156)](https://arxiv.org/abs/2512.24156) | 3rd place: a directed graph of states, heading for the shortest path to untested state-action pairs; median 30/52 levels, no learning |
| [Ecoffet et al., First return, then explore (Nature 2021)](https://www.nature.com/articles/s41586-020-03157-9) | Exploration fails mainly by forgetting how to return to promising states; remembering and returning solved Montezuma's Revenge and Pitfall |
| [BDR-Pro ARC-AGI-3 agent (2026 Kaggle)](https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3) | Masked state hashing + transition graph + BFS planning + replaying the best prefix after resets |
| Plan B dev tier (ours) | The local "don't repeat" mask failed; it only looked at the current screen |

## Design changes forced by the smoke test (before any sweep)

Smoke: novelty label with the map, 5,000 actions, seed 0, six games.

1. **Unlimited walk back ate the attempt.** On ft09 (attempts last ~36 moves)
   the first version walked back 31 moves on average, the map chose 85% of all
   actions, and exploration collapsed (32 distinct screens in 5k actions). Fix:
   the walk back is limited to half a typical attempt and targets the most
   recent discovery *within* that limit. After the fix, ft09 walks ~18 moves
   against a limit of 16 and the map chooses 35% of actions.
2. **Long walks drift on some games.** On dc22 and g50t every walk back stops
   matching the map ~10 steps in (state the screen does not show). The step
   check hands control back to Goose; the map chooses 7–9% of actions there.

Sanity check after the fix, one seed, 5k actions (not evidence):

| game | levels, novelty only | levels, novelty + map | map share of actions |
|---|---:|---:|---:|
| ls20 | 0 | 0 | 49% |
| tu93 | 0 | 3 | 29% |
| dc22 | 0 | 0 | 7% |
| g50t | 0 | 0 | 9% |
| ft09 | 1 | 2 | 35% |
| vc33 | 1 | 2 | 4% |

Throughput 127–142 act/s, same as without the map.

## Pre-registered dev test (written before it runs)

`make map-dev`: arms A1 (novelty label) and A4 (novelty label + map), games
ls20, dc22, g50t, tu93, ft09, lp85 (the Plan B dev set) plus re86 and wa30
(keyboard games nothing has ever solved), seeds 0–2, 100k actions. 48 runs.

**A4 goes to the 25-game Confirm tier if both hold:**

1. Levels summed over the 24 (game, seed) pairs: A4 ≥ A1, **and** paired
   (game, seed) wins exceed losses.
2. No game where A4 completes fewer levels than A1 on all 3 seeds.

Read beside the levels, never alone: unique states per action, late novelty,
the map's share of actions, and route completion vs abandonment
(`return_map_stats.json` in each run directory).

If it fails, the feature stays off (the default) and is kept, with this
document and its results, as a documented experiment.

## How to remove it

- **Switch off:** leave `EVAL_RETURN_MAP` unset. Nothing in the module runs.
  Verified: with it unset, the agent takes the same actions as before the
  feature existed (295/295 on ls20, both labels).
- **Delete:** every hook line in `custom_agents/action.py` ends in
  `# [return-map]`, 25 lines. Deleting them restores the previous agent byte
  for byte (checked with a diff):

  ```bash
  sed -i '/\[return-map\]/d' custom_agents/action.py
  ```

  Then delete `custom_agents/return_map.py` and `tests/test_return_map.py`, or
  move them to `legacy/` with this document. Remove the A4 case from
  `sweep.sh` and the `map-dev` and `map-confirm` targets from the Makefile.

## Tests

`tests/test_return_map.py` (14 tests): edges and shortest routes; stall routing
to the nearest untried button; arrival behaviour; abandoning a route when the
game disagrees or a move is unavailable; game-over marking and walking back;
the half-attempt limit; switching the walk back off; never routing through a
game-over move; click screens; level change; the sticky mask; the action
format the sampler expects.
