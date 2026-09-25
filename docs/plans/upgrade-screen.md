# Upgrade screen — candidate improvements to the novelty label and the map

Status: built and smoke-tested 2026-09-25; **not run**. Start with
`make upgrade-screen` (~17 h, pausable). Plain-language version with full
reasoning, pros and cons: `docs/reports/Goose_Upgrade_Options.docx`.

## Why a screen

Nine ideas, each with a real case for it, are too many to test properly (a
proper test is 3 seeds × 100k actions per arm). A screen runs every idea
briefly and in one sweep, ranks them against the adopted agent, and sends only
the best one to three candidates to a proper dev test. It can miss late gains
and can be fooled by luck on one seed; it is a shortlist, not a verdict.

## What our own data says (novelty-label corpora, seed 0)

| finding | measurement | points to |
|---|---|---|
| Progress bars that move with real changes are missed | live decoration mask empty on 12 of 14 games; the component detector finds a bar (row 0, 61–63 or column 0) on 12 of 25 games | `bars`, `map_bars` |
| That also explains the map's drift on dc22 and g50t | bar on row 63 unmasked there; a shorter walk back shows a different bar, so the screen looks new and the route is abandoned | `map_bars` |
| Clicks already known to be dead are repeated | share of all clicks that repeat a no-op on the same cell and colour 4+ times: su15 54%, vc33 51%, tn36 47%, lf52 23%, s5i5 21%, r11l 17% | `deadclick` |
| The map helps stuck games and hurts solved ones | Confirm, 2 seeds: tu93, vc33, bp35, lf52 up; ar25, tr87 down; the map chooses ~50% of actions on solved click games | `map_gated`, `map_stuck`, `map_objects` |
| Levels restart constantly | a game over every 20–200 moves on most games | `attempt`, `map_gated` |
| Most moves revisit well-known screens on stuck games | next screen already seen 5+ times: tu93 91%, dc22 84%, vc33 80%; seen 1–4 times: 5–20% on mid games | `graded` (small expected effect) |

An unguarded component detector also flagged mid-screen playfield rows on
sk48, s5i5 and tr87; `bars` therefore only accepts cells within 4 cells of the
screen edge, which kept every real bar found.

## The arms

Every arm uses `EVAL_LABEL=novel`. Reference arms are run fresh in the same
sweep at the same budget.

| arm | switches | reference | idea | main source |
|---|---|---|---|---|
| novelty | — | — | the adopted agent | Plan B |
| map | `EVAL_RETURN_MAP=1` | — | the return map as tested | Option 1 |
| bars | `EVAL_UPGRADES=bars` | novelty | component-level bar detection, border guard, sticky | 3rd place masks status bars; our 1.4× ft09 inflation |
| attempt | `bars,attempt` | novelty | 0.5 reward for screens new this attempt | NGU; Henaff et al. 2023 |
| graded | `graded` | novelty | reward 1/sqrt(visits) | Strehl & Littman 2008; Tang et al. 2017 |
| deadclick | `deadclick` | novelty | never re-click a cell+colour that did nothing 4 times this level | BDR-Pro (its contextual click rules, this one included: +68% mean score) |
| persist | `EVAL_RESET_ON_LEVEL=0` | novelty | keep the network across levels (existing ablation flag) | BDR-Pro cross-level transfer |
| map_stuck | `EVAL_MAP_RETURN=0` | map | stall routes only, no walk back | Option 1 Confirm |
| map_gated | `map_gated` | map | sliding-window UCB per game: walk back or not | Agent57 meta-controller |
| map_objects | `map_objects` | map | "untried" = an object never clicked; click it on arrival | 3rd place 5-tier segmentation |
| map_diverse | `map_diverse` | map | walk-back target weighted 1/sqrt(visits+1) | Go-Explore |
| map_bars | `bars` + map | map | bar-masked fingerprints for label and map | as `bars` |

## Budget

8 games (tu93, su15, ar25, tr87, dc22, g50t, vc33, ft09) × 2 seeds × 50k
actions × 12 arms = 192 runs. Four runs at once (measured best: 162 act/s total
against 140 for one run; parallel runs share the GPU and barely add up), about
17 hours.

## Pre-registered ranking (written before the screen runs)

Implemented in `experiments/upgrade_screen/rank.py`. Every arm is paired by
(game, seed) with `novelty`; map arms are also reported against `map`.

1. Rank by levels gained over `novelty`, summed over pairs; then paired wins
   minus losses; then AULC gain (levels weighted early on a log axis, T 100 to
   the cap).
2. **Promising** = more levels than `novelty`, more paired wins than losses,
   and no game worse on both seeds.
3. The best one to three promising arms go to a dev test: 3 seeds × 100k on
   the Plan B dev games plus re86 and wa30, the rule used for the map. If
   none is promising, the screen's answer is "none of these beat the novelty
   agent" and that is reported as it is.

## How to run, pause, resume

```bash
make upgrade-screen DRY_RUN=1
make upgrade-screen
make upgrade-screen-status
make upgrade-screen-pause                 # runs in progress finish and are kept
make upgrade-screen RESUME=results/screen/<stamp>/manifest.tsv
make upgrade-screen-rank MANIFEST=results/screen/<stamp>/manifest.tsv
```

Output: `results/screen/<stamp>/` with `manifest.tsv`, one results directory
per run, logs, and `leaderboard.md` / `.csv`.

## How to remove it

It is separate from the baseline, the novelty label and the map:

- `custom_agents/upgrades.py` (the candidates; the map variants subclass
  `ReturnMap`, which is never edited), `tests/test_upgrades.py`,
  `experiments/upgrade_screen/`.
- 26 hook lines in `custom_agents/action.py`, one in the root `Makefile`, two
  in `check_repo.py`, all ending in `# [upgrades]`:

  ```bash
  sed -i '/\[upgrades\]/d' custom_agents/action.py Makefile check_repo.py
  ```

  Verified: this restores each file byte for byte, and removing both
  `[upgrades]` and `[return-map]` lines from `action.py` restores the agent as
  it was before the map. With `EVAL_UPGRADES` unset nothing in the module runs.

## Checks done

- 17 unit tests (`tests/test_upgrades.py`): small-component ticks, the bar
  detector catching a bar the shared detector misses, the border guard, object
  labelling against brute force, object salience, each label, dead-click
  blocking and resampling, the bandit following the better arm and stopping
  walk-backs that do not pay, object frontier and arrival clicks, diverse
  targets, bar mask reaching the map fingerprint.
- End-to-end smoke of all 12 arms (3 games, 2k actions): every arm runs and
  its own counters show it active (bars found 59–63 cells on ft09, tu93, vc33;
  dead clicks blocked; attempt rewards given; object clicks made; the bandit
  already preferred not walking back on ft09 and vc33 and walked back about
  half the time on tu93). The smoke caught one crash (an empty statistic sent
  to TensorBoard), now fixed.
