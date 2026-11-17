# Upgrade screen 3: results (17 Nov 2026)

**In one line.** `SOT`, all three upgrades together, ranked first: 22 levels by
5,000 actions on the 25 games, against 15 for the adopted Goose and 15 for the
adopted Goose run again. By the rule written before the runs it clears the
noise. It is a candidate for a confirm, not a result.

**Exploratory: one seed, 5,000 actions a game. Nothing is adopted: no new
preset, no changed default.** The plan, the three upgrades, the checks and the
rule: [`upgrade-screen3.md`](upgrade-screen3.md). All 225 runs finished (16 Nov
23:50 to 17 Nov 04:46), none failed, all on commit `eefcde8`.

## The answer

| Arm | Levels by 5,000 | vs `base` | 95% interval | Noise |
|---|---:|---:|---|---|
| `base` (the adopted Goose) | 15 | | | |
| `base2` (the same again) | 15 | +0 | [-5, +5] | |
| **`SOT`** | **22** | **+7** | [+1, +13] | **clears** |
| `ST` | 21 | +6 | [+2, +10] | clears |
| `T` | 20 | +5 | [-1, +13] | inside |
| `SO` | 19 | +4 | [-2, +10] | inside |
| `O` | 19 | +4 | [+1, +8] | clears |
| `OT` | 18 | +3 | [-1, +7] | inside |
| `S` | 18 | +3 | [-2, +8] | inside |

1. **`SOT` ranked first and clears the noise**: 22 levels against 15 and 15,
   +7 with a 95% interval of [+1, +13]. It was better than `base` on 7 games
   and worse on 1.
2. **Two more arms clear the noise**: `ST` (21, +6, [+2, +10], no game worse)
   and `O` alone (19, +4, [+1, +8], no game worse). The other four are inside
   it. All seven upgraded arms scored above both runs of the baseline.
3. **The main effects are all positive, and none is settled**: T +2.50
   [+0.00, +5.50], S +2.00 [-1.25, +5.50], O +1.00 [-4.00, +5.00]. They point
   the same way as `SOT`, so they agree with it as the best combination, but no
   interval is above 0: the screen does not say which upgrade carries the gain.
4. **The setup looks right**: `base` completed 15 levels, the number the
   sanity anchor expected (near 15; 8 to 24 accepted), and every run's
   `run_config.json` has its arm's flags.

### How far to trust it

- **One seed.** Every number is one run per game. The two baseline runs, which
  are the same agent with the same seed, differ on 6 of 25 games by one level
  each (3 up, 3 down). That is a lower bound on chance: a new seed adds more.
- **Seven arms were tried against one `base` run.** One of seven clearing the
  noise by luck would not be a surprise. Against `base2` instead, `SOT`
  (+7, [+1, +13]) and `ST` (+6, [+2, +12]) still have intervals above 0; `O`'s
  touches 0 ([+0, +9]).
- **`T`'s own gain leans on one game**: 3 of its +5 are lp85 (5 levels against
  2), which is why its interval is wide. `SOT`'s +7 is spread over 7 games.
- **Where the levels are.** By tier, `base` to `SOT`: dev and seen 10 to 14,
  A/B-offline 3 to 6, untouched 2 to 2. On the 15 held-out games the two
  baselines have 5 and 5; the four arms with O have 7 to 8, `ST` has 7, and S
  alone and T alone have 5.
- **When the levels came.** O acts early: `SO` and `SOT` lead `base` by 3
  levels at 1,000 actions and by 5 at 2,000. T acts late: its main effect is
  about 0 at 2,000 actions and +2.5 at 5,000; `ST` was 1 level behind `base`
  at 2,000 and 6 ahead at 5,000.
- **T costs time.** It took a training step on 96% of actions and ran at 10 to
  11 actions a second where the others ran at 25 to 32 (4 runs sharing the
  GPU): about 3 times the time per action. S and O cost nothing that shows.

### What I would confirm next (not run)

**`SOT` against the adopted Goose on new seeds, with `SO` beside it.**

- 25 games x seeds 1, 2, 3 x 5,000 actions, three arms (`base`, `SO`, `SOT`):
  225 runs, about 5 hours, the same driver.
- Primary: `SOT` against `base` on levels by 5,000 actions, with the confirm
  rule of `tools/paired_compare.py` (strictly more levels, more pairs won than
  lost, no game worse on every seed), registered before the runs.
- Second question, same rule: `SO` against `base`, and `SOT` against `SO`. S
  and O cost no time and T costs three times the time per action, so this says
  whether the extra training is worth it. In this screen `SO` led as early as
  `SOT` but was 3 levels behind it at 5,000 actions. If the question is
  instead whether O belongs in the bundle, the third arm is `ST`.
- Two limits to state in its registration. It would be a repeat on new seeds,
  not a held-out test: every game has now been played with these upgrades. And
  5,000 actions is not the budget the adopted Goose was confirmed at; whether
  the gain lasts to 100,000 actions is a separate, much larger job (an `SOT`
  run of 100,000 actions takes about 3 hours with four sharing the GPU).

## Per arm

Ranked arms first to last between `base` and `base2`. "vs `base`" and its interval are on the score; the interval resamples games 10,000 times (seed 0), as `tools/paired_compare.py` does. Unique states: summed over the games. Median act/s: over the arm's runs, 4 runs sharing the GPU.

| Arm | Rank | Levels by 1,000 | by 2,000 | by 5,000 (score) | vs `base` | 95% interval | Games better / same / worse | Noise | Unique states | Median act/s | Runs finished |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|
| `base` | - | 6 | 10 | **15** | - | - | - | - | 60,677 | 31 | 25 of 25 |
| `SOT` | 1 | 9 | 15 | **22** | +7 | [+1, +13] | 7 / 17 / 1 | clears | 77,157 | 10 | 25 of 25 |
| `ST` | 2 | 5 | 9 | **21** | +6 | [+2, +10] | 6 / 19 / 0 | clears | 66,366 | 10 | 25 of 25 |
| `T` | 3 | 8 | 12 | **20** | +5 | [-1, +13] | 4 / 20 / 1 | inside | 62,654 | 10 | 25 of 25 |
| `SO` | 4 | 9 | 15 | **19** | +4 | [-2, +10] | 7 / 15 / 3 | inside | 76,953 | 25 | 25 of 25 |
| `O` | 5 | 8 | 12 | **19** | +4 | [+1, +8] | 4 / 21 / 0 | clears | 71,860 | 31 | 25 of 25 |
| `OT` | 6 | 8 | 12 | **18** | +3 | [-1, +7] | 4 / 20 / 1 | inside | 80,632 | 11 | 25 of 25 |
| `S` | 7 | 6 | 12 | **18** | +3 | [-2, +8] | 3 / 21 / 1 | inside | 61,378 | 32 | 25 of 25 |
| `base2` | - | 6 | 9 | **15** | +0 | [-5, +5] | 3 / 19 / 3 | - | 61,857 | 28 | 25 of 25 |

## By tier

Levels by 5,000 actions.

| Arm | dev and seen (10 games) | A/B-offline (8 games) | untouched (7 games) | all |
|---|---:|---:|---:|---:|
| `base` | 10 | 3 | 2 | 15 |
| `S` | 13 | 3 | 2 | 18 |
| `O` | 11 | 5 | 3 | 19 |
| `T` | 15 | 3 | 2 | 20 |
| `SO` | 11 | 6 | 2 | 19 |
| `ST` | 14 | 4 | 3 | 21 |
| `OT` | 11 | 5 | 2 | 18 |
| `SOT` | 14 | 6 | 2 | 22 |
| `base2` | 10 | 3 | 2 | 15 |

## Game by arm

Levels by 5,000 actions.

| Game | Tier | `base` | `S` | `O` | `T` | `SO` | `ST` | `OT` | `SOT` | `base2` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tu93 | dev and seen | 2 | 4 | 3 | 2 | 3 | 3 | 2 | 4 | 3 |
| ls20 | dev and seen | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| tr87 | dev and seen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lp85 | dev and seen | 2 | 2 | 2 | 5 | 1 | 3 | 1 | 1 | 1 |
| dc22 | dev and seen | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| g50t | dev and seen | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 0 |
| vc33 | dev and seen | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| ft09 | dev and seen | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 2 |
| m0r0 | dev and seen | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| cd82 | dev and seen | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| ar25 | A/B-offline | 1 | 2 | 1 | 1 | 2 | 1 | 1 | 2 | 0 |
| lf52 | A/B-offline | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| re86 | A/B-offline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s5i5 | A/B-offline | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| sc25 | A/B-offline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sp80 | A/B-offline | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| su15 | A/B-offline | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| tn36 | A/B-offline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp35 | untouched | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| cn04 | untouched | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| ka59 | untouched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r11l | untouched | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| sb26 | untouched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sk48 | untouched | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| wa30 | untouched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Main effects

| Upgrade | arms with it | arms without | mean score with | mean score without | main effect | 95% interval | effect by 1,000 actions | effect by 2,000 actions |
|---|---|---|---:|---:|---:|---|---|---|
| S `map_sched` | S SO ST SOT | base O T OT | 20.00 | 18.00 | **+2.00** | [-1.25, +5.50] | -0.25 [-2.00, +1.50] | +1.25 [-1.75, +4.50] |
| O `objclick` | O SO OT SOT | base S T ST | 19.50 | 18.50 | **+1.00** | [-4.00, +5.00] | +2.25 [-0.50, +5.75] | +2.75 [-1.25, +7.00] |
| T train more | T ST OT SOT | base S O SO | 20.25 | 17.75 | **+2.50** | [+0.00, +5.50] | +0.25 [-1.25, +1.75] | -0.25 [-1.75, +1.25] |

The same, pair by pair: the levels an upgrade adds to each of the four arms without it (the main effect is the mean of the four).

| Upgrade | added to |  |  |  |
|---|---|---|---|---|
| S `map_sched` | `base` to `S`: +3 | `O` to `SO`: +0 | `T` to `ST`: +1 | `OT` to `SOT`: +4 |
| O `objclick` | `base` to `O`: +4 | `S` to `SO`: +1 | `T` to `OT`: -2 | `ST` to `SOT`: +1 |
| T train more | `base` to `T`: +5 | `S` to `ST`: +3 | `O` to `OT`: -1 | `SO` to `SOT`: +3 |

## Time to level, each arm against `base`

From `tools/levels_by_budget.py`'s functions (budgets 1,000, 2,000 and 5,000). A level race is one level both runs started and at least one finished: the arm that took fewer actions on it wins. `base2`'s row is what chance alone does.

| Arm | levels by 1,000, arm - base | levels by 2,000, arm - base | levels by 5,000, arm - base | AULC, 100 to 5,000 actions, arm - base [95% interval] | level 1: arm faster / base faster / same or undecided | levels 2 and up: arm faster / base faster / same or undecided | levels 2 and up, arm - base [95% interval] |
|---|---:|---:|---:|---|---|---|---|
| S | +0 | +2 | +3 | +0.86 [-0.78, +2.79] | 7 / 4 / 2 | 5 / 0 / 0 | +5 [+4, +7] |
| O | +2 | +2 | +4 | +1.86 [-0.11, +4.10] | 8 / 4 / 3 | 4 / 0 / 0 | +4 [+3, +6] |
| T | +2 | +2 | +5 | +1.37 [-0.70, +3.87] | 8 / 6 / 1 | 3 / 1 / 0 | +2 [-3, +6] |
| SO | +3 | +5 | +4 | +2.98 [+0.25, +6.03] | 11 / 4 / 1 | 4 / 2 / 0 | +2 [-3, +7] |
| ST | -1 | -1 | +6 | +1.11 [-0.57, +3.01] | 9 / 5 / 1 | 5 / 0 / 1 | +5 [+2, +8] |
| OT | +2 | +2 | +3 | +2.56 [-0.02, +5.40] | 11 / 4 / 1 | 2 / 1 / 0 | +1 [-3, +3] |
| SOT | +3 | +5 | +7 | +3.64 [+1.63, +6.16] | 14 / 1 / 1 | 3 / 1 / 2 | +2 [-2, +6] |
| base2 | +0 | -1 | +0 | +0.10 [-1.23, +1.38] | 7 / 3 / 3 | 3 / 2 / 0 | +1 [-2, +4] |

Read with `base2`'s row in mind: the same agent run again was faster than
`base` on 7 first levels and slower on 3. Only `SOT`'s 14 against 1 is far from
that: on the 12 first levels both finished, `SOT` took a median of 0.38 of
`base`'s actions, where `base2` took 0.95 (`tools/levels_by_budget.py`). On
levels 2 and up the counts are small, 3 to 6 races an arm: `S`, `O` and `ST`
won all they decided (5, 4 and 5), `base2` won 3 of 5. On AULC only `SOT` and
`SO` have intervals above 0.

## `base` against `base2`: the size of chance

The two are the same agent with the same seed; they differ only because GPU runs stop being identical after a few hundred actions. So their gap is a LOWER bound on chance: a different seed would add more. Score 15 against 15 (+0, 95% interval [-5, +5]); `base2` better / same / worse on 3 / 19 / 3 games; levels by 1,000 actions 6 against 6, by 2,000 10 against 9. Its level races are the `base2` row of the table above.

Games where the two differ: tu93 (2 vs 3), ls20 (1 vs 0), ar25 (1 vs 0), lp85 (2 vs 1), ft09 (1 vs 2), su15 (0 vs 1).

Does it matter which of the two runs an arm is compared with? Not part of the rule, which names `base`; shown because every arm above is compared with the same one run.

| Arm | vs `base` [95% interval] | games better / same / worse | vs `base2` [95% interval] | games better / same / worse | vs the mean of the two [95% interval] |
|---|---|---|---|---|---|
| `SOT` | +7 [+1, +13] | 7 / 17 / 1 | +7 [+1, +13] | 7 / 17 / 1 | +7.0 [+2.0, +12.5] |
| `ST` | +6 [+2, +10] | 6 / 19 / 0 | +6 [+2, +12] | 5 / 20 / 0 | +6.0 [+2.5, +10.5] |
| `T` | +5 [-1, +13] | 4 / 20 / 1 | +5 [-3, +16] | 6 / 15 / 4 | +5.0 [-1.5, +14.0] |
| `SO` | +4 [-2, +10] | 7 / 15 / 3 | +4 [-2, +10] | 5 / 18 / 2 | +4.0 [-1.5, +9.5] |
| `O` | +4 [+1, +8] | 4 / 21 / 0 | +4 [+0, +9] | 5 / 19 / 1 | +4.0 [+1.0, +7.5] |
| `OT` | +3 [-1, +7] | 4 / 20 / 1 | +3 [-2, +8] | 5 / 18 / 2 | +3.0 [-1.0, +7.0] |
| `S` | +3 [-2, +8] | 3 / 21 / 1 | +3 [-3, +9] | 5 / 17 / 3 | +3.0 [-2.0, +8.5] |

## What each upgrade did

**S (`map_sched`)**: the five schedules in the order stay200, walk200, stay30, walk30, sweep30.

| Arm | pulls | share of decisions | new screens per decision | untried-first presses | moves chosen by the map |
|---|---|---|---|---:|---:|
| S | 591 / 431 / 646 / 355 / 342 | 27% / 18% / 25% / 15% / 14% | 0.266 / 0.194 / 0.326 / 0.259 / 0.212 | 6598 | 15.1% |
| SO | 679 / 367 / 486 / 380 / 326 | 29% / 17% / 22% / 17% / 15% | 0.442 / 0.256 / 0.384 / 0.271 / 0.275 | 6886 | 16.7% |
| ST | 663 / 426 / 499 / 394 / 332 | 28% / 20% / 22% / 17% / 14% | 0.310 / 0.244 / 0.343 / 0.275 / 0.256 | 6499 | 15.9% |
| SOT | 674 / 391 / 461 / 332 / 356 | 29% / 19% / 21% / 16% / 15% | 0.414 / 0.268 / 0.417 / 0.302 / 0.275 | 7544 | 17.2% |

For comparison, the arms with `map_gated`:

| Arm | walk-back choices | stay choices | moves chosen by the map |
|---|---:|---:|---:|
| base | 792 | 1198 | 7.9% |
| O | 810 | 1110 | 8.0% |
| T | 849 | 1207 | 7.1% |
| OT | 791 | 1130 | 8.6% |
| base2 | 870 | 1114 | 7.5% |

**O (`objclick`)**

| Arm | clicks redrawn by shape | clicks kept (Goose's cell) | moves that were Goose clicks | moves redrawn | mean shapes on a redrawn screen | games with no click |
|---|---:|---:|---:|---:|---:|---:|
| O | 37,953 | 4,105 | 34% | 31% | 53.8 | 6 |
| SO | 39,578 | 4,292 | 36% | 32% | 54.2 | 6 |
| OT | 36,020 | 3,916 | 32% | 29% | 54.8 | 6 |
| SOT | 35,664 | 3,852 | 32% | 29% | 56.1 | 6 |

**T (`EVAL_TRAIN_EVERY`, `EVAL_TRAIN_RECENT`)**: `wall_ms_med` is the median over a run's moves, so for an arm that trains on one move in five it is a move without a training step.

| Arm | training steps | steps per action | median over runs of `wall_ms_med` | median act/s |
|---|---:|---:|---:|---:|
| base | not counted (one per 5 actions) | 0.20 at most | 22.7 | 31 |
| S | not counted (one per 5 actions) | 0.20 at most | 23.3 | 32 |
| O | not counted (one per 5 actions) | 0.20 at most | 23.4 | 31 |
| T | 119,777 | 0.96 | 111.9 | 10 |
| SO | not counted (one per 5 actions) | 0.20 at most | 25.6 | 25 |
| ST | 119,740 | 0.96 | 114.2 | 10 |
| OT | 120,031 | 0.96 | 109.1 | 11 |
| SOT | 119,675 | 0.96 | 112.6 | 10 |
| base2 | not counted (one per 5 actions) | 0.20 at most | 23.8 | 28 |

- **S.** The map chose 15% to 17% of moves, against 7% to 9% with `map_gated`,
  and pressed an untried button first on 5% to 6% of moves. The bandit did not
  settle on one schedule in 5,000 actions: no schedule got more than 29% of
  the decisions. The two "stay" schedules found more new screens per decision
  than the two "walk" ones.
- **O.** It redrew about 3 moves in 10 (9 of every 10 clicks Goose drew). Six
  games have no click at all, so it cannot act there: g50t, ls20, re86, tr87,
  tu93, wa30. A redrawn screen had 54 to 56 shapes on average.
- **T.** 0.96 training steps per action against 0.20 at most. Speeds depend on
  what ran beside a run: a T arm took about 500 seconds for 5,000 actions, the
  others 150 to 190.

## Runs

- In the manifest: 225 of 225. Missing: none.
- Failures logged: 0.
- Flags in each run's `run_config.json`: as the arm table says, on every run.
- Runs that ended before the cap: none.
- Commit recorded by the runs: `eefcde8`.
- Recordings: `results/screen3/20261116_235041/runs/<arm>/<game>_s0/runs/<time>/<game>/` (`transitions/`, `metrics.json`, `run_config.json`, the stats files). One row per run: `results/screen3/20261116_235041/runs.csv`.
- Backup of the numbers, in git: [`upgrade-screen3-runs.csv`](upgrade-screen3-runs.csv),
  one row per run (arm, game, tier, levels, the action of each level-up, unique
  states, actions, act/s, the upgrade stats, run folder, commit). The
  recordings themselves (168 MB) are under `results/`, which is not in git.
- The tables again: `uv run python experiments/goose/screen3.py report --sweep
  results/screen3/20261116_235041` (writes `report.md` and `runs.csv` there;
  `--docs` also rewrites the CSV in this folder).

## Problems, and decisions made without Matt

**Problems.**

- **No run failed** and none was retried; the sweep was not paused.
- **The sweep took 4 h 55 min, not the 4.1 h projected.** In the sweep a T arm
  mostly ran beside other T arms and took about 112 ms a move (median), where
  the smoke run measured 83 to 96 ms. It ended at 04:46, far inside the 08:34
  line the projection was checked against, and the T arms ran as planned, at
  `EVAL_TRAIN_EVERY=1`.
- **One smoke check was restated before the start**, on the same smoke runs
  (plan page, check 4b): it compared single runs' speeds, and the last run of
  the queue had the GPU to itself.
- **Nothing was changed after the start.** No arm, no upgrade, no rule; no run
  was rerun. `git diff eefcde8 -- 'custom_agents/*.py' custom_agents/gb
  run_local.py compute_metrics.py experiments/goose/screen3.py` is empty (the
  only file under `custom_agents/` touched since is its README).

**Decisions.** The 18 choices made before the start are listed in the plan
page ("Decisions made without Matt"). Two more were made after the last run:

19. **Two tables were added to the report after the results were seen**: what
    each upgrade adds pair by pair, and each arm against `base2` and against
    the two baselines' mean. They describe; the rule and its numbers are as
    written before the runs.
20. **"What I would confirm next" is a recommendation only.** Nothing was
    registered and nothing was run.

**Limits.**

- One seed, and the held-out games are no longer unseen for these three
  upgrades (pre-registration section 11, entry 10).
- `base2` shares `base`'s seed, so their gap understates chance.
- Unique states are the published scorer's count, which masks no decoration on
  six games (CLAUDE.md, "Known measurement caveats"); read them next to the
  levels, never alone.
- GPU runs are not reproducible past a few hundred actions: a rerun of this
  sweep will not give these exact numbers.
