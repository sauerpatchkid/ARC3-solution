# Knowledge Planner: the dev sweep (19 Nov 2026)

**In one line.** On the seven dev games at two seeds, Goose with the planner
ended on 36 levels and Goose without it on 20. Thirteen of the 16 extra levels
are the planner's own, on ft09 (6 and 6 against 2 and 1) and lp85 (5 against 1
at one seed; neither arm cleared level 1 at the other). The other three are on
tu93 and vc33, where the planner did nothing, so they are chance. No run
failed and the planner made no error. Nothing in the agent was changed
afterwards.

**Status: exploratory.** These are the games the method was built and checked
on, there is no rule, and 14 pairs are few. It shows the agent runs as built
on the GPU and what a gap between two arms looks like at this size. It is not
the test.

The plan: `kp-plan.md`, Phase 3. Runs: `results/kp/dev/20261119_162448/`
(`verdict.md`, `runs.csv`), 19 Nov 16:24 to 17:24, 28 of 28 runs on commit
`b34051e`, none failed. Arms as in the test: `SOT` (Goose with upgrade screen
3's three upgrades) and `SOT-KP` (the same with `EVAL_KP=1`).

## Levels

| Game | Seed | `SOT` | `SOT-KP` | Level-ups at action: `SOT` | ... `SOT-KP` |
|---|---|---|---|---|---|
| **ft09** | 14 | 2 | **6, won** | 343, 2,301 | 926, 949, 986, 1,020, 1,067, 1,103 |
| **ft09** | 15 | 1 | **6, won** | 301 | 853, 876, 913, 947, 994, 1,030 |
| **lp85** | 14 | 1 | **5** | 134 | 134, 238, 283, 510, 1,524 |
| lp85 | 15 | 0 | 0 | none | none |
| tu93 (no clicks) | 14 | 3 | 4 | 1,356, 1,771, 4,286 | 1,011, 1,159, 2,430, 4,927 |
| tu93 (no clicks) | 15 | 3 | 4 | 2,009, 2,068, 3,415 | 2,036, 2,087, 3,049, 3,392 |
| vc33 | 14 | 2 | 3 | 186, 265 | 186, 265, 1,248 |
| vc33 | 15 | 2 | 2 | 21, 566 | 21, 1,445 |
| cd82 | 14 | 1 | 1 | 601 | 801 |
| cd82 | 15 | 1 | 1 | 479 | 1,267 |
| dc22 | 14 | 1 | 1 | 1,140 | 1,461 |
| dc22 | 15 | 1 | 1 | 3,797 | 2,620 |
| m0r0 | 14 | 1 | 1 | 710 | 2,182 |
| m0r0 | 15 | 1 | 1 | 529 | 579 |
| **All seven** | | **20** | **36** | | |

| | `SOT` | `SOT-KP` | Difference |
|---|---|---|---|
| Levels by 1,000 actions | 11 | 17 | +6 |
| Levels by 2,000 actions | 14 | 28 | +14 |
| Levels by 5,000 actions | 20 | 36 | +16 |
| Pairs better / same / worse | | | 6 / 8 / 0 |
| 95% interval, games resampled | | | [+3, +33] |

**Where the 16 come from.**

| Games | What the planner did | `SOT-KP` minus `SOT` |
|---|---|---|
| ft09 | cleared levels 2 to 6 in both runs | +9 |
| lp85 | cleared levels 2 to 4 at seed 14; never started at seed 15 | +4 |
| tu93 | nothing: no click action, the planner is dormant | +2 |
| vc33, cd82, dc22, m0r0 | nothing: it never engaged | +1 |

So **+13 is the planner and +3 is chance.** The 11 pairs on which the planner
did nothing (these ten and lp85 at seed 15) are two runs of the same agent,
and their totals came out 3 levels apart. That is the size of gap the test's
rule has to tell from a real one.

## What the planner did

| Game | Levels cleared by planning | Actions for each planned level, median | The game's human baseline for those levels |
|---|---|---|---|
| ft09 (2 runs) | 10 | 23, 37, 34, 47, 37 | 12, 23, 28, 65, 37 |
| lp85 (1 run) | 3 | 104, 45, 227 | 38, 31, 16 |

- **ft09:** the same 178 clicks as on the CPU and in the smoke: 135 probes and
  43 plan steps a run. Goose needed 926 and 853 actions for level 1, and the
  game was won at 1,103 and 1,030.
- **lp85, seed 14:** levels 2 to 4 by planning. It gave level 5 up at action
  766 (41 wrong predictions), and Goose cleared level 5 at action 1,524. On
  level 6 it had no rule that fits.
- **lp85, seed 15:** Goose never cleared level 1, in either arm. The run saw
  13 different screens in 5,000 actions. The planner cannot start without a
  cleared level.
- **cd82, dc22, m0r0, vc33:** it never engaged ("no clickable kind learned"),
  as Phase 2 found from recordings. 1% to 31% of level 1's screens sat on the
  slots.
- **No error inside the planner in 14 runs,** no move went by unseen, and every
  planned click was the move sent.
- **Cost:** 110 ms per action against 105 at the median. Where it never
  engages it still watches: 22 to 30 seconds of a 520-second run.

## Level 1 came later in the planner's runs: chance

On level 1 the planner only watches, so the two arms are the same agent. Yet
level 1 came later with the planner in 7 of the 11 click-game pairs, sooner in
1 and at the same action in 3. Two things say this is chance and not the
planner.

1. **The two runs of a pair are the same run until GPU runs stop repeating.**
   A planner run and its plain twin take identical moves until somewhere
   between action 67 and action 449 on the six click games (9 of the 12 pairs
   between 97 and 145). Two runs of the *same* arm and seed part at action 98
   (ft09) and 104 (lp85). If the watching planner changed Goose's moves, a
   pair would part at once.
2. **The same arm at the same seed moves as much.** On ft09 at seed 14, level
   1 came at action 493 in the smoke and 343 here for `SOT`, and at 499 and
   926 for `SOT-KP`.

3. **On the CPU, where a run repeats exactly, level 1 is the same with and
   without the planner.** Two comparisons on ft09, made after this sweep
   (`results/cpu_check/2026-11-19_kp/`, `level1.sh`):

   | Flags | Without the planner | With `EVAL_KP=1` | Result |
   |---|---|---|---|
   | the adopted Goose's | 788 actions | the end-to-end run of Phase 1.3 | IDENTICAL on all 780 recorded moves of level 1, through action 787. The planner's first click is move 788 |
   | `SOT`'s | 500 actions | 500 actions | IDENTICAL on every move: 489 of 489 |

   In both, the planner was watching a click game and booking every one of
   Goose's clicks.

So the planner's runs did not get a worse level 1. They drew a later one.

## The smoke's six runs, beside these

The smoke ran ft09, lp85 and tu93 at seed 14 an hour earlier
(`kp-bridge-check.md`). GPU runs do not repeat, so these are second samples of
the same six runs.

| Game, seed 14 | `SOT`: smoke, dev sweep | `SOT-KP`: smoke, dev sweep |
|---|---|---|
| ft09 | 1, 2 | 6, 6 |
| lp85 | 2, 1 | 5, 5 |
| tu93 | 3, 3 | 4, 4 |

## Beside upgrade screen 3

On these seven games `SOT` completed 12 levels at screen 3's seed 0, and 10 a
seed here (20 at two seeds). The test's sanity anchor expects `SOT` near 44
levels on 25 games and two seeds if it repeats screen 3; this is a first sign
it may land a little under.

## What was changed after it

Nothing in the agent, the bridge or the settings: the code every run loads is
the code of `b34051e`. Two things beside the runs:

- **Two of the smoke's checks** in `kp_test.py` were written wrongly and are
  corrected (`kp-bridge-check.md`, section 1.5). Read again with the corrected
  checks, the smoke passes all seven.
- **The verdict page** gives the training share for the baseline runs and the
  planner's runs apart.

## What it means for the test

- **ft09 carries.** Six levels in every planner run so far: three on the GPU
  (the smoke's and this sweep's two), one on the CPU.
- **lp85 is 0 or 5.** It depends on whether Goose clears level 1, which it did
  not at one seed of two here.
- **A gap of 3 levels on 11 pairs came from nothing.** On the test's 50 pairs,
  chance alone will move the totals by several levels. The rule's bar is built
  from the test's own baseline-against-baseline differences for that reason.
