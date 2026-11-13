# The carry test: results (13 Nov 2026)

**In one line.** **FAIL.** Carrying colour no-ops and goal hints from one level
to the next did not make later levels come sooner: on the six click games the
carry arm finished 23 later levels in fewer actions and the plain Goose 22. No
levels were lost (87 against 81, inside the noise). Nothing is adopted; the
adopted Goose stays `mb_gated_att`.

**Status.** Registered on 12 Nov, before any run
([`carry-inside-goose.md`](carry-inside-goose.md)). Run 12-13 Nov. Verdict by
the registered command, run once, on 13 Nov. Everything named below is in
`results/sweeps/` (manifest `sweep_20261112_135205.manifest`).

The two arms: **A5** is the adopted Goose (`mb_gated_att`'s settings by flag).
**A6** is A5 plus `carry_noops,carry_hints`.

## The verdict

| | The registered rule | Result | |
|---|---|---|---|
| **Primary** | On levels 2 and up of the six click games, A6 finishes more levels in fewer actions than A5, and the 95% interval of the difference has its lower end above zero | A6 faster on 23, A5 faster on 22: **+1, interval [-15, +18]** | **FAIL** |
| Safety | Levels at 100,000 actions on all seven games, the dev rule | 87 against 81; pairs 8 better / 15 same / 5 worse; no game worse on every seed | pass |
| **The test** | Primary and safety | | **FAIL** |

Command: `bash experiments/goose/carry_test.sh verdict` (output kept in
`carry_test_verdict.txt`). The 23 against 22 was counted a second time from
each run's own level-up times, without the tool, and agrees
(`carry_test_races.md`, which lists every race).

## The primary, game by game

Levels 2 and up: levels the carry arm finished in fewer actions, against levels
the plain arm finished in fewer actions. The first two columns are the
exploratory numbers the test was built on (the guided arm against the same-day
A0, [`time-to-level.md`](../3-glassbox/time-to-level.md)).

| Game | G4, seeds 0-2 | Dev sweep 2, seeds 3-6 | **This test, seeds 7-10** |
|---|---:|---:|---:|
| lp85 | 10 - 2 | (not run) | **11 - 5** |
| vc33 | 5 - 3 | 7 - 4 | **7 - 5** |
| ft09 | 2 - 2 | 2 - 2 | **3 - 1** |
| cd82 | 3 - 0 | 3 - 1 | **1 - 2** |
| dc22 | 2 - 1 | 3 - 1 | **1 - 5** |
| m0r0 | 2 - 1 | 2 - 2 | **0 - 4** |
| **Six click games** | 24 - 9 | 17 - 10 | **23 - 22** |
| tu93 (placebo) | 5 - 4 | 6 - 6 | 6 - 5 |

- **lp85 and vc33 lean to the carry in every sweep. The other leads did not
  repeat:** cd82, dc22 and m0r0 were 15 - 6 before and are 2 - 11 here.
- **Where both runs finished the level it is an even split:** 34 races, 17
  each, median time ratio 0.99. Where only one run finished it: 6 for A6, 5 for
  A5.
- **By level:** level 2 went 10 - 13, levels 3 and up 13 - 9.

## Did the test work?

| Check | Result |
|---|---|
| The run set | 56 of 56 runs, each in the manifest once, all on the registered commit `57e178c`, none ended on an error. 54 played 100,000 actions; two lp85 runs stopped earlier because they finished the game (below) |
| The arms | All 28 A5 runs: novelty label, map, `attempt,bars,map_gated`. All 28 A6 runs: the same plus `carry_hints,carry_noops`. Speed 144.3 and 142.5 actions a second |
| The carry acted | It changed 9.4% of A6's moves (253,747 of 2.69 million), carried 104 colours and set hints at 53 level-ups |
| Placebo 1: tu93 (buttons only) | The carry changed 0 of 379,679 moves. Races 6 - 5 |
| Placebo 2: level 1 (nothing to carry yet) | **13 - 7 for A6**, 4 the same; interval [-5, +17] |
| Same agent where the carry is idle | Six races ended at exactly the same action in both arms (five level 1s, and tu93 seed 10's level 2) |

**Placebo 2 is the size of the noise.** On level 1 the two arms are the same
agent, and they still split 13 - 7. The primary's +1 is far inside that.

**It ran in two parts**, 13:52 to 15:13 on 12 Nov (7 runs) and 22:49 to 08:02
(49 runs), paused between runs at Matt's request. No run was cut or run twice.
About 10.5 hours of running in all.

**Two runs won the game.** lp85 seed 8 (A6) finished its eighth and last level
at 35,664 actions, and lp85 seed 9 (A5) at 52,998. The registration's words
were "56 of 56 runs at the 100,000-action cap"; its run-set check
(`--expect 28`) counts a won game as a finished run and passed. This is noted
under "Changes after registration" in the plan.

## Reported, not gating

**Levels at the end of each run** (seeds 7, 8, 9, 10):

| Game | A5 | A6 | A5 total | A6 total |
|---|---|---|---:|---:|
| vc33 | 2, 4, 4, 4 | 4, 4, 6, 4 | 14 | 18 |
| ft09 | 2, 2, 2, 2 | 2, 2, 2, 2 | 8 | 8 |
| cd82 | 2, 2, 1, 1 | 1, 2, 2, 1 | 6 | 6 |
| dc22 | 3, 2, 3, 3 | 1, 3, 2, 1 | 11 | 7 |
| m0r0 | 2, 2, 2, 2 | 2, 2, 2, 2 | 8 | 8 |
| lp85 | 2, 5, 8, 5 | 5, 8, 5, 5 | 20 | 23 |
| tu93 (placebo) | 4, 4, 4, 2 | 5, 4, 4, 4 | 14 | 17 |
| **All seven** | | | **81** | **87** |

The +6 has a 95% interval of [-8, +18]. **Half of it (+3) is tu93**, where the
carry changed no move.

**Levels by budget, six click games:**

| By | A5 | A6 | A6 - A5 | 95% interval | Of which level 1: A5, A6 |
|---|---:|---:|---:|---|---|
| 2,000 actions | 13 | 19 | +6 | [+0, +16] | 12, 17 |
| 10,000 actions | 33 | 42 | +9 | [-2, +23] | 22, 24 |
| 30,000 actions | 47 | 53 | +6 | [-9, +27] | 23, 24 |
| 100,000 actions | 67 | 70 | +3 | [-9, +15] | 24, 24 |

- **The early lead is mostly level 1**, where the carry cannot act: 5 of the +6
  at 2,000 actions.
- Level 1 luck also reaches the later counts. On lp85 seed 7 the plain arm
  spent 89,688 actions on level 1, so it had no time left for later levels.
  The level races are not moved by this: they count the actions spent on each
  level.
- **AULC** (100 to 35,664 actions, the shortest run): 17.03 against 20.51,
  +3.48, interval [-0.35, +9.31]; pairs 13 better, 11 worse.

## What the carry did

A6's 28 runs (`carry_test_stats.md` has every run):

| Game | Moves changed | Colours carried | Dropped later | Level-ups with hints | Hints dropped | Later-level races |
|---|---:|---:|---:|---:|---:|---:|
| vc33 | 25.1% | 29 | 0 | 7 | 0 | 7 - 5 |
| cd82 | 20.7% | 8 | 1 | 6 | 3 | 1 - 2 |
| dc22 | 10.8% | 15 | 0 | 7 | 6 | 1 - 5 |
| ft09 | 3.6% | 8 | 0 | 8 | 4 | 3 - 1 |
| lp85 | 3.3% | 17 | 1 | 9 | 2 | 11 - 5 |
| m0r0 | 1.2% | 27 | 5 | 0 | 0 | 0 - 4 |
| tu93 | 0.0% | 0 | 0 | 16 | 8 | 6 - 5 |
| **All** | **9.4%** | **104** | **7** | **53** | **23** | |

- **How much it acted says nothing about how it did.** Its best game, lp85,
  had 3.3% of moves changed; cd82 and dc22 had 20.7% and 10.8% and went the
  other way.
- **The colour rule was rarely wrong:** 7 of 104 carried colours were dropped
  later (a click on one changed something), 5 of them on m0r0.
- 23 hint colours were dropped after 2,000 moves in which they did not change.

## dc22 and m0r0

On these two games the carry arm lost all eight level-2 races, and dc22 ended
with 7 levels against 11 (worse on 3 of its 4 seeds). This was picked out after
the fact, and the plain Goose's own earlier runs vary as much:

| | Plain Goose, 10 earlier runs | A5, this test | A6, this test |
|---|---|---|---|
| dc22: runs that never finished level 2 | 5 of 10 | 0 of 4 | 2 of 4 |
| m0r0: actions spent on level 2 | 18,900 to 91,291, median 50,496 (9 runs; one never finished) | 13,704 to 48,684 | 39,393 to 75,663 |

(Earlier runs: A0 on 27 Sep, and the same-day A0 arms of G4 and dev sweep 2.)

A6's runs look like the plain Goose's usual ones; A5 drew four good runs on
each game. On m0r0 the carry changed 1.2% of moves. So this is not read as
harm, and it is not evidence of safety on these two games either.

## What it means

1. **The registered answer is no.** As a Goose upgrade, with new seeds, the
   carry does not make levels after the first come sooner on these games.
2. **The lead found on 10 Nov did not repeat.** That lead was exploratory: the
   measure was chosen after the runs. Two things differ between then and now,
   and this test cannot separate them:
   - *Chance.* Level 1 alone went 13 - 7 here with two identical agents.
   - *The book.* On the five dev games the earlier guided arm also carried the
     LLM's rules; A6 has none. The earlier evidence was that the book adds
     little (at most 3.6 points of covered moves; G2's LLM-free arm did as well
     as the guided one).
3. **The interval is wide.** [-15, +18] on 45 races does not show that the
   carry does nothing. It shows that an effect, if there is one, is too small
   to see in 24 pairs of runs, and so too small to adopt on.
4. **For the report.** G4's +9 levels was about one standard deviation of
   chance, and the time-to-level lead was the best sign that the guide does
   something. Its registered repeat found nothing. The guide's effect on
   levels and on time to level is **not shown**.
5. **It is safe.** No levels were lost, and no game was worse on every seed.

## What happens to the code

- **No preset.** `mb_carry` is not created.
- `carry_noops` and `carry_hints` stay in `EVAL_UPGRADES`, **off by default,
  not adopted**, like `deadclick`. With them off the agents are unchanged
  (CPU check of 12 Nov, `carry-inside-goose.md`).
- **They are kept** (Matt, 13 Nov: "keep the carry switches"). Removing them
  later would need its own `cpu_check`.

## Limits

- **Not held out.** These are dev and seen games, and the carry's settings were
  tuned on the dev games. New seeds make this a fair repeat only.
- **Six games, four seeds.** lp85 alone holds 16 of the 45 races.
- **One comparison.** The two switches were tested together, never one alone.

## A correction to a commit message

Commit `57e178c` (the registration) says a diff of the code since `c2f8d85`
"prints nothing". It prints one line: the status text of
`custom_agents/README.md`. The only code that changed between the CPU check and
the test is the new launcher, `experiments/goose/carry_test.sh`.

## Files

All in `results/sweeps/` (not in git):

| File | What it is |
|---|---|
| `sweep_20261112_135205.manifest` | The 56 runs |
| `carry_test.log` | The sweep's log, both parts |
| `carry_test_verdict.txt` | What the verdict command printed |
| `carry_test_levels.md` | Safety: levels, the dev rule |
| `carry_test_click_games.md`, `carry_test_placebo.md` | The primary's tables, and tu93's |
| `carry_test_races.md` (`carry_test_races.py`) | The second count: every race, by level, by game, level 1 split out |
| `carry_test_stats.md` (`carry_test_stats.py`) | The run set, and what the carry did in every run |
