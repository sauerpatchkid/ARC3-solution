# The Knowledge Planner's test: results (19 Nov 2026)

**In one line.** **The planner clears the noise.** With it, Goose with the
screen 3 upgrades ended on 62 levels, against 44 and 47 for the two baseline
arms (25 games x seeds 11 and 12 x 5,000 actions). Its lead over `SOT` is +18
and the bar was 10. All of the lead is on ft09 (+9) and lp85 (+10), the two
games the method was built on. On the 15 held-out games the planner never
made a move, and the arms are level there (18 against 17 and 18). Nothing is
adopted.

**Status.** Registered on 19 Nov, before any run ([`kp-test.md`](kp-test.md)).
Run 19 Nov, 18:11 to 23:53: 150 of 150 runs, none failed. Verdict by the
registered command, run once, on 19 Nov at 23:54. Runs:
`results/kp/test/20261119_181131/` (`verdict.md`, `runs.csv`). What a pass
leads to is Matt's decision (`kp-plan.md`, section 2.3); no preset is created.

The three arms: **`SOT`** is Goose with upgrade screen 3's three upgrades.
**`SOT-KP`** is the same with the planner (`EVAL_KP=1`): a groupmate's agent
E20, run unedited behind Goose. **`SOT2`** is `SOT` again, to measure chance.

**Read it as:** a tuned-tier result, at 5,000 actions and two seeds.

## The verdict

| | The registered rule | Result | |
|---|---|---|---|
| (1) | `SOT-KP`'s levels are above both `SOT`'s and `SOT2`'s | 62 against 44 and 47 | met |
| (2) | `SOT-KP - SOT` is above the 97.5th percentile of the chance gap | **+18** against a bar of **10** | met |
| **The test** | (1) and (2) | | **CLEARS THE NOISE** |

Command: `uv run python experiments/kp/kp_test.py verdict`.

**The bar.** The chance gap is built from the test's 50 differences
`SOT2 - SOT`, one per game and seed. Nine of the 50 are not zero and they sum
to +3. Of the 10,000 chance gaps drawn from them, 8 are +18 or more. The
planner's lead over the second baseline, `SOT-KP - SOT2` = +15, is above the
bar too.

One pair makes most of the bar: on lp85 at seed 12, `SOT2` cleared 5 levels
unaided and `SOT` cleared 1. Without that pair the bar would be 6 (this is not
the rule, it only shows what the bar is made of).

## Where the 18 come from

| Games | What the planner did | `SOT` | `SOT-KP` | `SOT2` | `SOT-KP - SOT` |
|---|---|---:|---:|---:|---:|
| **ft09** | cleared levels 2 to 6 in both runs and won the game | 3 | **12** | 4 | **+9** |
| **lp85** | cleared levels 2 to 4 in both runs, and 6 and 7 at seed 12 | 2 | **12** | 6 | **+10** |
| The other 8 dev and seen games | nothing: it made no click | 22 | 20 | 19 | -2 |
| The 15 held-out games | nothing: it made no click | 17 | 18 | 18 | +1 |
| **All 25** | | **44** | **62** | **47** | **+18** |

So **+19 is the planner and -1 is chance.** On the 23 games where the planner
made no click, `SOT-KP` ended on 38 levels, `SOT` on 39 and `SOT2` on 37.

Of `SOT-KP`'s 62 levels, 18 were cleared by planning (ft09 10, lp85 8) and 44
by Goose.

## By tier, by budget, pair by pair

| Tier | Games | `SOT` | `SOT-KP` | `SOT2` |
|---|---:|---:|---:|---:|
| dev (tuned) | 8 | 23 | 32 | 22 |
| seen (tuned) | 2 | 4 | 12 | 7 |
| A/B-offline (held out) | 8 | 11 | 11 | 11 |
| untouched (held out) | 7 | 6 | 7 | 7 |
| **Tuned tiers** | 10 | 27 | **44** | 29 |
| **Held-out tiers** | 15 | 17 | **18** | 18 |

| Levels completed by | `SOT` | `SOT-KP` | `SOT2` | `SOT-KP - SOT` |
|---|---:|---:|---:|---:|
| 1,000 actions | 19 | 42 | 20 | +23 |
| 2,000 actions | 31 | 53 | 28 | +22 |
| 5,000 actions | 44 | 62 | 47 | +18 |

The lead is largest early. Every planned level came inside the first 1,000
actions (ft09 was won at action 793 and 763; lp85's last planned level came at
action 777 and 988). After that the baselines catch up a little.

| Kind of game | Games | `SOT` | `SOT-KP` | `SOT2` |
|---|---:|---:|---:|---:|
| Click games | 19 | 33 | 53 | 39 |
| No click (g50t, ls20, re86, tr87, tu93, wa30): the arms are the same agent | 6 | 11 | 9 | 8 |

| Difference | Levels | 95% interval, games resampled | Pairs better / same / worse | Games worse on every seed |
|---|---:|---|---|---|
| `SOT-KP - SOT` | +18 | [-4, +48] | 8 / 37 / 5 | ls20 |
| `SOT-KP - SOT2` | +15 | [-1, +37] | 7 / 40 / 3 | none |
| `SOT2 - SOT` | +3 | [-5, +13] | 5 / 41 / 4 | none |

- **The interval with games resampled reaches below zero,** as the test page
  said it would: the gain sits on two games of 25, and a resample that leaves
  both out has no gain.
- **ls20 is chance.** It offers no click, so the planner is dormant there. The
  planner's arm cleared no level (0 and 0) against 1 and 1 for `SOT` and 1 and
  0 for `SOT2`: three of the six ls20 runs cleared level 1.

**Level races** (which arm finished a level in fewer actions; click games):

| Levels | Against | Races | `SOT-KP` faster | Baseline faster | Same or undecided | 95% interval of the difference |
|---|---|---:|---:|---:|---:|---|
| 2 and up | `SOT` | 10 | 7 | 2 | 1 | [-1, +10] |
| 2 and up | `SOT2` | 14 | 11 | 3 | 0 | [+0, +16] |
| 1 | `SOT` | 30 | 13 | 9 | 8 | [-7, +14] |
| 1 | `SOT2` | 30 | 13 | 9 | 8 | [-4, +12] |

`SOT2` against `SOT`: 3 against 3 on levels 2 and up, 10 against 9 on level 1.
Races are few because a level is raced only where both arms entered it, and
the baselines seldom got past level 2 on ft09 and lp85.

## Every game

Levels by 5,000 actions, written `seed 11, seed 12`. The last column is
upgrade screen 3 at the same budget, which is another seed (0).

| Game | Tier | `SOT` | `SOT-KP` | `SOT2` | Screen 3, seed 0: `SOT` / adopted Goose / again |
|---|---|---|---|---|---|
| **ft09** | dev | 2, 1 | **6, 6** | 2, 2 | 1 / 1 / 2 |
| **lp85** | seen | 1, 1 | **5, 7** | 1, 5 | 1 / 2 / 1 |
| cd82 | dev | 1, 1 | 1, 1 | 1, 1 | 1 / 1 / 1 |
| dc22 | dev | 1, 1 | 1, 1 | 1, 1 | 1 / 0 / 0 |
| g50t | dev | 1, 0 | 1, 1 | 0, 1 | 1 / 0 / 0 |
| m0r0 | dev | 1, 1 | 1, 1 | 1, 1 | 1 / 1 / 1 |
| tr87 | dev | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| tu93 | dev | 4, 4 | 3, 4 | 4, 2 | 4 / 2 / 3 |
| vc33 | dev | 3, 2 | 2, 3 | 2, 3 | 3 / 2 / 2 |
| ls20 | seen | 1, 1 | 0, 0 | 1, 0 | 1 / 1 / 0 |
| ar25 | A/B-offline | 2, 1 | 2, 1 | 2, 1 | 2 / 1 / 0 |
| lf52 | A/B-offline | 1, 1 | 1, 1 | 1, 1 | 1 / 0 / 0 |
| re86 | A/B-offline | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| s5i5 | A/B-offline | 1, 1 | 1, 1 | 1, 1 | 1 / 1 / 1 |
| sc25 | A/B-offline | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| sp80 | A/B-offline | 1, 1 | 1, 1 | 1, 1 | 1 / 1 / 1 |
| su15 | A/B-offline | 1, 1 | 1, 1 | 1, 1 | 1 / 0 / 1 |
| tn36 | A/B-offline | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| bp35 | untouched | 0, 0 | 0, 1 | 0, 1 | 0 / 0 / 0 |
| cn04 | untouched | 1, 1 | 0, 1 | 1, 1 | 1 / 1 / 1 |
| ka59 | untouched | 0, 0 | 0, 1 | 0, 0 | 0 / 0 / 0 |
| r11l | untouched | 1, 1 | 1, 1 | 1, 1 | 1 / 1 / 1 |
| sb26 | untouched | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| sk48 | untouched | 1, 1 | 1, 1 | 1, 1 | 0 / 0 / 0 |
| wa30 | untouched | 0, 0 | 0, 0 | 0, 0 | 0 / 0 / 0 |
| **All 25** | | **24, 20** | **28, 34** | **22, 25** | **22 / 15 / 15** |

**`SOT` beside upgrade screen 3.** `SOT` scored 22 levels at screen 3's one
seed. Here its four totals of one seed are 24, 20, 22 and 25, so it repeats
that score on new seeds. This test has no adopted-Goose arm, so it does not
compare `SOT` with the adopted Goose.

## What the planner did

| Game | Levels cleared by planning | Clicks: probe / verify / plan | Actions for each planned level | The game's human baseline for those levels |
|---|---:|---|---|---|
| ft09 (2 runs) | 10 | 270 / 0 / 86 | L2 to L6: 23, 37, 34, 47, 37, in both runs | 12, 23, 28, 65, 37 |
| lp85, seed 11 | 3 | 50 / 551 / 120 | L2 to L4: 104, 45, 227 | 38, 31, 16 |
| lp85, seed 12 | 5 | 52 / 651 / 173 | L2 to L4: 27, 73, 165; L6: 126; L7: 75 | 38, 31, 16; 60; 26 |

- **ft09.** The same 178 clicks as in every earlier run: 135 probes and 43
  plan steps. Goose needed 616 and 586 actions for level 1, and the game was
  won at action 793 and 763. The rule it used, in the reference's words:
  "every multi-colour pattern describes its neighbourhood". It was confirmed
  at all five clears and never refuted.
- **lp85, seed 11.** Levels 2 to 4 by planning. It gave level 5 up at action
  1,033 after 41 wrong predictions, and Goose cleared level 5 at action 2,692.
  On level 6 no rule it had left fitted. Five levels.
- **lp85, seed 12.** Levels 2 to 4 by planning. It gave level 5 up at action
  655, and Goose cleared it at action 787. The planner then cleared levels 6
  and 7, which it had not done in any earlier run or offline check. On level 8
  no rule fitted. Seven levels by action 988.
- **Why the two seeds differ: which rule survived.** At seed 12 the rule "for
  every mark, some nearest dynamic object has the same colour" was confirmed
  at all seven clears. At seed 11 that rule was refuted at level 2's clear,
  and the planner went on with "for every frame, every enclosed dynamic object
  has the same colour". That second rule gives it nothing to plan to on level
  6: at seed 12 it was confirmed at every clear but level 6's, and never
  refuted. Why the first rule was refuted at seed 11 is not looked into.
- **Everywhere else it made no click:** 46 of its 50 runs. It is dormant on
  the six games that offer no click.
- **No error inside the planner in 50 runs.** No move went by unseen, and
  every planned click was the move sent.
- **Cost.** 113 ms per action against 109 at the median, and a run took 9.2
  minutes. Where it never engaged it still watched: 25 seconds of a run at the
  median, 36 at most. The slowest run was lp85 at seed 12 (698 seconds, 211 of
  them inside the planner).

### At the level clears

| Game | Clears the planner learned from | Solved board fits the slots | At least one rule consistent | Rule carried in: confirmed again |
|---|---:|---|---|---|
| ft09 | 10 | 10 of 10 | 10 of 10 | 8 of 8 |
| lp85 | 12 | 12 of 12 | 12 of 12 | 9 of 10 (refuted once: seed 11, level 2) |

## The funnel, per game

Over the levels after the first that the planner's runs entered. Each step
counts a level only if it passed every step before it.

| Game | Entered | Slots held | Had a rule | Model settled | Plan found | Cleared by planning | Cleared by Goose after a fallback | Not cleared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ft09** | 10 | 10 | 10 | 10 | 10 | **10** | 0 | 0 |
| **lp85** | 12 | 12 | 12 | 11 | 10 | **8** | 2 | 2 |
| The other 14 click games with a level 1 cleared | 29 | 0 | 0 | 0 | 0 | 0 | 4 | 25 |
| sb26, sc25, tn36 | 0 | | | | | | | |

- **lp85's four levels not planned:** level 5 at both seeds (a plan was found,
  then 41 wrong predictions; Goose cleared both), level 6 at seed 11 (the
  model never settled) and level 8 at seed 12 (no plan was found).
- **The 14 games** are ar25, bp35, cd82, cn04, dc22, ka59, lf52, m0r0, r11l,
  s5i5, sk48, sp80, su15 and vc33. On none of them did 95% of a later level's
  screens stay on its slots.
- **sb26, sc25 and tn36:** no arm cleared level 1 in 5,000 actions, so the
  planner had nothing to learn from.

## Why it did not engage on the held-out games

The planner reads a level as a fixed set of slots, taken from the level's
first screen, and reads a later screen only if every object sits on a slot.
The table is for level 1 in the planner's own two runs.

| Game | Tier | Level 1's screens on the slots | Objects on the first screen | What stopped it |
|---|---|---|---:|---|
| ar25 | A/B-offline | 2%, 3% | 4 | no clickable kind learned |
| s5i5 | A/B-offline | 7%, 5% | 12 | no clickable kind learned |
| su15 | A/B-offline | 7%, 8% | 23 | no clickable kind learned |
| sp80 | A/B-offline | 4%, 17% | 3 | no clickable kind learned; at seed 12 a click effect was learned and the solved board did not fit |
| **lf52** | A/B-offline | 50%, 46% | 34 | no goal rule: the solved board did not fit (see below) |
| sc25 | A/B-offline | 100%, 100% | | level 1 never cleared |
| tn36 | A/B-offline | 5%, 4% | | level 1 never cleared |
| bp35 | untouched | 7%, 7% | 158 | no clickable kind learned (level 1 cleared at one seed) |
| cn04 | untouched | 2%, 2% | 2 | no clickable kind learned (level 1 cleared at one seed) |
| r11l | untouched | 2%, 2% | 30 | no clickable kind learned |
| sk48 | untouched | 5%, 8% | 5 | no clickable kind learned |
| ka59 | untouched | 100%, 100% | 1 | no goal rule: it sees one object, so every screen fits, and no rule was left standing (level 1 cleared at one seed) |
| sb26 | untouched | 100%, 100% | | level 1 never cleared |

re86 and wa30 offer no click. The four dev games where it never engages are as
Phase 2 found them: cd82 10% and 7%, dc22 30% and 26%, m0r0 1% and 2%, vc33
22% and 57%.

Three things stopped it on the 13 held-out games that offer clicks:

1. **The screen does not stay on the slots** (8 games: ar25, bp35, cn04, r11l,
   s5i5, sk48, sp80, su15). Things move, grow or appear, and under 20% of
   level 1's screens fitted. So it almost never saw what a click does: it
   learned a clickable kind at 1 of the 14 level-1 clears.
2. **Goose did not clear level 1 in 5,000 actions** (3 games: sb26, sc25, tn36,
   in every arm). On sb26 and sc25 every screen did sit on the slots, so
   whether the planner would work there is still open. It needs a level 1 that
   Goose does not deliver at this budget.
3. **No goal rule was confirmed** (2 games: ka59, lf52). The solved board was
   off the slots (lf52), or no rule in its rule space was consistent with it
   (ka59).

**lf52 came closest.** It behaves like lp85: clicks move pieces, 34 objects.
From two of the four baseline recordings (next section) the planner learned
lp85's own rule there ("for every mark, some nearest dynamic object has the
same colour"), started on level 2 and made one probe. The screen then left the
slots and it handed the level back.

## The planner alone against Goose, level by level

The offline takeover (`kp-plan.md`, section 6.3), run after the test on the two
baseline arms' own recordings: each recording plays level 1, then the planner
plays alone to action 5,000. It runs on the CPU and repeats exactly, so the
planner's part has no seed noise. 76 recordings of the 19 click games, 57 with
a level 1 inside 5,000 actions, no error and no replay mismatch.

| Game | Recordings with a level 1 | Levels the planner reached | Levels the same recording's Goose reached |
|---|---:|---|---|
| **ft09** | 4 | 6 from all four: the game won | 1, 2, 2, 2 |
| **lp85** | 4 | 4 from all four; it stops on level 5 | 1, 1, 1, 5 |
| The other 13 games with a level 1 | 49 | 1 from every one: it never cleared a level | 1 to 3 |

No baseline run cleared level 1 on ka59, sb26, sc25 or tn36.

| Game | Level | The planner: actions (recordings that cleared it) | Goose in the same recordings: actions (recordings that cleared it) | Human baseline |
|---|---:|---|---|---:|
| ft09 | 2 | 23 (4 of 4) | 1,532, 3,179, 4,139 (3 of 4) | 12 |
| ft09 | 3 | 37 (4 of 4) | none (0 of the 3 that reached it) | 23 |
| ft09 | 4, 5, 6 | 34, 47, 37 (4 of 4) | never reached | 28, 65, 37 |
| lp85 | 2 | 104 at seed 11, 27 at seed 12 (4 of 4) | 723 (1 of 4) | 38 |
| lp85 | 3 | 45, 73 (4 of 4) | 929 (1 of the 1 that reached it) | 31 |
| lp85 | 4 | 227, 165 (4 of 4) | 328 (1 of 1) | 16 |
| lp85 | 5 | given up (0 of 4) | 360 (1 of 1) | 41 |

- **ft09:** from Goose's level 1 the planner needs 178 actions for the rest of
  the game. Goose alone needed 1,532 to 4,139 actions for level 2 and never
  cleared level 3.
- **lp85:** the two baseline arms share a level 1 at each seed (it comes at
  action 213 and 166, before two GPU runs part), so these are two different
  starts, each seen twice. Three of the four recordings never cleared level 2
  in the 4,787 to 4,834 actions they had left. The fourth is the `SOT2` run
  that reached 5 levels by itself.
- **Online and offline agree.** In the test's own runs the planner took the
  same actions for levels 2 to 4 of lp85 (104, 45, 227 and 27, 73, 165) and
  the same 178 clicks on ft09.

## What the written expectation got right and wrong

| Written before the runs (`kp-test.md`) | What happened | |
|---|---|---|
| ft09: 6 and 6 for `SOT-KP` against 1 or 2 each for `SOT` | 6 and 6 against 2 and 1 (`SOT2`: 2 and 2) | right |
| lp85: 4 or 5 where Goose clears level 1, so 0 to 10 over two seeds | 5 and 7: 12 | **too low.** It cleared levels 6 and 7 at seed 12, which it had never done |
| lp85: 0 to 2 for `SOT` | `SOT` 1 and 1; `SOT2` 1 and 5 | **wrong for one run of four.** The plan's own draft had said "1 to 5" |
| The other eight dev and seen games: no effect | the planner made no click; 20 levels against 22 and 19 | right |
| The 15 held-out games: no number; it engages only where pieces keep to their places, which held on 2 of the 6 click games it was built on | it engaged on none of the 13 that offer clicks; 18 levels against 17 and 18 | nothing was predicted. The share was lower than on the games it was built on |
| `SOT` and `SOT2` near 40 each | 44 and 47 | a little low |
| `SOT-KP - SOT` about +13, between +8 and +18 | +18 | the top of the range |
| The bar: between 4 and 12 levels | 10 | right |
| A pass is likely if both games carry | both carried, and it passed | right |

## Did the test work?

| Check | Result |
|---|---|
| The run set | 150 of 150 runs, each in the manifest once. None failed and none was retried (there is no `failed.tsv`). 148 played 5,000 actions; the two `SOT-KP` runs of ft09 stopped at 793 and 763 because they won the game |
| The frozen code | Started from `f0a1d3a` with a clean tree. Code hash `aae8104ab8d0c6bd` and settings hash `d468480ec9387b71`, as registered, and the same after the runs. No file a run loads differs from `527b8a4` |
| The arms | `SOT`'s flags in all 150 `run_config.json`. The planner on in the 50 `SOT-KP` runs only, with the committed settings |
| Training | A step on at least 87% of actions in every baseline run. The lowest `SOT-KP` run is 66%: ft09 at seed 12, won in 763 actions (Goose takes no step on a level the planner clears before 64 moves are stored) |
| Sanity anchor | `SOT` 44 and `SOT2` 47: inside 28 to 66, and near the 44 expected if `SOT` repeats screen 3 |
| The verdict | Run once. `kp_verdict.py` is unchanged (sha256 starts `bb0ef307151e0f3f`). The totals by budget, the pair counts and the interval were counted a second time by `tools/levels_by_budget.py` and agree |
| Changes after the start | None |
| A look before the end | At 21:53, with 95 runs in, the tables of the 31 complete pairs were printed for Matt (40 against 31 and 33). It wrote no file, gave no verdict, and nothing was changed, stopped or rerun |

## What it means

- **On ft09 and lp85 the planner turns later levels from thousands of actions
  into tens.** ft09 is won in under 800 actions in both runs, and from all
  four baseline recordings offline. On lp85 it clears levels 2 to 4 every
  time, and sometimes two more.
- **It reaches no other game of the 25.** That was known for the other eight
  games it could be built on, and the test adds the 15 held-out games: no
  click on any of them. Most are games whose pieces move.
- **It costs almost nothing where it does not engage.** On those 46 runs its
  arm is 1 level behind one baseline and 1 ahead of the other, and 4% slower
  per action.
- **So the result is narrow and large.** It engaged on 2 of the 19 click
  games, and there the gain is 19 levels in 4 runs.

## Limits

1. **Tuned tier.** ft09 and lp85 are the games the method was built on, by the
   groupmate, with the games' source code read. The held-out tiers were the
   clean test of whether it reaches further, and it did not.
2. **Two seeds, 5,000 actions.** A short-budget test: a pass is a candidate
   for a longer confirm. It says nothing about 100,000 actions, where Goose
   alone has more time to clear the same levels.
3. **The planner acted in 4 of its 50 runs.** The +18 rests on those four. The
   offline takeover repeats ft09's five planned levels and lp85's levels 2 to
   4, action for action, from eight baseline recordings, so the mechanism does
   not rest on chance. The level count still has four runs behind it.
4. **The bar is a yardstick.** It came from 50 differences, and one pair made
   most of it.
5. **The base is `SOT`,** which is exploratory itself and not adopted.

## Open, Matt's

From `kp-plan.md`, section 2.3. Nothing below is decided or done.

1. **What the pass leads to.** No preset exists. One would have to carry
   `SOT`'s three switches too, and `SOT` is not adopted.
2. **How the report names the method and credits the groupmate,** and whether
   the README's "four parts" become five.
3. **Whether Phase 5 (an LLM proposes goal rules) or Phase 6 (a clean port)
   is done.** Both are in `future/README.md` as ideas, not built.
4. **`SOT`'s own confirm.** This test adds 100 `SOT` runs on new seeds
   (44 and 47 levels), which that confirm could be read beside.

## Files

| | |
|---|---|
| The test and its rule | [`kp-test.md`](kp-test.md) |
| The verdict page and one row per run | `results/kp/test/20261119_181131/verdict.md`, `runs.csv` |
| The runs | `results/kp/test/20261119_181131/runs/<arm>/<game>_s<seed>/`; a planner run's `kp/` folder holds `kp_stats.json`, `events.jsonl` and `store.json` |
| The offline takeover | `results/kp/phase4/` (`takeover.md`, `takeover.tsv`, `takeover_clears.tsv`, `funnel.md`) |

```bash
uv run python experiments/kp/kp_test.py verdict        # the verdict page again
uv run python tools/levels_by_budget.py --base results/kp/test/20261119_181131/manifest.tsv:SOT \
    --new results/kp/test/20261119_181131/manifest.tsv:SOT-KP --budgets 1000,2000,5000 --by-game
uv run python tools/kp_offline.py takeover --registered --out results/kp/phase4 \
    --runs results/kp/test/20261119_181131/manifest.tsv:SOT results/kp/test/20261119_181131/manifest.tsv:SOT2
```
