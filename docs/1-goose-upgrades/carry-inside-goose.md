# Carry inside Goose: what one level teaches about the next (12 Nov 2026)

**In one line.** The part of Glassbox that helped needs no LLM: colours that
do nothing when clicked, and what shrank before a level was won, carried to the
next level. It is now a Goose upgrade, two switches, off by default, and it
makes the same moves as Glassbox's LLM-free arm. Its test is drafted below and
waits for Matt's go.

**Status.** Built and checked on the CPU on 12 Nov. **The test is a draft: not
registered, not run.** It needs about 11 GPU-hours.

## Why

- G4 adopted the guided Goose on weak evidence: 110 levels against 101, about
  one standard deviation of chance (`docs/3-glassbox/g4-confirm.md`).
- Looked at level by level, the guide makes levels after the first come sooner:
  36 level races against 17 in G4, where two unguided arms split 24 against 27
  (`docs/3-glassbox/time-to-level.md`, exploratory).
- The LLM is not what does it. Outside the 8 dev games the guide used no LLM
  output. In G2 the arm with no LLM rules anywhere did as well as the guided
  one. In the counterfactual the LLM's rules added at most 3.6 points of
  covered moves (`docs/3-glassbox/funnel.md`).

So the useful part belongs with the explorer. On 12 Nov Matt put it on the
week's list in place of "a reset watchdog and ACTION7", which three probes
found little use for (`input-probes.md`).

## What it is

Two switches in `EVAL_UPGRADES`, each usable alone. Code:
`custom_agents/carry.py` (the upgrade) and a few lines in `upgrades.py` (the
wiring). No new hook line in `action.py`: it uses the hooks the other upgrades
use, and one existing line now hands over the agent's seed.

| Switch | The problem | What it does |
|---|---|---|
| `carry_noops` | On each new level Goose finds out again which colours do nothing when clicked | A colour clicked 20 times or more on a level, whose clicks never changed a live cell, is carried: on later levels a click on it keeps a tenth of its probability. The first such click that changes a live cell drops it |
| `carry_hints` | Goose has no idea what a level is about | The two colours that vanished or shrank most before a level was won (never the background) become hints: on the next level a click on one counts double. A hint is dropped after 2,000 moves in which that colour did not change |

"Never changed a live cell" is judged at the level's end, under the agent's
decoration mask of that moment with each bar run to the screen's edge. Weights
reach Goose by rejection sampling with the upgrade's own random generator, so
Goose's own random stream is untouched.

Not in it: the book's no-op rules and the map's "a rule covers this button"
(both need the LLM's rules), "hints on button games" and "keep the network
across levels" (the review's other List 1 item 2 ideas; separate switches if
this one passes).

## It is the same agent as Glassbox's LLM-free arm

`carry.py` is a port of Glassbox's S5 v2 guide run with no rule book
(`gb/s5_act.py`, the arm called GB-free), copied so that the Goose upgrade does
not depend on Glassbox. Three checks say the copy is exact and that nothing
else moved:

| Check | How | Result |
|---|---|---|
| Same moves as GB-free on made-up levels | `tests/test_carry.py`: both fed the same 2,100 moves over 3 levels, 3 seeds; every move chosen, every colour carried or dropped and every hint must agree | pass |
| Same moves as GB-free on a real game | CPU, ft09, seed 0, 3,000 actions: Goose with `bars,map_gated,attempt,carry_noops,carry_hints` against `mb_gated_att` with `EVAL_GB=1` and no book | **identical on all 2,934 recorded moves.** Level 1 ended at action 787 in both; the upgrade carried two colours, set two hints and changed 1,114 of 2,935 moves, the guide's own counts |
| Off by default | CPU, 2,000 actions, before and after the change: `goose` on ft09, `mb_gated_att` on tu93 | **identical: 1,970 of 1,970 moves and 1,960 of 1,960** |

The three CPU comparisons, their commands and their recordings are kept in
`results/cpu_check/2026-11-12_carry/` (`RESULT.md`). "Before" is commit
`2443dd3`; "after" is the code of the commit that added the upgrade.

## The test (DRAFT: not registered, not run)

**Question.** As a Goose upgrade, does the carry make levels after the first
come sooner, without losing levels?

| | |
|---|---|
| Arms | `A5`: the adopted Goose (`mb_gated_att`'s three settings by flag). `A6`: A5 + `carry_noops,carry_hints`. Same night, interleaved per game and seed |
| Games | The six dev and seen games with clicks that reach a second level: vc33, ft09, cd82, dc22, m0r0, lp85. Plus **tu93 as a placebo**: buttons only, so the carry cannot act there |
| Seeds | 7, 8, 9, 10 (G4 used 0-2, dev sweep 2 used 3-6) |
| Budget | 100,000 actions a run; 56 runs; about 11 GPU-hours, one run at a time |
| **Primary** | On levels 2 and up of the six click games, A6 finishes more levels in fewer actions than A5 (`tools/levels_by_budget.py`, the level races), with the 95% interval of the difference (games resampled) above zero |
| Safety | Levels at 100,000 actions under the dev rule: A6 completes at least as many as A5, wins at least as many pairs as it loses, and no game is worse on every seed (`tools/paired_compare.py --rule dev --expect 28`) |
| Reported | Level 1 races and tu93's races (both should split evenly); levels by 2,000, 10,000 and 30,000 actions; AULC; what the carry did (colours carried and dropped, hints set and dropped, share of moves changed) |
| PASS | Primary and safety. Then the bundle gets a preset (`mb_carry`) and becomes the report's last Goose upgrade |
| FAIL | The exploratory result did not repeat. Nothing is adopted; the report says so |

What the same games showed before (exploratory, the guided arm against the
same-day A0, levels 2 and up):

| | Six click games | tu93 |
|---|---|---|
| G4, seeds 0-2 | 24 against 9 | 5 against 4 |
| Dev sweep 2, seeds 3-6 (no lp85) | 17 against 10 | 6 against 6 |

The command, once registered:

```bash
GAMES="vc33 ft09 cd82 dc22 m0r0 lp85 tu93" SEEDS="7 8 9 10" CAP=100000 ARMS="A5 A6" AGENT=goose \
    nohup bash sweep.sh > results/sweeps/carry_test.log 2>&1 &
```

`touch results/sweeps/STOP` pauses it after the run in progress; `RESUME=<its
manifest>` continues it.

**Two things to know before saying go.**

1. **These games are not held out.** The carry's settings were tuned on the dev
   games (S5 v2, 3 Nov). New seeds make it a fair repeat, not a held-out test.
   A held-out version would add ar25, sp80, s5i5 and r11l (about 5 more
   GPU-hours for 3 seeds). The tier rule keeps those games for the confirm,
   which is over; using them again is Matt's call.
2. **Six games is few.** In G4, lp85 alone gave 10 of the 24 wins. The interval
   resamples games and will be wide.
