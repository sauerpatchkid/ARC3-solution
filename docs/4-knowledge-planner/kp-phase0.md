# Knowledge Planner, Phase 0: three checks before anything is built (19 Nov 2026)

**In one line.** The groupmate's planner runs here as received and gives the
report's numbers (E20: 4.83 levels per episode against the report's 4.33, and
30 of 36 episodes are the report's to the action). All 180 recordings of the
ten dev and seen games play again on the engine move for move (886,873 moves).
And the planner works after this project's Goose: starting from Goose's own
level 1 it won ft09 from 20 of 20 recordings and reached level 4 of lp85 from
15 of 15, the four played with `objclick` included. **Go.**

**Status: checks, not tests.** CPU only. No agent was changed and nothing is
built yet except the driver. ft09 and lp85 are the two games the method was
developed on, so everything here is a tuned-tier result.

The plan: `kp-plan.md`, Phase 0. The tool: `experiments/kp/run_reference.py`.
The reference: `custom_agents/kp_ref/` (the groupmate's four files, byte for
byte; hashes in its README). Tables and logs: `results/kp/phase0/`.

| Check | The bar, from the plan | Result |
|---|---|---|
| 0.2 The reference gives the report's numbers | E20's mean at least 4.0 levels over 12 episodes; every episode that clears level 1 reaches level 6 on ft09 and level 4 on lp85 | **PASS.** 4.83; none stopped short |
| 0.3 Recordings play again on the engine | every move matches on ft09 and lp85 | **PASS.** 176,672 of 176,672 there; 886,873 of 886,873 on all ten games |
| 0.3 The solved board is among the returned frames | on ft09 and lp85 | **PASS.** 24 of 24 and 27 of 27 level-ups |
| **0.4 The go/no-go** | ft09 to level 5 from at least 80% of its prefixes and from 3 of the 4 `objclick` ones; lp85 to level 4 from at least 60% and from 2 of the 4 | **PASS.** 20 of 20 and 4 of 4; 15 of 15 and 4 of 4 |

A *prefix* is a recorded Goose run up to its first level-up.

## 1. The reference gives the report's numbers

**What was run.** The reference's three agents, unedited and with nothing
replaced, in the reference's own `run_episode`: E0 (its small Goose), E18 and
E20, on its two games x seeds 0 to 5 x 1,500 actions. The repo's offline engine,
the reference's own game ids (`ft09-0d8bbf25`, `lp85-305b61c3`, both found
offline), scorecard `None`.

Levels completed per episode:

| Game | Agent | s0 | s1 | s2 | s3 | s4 | s5 | Mean | The report (Appendix D) |
|---|---|---|---|---|---|---|---|---|---|
| ft09 | E0 | 1 | 1 | 1 | 1 | 1 | 1 | 1.00 | 1 0 1 1 1 1 |
| ft09 | E18 | 3 | 3 | 3 | 3 | 3 | 3 | 3.00 | 3 0 3 3 3 3 |
| ft09 | E20 | 6 | 6 | 6 | 6 | 6 | 6 | 6.00 | 6 0 6 6 6 6 |
| lp85 | E0 | 0 | 2 | 1 | 1 | 1 | 1 | 1.00 | 0 2 1 1 1 1 (same) |
| lp85 | E18 | 0 | 3 | 3 | 3 | 3 | 3 | 2.50 | 0 3 3 3 3 3 (same) |
| lp85 | E20 | 0 | 4 | 4 | 5 | 5 | 4 | 3.67 | 0 4 4 5 5 4 (same) |
| both | E0 | | | | | | | 1.00 | 0.92 |
| both | E18 | | | | | | | 2.75 | 2.50 |
| both | E20 | | | | | | | **4.83** | 4.33 |

- **30 of 36 episodes are the groupmate's own rows to the action:** the same
  levels and the same action for the first and the second level clear. That is
  every lp85 episode and ft09 seeds 2 to 5. Their rows are
  `results/runs_E0.csv`, `runs_E18.csv` and `runs_E20.csv` in their hand-in;
  they equal the plan's Appendix D.
- **The six that differ are ft09 seeds 0 and 1, and the difference is E0's
  level 1.** On seed 0 it clears at action 542 here and 768 there. On seed 1 it
  clears at action 1,180 here, and there it never cleared. E0 is a small CNN
  that trains as it plays, so another machine changes the last bits of its
  numbers and, sooner or later, a click. The plan expected this.
- **From the level-up on, the planner does exactly what it does in the report.**
  E20 takes 23 actions for ft09's level 2 on all six seeds here and on all five
  there. So the two extra ft09 episodes add levels (E18 3, E20 6), and the means
  come out above the report's.

## 2. Recordings play again on the engine

**What was run.** Each recording's first 5,000 actions, re-driven on the local
engine with the run's own game seed: `env.reset()`, then each recorded move,
and a RESET wherever the action number jumps by 2 (a RESET is an action but not
a row). Before every move the engine's screen must be the recorded one, and
after it the recorded next one.

**The recordings** (every unguided arm recorded since 6 Oct, dev and seen games
only): upgrade screen 3's nine arms (seed 0); the adopted Goose in G4 (seeds 0
to 2), dev sweep 2 (seeds 3 to 6) and the carry test (seeds 7 to 10).

| Game | Recordings | Moves replayed | Moves matched |
|---|---|---|---|
| cd82 | 20 | 99,012 | 99,012 |
| dc22 | 20 | 99,234 | 99,234 |
| ft09 | 20 | 97,717 | 97,717 |
| g50t | 16 | 79,385 | 79,385 |
| lp85 | 16 | 78,955 | 78,955 |
| ls20 | 12 | 59,414 | 59,414 |
| m0r0 | 20 | 99,356 | 99,356 |
| tr87 | 16 | 79,384 | 79,384 |
| tu93 | 20 | 96,044 | 96,044 |
| vc33 | 20 | 98,372 | 98,372 |
| **All ten** | **180** | **886,873** | **886,873** |

**Every move matches, in 180 of 180 recordings.** A recorded run can stand in
for a live one on the CPU. The offline tools of Phase 2 rest on this.

## 3. What a level-completing move returns

The reference takes the solved board from the second-to-last frame the
level-completing move returns. It can use that frame only if its objects sit on
the level's slots (the centres of the objects on the level's first screen).
Counted over every level-up in the replayed recordings:

| Game | Level-ups seen | Frames the move returned | Second-to-last frame fits the old level's slots |
|---|---|---|---|
| **ft09** | 24 | 2 | **24 of 24** |
| **lp85** | 27 | 2 | **27 of 27** |
| cd82 | 18 | 16 | 0 of 18 |
| dc22 | 9 | 2 | 2 of 9 |
| m0r0 | 20 | 2 | 0 of 20 |
| vc33 | 42 | 2 | 0 of 42 |
| tu93 (no clicks) | 52 | 9 or 14 | 52 of 52 |
| g50t (no clicks) | 5 | 9 | 0 of 5 |
| ls20 (no clicks) | 9 | 2 | 0 of 9 |
| tr87 (no clicks) | 1 | 37 | 0 of 1 |

- **On ft09 and lp85 the move returns two frames: the solved board, then the
  next level's first screen.** The solved board fits the slots every time.
- **On the other four click games it almost never does** (2 of 89 level-ups,
  both on dc22). With no usable solved board the reference learns no goal rule
  there, so it will not plan. This says, before Phase 2 measures it, that the
  planner as received stays off on cd82, dc22, m0r0 and vc33.
- **The setting `win_frame = "last_valid"` has little to work with.** On dc22,
  m0r0 and vc33 the move returns only those two frames. On cd82 none of the 15
  earlier frames fits either.
- tu93's solved boards fit, but tu93 offers no click, so the planner is dormant
  there.

Why the frame does not fit (an object that appears, vanishes or leaves its
slot at the winning screen, or a counter) is Phase 2's question.

## 4. The go/no-go: the planner after this project's Goose

**What was run.** The reference's E20 in its own `run_episode`, with one
replacement made at run time: its explorer hands out the recording's next move,
after checking that the engine's screen is the recorded one. So the recording
plays level 1. From the first level-up the planner plays alone. If it asks its
explorer for a move, the episode stops ("gave up"). 5,000 actions in all, the
prefix included.

"Level 1 at action" is the number of the level-completing move. "Actions for
each planned level" runs from one level-up to the next, RESETs included.

### ft09: 20 prefixes

| Recording | Arm | Seed | Level 1 at action | Levels by action 5,000: the planner | ... the recording's Goose | Actions for each planned level | How it stopped |
|---|---|---|---|---|---|---|---|
| screen 3 | **SOT** | 0 | 1,381 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | O | 0 | 688 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | SO | 0 | 756 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | OT | 0 | 1,208 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | S | 0 | 917 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | ST | 0 | 431 | 6 | 2 | 23 37 34 47 37 | won the game |
| screen 3 | T | 0 | 263 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | base | 0 | 1,659 | 6 | 1 | 23 37 34 47 37 | won the game |
| screen 3 | base2 | 0 | 803 | 6 | 2 | 23 37 34 47 37 | won the game |
| G4 | A0-fresh | 0 | 1,234 | 6 | 1 | 23 37 34 47 37 | won the game |
| G4 | A0-fresh | 1 | 167 | 6 | 1 | 23 37 34 47 37 | won the game |
| G4 | A0-fresh | 2 | 979 | 6 | 1 | 23 37 34 47 37 | won the game |
| dev sweep 2 | A0-fresh | 3 | 1,115 | 6 | 1 | 23 37 34 47 37 | won the game |
| dev sweep 2 | A0-fresh | 4 | 1,298 | 6 | 1 | 23 37 34 47 37 | won the game |
| dev sweep 2 | A0-fresh | 5 | 2,025 | 6 | 1 | 23 37 34 47 37 | won the game |
| dev sweep 2 | A0-fresh | 6 | 1,069 | 6 | 2 | 23 37 34 47 37 | won the game |
| carry test | A5 | 7 | 514 | 6 | 1 | 23 37 34 47 37 | won the game |
| carry test | A5 | 8 | 796 | 6 | 2 | 23 37 34 47 37 | won the game |
| carry test | A5 | 9 | 1,156 | 6 | 1 | 23 37 34 47 37 | won the game |
| carry test | A5 | 10 | 543 | 6 | 1 | 23 37 34 47 37 | won the game |

### lp85: 15 prefixes

| Recording | Arm | Seed | Level 1 at action | Levels by action 5,000: the planner | ... the recording's Goose | Actions for each planned level | How it stopped |
|---|---|---|---|---|---|---|---|
| screen 3 | **SOT** | 0 | 170 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| screen 3 | O | 0 | 657 | 4 | 2 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| screen 3 | SO | 0 | 717 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| screen 3 | OT | 0 | 170 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| screen 3 | S | 0 | 438 | 4 | 2 | 27 73 165 | gave up on level 5: 41 wrong predictions |
| screen 3 | ST | 0 | 173 | 4 | 3 | 27 73 165 | gave up on level 5: 41 wrong predictions |
| screen 3 | T | 0 | 173 | 4 | 5 | 27 73 165 | gave up on level 5: 41 wrong predictions |
| screen 3 | base | 0 | 438 | 4 | 2 | 27 73 165 | gave up on level 5: 41 wrong predictions |
| screen 3 | base2 | 0 | 438 | 4 | 1 | 27 73 165 | gave up on level 5: 41 wrong predictions |
| G4 | A0-fresh | 0 | 701 | 4 | 2 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| G4 | A0-fresh | 1 | 720 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| G4 | A0-fresh | 2 | 1,055 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| carry test | A5 | 8 | 620 | 4 | 2 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| carry test | A5 | 9 | 434 | 4 | 1 | 104 45 227 | gave up on level 5: 41 wrong predictions |
| carry test | A5 | 10 | 724 | 4 | 2 | 104 45 227 | gave up on level 5: 41 wrong predictions |

Not a prefix: the carry test's seed 7 on lp85 cleared no level inside 5,000
actions. No recording failed to replay.

- **ft09: PASS.** Level 5 or more from 20 of 20 prefixes (the bar: 80%), and
  from 4 of the 4 played with `objclick` (the bar: 3).
- **lp85: PASS.** Level 4 or more from 15 of 15 prefixes (the bar: 60%), and
  from 4 of the 4 played with `objclick` (the bar: 2).

### What it shows

- **On ft09 the planner won the game from every prefix.** Levels 2 to 6 took
  23, 37, 34, 47 and 37 actions, 178 in all, the same in all 20. The first four
  are the report's numbers. Goose needed 167 to 2,025 actions for level 1, so
  the game was won by action 345 to 2,203. In the same recordings Goose alone
  had 1 level by action 5,000 in 16 runs and 2 in 4.
- **How Goose played level 1 made no difference on ft09.** The planner learned
  the same rule from all 20 ("every multi-colour pattern describes its
  neighbourhood"; in `SOT`'s prefix confirmed on 5 of 5 levels, with 30 other
  rules rejected, as in the report) and probed the same 14 objects on level 2.
- **On lp85 it reached level 4 from every prefix and gave up on level 5,**
  always at the reference's cap of 40 wrong predictions.
- **On lp85 the play of level 1 does matter.** In 10 prefixes the planner took
  104, 45 and 227 actions for levels 2 to 4, the report's own numbers (its
  "226" leaves out one RESET). In the other 5, screen 3's arms without
  `objclick`, it took 27, 73 and 165. There level 1 had shown 3 of level 2's
  objects doing something, not 6, so there was less to probe, and one rule
  planned all three levels where the others changed rule after level 2.
- **The planner alone is not always ahead on lp85.** By action 5,000 the
  recorded Goose had 1 or 2 levels in 13 of the 15 runs, and 3 and 5 in the
  other two (`ST` and `T`). This driver stops when the planner gives up; in the
  real agent Goose takes the level back. lp85 stays the least certain number
  in the plan's expectation.
- An episode took 27 to 98 seconds.

## 5. The games with no click action

From the engine probe's table of 10 Nov
(`results/glassbox/engine_probe/all25_2000_random_moves.md`, column "Actions
offered"): **g50t, ls20, re86, tr87, tu93 and wa30** never offer ACTION6. The
planner is dormant on these six and can act on at most the other 19. They are
the same six on which screen 3's `objclick` arms never sampled a click.

## Four things the plan has wrong or out of date

1. **The pandas stand-in (plan 4.3) must not stay in place.** The plan says to
   register an empty stand-in module before importing the reference. Left
   there, it stops torch the first time an optimizer is built: torch looks
   pandas up by name and fails with `pandas.__spec__ is None`. The driver gives
   the stand-in a spec and removes it as soon as the three modules are loaded.
   The bridge must do the same, or Goose would stop at its first level.
2. **lp85 has 15 prefixes, not 16** (the carry test's seed 7).
3. **Action numbers.** Here a level-up is numbered by the move that completed
   the level. Upgrade screen 3's table, and the plan's Appendix B.4, give one
   more, the action at which the new level is first seen: ft09 `SOT` 1,381
   here, 1,382 there.
4. **Section 4.7 and Appendix C say E21 to E23 were not received.** They are in
   the same hand-in (`Final_Working_Results_Nov16.zip`), with E24, the
   notebook, the report and the result CSVs. The scope stays E20.

## What it means for the next phases

- **Phase 1 goes ahead:** the bridge and the hook.
- **Expect Phase 2's funnel to show the planner engaging on ft09 and lp85
  only,** among the dev and seen games. Section 3 says why: elsewhere the
  solved board is not on the level's slots. That is the plan's own "likeliest
  outcome".

## Run it again

```bash
uv run python experiments/kp/run_reference.py repro          # about 8 minutes with 6 at a time
uv run python experiments/kp/run_reference.py replay-check   # about 2 minutes with 8 at a time
uv run python experiments/kp/run_reference.py prefix --games ft09,lp85   # about 5 minutes with 8 at a time
```

Every episode runs in its own process with `PYTHONHASHSEED=0`, one torch
thread and no GPU, so the tables come out the same each time. Output:
`results/kp/phase0/` (`repro`, `replay_check`, `level_ups`, `prefix`: a `.tsv`
and a `.md` each; `prefix/<recording>/` holds the reference's own store and
logs for every prefix).
