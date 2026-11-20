# Knowledge Planner, Phase 1: the bridge and the hook (19 Nov 2026)

**In one line.** The groupmate's planner now runs inside Goose behind
`EVAL_KP=1`, with its files unedited. Switched off, it changes nothing: both
standard agents are identical on every move on the CPU. Switched on, it won
ft09 on the CPU in 965 actions (the adopted Goose alone had 1 or 2 levels by
action 5,000), and the reference in its own harness makes the same 178 planned
clicks at the same steps. The GPU smoke is the one part not run yet.

**Status: checks, not tests.** CPU only so far. Nothing is committed.

The plan: `kp-plan.md`, Phase 1. Recordings and scripts of the CPU checks:
`results/cpu_check/2026-11-19_kp/` (its `RESULT.md` has the commands).

| Check | The bar, from the plan | Result |
|---|---|---|
| 1.2 (a) Off: `goose` on ft09, and the adopted Goose's flags on tu93, before and after the change | identical on every move | **PASS.** 1,970 of 1,970 and 1,960 of 1,960 |
| 1.2 (b) Dormant: `SOT` with `EVAL_KP=1` on tu93 (no clicks) against `SOT` | identical on every move | **PASS.** 490 of 490 |
| 1.2 (c) Passive: `SOT` with `EVAL_KP=1 EVAL_KP_PLAN=0` on cd82 against `SOT` | identical on every move | **PASS.** 495 of 495 |
| 1.3 End to end with planning: the adopted Goose's flags with `EVAL_KP=1` on ft09, 5,000 actions | *expected:* the game won, levels 2 to 6 by planning, each in under 150 actions | **As expected.** Won at action 965; levels 2 to 6 in 23, 37, 34, 47 and 37 actions |
| 1.4 The bridge against the reference, on 1.3's recording | the same planned clicks at the same steps, and the same count | **PASS.** 957 of 957 moves, 178 of 178 planned clicks |
| 1.5 GPU smoke | | not run: it needs Matt's word |
| 1.6 The bridge against the reference on the smoke's recordings | | after the smoke |
| The repo | `make check`; the live tests | **PASS.** 352 tests pass (34 of them new) |

## What was built

| File | What it is |
|---|---|
| `custom_agents/kp_ref/` | The groupmate's four files, byte for byte (Phase 0) |
| `custom_agents/kp_bridge.py` | The bridge: it hands the reference each screen, takes its planned clicks, and tells it which move Goose made |
| `custom_agents/action.py` | 18 lines tagged `# [kp]`, removable with `sed -i '/\[kp\]/d'` |
| `experiments/kp/kp.toml` | The settings. As committed: the reference's behaviour |
| `check_repo.py` | Two additions tagged `# [kp]`: `kp_bridge` is a local module, and the baseline may import it only on tagged lines |
| `tests/kp_games.py`, `tests/test_kp_bridge.py` | Toggle, a made-up game of ft09's kind, and 18 tests of the bridge |
| `experiments/kp/kp_test.py`, `kp_verdict.py` | The launcher of the smoke, the dev sweep and the test, and the test's rule (16 tests on made-up runs) |
| `experiments/kp/run_reference.py`, `tools/kp_offline.py` | The CPU tools of Phases 0 and 2 |

**How the bridge works.** The reference's files are loaded as they are, and
three of its functions are replaced on the loaded module:

1. Its explorer (its own small Goose) answers "this move is Goose's".
2. Its per-level model builder does nothing. It built a CNN nobody uses, and
   it reseeded torch's global random generator, which Goose itself draws from.
3. Its screen parser is put behind a memory of the screens already parsed.

With those in place the reference draws no random number. One step: Goose
samples its move and every other override runs; then the bridge asks the
reference. If it answers with a click (a probe, a verify press or a plan
step), that click is sent. Otherwise Goose's move stands, and if it was a
click the reference is told, the way it books its own explorer's clicks.

Until a game first offers a click the reference is not called at all. An error
anywhere inside it switches the planner off for the rest of the run and is
counted; Goose plays on.

## The checks, one by one

**1.2 (a) Switched off, nothing changes.** `tools/cpu_check.py`, 2,000
actions, seed 0, 4 threads, before the change and after it:

| Agent | Game | Result |
|---|---|---|
| `goose`, no flags | ft09 | IDENTICAL on every move: 1,970 of 1,970 |
| the adopted Goose's flags | tu93 | IDENTICAL on every move: 1,960 of 1,960 |

Today's "before" recordings also equal the ones the screen 3 check made on 16
Nov with the same code, so the machine has not changed under the comparison.

**1.2 (b) and (c) Under `SOT`'s flags the planner leaves Goose's own moves
alone.** 500 actions each, the same tool. The baselines were recorded before
the hook existed.

| Run | Against | Result | What the planner did meanwhile |
|---|---|---|---|
| `SOT` with `EVAL_KP=1` on tu93 | `SOT` on tu93 | IDENTICAL on every move: 490 of 490 | Nothing: tu93 offers no click, and the reference was never called |
| `SOT` with `EVAL_KP=1 EVAL_KP_PLAN=0` on cd82 | `SOT` on cd82 | IDENTICAL on every move: 495 of 495 | It watched 496 screens, parsed 363 of them, and went through a level clear at action 9, which is where the reference as received would have reseeded torch |

(c) is the proof that the bridge leaves `SOT`'s random numbers alone. On cd82
the planner learned nothing it could use (33 of 496 screens fit the slots), as
Phase 2 found for that game.

**1.3 End to end on the CPU.** The adopted Goose's flags with `EVAL_KP=1` on
ft09, seed 0. The game was won at action 965, in 10 minutes.

| Level | Cleared by | Actions | Probes | Plan steps |
|---|---|---|---|---|
| 1 | Goose | 787 | | |
| 2 | planning | 23 | 17 | 6 |
| 3 | planning | 37 | 28 | 9 |
| 4 | planning | 34 | 23 | 11 |
| 5 | planning | 47 | 39 | 8 |
| 6 | planning | 37 | 28 | 9 |

- No error inside the planner, no fallback to Goose, 2 wrong predictions.
- All 958 screens sat on their level's slots.
- The rule learned from level 1 ("every multi-colour pattern describes its
  neighbourhood") was confirmed again at each of the four later clears the
  planner saw.
- The reference took 11.2 seconds of the run's 597: 306 screens parsed, 4,066
  parses asked for.

**1.4 The bridge makes the reference's moves.** The recording of 1.3 was
replayed with the reference in its own harness and the recording standing in
for its explorer for the whole run. Every one of the 957 recorded moves came
out the same, the reference chose exactly the 178 moves the bridge reported as
planned, and it asked for the run's last move (the winning click, which no
recording holds) itself.

Two more checks of the same thing, not in the plan:

- **On 36 offline episodes** (Phase 2): the takeover run through the bridge
  gives the rows Phase 0 got with the reference in its own harness, column for
  column, on all 20 ft09 and 16 lp85 recordings.
- **On Toggle**, in the tests: the bridge and the reference's own
  `run_episode` send the same clicks, move for move, through three levels and
  two game overs.

**The launcher, tried without the GPU.** `TRIAL=60` runs every command of
`kp_test.py` on 60-action runs, and with `CUDA_VISIBLE_DEVICES=` they run on
the CPU. The smoke played and scored its six runs and printed its checks. The
dev sweep's 28 runs were started detached, paused after 6, resumed, and
finished; `verdict` wrote its tables from them.

## Where the build departs from the plan

1. **The pandas stand-in is removed once the reference is loaded** (Phase 0's
   finding). Left in place it stops torch at the first optimizer.
2. **`step()` takes Goose's action counter as a fifth argument.** The plan's
   rule "drop a pending click that is more than one step old" needs to know
   that a move went by unseen.
3. **`kp.toml` holds three settings: `plan`, `parse_cache` and `mask`.** The
   plan's `win_frame` and `try_unconfirmed` are not built: Phase 2 found no
   level that calls for either (`kp-offline-dev.md`, section 5). `mask =
   "agent"` is built, was tried, and is not adopted.
4. **A run records more than the plan lists.** `kp_stats.json` has one record
   per level: screens, how many fit the slots, clicks by kind, why the planner
   left the level to Goose, how the level ended, and what the planner learned
   at its clear. The funnel is read from it.
5. **The test's rule is in `kp_verdict.py`, not in `kp_test.py`.** `kp_test.py`
   is among the files a resume checks, so a fix to a table would otherwise
   block a paused test. `kp_test.py verdict` calls it.
6. **The offline tools drive the bridge, not the reference in its own
   harness.** They measure the component the test runs, and the 36 episodes
   above show the two agree.
7. **The dev sweep may start from uncommitted code** (it says so in its
   `sweep.json`, with a hash of the code). The test may not.

## The GPU smoke: what it will run

`uv run python experiments/kp/kp_test.py smoke`: ft09, lp85 and tu93 x seed 14
x `SOT` and `SOT-KP` x 5,000 actions. Six runs, 4 at a time, about 15 minutes.
Then its checks: every run played to its cap or a win; `SOT`'s flags in every
`run_config.json`; a training step on nearly every action; no error inside the
planner; an `SOT-KP` run at most 1.5 times as slow per action as its `SOT`
run; and the same speed on tu93, where the planner is dormant.
