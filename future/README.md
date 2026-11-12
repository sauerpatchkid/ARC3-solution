# future/: methods we may add

Nothing in this folder is built. It holds what could come next, so that the
live code (`custom_agents/`) only ever contains what has been tested.

- [`improvement-lists-2026-11-10.md`](improvement-lists-2026-11-10.md): the
  review of 10 Nov 2026. Where the project stands against published work, why
  Glassbox's LLM adds little, and nine ideas in three lists. Every idea below
  points into it.

## This week (to Tue 17 Nov 2026)

Chosen by Matt on 10 Nov and changed by him on 12 Nov (item 4 replaced, item 5
put on hold). Taken one at a time, in this order. GPU runs are asked for one by
one.

| # | Item | From | Needs | Status |
|---|---|---|---|---|
| 1 | Re-score G4 (150 runs) and dev sweep 2 (64 runs) on time-to-level: levels by 2,000, 10,000 and 30,000 actions, and AULC | List 2, item 3a | CPU only | **done 10 Nov**: [`time-to-level.md`](../docs/3-glassbox/time-to-level.md) |
| 2 | Count frozen runs in the 150 G4 recordings | Suggested order, step 1 | CPU only | **done 10 Nov**: [`input-probes.md`](../docs/1-goose-upgrades/input-probes.md) |
| 3 | Probe which games return more than one frame per move, and what ACTION7 does | List 1, item 3 | CPU only (random moves on the engine) | **done 10 Nov**: the same write-up |
| 4 | Goose: carry colour no-ops and goal hints inside Goose. It replaced "a reset watchdog and ACTION7" (Matt, 12 Nov), which items 2 and 3 found little use for | List 1, item 2 | `cpu_check`; about 11 GPU-hours for its test | **built and checked on the CPU 12 Nov; its test waits for Matt's go**: [`carry-inside-goose.md`](../docs/1-goose-upgrades/carry-inside-goose.md) |
| 5 | LLM plays first, Goose takes over: a pilot on the dev games | List 3, item 1 | the instructor's OK (it differs from the 2 Nov idea statement); a model | **on hold** (Matt, 12 Nov) |
| 6 | Owed for the report: G4's GB recordings replayed with the agent's mask; the funnel table; S6 pages for figures | plan §7.4 item 20 | CPU only | **done 12 Nov**: [`funnel.md`](../docs/3-glassbox/funnel.md), and "S5 online" in [`g4-confirm.md`](../docs/3-glassbox/g4-confirm.md) |

**What items 1 to 3 found**, and why item 4 changed:

- **Item 1** (exploratory): counted by action budget, nothing shows early.
  Counted level by level, the guided agent finished levels after the first in
  fewer actions on 36 against 17 in G4, where two unguided arms split 24
  against 27, and the LLM-free arm did as well as the guided one.
- **Items 2 and 3:** no G4 run is frozen (a watchdog firing after 300 frozen
  moves could have cut 179 of 7.39 million moves). ACTION7 is an undo on sk48,
  ar25 and bp35, does nothing on sb26 and su15, and is something else on lf52;
  under random play it reaches fewer screens on five of the six games.
  In-between frames hide something on ft09 (every dead click flashes), sc25
  and cd82 only.

So the watchdog and ACTION7 had little to act on, and the carry had evidence
for it.

## All nine ideas

| Idea | In the review | Work | GPU-hours | This week? |
|---|---|---|---|---|
| ACTION7 (undo) and a reset watchdog | List 1, item 1 | 1 day | 6 to 7 | no: taken off the list on 12 Nov, the probes found little to act on |
| Carry colour no-ops and goal hints inside Goose; keep the network across levels | List 1, item 2 | 1 to 2 days | about 11 | **yes** (item 4): the carry is built, its test is drafted; "keep the network" is not built |
| Animation frames and the ends of bars | List 1, item 3 | 1 to 2 days | 4 to 8 | the probe, done (item 3): frames matter on sc25 at most, and would hurt on ft09 |
| A stronger rule writer through a hosted API | List 2, item 1 | 0.5 to 1 day | none | no (needs an exception to "everything is the local model") |
| An LLM-written, replay-checked goal test | List 2, item 2 | 2 to 3 days | about 12 | no (only worth it after the stronger writer) |
| Time-to-level re-score; the LLM-free arm on the 6 rule games | List 2, item 3 | 0.5 day | 3.5 | **the re-score, done** (item 1); the arm, no |
| LLM plays first, Goose takes over | List 3, item 1 | 2 to 3 days | 4 to 8 | on hold (item 5; Matt, 12 Nov) |
| LLM plans at stalls, checked by the game | List 3, item 2 | 3 to 4 days | about 12 | no |
| A small prior trained on solved-level traces | List 3, item 3 | 1 to 2 weeks | Colab | no (future work) |

## Small fixes

Found while reading the code on 10 Nov. None changes what an agent does.

| Fix | Status |
|---|---|
| Root README rebuilt around the four parts; `custom_agents/README.md` added; this folder added | done 10 Nov |
| CLAUDE.md's status section cut to tables that link to the write-ups | done 10 Nov |
| Glassbox's README and `gb/__init__.py` listed `h2_schedule.py`, which was never built | done 10 Nov |
| `glassbox.toml`: the 35 settings no code reads are now labelled `[unread]` (comments only: the config hash is unchanged) | done 10 Nov |
| `make local` ignored `AGENT` (it always ran `goose`); `make metrics` labelled every row `goose` | done 10 Nov |
| `sweep.sh`'s comments named arms and a make target that were removed | done 10 Nov |
| `tests/conftest.py` in place of the same `sys.path` lines in 26 test files (54 lines gone; 261 tests pass) | done 10 Nov |
| `gb/s5_act.py` built its settings twice with the same expression: one helper, `settings_from()` | done 10 Nov |
| `action.py`: the module docstring sat after the imports and said 6 action outputs (there are 5); `_has_time_elapsed` called seconds "hours" | done 12 Nov, with item 4 (one `cpu_check` covered both: identical on every move) |

## Build notes for the chosen items

What the code looks like today at the places each item touches.

**Item 1, the re-score.** Every scored run has a `metrics.json` with
`levelup_events` (`[[action, level], ...]`, on the actions-taken clock).
`analyze_curves.py` already has `level_at()` and `aulc()`. The runs:
`results/glassbox/g4/online/manifest.tsv` (arms `A0-fresh`, `GB`) and
`results/glassbox/dev2/online/manifest.tsv`. The metric was chosen after the
runs, so the write-up is labelled exploratory.

**Item 4, the carry.** Plan, checks and the drafted test:
[`carry-inside-goose.md`](../docs/1-goose-upgrades/carry-inside-goose.md). The
code is `custom_agents/carry.py`; the arms of its test are `A5` and `A6` in
`sweep.sh`.

**ACTION7 and the watchdog (not built; these notes are kept).**

- `run_local.py` (`ShimFrame`) drops every action the harness enum cannot
  name, so the six games in `tiers.NO_ACTION7` (ar25, bp35, lf52, sb26, sk48,
  su15) play without ACTION7.
- The action index is fixed everywhere: 0 to 4 are ACTION1 to ACTION5 and
  `5 + 64*y + x` is a click. It is in the recordings (`eval_common.py`), the
  scorer, `return_map.py` (`unpack`), `upgrades.py` (dead clicks) and
  `gb/s5_act.py` (`CLICKS = 5`). A sixth button needs an index past the clicks
  (4101), and every "5 or more means a click" test has to learn it.
- Goose's network has 5 button outputs (`ActionModel.num_action_types`). A
  sixth changes the network, so it needs its own flag, off by default, and a
  new preset; `goose` and `mb_gated_att` must pass `tools/cpu_check.py` with
  the flag off.
- The agent sends RESET only after GAME_OVER (`action.py`), and
  `run_local.play` stops a run after more than 10 resets in a row. Two resets
  in a row restart the whole game, so the watchdog must not fire twice running.

**Item 5, LLM plays first (on hold).** The harness is other people's code
([Tufa Labs](https://tufalabs.ai/research/duck-harness/),
[Franzen](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0). Our part
is the relay: cap the LLM per game, save its action list, replay it as a
prefix in `run_local.py`, then let `mb_gated_att` continue to 100,000 actions.
It changes what the LLM does, so it is a reference arm, and the instructor has
to allow it under the 2 Nov idea statement before anything is built.

## How an idea leaves this folder

1. **Before building:** a page here says what changes, behind which flag, and
   how it is tested (games, seeds, the rule for the verdict).
2. **Built:** the code goes where it belongs, off by default. A Goose upgrade
   goes in `custom_agents/` behind its own `EVAL_*` flag and line tag. A
   Glassbox stage goes in `custom_agents/gb/`.
3. **Tested:** the write-up goes in `docs/1-goose-upgrades/` or
   `docs/3-glassbox/`. If it is adopted it gets a new preset. If not, its code
   moves to `legacy/` and it becomes a row under "previously tried methods".
