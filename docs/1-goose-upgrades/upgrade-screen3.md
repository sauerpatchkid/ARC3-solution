# Upgrade screen 3: three Goose upgrades, alone and combined (16-17 Nov 2026)

**In one line.** Three new Goose upgrades, each behind a switch that is off by
default: a bandit over map schedules (S), clicks chosen by shape (O), and more
training on fresher data (T). A 9-arm screen asks whether any of them, alone or
combined, lets the adopted Goose complete more levels in 5,000 actions.

**Status.** Built and checked on 16 Nov (the checks are below). The sweep ran
from 16 Nov 23:50 to 17 Nov 04:46 on commit `eefcde8`, the commit that holds
this page as it was before the first run: 225 of 225 runs, none failed.
**Result: `SOT` ranked first (22 levels against 15 and 15) and clears the
noise by the rule below; so do `ST` and `O`.** Exploratory: one seed, 5,000
actions a game. Nothing is adopted from it: no new preset, no changed default.
Results: [`upgrade-screen3-results.md`](upgrade-screen3-results.md).

*This is Matt's overnight job of 16 Nov ("upgrade screen 3"). He was asleep
while it ran, so every choice his brief left open is listed under "Decisions
made without Matt".*

## The problem

The adopted Goose (A0, `mb_gated_att`) is slow at the start of a game. Over 25
games x 3 seeds it completed 25 and 27 levels by 2,000 actions and 62 and 61 by
10,000 in its two sweeps, against 112 by 100,000. Three things in it are slow by
design, and each upgrade goes at one of them. The reasons are the brief's.

| | The problem | The upgrade |
|---|---|---|
| **S** | `map_gated` only chooses "walk back or stay" after a game over, and the map only steps in after 200 decisions with no new screen. That is slow at 5,000 actions. The strongest no-LLM explorers search the state graph systematically and reach first levels sooner | `map_sched`: a bandit over five map schedules |
| **O** | Goose picks a click cell in proportion to the click head's output, so a large background takes most clicks and small objects are found slowly. Blind Squirrel and Rudakov's graph explorer click shapes rather than cells | `objclick`: clicks chosen by shape |
| **T** | Goose takes one gradient step per 5 actions on a uniform batch, so with a couple of thousand stored moves a new one waits about 150 actions for its first update. Goose already resets its network at each level, which is what keeps a high update rate safe (the BBF result) | `EVAL_TRAIN_EVERY`, `EVAL_TRAIN_RECENT`: train more, on fresher data |

## The three upgrades

All three: off by default; with the new switches off, `goose` and
`mb_gated_att` take exactly the same moves as before (check 2); unit tests in
`tests/`; what each did is written to the run's stats files.

### S = `map_sched` (`custom_agents/upgrades.py`, class `UpgradedReturnMap`)

An option in `EVAL_UPGRADES` that takes `map_gated`'s place (the two together
are refused). `return_map.py` is not edited. One sliding-window UCB bandit, kept
per game, picks one of five schedules:

| Schedule | Stall | Walk back after a game over | Untried button first |
|---|---:|---|---|
| `stay200` | 200 | no | no (what `map_gated` calls "stay") |
| `walk200` | 200 | up to 50% of a typical attempt | no (what `map_gated` calls "walk") |
| `stay30` | 30 | no | no |
| `walk30` | 30 | up to 50% | no |
| `sweep30` | 30 | up to 80% | yes |

- **Stall**: decisions without a new screen before the map routes to the
  nearest frontier.
- **Walk back**: the share is the map's `return_limit()`; "no" clears the
  pending walk back, as the gate does.
- **Untried button first**: when the map has no move of its own (no route step,
  no walk back, no stall route) and the current screen has a button never
  pressed from it, one of those is pressed, at random, in place of Goose's
  pick. No move is ever forbidden: once every button has been pressed there,
  Goose picks as usual.
- **When it chooses**: at every game over, and after 100 decisions without one.
  The schedule that just ran is credited with new screens per decision over
  that stretch (the gate's measure). It starts on `stay200`, tries each
  schedule twice, then follows UCB with the gate's formula and a window of 50.
- The route-step check is unchanged (it is what catches the hidden state on
  g50t). Glassbox's `GbMap` subclasses this class and is unchanged.
- **Stats** (`return_map_stats.json`): pulls, decisions, new screens and mean
  reward per schedule; untried-first presses.

### O = `objclick` (`custom_agents/upgrades.py`, class `ObjClick`)

An option in `EVAL_UPGRADES`. It acts only when Goose's own draw is a click, so
the split between buttons and clicks is untouched, and it acts before
`deadclick`.

- The screen's shapes are its 4-connected patches of one colour
  (`gridtools.label_components`). Cells under the decoration mask count
  together as one shape. Labels are kept for the last 256 screens.
- With probability 0.9 the cell is drawn again with weight = its click
  probability / the size of its shape. That is: a shape in proportion to its
  mean score, then a cell inside it in proportion to its own score. With
  probability 0.1 Goose's cell is kept, so every cell stays reachable.
- It draws from its own generator, seeded from the agent's seed, so Goose's
  random stream is untouched. No fixed click order. If the weights are not
  finite or sum to 0, Goose's cell is kept.
- It is not the removed `map_objects`: that changed what the map called
  untried; this changes only where a click lands.
- **Stats** (`upgrades_stats.json`): clicks redrawn, clicks kept, mean shapes
  on a redrawn screen.

### T = train more, on fresher data (`custom_agents/train_more.py`; lines tagged `# [train]` in `action.py`)

| Flag | Default | What it does | In the T arms |
|---|---:|---|---:|
| `EVAL_TRAIN_EVERY` | 5 | train every N actions | 1 |
| `EVAL_TRAIN_RECENT` | 0 | this many of the newest stored moves go into every batch of 64; the rest are drawn uniformly from the others | 16 |

- 10 added lines in `action.py`, none edited: `sed -i '/\[train\]/d'
  custom_agents/action.py` restores the file byte for byte.
- At the defaults the agent holds nothing from the module and trains as
  before, with the same `np.random` calls in the same order.
- Both values go to `run_config.json`; the training steps taken go to
  `train_stats.json`.
- Mixed precision, fused Adam and positive over-sampling are left out: they
  would blur what caused a difference.

## The arms

Every run: `run_local.py --agent goose --offline` with `EVAL_LABEL=novel
EVAL_RETURN_MAP=1 EVAL_SEED=0 EVAL_MAX_ACTIONS=5000 PYTHONHASHSEED=0`, plus:

| Arm | `EVAL_UPGRADES` | Training flags |
|---|---|---|
| `base` | `bars,map_gated,attempt` | none (A0 by flags, as `sweep.sh`'s arm A5) |
| `S` | `bars,map_sched,attempt` | none |
| `O` | `bars,map_gated,attempt,objclick` | none |
| `T` | `bars,map_gated,attempt` | `EVAL_TRAIN_EVERY=1 EVAL_TRAIN_RECENT=16` |
| `SO` | `bars,map_sched,attempt,objclick` | none |
| `ST` | `bars,map_sched,attempt` | as `T` |
| `OT` | `bars,map_gated,attempt,objclick` | as `T` |
| `SOT` | `bars,map_sched,attempt,objclick` | as `T` |
| `base2` | as `base` | none (the baseline again, to measure chance) |

9 arms x 25 games (`benchmark.ALL_GAMES`) x seed 0 x 5,000 actions = **225
runs**, 4 at a time. `--agent goose` with flags, not the preset: a preset
refuses other flags. Games are the outer loop with the four tiers of `tiers.py`
taken in turn, so a sweep that is cut short still covers every tier; arms run
inside a game in the order `base S O T SOT SO ST OT base2`.

All 25 games are used outside a confirm, held-out tiers included, at Matt's
request: pre-registration section 11, entry 10, dated 16 Nov
(`docs/3-glassbox/glassbox-prereg.md`). Results are reported by tier.

## The checks before any sweep run

| # | Check | Result |
|---|---|---|
| 1 | "Before" CPU runs started before any edit under `custom_agents/` (2,000 actions, 4 threads): `goose` on ft09, the adopted Goose's three flags on tu93 | started 16 Nov 23:04, on commit `dc022d2`; both had printed their `[run_local] agent=` line before the first edit |
| 2 | The same two runs on the finished code, compared move for move (`tools/cpu_check.py compare`) | **IDENTICAL on every move**: `goose`/ft09 1,970 of 1,970; the adopted Goose's flags/tu93 1,960 of 1,960 (39 game overs, so the gate chose 39 times). Recordings: `results/cpu_check/2026-11-16_screen3/` |
| 3 | `uv run python -m pytest tests -q` and `make check` | 318 passed (284 before, + 34 new: 18 in `test_upgrades.py`, 6 in `test_train_more.py`, 10 in `test_screen3.py`), 1 expected failure as before; `make check`: all checks passed |
| 4a | Smoke, GPU: arm `SOT` on all 25 games, 300 actions, `--offline` | **PASS**: 25 of 25 games loaded with `--offline` and ran without a crash |
| 4b | Smoke, GPU: all eight settings on ft09 and tu93, 1,000 actions: each upgrade acted, speed per arm | **PASS**: 16 of 16 runs; every upgrade acted where it should and nowhere else (table below) |

Smoke run: `results/screen3_smoke/20261116_234125/` (41 runs, 6.9 minutes, 4 at
a time; not part of the sweep). The GPU was free (2.6 GB of 32.6 GB in use, 0%
busy) and the disk had 889 GB free. Part 4b, ft09 | tu93:

| Arm | Mean ms per action | act/s | `wall_ms_med` | Clicks redrawn | Schedules used | Untried-first presses | Training steps |
|---|---:|---|---|---|---|---|---|
| `base` | 25.6 | 46.6 \| 27.1 | 9.6 \| 24.0 | - | 0 \| 0 | - | - |
| `S` | 21.9 | 46.9 \| 33.9 | 9.0 \| 19.0 | - | 5 \| 5 | 0 \| 62 | - |
| `O` | 20.6 | 47.4 \| 35.0 | 9.1 \| 18.4 | 902 \| 0 | 0 \| 0 | - | - |
| `T` | 83.0 | 12.8 \| 12.1 | 88.6 \| 89.5 | - | 0 \| 0 | - | 798 \| 891 |
| `SO` | 34.2 | 31.8 \| 24.3 | 11.2 \| 24.6 | 896 \| 0 | 5 \| 5 | 0 \| 77 | - |
| `ST` | 91.1 | 11.0 \| 11.5 | 96.8 \| 89.0 | - | 5 \| 5 | 0 \| 80 | 854 \| 891 |
| `OT` | 69.1 | 12.4 \| 17.2 | 84.7 \| 28.9 | 889 \| 0 | 0 \| 0 | - | 912 \| 891 |
| `SOT` | 96.0 | 10.7 \| 11.2 | 97.9 \| 89.6 | 889 \| 0 | 5 \| 5 | 0 \| 30 | 912 \| 891 |

- **O** redrew 889 to 902 of 1,000 moves on ft09 (a click-only game) and none
  on tu93 (buttons only). It is not slower than `base` on ft09 (47.4 against
  46.6 act/s); the labelling costs 0.2 to 0.3 ms a click, measured alone.
- **S** used all five schedules on both games. It pressed untried buttons
  first 30 to 80 times on tu93 and never on ft09, which has no buttons.
- **T** took 798 to 912 training steps in 1,000 actions, where an arm without
  it takes 200 at most. The median move of the T runs took 6 times as long as
  the median move of the others (89 ms against 15 ms).
- **Speeds are with 4 runs sharing the GPU.** A T arm alone takes about 29 ms
  a move (`OT` on tu93 ran last, with nothing beside it); with three other
  runs it takes 70 to 96 ms. In part 4a, four `SOT` runs at a time took 72 ms a
  move on average.
- **One check was restated after the smoke run, on the same runs.** As first
  written it compared each T run with `base` on the same game and printed FAIL
  for `OT` on tu93 (28.9 ms against `base`'s 24.0 ms): that run was the last
  in the queue and had the GPU to itself. The check now compares the median of
  the T runs with the median of the others, and counts training steps, which
  is the direct evidence. No run was repeated: `screen3.py smoke --evaluate
  results/screen3_smoke/20261116_234125` prints the table again.

## The rule (written before the first sweep run)

- An arm's **score** is the levels completed by 5,000 actions, summed over the
  25 games (the engine's count, as `compute_metrics.py` reports it).
- Arms are **ranked** by score. Ties go to more levels by 2,000 actions, then
  to more games better than `base`.
- An arm **clears the noise** only if its score is above both `base` and
  `base2`, and the 95% bootstrap interval for its difference from `base`
  (games resampled, as `tools/paired_compare.py` does) is above 0. Otherwise it
  is **inside the noise**, whatever its rank.
- The **main effect** of an upgrade is the mean score of the four arms that
  have it minus the mean of the four that do not (`base2` left out), with the
  same kind of interval.
- The **best combo** is the top-ranked of the seven upgraded arms, stated with
  whether it clears the noise and whether the main effects agree with it.
- **Sanity anchor.** One seed at 5,000 actions should put `base` near 15
  levels (from A0's 25 and 27 by 2,000 and 62 and 61 by 10,000 over three
  seeds). If `base` is under 8 or over 24, the setup is checked (engine mode,
  the flags in `run_config.json`) and the results page says so at the top.
- No upgrade is changed after sweep results are seen, and no run is rerun to
  get a better number. If a bug makes an upgrade's runs invalid, it is fixed
  and all of that upgrade's arms are rerun if time allows; otherwise those
  arms are reported as invalid.

It is a screen, not a confirm: whatever ranks first is a candidate for a
registered confirm with more seeds, not a result.

## Running it

`experiments/goose/screen3.py` holds the run set and runs it detached:

```bash
uv run python experiments/goose/screen3.py plan      # the arms, their flags, the run count, a time estimate
uv run python experiments/goose/screen3.py smoke     # checks 4a and 4b (results/screen3_smoke/)
uv run python experiments/goose/screen3.py start     # begin, detached; refuses uncommitted code
uv run python experiments/goose/screen3.py status    # runs done, runs in progress, time left
uv run python experiments/goose/screen3.py pause     # no new run starts; the runs in progress finish
uv run python experiments/goose/screen3.py resume    # continue after a pause, a crash or a reboot
uv run python experiments/goose/screen3.py report    # the tables, also on a partial manifest
```

Output: `results/screen3/<stamp>/`. A run enters `manifest.tsv` only after it
has been played to its cap (or to the end of the game) and scored, so a pause
loses nothing and a crash loses only the runs in progress. A failed run is
logged in `failed.tsv` and skipped; when the queue is empty the failed runs are
tried once more and the tables are written
(`experiments/goose/screen3_report.py`: `report.md`, `runs.csv`). `resume`
refuses to run if the arms or the code the runs depend on have changed since
the start. `TRIAL=1` does all of this on a tiny set under
`results/screen3_trial/`.

## Time

- The job started on 16 Nov at 23:04 PST; the results page is due 10 hours
  later, on 17 Nov at 09:04.
- The sweep starts on 16 Nov a few minutes before midnight, under an hour into
  the job (the aim was within 2.5 hours).
- **Projection: about 4.1 hours** (`screen3.py plan`, from the smoke run's
  speeds), so the last run should be in between 04:00 and 05:00. That is well
  before 08:34, 30 minutes before the deadline, so **the T arms keep
  `EVAL_TRAIN_EVERY=1`**; the fallback to 2 is not used.
- No arm changes once the sweep has started. If the sweep is cut short it is
  still paired by game, and `resume` continues it.

## Decisions made without Matt

Where the brief left a choice open, the option that leaves the existing agents
untouched was taken.

**The upgrades**

1. **T holds nothing at the defaults.** With both flags at today's values the
   agent's `train_more` is `None`, so an unflagged run's `run_config.json` and
   output files are what they were. The T arms write `train_every` and
   `train_recent` to `run_config.json`; `train_frequency`, which was always
   there, shows the value in use.
2. **T's batch uses Goose's own draw.** Goose draws 64 stored moves uniformly,
   in random order. The 16 newest go in, and the batch is filled with the first
   48 of Goose's draw that are not among them. That is a uniform draw from the
   others, and it needs no second random call.
3. **S tries the schedules in turn**, each once and then each again
   (`stay200 walk200 stay30 walk30 sweep30`, twice), not each twice in a row:
   two tries at two different times say more about a schedule than two in a
   row, because new screens are easiest to find at the start of a level.
4. **S, a stretch with no decision in it** (a game over straight after a
   choice) credits nothing and keeps the schedule.
5. **S, a level change or a map restart** (the mask grew) starts the stretch's
   counts again. Two quirks are shared with `map_gated` and left as they are: a
   new level's first screens are all new and are credited to whichever
   schedule is running, and so are the screens found again after a map restart.
6. **S's untried-button press uses the map's random generator**, as the map's
   own "untried button at a frontier" does. An arm with `map_sched` is a
   different agent anyway, so Goose's stream need not be kept.
7. **`map_sched` with `EVAL_GB=1` is refused.** `GbMap` builds its map from
   `gated` and `bars` only and would drop the schedule without saying so.
   `gb/s5_act.py` is not edited.
8. **Two small refactors touch the adopted agent's code, not its behaviour.**
   The gate's UCB formula moved into one function (`ucb_choice`) that the gate
   and the schedules both call, and `Upgrades.key` gets its mask from a
   one-line helper (`live_mask`) that `objclick` also uses. Both are covered by
   check 2 and by a new test that pins the gate's choices on two made-up
   reward streams, recorded from the code before the change.
9. **O decides "keep or redraw" before it labels the screen**, so the 1 click
   in 10 that keeps Goose's cell costs nothing. "Clicks redrawn" counts the
   redraws; "mean shapes" is over the screens it redrew on.
10. **Extra stats.** S also records decisions and new screens per schedule; O
    also records how many redrawn cells differ from Goose's, the screens
    labelled, and the clicks kept because no weight was usable.

**The screen**

11. **Arms are compared on the games every arm has finished.** For a complete
    sweep that is all 25; a partial sweep is still paired by game, and the
    report says which games it used.
12. **The tie-break "games better than `base`"** is counted on levels by 5,000
    actions.
13. **"The main effects agree with the best combo"** means: positive for every
    upgrade the combo has, not positive for every upgrade it lacks.
14. **A run enters the manifest only if it ended at its cap or on a win**
    (`run_end.json`). A run still going after 60 minutes is killed and counted
    as failed.
15. **`start` refuses uncommitted code**, and `resume` refuses if the arms or
    the code the runs depend on changed since the start (a commit that touches
    only docs is fine).
16. **The driver is two files**: `screen3.py` (the runner and its commands) and
    `screen3_report.py` (the tables), one problem each.
17. **The smoke run's speed** is the mean wall-clock time per move after a
    run's first 100 moves. The median the brief names (`wall_ms_med`) is shown
    too, but for an arm that trains on one move in five the median move has no
    training step in it.
18. **The time estimate is `plan`'s**, from the smoke run's speeds.

## Changes after the start

No arm, upgrade, flag, game or rule was changed, and no run was rerun. Notes
written on 17 Nov, after the last run:

1. **The Status paragraph at the top** was rewritten with the result. The rest
   of this page is as it was committed before the first run (`eefcde8`).
2. **The sweep took 4 h 55 min**, not the 4.1 h projected under "Time": in the
   sweep the T arms mostly ran beside other T arms. It ended at 04:46.
3. **Two tables were added to the report** (`screen3_report.py`) after the
   results were seen: what each upgrade adds pair by pair, and each arm
   against `base2`. They describe; the rule above is unchanged.
