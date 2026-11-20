# The Knowledge Planner's test (written before its first run)

**The question.** Does the knowledge planner let Goose with the screen 3
upgrades (`SOT`) complete more levels in 5,000 actions?

**Status: written and frozen on 19 Nov 2026, before any run of the test.**
Nothing under "The test" is edited once the first run starts; corrections go
under "Changes after the start". It starts on Matt's word (19 Nov: "commit,
push kp to origin, and start the test").

The planner is a groupmate's agent E20, run unedited behind Goose
(`custom_agents/kp_bridge.py`, `EVAL_KP=1`): `kp-plan.md` has the method,
`kp-phase0.md`, `kp-bridge-check.md` and `kp-offline-dev.md` what was checked
before this page, and `kp-dev-sweep.md` the dev sweep.

## The test

| | |
|---|---|
| Arms | `SOT`: Goose with upgrade screen 3's three upgrades, by flags (`EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_sched,attempt,objclick EVAL_TRAIN_EVERY=1 EVAL_TRAIN_RECENT=16`). `SOT-KP`: the same with `EVAL_KP=1` and `experiments/kp/kp.toml`. `SOT2`: `SOT` again, to measure chance |
| Games | All 25. The held-out tiers are included at Matt's request of 19 Nov, noted in the pre-registration as upgrade screen 3's use of them was (`docs/3-glassbox/glassbox-prereg.md`, section 11, entry 11). Results are reported by tier |
| Seeds | 11 and 12. No run of this project used them before |
| Budget | 5,000 actions a run |
| Runs | 3 arms x 25 games x 2 seeds = 150, 4 at a time, about 5.5 hours, pausable |
| Score | Levels completed by 5,000 actions, summed over an arm's 50 runs (the engine's count, as `compute_metrics.py` reports it) |
| **The rule: `SOT-KP` clears the noise if** | (1) its score is above both `SOT`'s and `SOT2`'s, **and** (2) `SOT-KP - SOT` is above the 97.5th percentile of the chance gap. The chance gap is the difference between two 50-run totals of the same agent: this test's 50 differences `SOT2 - SOT`, one per game and seed, drawn again with replacement, each with a sign at random, and summed, 10,000 times (seed 0). Upgrade screen 3's `base2 - base` is not added to them: it is the adopted Goose's noise |
| Otherwise | The planner is inside the noise at this budget, whatever its total |
| Reported, not part of the rule | By tier. The click games and the six games with no click apart (g50t, ls20, re86, tr87, tu93, wa30: there the two arms are the same agent). Levels by 1,000 and 2,000 actions. The 95% interval of `SOT-KP - SOT` with games resampled (upgrade screen 3's interval). Pairs better, same and worse, and games worse on both seeds, with the same counts for `SOT2` against `SOT`. Level races on levels 2 and up of the click games. Levels cleared by planning and by Goose. Actions for a planned level beside the game's own human baseline. The funnel per game. Errors inside the planner, why it left levels to Goose, run times. `SOT` and `SOT2` beside upgrade screen 3's `SOT`, `base` and `base2`, which are another seed |
| Sanity anchor | Every baseline run's `run_config.json` shows `SOT`'s flags, and its `train_stats.json` a training step on nearly every action. `SOT` and `SOT2` should each land near 44 levels if `SOT` repeats its screen 3 score (22 at one seed, so loosely). Under 28 or over 66: the set-up is checked (engine mode, flags) and the results page says so at the top. A low total with the right flags is a finding about `SOT` on new seeds, not an error, and the planner's comparison stands either way |
| Expectation, written before the runs | **ft09:** 6 and 6 for `SOT-KP` against 1 or 2 each for `SOT`. Every planner run so far won the game (3 on the GPU, 1 on the CPU, and 20 of 20 from recordings). **lp85:** 4 or 5 in a run where Goose clears level 1, and 0 where it does not (it did not at one of the dev sweep's two seeds, in either arm), against 0 to 2 for `SOT`. So 0 to 10 over the two seeds, the least certain number here. **The other eight dev and seen games:** no effect. On cd82, dc22, m0r0 and vc33 the planner never engages (`kp-offline-dev.md`), and tu93, tr87, g50t and ls20 offer no click. **The 15 held-out games:** nothing is known. Two offer no click (re86, wa30). On the other 13 the planner engages only where the pieces keep to their places, which held on 2 of the 6 click games it was built on; no number is put on them. **Totals:** `SOT` and `SOT2` near 40 each (the dev sweep's `SOT` had 10 levels a seed on seven games where screen 3's had 12). `SOT-KP - SOT` about +13, between +8 and +18, almost all of it on ft09 and lp85. **The bar:** between 4 and 12 levels. In the dev sweep the 11 pairs on which the planner did nothing came out 3 levels apart. So a pass is likely if both games carry and about even if only ft09 does |

`uv run python experiments/kp/kp_test.py verdict` applies the rule
(`experiments/kp/kp_verdict.py`; tested on made-up runs). It gives no verdict
before all 150 runs are in.

## Why not upgrade screen 3's interval as the rule

That interval resamples games, which suits an upgrade that acts on every game.
The planner is expected to act on two of the 25. About one resample in eight
leaves both out ((23/25)^25 = 0.12), and half of those fall at or below zero,
so the interval's lower end reaches zero even with the whole expected gain.
The chance gap asks the question this test is about: is the total further
ahead than two runs of the same agent come apart? Screen 3's interval is still
reported, and it will say, correctly, that the gain is not spread over the
games.

Both rules were checked by simulation before this page (`kp-plan.md`,
Appendix B.5). With no effect anywhere the chance-gap rule passed 1% to 5% of
the time; with the expected gain on ft09 and lp85, 83% to 99%; with a gain on
ft09 only, 53% to 75%. The bar averaged 6 to 9 levels.

## What makes the run set valid

- Every run played to its cap or to the end of the game; none ended on an
  error.
- Every run on the frozen commit and the frozen settings.
- A run cut by a stop, a crash or a reboot is in no manifest and is rerun
  whole. A pause lets the runs in progress finish.
- No conclusion before all 150 are in. Nothing is changed after results are
  seen, and no run is rerun to get a better number.

## Four limits

1. **ft09 and lp85 are the games the method was built on** (by the groupmate,
   with the games' source code read), and they sit in the dev and seen tiers.
   A gain there is a tuned-tier result. The held-out tiers are the clean test
   of whether it reaches further.
2. **Two seeds, 5,000 actions.** It is a short-budget test: a pass is a
   candidate for a longer confirm, as upgrade screen 3 says of its own winner,
   and it says nothing about 100,000 actions.
3. **The chance gap is estimated from 50 differences,** so the bar itself moves
   from test to test. It is a yardstick, not an exact test, and a real gain on
   one game can fall short of it.
4. **The base is `SOT`,** which is itself one seed of an exploratory screen.
   The test compares `SOT` with and without the planner. It has no
   adopted-Goose arm and says nothing about `SOT` against the adopted Goose.

Nothing is adopted from the test and no preset is created. What a pass leads
to is Matt's decision (`kp-plan.md`, section 2.3).

## What is frozen

| | |
|---|---|
| Commit | `527b8a4`: the code every run loads, and the launcher. This page and the pre-registration note are the commit after it; they change no such file |
| The planner | E20 as received: `custom_agents/kp_ref/`, four files, hashes in its README. No setting departs from it (`kp-offline-dev.md`, section 5) |
| Settings | `experiments/kp/kp.toml`, hash `d468480ec9387b71` (in every planner run's `run_config.json` as `kp_config_hash`) |
| Code the runs depend on | hash `aae8104ab8d0c6bd` (`kp_test.py plan` prints it, the sweep's `sweep.json` records it, and a resume refuses to go on if it changed) |
| The rule's code | `experiments/kp/kp_verdict.py`, sha256 starting `bb0ef307151e0f3f`. It is not among the files a resume checks, so a table can be mended while the test is paused; any change to it after the start is listed below |
| Every run | `run_local.py --agent goose --offline` with its arm's flags and none of the caller's `EVAL_*` settings |
| Every planner run | starts with an empty knowledge store; no LLM; `EVAL_GB` unset (the two are refused together) |

## How it is run

```bash
uv run python experiments/kp/kp_test.py plan      # the arms, their flags, the run count
uv run python experiments/kp/kp_test.py start     # begin, detached; it refuses uncommitted code
uv run python experiments/kp/kp_test.py status    # also: pause, resume
uv run python experiments/kp/kp_test.py verdict   # once, when every run is in
```

## Changes after the start

None.
