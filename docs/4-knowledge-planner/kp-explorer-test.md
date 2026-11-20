# The explorer's test (written before its first run)

**The questions.** Does the explorer let Goose with the planner (`SOT-KP`)
complete more levels in 5,000 actions on the 25 games? And where it plays, do
the levels come sooner?

**Status: written and frozen on 20 Nov 2026, before any run of the test.**
Nothing under "The test" is edited once the first run starts; corrections go
under "Changes after the start". It starts on Matt's word (20 Nov: "commit and
push, then run the registered test on all 25 games").

The explorer is a groupmate's agent E21, run unedited behind Goose
(`custom_agents/kp_bridge.py`, `[explore] keys = "e21"`): on a game with keys
Goose plays level 1 and E21 plays the levels after it; on a game with clicks
only nothing changes. `kp-explorer.md` has how it was chosen, what was checked
before this page, and its dev sweep.

## The test

| | |
|---|---|
| Arms | `SOT-KP`: Goose with upgrade screen 3's three upgrades, by flags (`EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_sched,attempt,objclick EVAL_TRAIN_EVERY=1 EVAL_TRAIN_RECENT=16`), and the planner (`EVAL_KP=1`, `experiments/kp/kp.toml`): the arm that passed the planner's test. `SOT-KP21`: the same with `experiments/kp/kp_e21.toml`, which turns the explorer on. `SOT-KP2`: `SOT-KP` again, to measure chance |
| Games | All 25. The held-out tiers are included at Matt's request of 20 Nov, noted in the pre-registration as the planner's test's use of them was (`docs/3-glassbox/glassbox-prereg.md`, section 11, entry 12). Results are reported by tier, and by whose games they are for E21 |
| Seeds | 16 and 17. No run of this project used them before |
| Budget | 5,000 actions a run |
| Runs | 3 arms x 25 games x 2 seeds = 150, 4 at a time, about 5.5 hours, pausable |
| Score | Levels completed by 5,000 actions, summed over an arm's 50 runs (the engine's count, as `compute_metrics.py` reports it) |
| **Rule A, more levels: `SOT-KP21` clears the noise if** | (1) its score is above both `SOT-KP`'s and `SOT-KP2`'s, **and** (2) `SOT-KP21 - SOT-KP` is above the 97.5th percentile of the chance gap. The chance gap is the difference between two 50-run totals of the same agent: this test's 50 differences `SOT-KP2 - SOT-KP`, one per game and seed, drawn again with replacement, each with a sign at random, and summed, 10,000 times (seed 0). This is the planner's test's rule |
| **Rule B, sooner where it plays: `SOT-KP21` clears the noise if** | on the levels after the first of the 18 games with keys, (1) its net of level races is above zero against `SOT-KP` and against `SOT-KP2`, **and** (2) its net against `SOT-KP` is above the 97.5th percentile of the chance gap built in the same way from the nets of `SOT-KP2` against `SOT-KP`, one per game and seed. A level race is a level both runs of a pair started and at least one finished; the run that took fewer actions on it wins it, counted from the level-up before it (`tools/levels_by_budget.py`, `level_races`). The net is races won minus races lost |
| The 18 games with keys | ar25, bp35, cd82, cn04, dc22, g50t, ka59, lf52, ls20, m0r0, re86, sb26, sc25, sk48, sp80, tr87, tu93, wa30: the games on which Goose pressed a key in the planner's test. On the other seven (ft09, lp85, r11l, s5i5, su15, tn36, vc33) Goose sends clicks only, E21 calls the planner itself, and the two arms are the same agent |
| How the two are read | Rule A is the project's measure and the headline. Rule B asks the narrower thing: where the explorer plays, are its levels sooner than Goose's? Passing B alone is not "more levels" |
| Otherwise | Inside the noise at this budget, each rule on its own |
| Reported, not part of a rule | By tier. The games with keys apart from the games with clicks only. The games with keys in three groups: this project's dev and seen games; held-out here but used by E21's author (ar25, bp35, cn04, ka59, lf52); held-out here and used by nobody for E21 (re86, sb26, sc25, sk48, sp80, wa30). Levels by 1,000 and 2,000 actions. The 95% interval with games resampled. Pairs better, same and worse, and games worse on both seeds, for each of the three comparisons. Level races on level 1. What the explorer did per game: levels entered after the first, cleared by it, cleared by Goose, moves made. Who cleared the levels: the planner, the explorer, Goose. Errors, run times |
| Sanity anchors | Every run's `run_config.json` shows `SOT`'s flags and its arm's settings hash. In the `SOT-KP21` runs the explorer's path is taken on the 18 games with keys and on no other. `SOT-KP` and `SOT-KP2` should each land near 62 levels, the planner's test's score. Under 40 or over 80: the set-up is checked and the results page says so at the top |
| Expectation, written before the runs | **The seven games with clicks only:** no effect. ft09: 6 levels in every run of every arm (every planner run so far won it). lp85: 4 to 7 levels where Goose clears level 1, and it is the game that moves the total most. **cd82:** 2 levels in each explorer run against 1 in each control run, so +2. It did that from 7 of 7 recordings and in 2 of 2 GPU runs. **tu93:** 4 levels in each explorer run; a control run has 4 in about two of three, so +0 to +2, and its levels 2 to 4 sooner. **dc22, g50t, ls20, m0r0, tr87:** the same count; no arm has cleared a second level there. **The 11 held-out games with keys:** nothing is known from this project. In its author's own runs E21 cleared a second level on none of ar25, bp35, cn04, ka59 and lf52. Goose reached level 2 of ar25 in 3 of the planner's test's 6 runs, and there the explorer takes the level from Goose and may lose it. On re86, sb26, sc25 and wa30 no arm has cleared level 1, so the explorer will not play. That leaves sp80 and sk48 as the two games nobody used for E21 on which it will. A guess for the 11: between -2 and +3. **Totals:** `SOT-KP` and `SOT-KP2` near 62 each; `SOT-KP21 - SOT-KP` about +3, between -1 and +7. **Rule A's bar:** between 5 and 9 levels (5 on the dev sweep's ten games, 10 in the planner's test). So rule A is more likely to say "inside the noise" than not, even if the explorer does on cd82 what it has done so far. **Rule B:** a net of about +8 from cd82 and tu93 if they repeat, plus or minus what the held-out games give, against a bar of 4 to 7. About even, leaning to a pass |

`E21TEST=1 uv run python experiments/kp/kp_test.py verdict` applies both rules
(`experiments/kp/kp_explorer_verdict.py`; tested on made-up runs). It gives no
verdict before all 150 runs are in.

## Why two rules

The explorer plays a few levels of a few games. In its dev sweep it ended 2
levels ahead on ten games, and two runs of the same `SOT-KP` arm came 2 apart
on those games. A 50-run total cannot tell such a gain from chance at two
seeds, whatever is true. So rule A can pass only if the explorer also adds
levels on the held-out games. That is the question worth asking of them, and a
"no" is an answer too.

Rule B asks about what the explorer was seen to do: on tu93 it finished levels
2 to 4 sooner than Goose in 6 of 6 races, and on cd82 it finished a level
Goose did not. The measure was chosen after seeing that sweep, so on cd82 and
tu93 this test is a repeat on new seeds, and on the other games with keys it
is new.

Both rules were run on the dev sweep's own runs before this page, with the
planner's test's `SOT-KP` runs standing in for `SOT-KP2` (ten dev and seen
games, seeds 11 and 12, tuned tier): rule A, +2 against a bar of 5, inside the
noise; rule B, +8 and +8 against a bar of 6, clears.

## What makes the run set valid

- Every run played to its cap or to the end of the game; none ended on an
  error.
- Every run on the frozen commit and the frozen settings.
- A run cut by a stop, a crash or a reboot is in no manifest and is rerun
  whole. A pause lets the runs in progress finish.
- No conclusion before all 150 are in. Nothing is changed after results are
  seen, and no run is rerun to get a better number.

## Six limits

1. **Where the explorer was chosen is in the test.** cd82, tu93 and the other
   dev and seen games are where E21 was picked over E22 and E24 and checked.
   Five held-out games were used by E21's author. Only six held-out games with
   keys were used by nobody for it, and Goose has cleared level 1 on two of
   them.
2. **Two seeds, 5,000 actions.** A pass is a candidate for a longer confirm
   and says nothing about 100,000 actions.
3. **The chance gaps are estimated from 50 differences (rule A) and 36 (rule
   B),** so each bar moves from test to test. They are yardsticks, not exact
   tests.
4. **The base is `SOT-KP`,** which is not adopted. The test compares it with
   and without the explorer and says nothing else.
5. **Rule B's measure was chosen after the dev sweep.**
6. **It is an explorer, not a planner.** It takes Goose's place on the later
   levels of games with keys. Whatever Goose would have done there is not
   done.

Nothing is adopted from the test and no preset is created. What a pass leads
to is Matt's decision.

## What is frozen

| | |
|---|---|
| Commit | `4e310f7`: the code every run loads, the launcher and the rules. This page and the pre-registration note are the commit after it; they change no such file |
| The planner and the explorer | E20 and E21 as received: `custom_agents/kp_ref/`, hashes in its README (E21: `76b262c953c60339...`) |
| Settings | `experiments/kp/kp.toml`, hash `d468480ec9387b71` (`SOT-KP`, `SOT-KP2`); `experiments/kp/kp_e21.toml`, hash `991f479318fce2d0` (`SOT-KP21`). Each is in its runs' `run_config.json` as `kp_config_hash` |
| Code the runs depend on | hash `bf4882f94f0ade61` (`E21TEST=1 kp_test.py plan` prints it, the sweep's `sweep.json` records it, and a resume refuses to go on if it changed) |
| The rules' code | `experiments/kp/kp_explorer_verdict.py`, sha256 starting `42464b7b9bee7142`. It is not among the files a resume checks, so a table can be mended while the test is paused; any change to it after the start is listed below |
| Every run | `run_local.py --agent goose --offline` with its arm's flags and none of the caller's `EVAL_*` settings |
| Every run | starts with an empty knowledge store; no LLM; `EVAL_GB` unset |

## How it is run

```bash
E21TEST=1 uv run python experiments/kp/kp_test.py plan      # the arms, their flags, the run count
E21TEST=1 uv run python experiments/kp/kp_test.py start     # begin, detached; it refuses uncommitted code
E21TEST=1 uv run python experiments/kp/kp_test.py status    # also: pause, resume
E21TEST=1 uv run python experiments/kp/kp_test.py verdict   # once, when every run is in
```

## Changes after the start

None.
