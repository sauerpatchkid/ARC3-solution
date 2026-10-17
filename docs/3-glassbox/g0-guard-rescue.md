# G0: guard rescue — result (16 Oct 2026)

*Judged by pre-registration section 3 (`glassbox-prereg.md`, registered in
`341ebec`). A diagnostic, not a gate: nothing passes or fails.*

## In one line

**No rule was rescued: 0 guards on 12 rules in scope. The expectation (m0r0
gains a trusted rule) is not met.** The guard fitter never had a case to fit:
m0r0's 11 near-misses are right on every case they apply to, and miss trust only
because 8 of those cases (on ACTION1; 1 on ACTION2) were seen to end two ways.
Most of those second endings happen at move 137 of an attempt: a timed event
the screen does not show, which no screen fact can guard.

## The run

- `make gb-guards RUN=cp1_import G0=1`, 16 Oct 17:03, commit `341ebec`, CPU,
  about 20 seconds. Code unchanged since `c051e50` (`git diff` empty), config
  hash `44a9ce1b5ebe03ad`, as registered. Log:
  `results/glassbox/cp1_import/s3_check/g0_log.txt`.
- **Valid.** On every game "before" equals what the CP1 import recorded
  (`before_matches_import`: book true on all 8; T1 true on 7, none on tu93,
  see below), so before is CP1's registered book and T1.
- Two notes, entered in section 11: the manifest says `341ebec+dirty` only
  because of the `ARC-AGI-3-Agents` submodule's local edits, which G0 does not
  use; tu93's T1 is missing from the run's summary (its transfer level is 3, the
  code looked for 2; fixed after the run), and its book is the same before and
  after, so its T1 is the import's.

## Per game

Training level book (cell coverage, case coverage) and T1 on the transfer level
(per-move rate; the differences from "nothing" and "memory" with their 95%
intervals; wrong share of changing cases). Before = after everywhere, so one
column each.

| Game | Rules in scope | Guards | Rescued | Exact exceptions | Cell coverage | Case coverage | T1 rate | T1 − nothing | T1 − memory | T1 wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| cd82 | 1 | 0 | 0 | 0 | 5.2% | 8.3% | 14.9% | +0.018 [0.017, 0.018] | −0.024 [−0.027, −0.022] | 0.0% |
| dc22 | 0 | — | — | — | 11.8% | 14.7% | 25.5% | +0.101 [0.097, 0.105] | +0.021 [0.004, 0.037] | 0.0% |
| ft09 | 0 | — | — | — | 97.7% | 56.0% | 18.1% | +0.151 [0.146, 0.156] | +0.033 [0.027, 0.039] | 0.0% |
| g50t | 0 | — | — | — | 17.7% | 0.0% | 9.1% | +0.000 [0, 0] | −0.059 [−0.062, −0.056] | 0.0% |
| m0r0 | 11 | 0 | 0 | 0 | 0.0% | 0.0% | 6.2% | +0.000 [0, 0] | −0.392 [−0.402, −0.383] | 0.0% |
| tr87 | 0 | — | — | — | 0.0% | 0.0% | 0.0% | +0.000 [0, 0] | −0.476 [−0.495, −0.462] | 0.0% |
| tu93 | 0 | — | — | — | 0.0% | 0.0% | 3.6% (import) | +0.000 [0, 0] | −0.815 [−0.834, −0.795] | 0.0% |
| vc33 | 0 | — | — | — | 10.6% | 0.0% | 75.1% | +0.000 [0, 0] | +0.000 [0, 0] | 0.0% |

**Every guard found, in words:** none. **Why none**, rule by rule
(`s3_check/<game>/guards.jsonl`):

| Game | Rules | Right cases | Wrong cases | Why no guard |
|---|---|---|---|---|
| m0r0 | 8 on ACTION1 (ids 2, 4, 6–11) | 272 each | 0 | no wrong case (a guard needs 3) |
| m0r0 | 3 on ACTION2 (ids 28, 31, 32) | 289 each | 0 | no wrong case |
| cd82 | 1 on clicks on orange (id 80) | 102 | 1 | 1 wrong case (needs 3) |

## Why m0r0's near-misses are near-misses

They are admitted, exact on every distinct case (with each case's commonest
ending), and not trusted for one reason: *"it applies to 8 cases that have been
seen to end in more than one way"* (ACTION1; 1 case on ACTION2). Their claimed
exactness, 98.5% and 99.8%, counts moves: 9 of ACTION1's 606 moves and 1 of
ACTION2's 575 ended the other way.

Those other endings are not at the end of an attempt (none is the attempt's
last move). They cluster at a fixed time: of the 14 minority moves on m0r0's 13
conflicting cases (all actions), 9 are at move 137 of their attempt, in 2 more
the majority ending is the one at move 137, and the other 3 are at moves 139,
144 and 145. So something happens at move 137 of every
attempt that the screen does not show beforehand. The same screen and action
then end two ways, and S1's facts (all read off the screen) cannot tell them
apart. The plan's risk table names this case ("hidden state ... can't be
guarded by what's on the screen").

## What it means for week 1

The registered reading is: *"no rescues means the features are the problem
(S1 gets the remaining time)"*. The data do not support that reading: the
features were never tried, because no rule in scope had wrong cases to
separate. G0 measured something else: **on CP1's candidates, every near-miss
but one is a near-miss because of hidden state, not because of a missing
condition.** Which way week 1 goes is Matt's call. What the data point to:

- **A move-counter fact** (moves since the attempt started), the plan's cut-line
  option in §5.3. It would need S3 to grade such rules move by move (`api.t` is
  None when rules are checked on cases, which merge moves made at different
  times).
- **The trust bar's treatment of conflicting cases.** m0r0's rules are exact
  wherever the screen decides the outcome; the tiers (§5.3 step 4) could say
  what such a rule may do.
- **The guard fitter is still untested on real data.** Its first real inputs
  will be G1's candidates, where near-misses with wrong cases may appear.
