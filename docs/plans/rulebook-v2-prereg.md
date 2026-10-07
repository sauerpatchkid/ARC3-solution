# Rulebook v2: pre-registration

*Registered 7 Oct 2026, before any Rulebook v2 answer was generated. The plan is
`docs/plans/rulebook-v2.md`; this file is the part of it that results are judged
against. The commit that adds a section is its timestamp: a gate counts only if
its section was committed before the run it judges.*

**How to use this file.** A run is judged by the text here, not by the plan's
prose and not by what the numbers turn out to suggest. Nothing in a registered
section is edited after its run starts. Corrections go in section 10 with a date
and a reason. Sections marked OPEN are not yet binding and must be settled, in
this file, before the run they affect.

| Section | Covers | Status |
|---|---|---|
| 1 | v1 is closed | registered |
| 2 | game tiers | registered |
| 3 | CP0 near-miss refinement day | registered; run follows this commit |
| 4 | CP1 gate (Tier 0a v2) | registered, with one OPEN item (4.4) |
| 5 | CP2 gates and Confirm 1 | registered |
| 6 | CP3 gate | registered, with one OPEN item (6.4) |
| 7 | CP4, CP5 | to be added before their runs |
| 8 | honesty rules | registered |
| 9 | failure taxonomy | registered |

---

## 1. v1 is closed

- **Stage A** (rerun with Qwen3.8-27B, 4 Oct 2026): NO-GO, 1 of 3 games
  (`docs/plans/rulebook-stageA.md`).
- **Tier 0a** (6 Oct 2026): G1 FAIL, 2 of 8 games (`docs/plans/rulebook-tier0a.md`).

Both verdicts are final. v2 metrics may be computed on v1's rules for
comparison; v1's verdicts are never re-scored. v1's result files are frozen:
registered as `rulebook_v1_results` in `artifacts/registry.json` (66 files with
hashes), read-only on disk, with a copy in
`results/rulebook_v1_frozen_2026-10-07.tar.gz`. The code that produced them is
the git tag `rulebook-v1`.

v1's pre-registration said that after a G1 fail there would be "one day to try
other settings on these dev games, then narrow the build to the games and
groups that pass". **This plan replaces that fallback.** The one day is used
(section 3). Instead of narrowing, v2 changes what a rule is (mechanism and
level bindings apart, partial claims) and tests that under new gates. The change
is motivated by v1's own diagnosis, and every v2 gate below was registered
before any v2 model run.

## 2. Game tiers

| Tier | Games | Use |
|---|---|---|
| Dev (8) | tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82 | building, tuning, offline gates |
| Seen (2) | ls20, lp85 | used in Stage A or the smoke test; reported separately |
| A/B-offline (8) | ar25, lf52, re86, s5i5, sc25, sp80, su15, tn36 | the offline held-out A/Bs of CP3 and CP5; otherwise only inside a confirm |
| Untouched (7) | bp35, cn04, ka59, r11l, sb26, sk48, wa30 | only ever run inside a confirm |

The two held-out tiers were drawn so that each gets 3 of the 6 games that play
without ACTION7: Python 3.12, `random.Random(2026)`, `sample` 3 of the sorted
six (ar25, bp35, lf52, sb26, sk48, su15), then 5 of the sorted other nine. The
draw was re-run on this machine on 6 Oct 2026 (Python 3.12.3) and gives the
lists above. **The lists above are what counts.** They are in code as
`custom_agents/wm/tiers.py`.

## 3. CP0: the near-miss refinement day

This is the one day of other settings v1 allowed. It is tuned on dev games and
is labelled that way wherever it is reported.

**Question.** Tier 0a left movement rules at 90–97% and its settings never gave
an admitted rule a second try. Does showing the model a rule's own failures
turn a near-miss into an exact rule?

**Groups (7), fixed.** tu93 ACTION1, tu93 ACTION3; m0r0 ACTION1, ACTION2,
ACTION3, ACTION4; dc22 ACTION4.

**Input, fixed.** Tier 0a's frozen training-level evidence
(`results/rulebook/tier0a/<game>_train.npz`) and its recorded candidates. No new
evidence, no new recordings. Known from those files when this was written: the
best rule per group is right on 91.5%, 91.1%, 95.7%, 95.7%, 91.4%, 89.6% and
96.6% of the moves it applies to, and six of the seven groups contain cases with
conflicting outcomes (2, 3, 8, 1, 1, 3 and 0 cases).

**Method, fixed** (`tools/wm_refine.py`; REx-style, Tang et al. 2024):

- An arm is a rule for the group that applies to at least 20 recorded moves with
  gain > 0. The starting arms are Tier 0a's candidates that meet this.
- Each step draws, for every arm, θ ~ Beta(1 + C·h, 1 + C·(1 − h) + N), where h
  is the arm's accuracy (the exactly-right share of the recorded moves it applies
  to), N is how many times it has been refined, and C = 20. The arm with the
  highest draw is refined. Draws use a fixed seed per group.
- One refinement call shows: the group as Tier 0a showed it, with 3 recorded
  moves instead of 6; the rule, its code and its scores; and up to 6 of its
  failing cases, wrong predictions first, then moves of the group it did not
  cover. (The server accepts 10 pictures per prompt: 1 + 3 + 6.)
- Model and settings as Tier 0a: `cyankiwi/Qwen3.8-27B-AWQ-INT4`, thinking on,
  20,480-token answers, one code-only retry when an answer runs out of room, one
  repair attempt when the code cannot run. One candidate per call. A retry or
  repair belongs to its call and is not counted separately.
- The child is checked on every case of the level, exactly as in Tier 0a. It
  joins the arms if it qualifies as one and its code is new.
- **Budget:** 12 refinement calls per group, 84 in all. A group stops at its
  first exact rule. The run is made once. If it is interrupted it is resumed
  from its answer cache with the same settings; no setting is changed and no
  second pass is made.

**"Exact", fixed.** v1's *plan-eligible*, unchanged: admitted (applies to ≥ 20
moves, ≥ 95% right, gain > 0, ≤ 5 ms per move), right on every case it applies
to, and none of those cases has conflicting outcomes.

**Expectation and what follows.** 2–4 of the 7 groups reach exact.

| Groups that reach exact | Reading | Consequence for CP1 |
|---|---|---|
| 3 or more | feedback does convert near-misses | CP1 keeps refinement central |
| exactly 2 | as expected | CP1 goes ahead as written |
| 1 or fewer | the misses are hidden state or edge cases | CP1 prioritises partial claims and `api.t` |

**Reported beside it, not part of the count:**

- *exact up to conflicts*: a rule right on every case's usual outcome whose cases
  include some that have also been recorded ending differently. No rule that
  reads only the screen and the action can be exact on those;
- for each group's best rule: the share of the group's changing cases it covers
  (so a rule made exact by covering less is visible), and how it does untouched
  on the transfer level;
- calls used, tokens, answers that ran out of room, wall clock.

## 4. CP1 gate: Tier 0a v2

### 4.1 Held equal to Tier 0a

The same 8 dev games and seed-0 recordings; the same training level (the first
level with ≥ 500 moves) and transfer level (the next level with ≥ 50 distinct
cases); the same model, thinking on, 20,480-token answers. Differences in the
result then come from the rule contract and the refinement policy.

### 4.2 New in v2, fixed

- The first 300 moves of the transfer level are the **fit split**: re-binding
  (T1) and repair (T2) see only those. **Scores use the rest.** The baselines
  are scored on the same moves, and "memory" may also use the fit split, so
  every arm sees the same evidence.
- The fit split's decoration mask is the ticker scan over the fit moves joined
  with the training level's mask. The score split uses the transfer level's own
  mask, as v1 did. (`LevelEvidence.split`)
- Budget per action group: 4 first-round candidates, then up to 8 refinement
  calls with the section 3 bandit, stopping at the first trusted rule.
- Trusted (v2): admitted (≥ 20 moves, ≥ 95% claimed-exact, gain > 0, ≥ 20
  changed cells predicted right in total, ≤ 5 ms per move), 100% claimed-exact
  on every level seen so far using each level's own binding, and no conflicting
  cases. At most 4 rules per group.

### 4.3 The gate

| | Criterion | v1, for comparison |
|---|---|---|
| Primary | On the transfer level, the re-bound book (T1: one-step exact rate, unclaimed cells read as unchanged) beats both "nothing" and "memory", with the 95% interval of each difference above zero, on **≥ 3 of 8** games | 1 of 8 (dc22) |
| Safety | T1 wrong predictions ≤ 5% of the transfer level's changing cases, on **every** game | ft09: 59% |

**Pass = primary and safety.** Reported, not gating: claimed-cell trusted
coverage ≥ 80% (target: ≥ 3 of 8), strict case coverage, T0 and T2, tokens,
the rate of answers that ran out of room.

### 4.4 OPEN: how the interval is computed

The plan says "computed as in Stage A's R1". Stage A's R1 is a paired bootstrap
of the difference in exact-prediction rate that resamples whole **runs** (2,000
resamples, seed 0, 2.5th–97.5th percentile). Stage A had many runs per game;
Tier 0a v2 has one recording per game, so there is nothing to resample at that
level. To be settled here before the CP1 run. Options:

1. *(recommended)* resample **attempts**, the stretches of play between game
   overs, within the score split; if a level has fewer than 20 attempts, resample
   blocks of 50 consecutive moves instead;
2. score the transfer level on all three recorded seeds and resample runs (only
   3 units);
3. resample distinct cases (treats cases as independent, which they are not).

## 5. CP2: rule-guided Goose

Arms: **A0** = `mb_gated_att` (its 75 confirm runs are reused, 112 levels);
**C2** = full guidance; **C2-free** = the LLM-free carry-over (Goose's
dead-click filter re-keyed by colour and carried across levels, plus goal
hints). Per-run LLM budget for C2: 16 requests, 48 completions, 400,000
generated tokens, 12,288-token answers. Guidance defaults: α = 0.1, β = 2, γ = 1.

### 5.1 Offline counterfactual (before any online run)

On the `mb_gated_att` confirm recordings, the first 2,000 actions of each level
from level 2 up. *Avoidable waste* = actions covered by the previous level's
re-bound no-ops that indeed did nothing. *Masking risk* = covered actions that
did something. **Expectation:** avoidable waste ≥ 10% on the games with carried
click no-ops (ft09, vc33, m0r0), and masking risk ≤ 1%. Both are counted in
actions (action numbers), not recorded rows, so the 6 Oct change to what is
recorded does not move them.

### 5.2 Dev gate

8 dev games × seeds 0–1 × 100,000 actions. **Pass:** C2 levels ≥ A0; 16 of 16
runs complete; paired wins ≥ losses; no game worse on both seeds.
Tool: `tools/paired_compare.py --rule dev --expect 16`. Mechanism check,
reported: avoidable waste in the first 2,000 actions of new levels is lower than
A0's. If C2-free matches C2, that is reported plainly as "the LLM adds nothing at
this rung".

### 5.3 Confirm 1 (only if 5.2 passes)

25 games × 3 seeds × 100,000 actions. **Adopt:** strictly more than 112 levels;
paired wins > losses; no game worse on every seed; 75 of 75 runs complete.
Tool: `tools/paired_compare.py --rule confirm --expect 75`. Reported by tier.
The confirm is never shortened.

**Sealing.** Per-game results on the A/B-offline and untouched tiers go to a
sealed file and stay closed until CP3's artifacts are frozen and registered.
Before then only the aggregate and the dev and seen tiers are read. Opening the
file is logged in `artifacts/registry.json`.

## 6. CP3: the cross-game layer

### 6.1 Building rules

Artifacts (helper library, schema catalog, playbook, Goose prior) are built only
from the 8 dev games and the 2 seen games, or from external games. They are
frozen and registered in `artifacts/registry.json` by 28 Oct 2026 and before any
sealed held-out result is opened. Limits: ≤ 30 helpers and ≤ 1,500 tokens of
signatures and docstrings; ≈ 25 playbook entries, ≈ 1,000 tokens.

### 6.2 The gate

Offline held-out A/B on the 8 A/B-offline games: the CP1 pipeline with and
without the frozen layer, at the same budget. **Pass:** more trusted rules **and**
higher cell coverage on ≥ 5 of 8 games, with transfer-level wrong predictions
and the rate of answers that ran out of room no higher.

### 6.3 Reported, and the prior's own rule

- Leave-one-dev-game-out for the library and the playbook: non-negative effect
  on ≥ 6 of 8 (reported).
- Goose prior: adopted only if at least as good as a fresh start on ≥ 6 of 8
  leave-one-game-out folds (20,000-action runs; early effective-action rate over
  the first 1,000 actions of each level, and levels). Dev games only.
- Online, CP2 + layer on dev games: a regression check only (no game worse on
  both seeds). The real held-out online test is Confirm 2, untouched tier.

### 6.4 OPEN: "no higher"

Whether "wrong predictions and overrun rate no higher" is judged per game or
pooled over the 8 games is not stated in the plan. To be settled here before
the CP3 A/B. Default if nothing else is written: pooled over the 8 games.

## 7. CP4 and CP5

Their gates (plan sections 8.4 and 9.3) are added here before their runs.

## 8. Honesty rules

Kept from v1: pre-registration before each run; anything tuned on dev games is
labelled as such; the "off" path is checked for isolation
(`tools/cpu_check.py`); the world model uses its own random stream; the answer
cache makes offline reruns exact; `paired_compare --expect` blocks a verdict
when runs are missing.

New in v2:

1. Cross-game artifacts come only from the dev and seen games (or external
   games) and are frozen and hashed in the registry before any held-out use.
2. The tiers of section 2.
3. Sealed per-game held-out results in Confirm 1 (section 5.3).
4. v1's "never share learned rules across scored runs" becomes: **no learning
   between scored runs**; frozen, declared artifacts built beforehand are
   allowed and listed in the registry.

## 9. Failure taxonomy

v1's flags stand (logging failure; unsupported action; no rules admitted; low
coverage; observation aliasing; no cleared level; completion test failed to
fit; completion test didn't transfer; search limit or watchdog; plan mismatch;
intervention budget exhausted). v2 adds:

| Flag | Means |
|---|---|
| level-bound rule | the lint flags literal colours or coordinates; the rule is left out of T1 |
| binding ambiguous | more than one binding is exact on the fit split |
| binding failed | no binding is exact on the fit split |
| claim too narrow | the rule is exact but fails the 20-cell floor or adds no unclaimed cells |
| schema mismatch | no schema meets the trust bar for the group |
| negative transfer | a frozen cross-game artifact makes a held-out game worse |

## 10. Changes after registration

*(none)*
