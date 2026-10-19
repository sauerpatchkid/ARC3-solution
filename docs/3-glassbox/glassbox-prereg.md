# Glassbox: pre-registration

*The plan is `docs/3-glassbox/glassbox-plan.md`; this file is the part of it
that results are judged against. The commit that marks a section **registered**
in the table below is its timestamp: a gate counts only if its section was
registered before the run it judges.*

**How to use this file.** A run is judged by the text here, not by the plan's
prose and not by what the numbers turn out to suggest. Nothing in a registered
section is edited after its run starts. Corrections go in section 11 with a date
and a reason. Items marked OPEN are not binding until settled, in this file,
before the run they affect.

| Section | Covers | Status |
|---|---|---|
| 1 | Rulebook v2 is closed | DRAFT (16 Oct): Matt to review, then mark registered |
| 2 | Game tiers | DRAFT (16 Oct), copied unchanged from v2 |
| 3 | G0: guard rescue (a diagnostic) | **REGISTERED 16 Oct 2026** (Matt), before any G0 run: the as-built procedure, config hash `44a9ce1b5ebe03ad`, code `c051e50`. **Run 16 Oct** (`341ebec`); result: `g0-guard-rescue.md` |
| 3b | G0b: the bar mask (a diagnostic) | **REGISTERED 16 Oct 2026** (Matt), before any G0b run: config hash `2f789d8dca149dfe`, code `78c461c`; its control equals CP1 on all 8 games. **Run 16 Oct** (`763d253`); result: `g0b-bar-mask.md` |
| 4 | G1: the offline rule test | **REGISTERED 17 Oct 2026** (Matt), before any G1 run: mask on, judged against G0b; config hash `5b7a7b2684e37ce5`, code `bd0552b`, vocabulary `glassbox_vocab_v1`. **Run 17 Oct** (`af208c7`): FAIL, 3 of 8, safety met; result: `g1-offline-rules.md` |
| 5 | The offline counterfactual and G2 | **5a REGISTERED 18 Oct 2026** (Matt: "go with your recommendations ... register 5a and run it"), before any replay of a dev game: book `pool_g0b_g1`, config hash `6c5d95983cf6fa97`, code `d111172`. 5b: DRAFT, its OPEN items settled 18 Oct (Matt), to be registered before G2 runs (target Tue 27 Oct) |
| 6 | G3: templates on held-out games | to be written before G3 runs (target Fri 6 Nov) |
| 7 | G4: the confirm | to be written before G4 runs (target Mon 16 Nov) |
| 8 | Honesty rules | DRAFT (16 Oct): v2's, plus one new rule |
| 9 | Failure taxonomy | DRAFT (16 Oct) |
| 10 | Frontier showcase limits | DRAFT (16 Oct), copied unchanged from v2's section 7.2 |
| 11 | Changes after registration | empty |

---

## 1. Rulebook v2 is closed

Recorded in v2's pre-registration, section 10, on 16 Oct 2026
(`docs/2-llm-history/rulebook-v2-prereg.md`, commit `8c067fa`):

> CP0 (7 Oct) and CP1 (15 Oct; gate FAIL, primary 2 of 8, safety met on all 8)
> are final and will never be re-scored. CP2–CP5 were not run. Sections 5–7 of
> that file are withdrawn unrun: no result will be judged by them. The successor
> plan is Glassbox, motivated by CP1's diagnosis (rule-writing yield, not the
> form of the rules, now limits the result). Its gates are registered here before
> any Glassbox run. The tier lists and the honesty rules carry over unchanged.

v2's code as run is git tag `rulebook-v2`; it is frozen in `legacy/rulebook/`.
Its registered results (`rulebook_v1_results`, `rulebook_v2_cp0_nearmiss`,
`rulebook_v2_cp1_tier0a` in `artifacts/registry.json`) are never rewritten.

## 2. Game tiers

Copied unchanged from v2's section 2 (registered 7 Oct 2026). **The lists are
what counts.** They are in code as `tiers.py` (moved from
`custom_agents/wm/tiers.py` on 16 Oct, lists unchanged; `tests/test_tiers_registry.py`
pins them).

| Tier | Games | Use in Glassbox |
|---|---|---|
| Dev (8) | tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82 | building, tuning, G0, G1, G2, dev sweep 2 |
| Seen (2) | ls20, lp85 | smoke tests (including H3's smoke test); reported separately |
| A/B-offline (8) | ar25, lf52, re86, s5i5, sc25, sp80, su15, tn36 | G3 (offline held-out A/B of the templates); otherwise only inside G4 |
| Untouched (7) | bp35, cn04, ka59, r11l, sb26, sk48, wa30 | only ever run inside G4 |

The two held-out tiers were drawn so that each gets 3 of the 6 games that play
without ACTION7: Python 3.12, `random.Random(2026)`, `sample` 3 of the sorted
six (ar25, bp35, lf52, sb26, sk48, su15), then 5 of the sorted other nine.
Cross-game artifacts (H1's vocabulary, H3's templates) may be built from dev and
seen games only (`tiers.BUILD_FROM`).

## 3. G0: guard rescue (a diagnostic, not a gate)

*REGISTERED 16 Oct 2026 (Matt), before any G0 run: the commit that marks this
section registered in the table above is its timestamp. The code it judges was
built and tested on synthetic games only; it has never run on CP1's candidates.*

**Question.** CP1 left rules that are right on almost every move but never
exact. Can S3's guard fitter, with S1's relation features and no LLM, learn the
condition under which such a rule fails, and so make it trusted?

**What runs.** No LLM, no GPU. `make gb-guards RUN=cp1_import G0=1`, once,
with a clean tree, on a commit where the code G0 runs is unchanged since commit
`c051e50` (`git diff c051e50 -- custom_agents/gb tools/gb_offline.py
experiments/glassbox` prints nothing): `custom_agents/gb/s3_guards.py` VERSION
1, `s1_see.py` VERSION 3, `h1_vocab.py`, `gb/core`. The run's manifest records
the commit. Its inputs are CP1's registered candidates and evidence splits
(`results/rulebook/v2/tier0a_v2/`, read only) as imported into
`results/glassbox/cp1_import/` by `make gb-import-cp1` (437 of 437 recomputed
numbers equal CP1's), for the 8 dev games:

1. **S1** (VERSION 3) on the training level's cases: its feature bits and the
   level's mover. (VERSION 3, 16 Oct: on a tie, an object sits on the
   background; before, a tie went to the lower colour number. On the 8 games
   this changed the feature bits of 398 of 109,402 cases: dc22 88 of 1,069
   training cases, vc33 4 of 2,890, ft09 2 of 581, and some of their transfer
   cases; m0r0's are unchanged. "Explained" is unchanged. See
   `tests/test_gb_vocab.py`.)
2. **Scope:** every **admitted-but-not-trusted** rule, as re-graded on the
   training level with CP1's binding (`s3_check/<game>/rules.jsonl`): v2's
   admission bar met (≥ 20 moves, ≥ 95% claimed-exact, gain > 0, ≥ 20 changed
   cells right, ≤ 5 ms per move) and not trusted.
3. **The guard.** R = the distinct applicable cases with every claimed cell
   right, W = the others; fewer than 3 W cases: no guard. W and R are each
   sorted by case number and shuffled by one `random.Random(0)`, W first; the
   first 30% of each (rounded half up) is held out. A literal is one of S1's
   facts or its negation; a guard is one literal or two joined by AND, and fits
   if it is true on every fit W case and false on every fit R case. The guards
   that fit are tried in order (fewer literals; geometry, then the subject and
   the action, then colour, by the worse literal and then the sum; then S1's
   order of the facts, a fact before its negation), and the first that also
   separates the held-out cases is the guard. Directions are the screen's. The
   mover and any colour in a guard are PARAMS slots: the rule's own `Colour()`
   slot when it is bound to that colour, else a new one.
4. **Two new rules** per guard, as source wrapping the original: the
   **narrowed** rule (base ∧ ¬guard) and the **exception** (base ∧ guard,
   predicting that the base's claimed cells stay as they are; `PRIORITY = 1`).
5. **Replay.** Each new rule is graded on every case of the training level with
   CP1's binding plus the new slots' values (`bind.run_rule`, `book.trust`). A
   rule **becomes trusted** when its narrowed rule is v2-trusted there. An
   exception claims no changed cell, so v2's bar (gain > 0) never admits it; it
   is reported as **exact** when it is exact on ≥ 20 moves with no conflicting
   case (v1's known-no-op bar).
6. **The book**, composed as CP1 composed it (`cp1_import.py`: group order,
   lint, gain, order; `book.compose` selecting): before, from the trusted rules;
   after, from those plus every narrowed rule that becomes trusted.
7. **T1.** The rules of CP1's book keep CP1's T1 bindings. A new rule in the
   book after is re-bound on the transfer level's fit split as CP1 re-bound its
   rules (`bind.search` preferring the training binding, `bind.choose`; in T1 if
   trusted on both levels and lint-clean). Both books are scored on the score
   split with the registered interval (2,000 resamples, seed 0).

**Settings.** `experiments/glassbox/glassbox.toml` config hash
`44a9ce1b5ebe03ad`; `[see]` `a31392fb68e47122`; `[check]` `635ce93f9ee63cd1`
(`guard_max_literals = 2`, `guard_holdout = 0.30`, `guard_min_wrong = 3`,
`guard_seed = 0`, `guard_colour_features = "params"`, `auto_exception = true`).

**Void.** A game's G0 numbers are void if its "before" (the book's numbers on
the training level, and T1) differs from what the import recorded (the run's
summary's `before_matches_import`: neither may be false; a game without a
transfer level has no T1 to compare), or if the run's code,
inputs or settings differ from the above. A void game is rerun after the cause
is found, and the rerun is entered in section 11.

**Reported per game** (from `s3_check/<game>_L<k>.guards.json` and its two
`.jsonl` files). The rules looked at; every guard found, in words, with its R
and W counts, and why none for the rest; the rules that become trusted, each
with its claimed exactness and moves before and after and whether it is in the
book and in T1; the exact exceptions; the composed book's cell coverage and
case coverage on the training level, before and after; T1 on the transfer level
(the per-move rate, the differences from "nothing" and "memory" with their
intervals, and the wrong share), before and after; each new rule's T1
re-binding and its outcome (a guard that fails on the next level is flagged
"guard overfit", section 9).

**Expectation, written before running.** m0r0 gains at least one trusted rule
(it has 11 admitted rules, the best at 99.8% claimed-exact). tu93 is not
expected to change: CP1 admitted no rule there (its best was 77%), so it is
outside G0's scope.

**What it decides.** Nothing is passed or failed. It steers week 1: no rescues
means the features are the problem (S1 gets the remaining time); many rescues
means guards are central (the exception prompts get it).

**Settled (Matt, 16 Oct, before any G0 code or run).** 3a: the scope stays
"admitted-but-not-trusted" and tu93 is dropped from the expectation (the plan's
draft named "tu93 or m0r0", but no tu93 rule meets the admission bar). 3b: the
hold-out seed is 0, as written in step 3. 3c (at registration): S1's tie rule
is fixed before G0 (step 1); the as-built details of the fitter (steps 3–7)
are registered as written here.

## 3b. G0b: the bar mask (a diagnostic, not a gate)

*REGISTERED 16 Oct 2026 (Matt: "follow your recommendation"), before any G0b
run: the commit that marks this section registered in the table above is its
timestamp. It follows from G0's result (section 3; `g0-guard-rescue.md`).*

**Question.** G0 found that m0r0's 11 near-miss rules are right on every case
they apply to and miss trust only because some cases were seen to end two ways,
and traced those to the decoration mask stopping 6 cells short of m0r0's
move-budget bar. If S0 runs each bar in the mask to the screen's edges, and
nothing else changes, do those rules become trusted, and what do the book and
T1 become?

**Known before registration** (evidence only: no rule has been run on S0's
evidence). With S0 on, per game (training / fit / score split): conflicting
cases m0r0 13 → 0 / 0 / 758 → 0; tu93 11 → 0 / 0 / 96 → 0; ft09 0 / 0 / 33 → 0;
g50t 93 / 10 / 1,826, unchanged; dc22, vc33, cd82, tr87 none either way. Mask
cells on the training level: m0r0 122 → 128, tu93 60 → 64, vc33, cd82, tr87
63 → 64; dc22, g50t 64 and ft09 0 unchanged. Cases on the training level:
m0r0 1,385 → 1,275, tu93 124 → 112, the others unchanged.

**The control, run before registration** (`make gb-g0b-control`, code
`78c461c`). With S0's bars off, S0 → S3 → S4 gives CP1's book members and
numbers, T0, T1 and every T1 re-binding (found again by S4's own search) on
all 8 dev games, and S0 rebuilds CP1's evidence array for array. So whatever
G0b changes, the mask changed.

**What runs.** No LLM, no GPU. `make gb-g0b`, once, on a commit where the code
it runs is unchanged since `78c461c` (`git diff 78c461c -- custom_agents/gb
tools/gb_offline.py experiments/glassbox` prints nothing), into
`results/glassbox/g0b_bars/`, for the 8 dev games:

1. **S0** (`s0_mask.py` VERSION 1): each game's evidence rebuilt from the
   recording CP1 used, at CP1's training and transfer levels and splits (the
   transfer level's first 300 moves are its fit split), with each bar in the
   decoration mask run to the screen's edges: a run of at least 8 masked cells
   along a row within 4 cells of the top or bottom edge (or a column within 4
   of the left or right) is masked in full. The candidates are CP1's, as
   imported (`cp1_import/s2_write/`), unchanged.
2. **S1** (VERSION 3) on every split.
3. **S3 grading** (`s3_check.py` VERSION 1): every candidate replayed on the
   training level with the binding CP1 gave it, graded by v2's trust; the book
   composed as CP1 composed it.
4. **S4** (`s4_carry.py` VERSION 1): each book rule re-bound on the fit split by
   CP1's procedure (`bind.search`, `bind.choose`; in T1 if trusted on both
   levels and lint-clean); T0 and T1 on the score split, per move, with the
   registered interval (2,000 resamples, seed 0).
5. **The guard fitter**, section 3's procedure unchanged, on the rules that are
   still admitted but not trusted.

**Settings.** `experiments/glassbox/glassbox.toml` config hash
`2f789d8dca149dfe`; `[mask]` `5561d138585be439` (`complete_bars = true`,
`bar_min_length = 8`, `band = 4`, `fit_moves = 300`); `[see]`
`a31392fb68e47122`; `[check]` `635ce93f9ee63cd1`.

**Reported per game, beside CP1's numbers** (section 3's "before"). S0's report
(mask cells, cases and conflicting cases per split); the admitted and trusted
rules, naming which of m0r0's 11 near-misses become trusted, with claimed
exactness and moves; the book's members, cell coverage and case coverage on
the training level; T0 and T1 (the per-move rate, the differences from
"nothing" and "memory" with their intervals, the wrong share); each book rule's
re-binding and flag; the guard fitter's result. S0 masks more cells for every
arm, so "nothing" and "memory" are scored on the same evidence: within G0b the
book is compared with them; CP1's T1 is shown beside, with its own baselines.

**Expectation, written before running.** m0r0: all 11 near-miss rules become
trusted, and its book's cell coverage on the training level rises above 0%
(it is 0% in CP1); its T1 is uncertain (the fit split is 300 moves). tu93: no
newly trusted rule (CP1 admitted none; its best was 77%, which conflicting
cases did not cause). The other six: the same books as CP1 and T1 within 1
percentage point. No game's T1 wrong share above 0%.

**Void.** The run is void if the control does not give CP1's results at the
run's commit (`make gb-g0b-control` prints "ALL EQUAL TO CP1"), or if the run's
code, inputs or settings differ from the above.

**What it decides.** Nothing passes or fails. It informs whether G1 runs with
S0 on, which section 4 decides before G1, naming the mask as a difference from
CP1 if it does.

## 4. G1: the offline rule test

*REGISTERED 17 Oct 2026 (Matt: "run G1 with the mask on and judge it against
G0b"), before any G1 run: the commit that marks this section registered in the
table above is its timestamp. S2 was built and tested with a mock LLM and on
one seen game (lp85, the smoke test below); it has not written a rule for a dev
game.*

**Question.** With S1's facts in the prompt, H1's vocabulary, one mechanism per
rule and the guard fitter, does the local LLM write rules that explain more of
each game's training level than CP1's rules did on the same evidence?

**The baseline is G0b, not CP1** (Matt, 17 Oct). G1 runs with S0 on. G0b
(section 3b) is CP1's candidates on that same evidence, so it already holds the
mask's effect (m0r0: 0% to 53.5% of changed cells). Judged against G0b, a gain
is the new rule writing's, not the mask's.

**Held equal to CP1** (and so to G0b): the 8 dev games; the seed-0 recordings
of A0's confirm (CP1's corpora); the same training level (the first with
>= 500 moves), transfer level (the next with >= 50 distinct cases) and fit split
(the transfer level's first 300 moves); the same model
(`cyankiwi/Qwen3.8-27B-AWQ-INT4`, served as `rulebook`) with thinking on,
20,480-token answers, one code-only retry and one repair, Qwen's sampling and
CP1's seeds (1 for the first round, 100 + step for later calls, + 50 for a
retry); 4 first-round candidates and up to 8 more calls per group, chosen by
CP0's bandit on claimed-cell exactness, residual prompts once a group has a
trusted lint-clean rule; at most 12 groups per game, those whose moves ever
change the board (under S0 m0r0's ACTION5 and orange-click groups change only
bar cells, so they get no calls); every candidate's PARAMS fitted by
`bind.search` and graded by v2's trust; the book composed as CP1 composed it;
T1 by CP1's re-binding (S4 v0, shown to reproduce CP1 exactly).

**What is new in G1** (what it tests): S2 v1 (`gb/s2_write.py`): S1's facts
under every move shown, H1's card in the instructions (`glassbox_vocab_v1`,
registered with this section), the one-mechanism instruction, and the guard
fitter tried once on a near-miss (>= 95% claimed-exact) before a call is spent
on it. After S2, the guard fitter (section 3's procedure) runs once more on the
rules still admitted but not trusted. **Not in G1** (the plan's cut lines):
exception prompts, two-level prompts and T2, priority composition, trust tiers.

**What runs.** `make gb-g1`, once, with the server up
(`experiments/llm/serve.sh`), on a commit where the code it runs is unchanged
since `bd0552b` (`git diff bd0552b -- custom_agents/gb tools/gb_offline.py
tools/gb_gate.py experiments/glassbox` prints nothing), into
`results/glassbox/g1/`: S0 (`--from cp1`) -> S1 -> S2 -> S3 grading -> S4 ->
the guard fitter -> `tools/gb_gate.py g1 --run g1 --base g0b_bars`. Answers are
cached (`s2_write/llm_cache.jsonl`); if the run stops (a power cut, a server
crash), it is restarted with the same command and the cached answers are
reused, so it continues where it stopped. A group that fails for a machine or
server reason is rerun the same way before the verdict; one that fails for its
rules' own reasons is a result.

**Settings.** `experiments/glassbox/glassbox.toml` config hash `5b7a7b2684e37ce5`;
`[write]` `75d996c8d133b7a9`; `[mask]` `5561d138585be439`; `[see]`
`a31392fb68e47122`; `[check]` `635ce93f9ee63cd1`; `[vocab]` `7ec1234e7b9c566b`.

**The smoke test, run before registration** (lp85, a seen game, 17 Oct; code
`bd0552b`; `results/glassbox/smoke_lp85/`). S0 -> S1 -> S2 -> S3 -> S4 -> the
guard fitter ran end to end on lp85's 2 groups with the server: 24 candidates,
2 trusted, a book of 1 rule explaining 45.9% of the training level's changed
cells, T1 34.0% per move (equal to "nothing"), 0% wrong; 391,501 tokens in about
45 minutes. 6 of the 8 first-round answers finished (the code-only retry wrote
the other 2), and every answer called an H1 primitive.

**Primary.** On at least 5 of the 8 dev games, the training level's composed
book (S3's book from S2's candidates plus every narrowed rule the guard fitter
rescues) has a **higher cell coverage than G0b's** (`g0b_bars`, its guard
fitter's "after" book; G0b's write-up, per game: tu93 0.0%, tr87 0.0%, dc22
11.8%, g50t 17.7%, vc33 10.6%, ft09 97.7%, m0r0 53.5%, cd82 5.2%). Higher means
strictly, at the 4 decimals stored.

**Safety.** On every one of the 8 games, T1 (the same book re-bound on the fit
split, scored on the score split) is wrong on at most 5% of the score split's
changing cases.

**Pass = primary and safety.** `tools/gb_gate.py` computes both and writes
`g1/g1_verdict.{json,md}`.

**Reported, not gating.** CP1's own criterion per game (T1 beats "nothing" and
"memory" with both 95% intervals above zero; G0b: dc22 and ft09); case
coverage; trusted rules per game; T1 per move beside G0b's; LLM calls, tokens,
answers that ran out of room; H1 use (the share of graded candidates that call a
primitive, and of trusted rules); guard rescues, in the loop and after.

**Ablations** (reported later, cannot change the verdict): GB−facts and
GB−vocab (first-round calls only), and GB−guards (re-scored from the cache),
run after G1 if GPU time allows.

**Expectation, written before running.** Per game, against G0b: tu93 gains a
trusted movement rule and coverage above 0% (likely); g50t and cd82 gain a
little (likely); dc22, ft09 and m0r0 stay about where they are (ft09 at 97.7%
and m0r0 at 53.5% leave little room; strictly higher is uncertain); tr87 may
stay at 0%; vc33 is uncertain. So the primary is uncertain: about 45% (the
plan's 55% was against CP1, before G0b gave m0r0 its 53.5%). Safety is expected
to hold, with m0r0 the risk (4.3% wrong in G0b).

**Void.** If the model, the code or the settings differ from the above, or if
the verdict is computed while a group's failure for a machine or server reason
has not been rerun.

**What it decides.** Pass: G1's book goes on to G2 (the online gate). Fail: as
the plan's §6.3 says, M2 still goes ahead with whatever trusted rules exist, and
G1 is reported as a diagnosis with its funnel.

## 5. The offline counterfactual and G2

*Written 18 Oct 2026 before either runs. **5a is REGISTERED** (18 Oct; the
commit that marks it so in the table above is its timestamp). 5b is a DRAFT:
its OPEN items were settled by Matt on 18 Oct ("go with your recommendations",
recorded below as SETTLED); it is registered, with its own hashes, before G2
runs.*

**What is built (18 Oct).** S5 ACT v1 (`custom_agents/gb/s5_act.py`): PRUNE
(carried colour no-ops, no LLM; the book's no-op rules, re-bound on every level
after S4's online wait), goal HINTS, and GbMap (buttons a re-bound no-op rule
covers count as tried). The hook is `EVAL_GB=1` in `custom_agents/action.py`
(16 lines tagged `# [gb]`; `sed -i '/\[gb\]/d'` restores the file byte for
byte), and `EVAL_GB_BOOK=<run>` names the book. With `EVAL_GB` unset the agent
is unchanged: `tools/cpu_check.py`, 2,000 CPU moves before and after, for
`goose` (ft09) and `mb_gated_att` (tu93): **identical on every move** (1,970
and 1,960 recorded moves; 18 Oct, `bde037e` against the hook as committed). **Not
built:** PROBE (matrix and disagreement probes), IMAGINE, contact goals, H2
(online LLM calls), H3. So in v1 the book steers Goose **only through its
no-op rules**; its changing rules (tu93's 59%, tr87's 35.5%) are not used
online until probes or IMAGINE exist. This is stated with every result.

### 5a. The offline counterfactual (no GPU, no LLM)

*REGISTERED 18 Oct 2026 (Matt), before any replay of a dev game. The replay was
built and tested on synthetic boards and on lp85 (a seen game) only.*

**Question.** On A0's own recordings, how many of the moves Goose made in the
first actions of each new level would S5's carried no-ops have covered, and how
many of those did nothing (avoidable waste) against something (masking risk)?

**What runs.** `make gb-cf BOOK=pool_g0b_g1 CF=1`, once, on a quiet machine
(the re-binding search works to a time budget), on a commit where the code it
runs is unchanged since `d111172` (`git diff d111172 -- custom_agents/gb
tools/gb_offline.py experiments/glassbox` prints nothing). That is
`tools/gb_offline.py --stage replay --run cf_gb --book pool_g0b_g1 --games <the
8 dev games> --seeds 0,1,2 --cf`, then the same with `--run cf_free` and no
`--book` (the GB-free arm), on the 24 recordings of A0's
confirm (`results/confirm_upgrade/20260927_221915/manifest.tsv`, arm
`mb_gated_att`; all 24 are logged). For each recording, level by level, the
guide sees every recorded move in order (`s5_act.replay`): colour no-ops carried
at each level-up (>= 20 clicks on the level, none changing anything), the
book's no-op rules re-bound on each level's own moves once S4's wait ends
(>= 10 distinct changing cases of the rule's family, or 1,000 moves), goal
hints. "Changed" is judged under the level's decoration mask as S0 builds it.

**Measured** (Rulebook v2's definitions, kept): in the first 2,000 actions of
each level from level 2 up (action numbers counted from the level's first
recorded move), AVOIDABLE WASTE = covered moves that changed nothing, MASKING
RISK = covered moves that changed something, each as a share of the window's
recorded moves, pooled over seeds and levels per game; also what covered them
(colour or rule) and the guide's events (carried, dropped, re-bound).

**Expectation, kept from v2 (plan §6.2), written before running.** Avoidable
waste >= 10% on the games with carried click no-ops (ft09, vc33, m0r0), and
masking risk <= 1% on every game, for the GB arm. Also expected: GB-free covers
most of what GB covers (the book's no-op rules are mostly "clicking colour X
does nothing", which the colour rule finds without the LLM).

**Known before registration.** The replay was smoke-tested on lp85 (a seen
game) with the lp85 smoke run's book, seed 0: levels 2-6, 9,902 moves in the
windows, 1,066 covered (10.8%), all avoidable waste, masking 0, all from colour
no-ops; its one no-op rule was re-bound on levels 1 and 3. No dev game has been
replayed; the tool refuses dev games without `--cf`. Also run before
registration, online and not a replay: two ft09 smoke runs (`EVAL_GB=1
EVAL_GB_BOOK=pool_g0b_g1 --agent mb_gated_att`, seed 0, 20,000 actions, GPU;
`results/runs/20261018_125714/` and `20261018_130128/`): no errors; the
guide changed 21.5% and 13.3% of Goose's moves (covered no-ops and goal
hints); no contradiction; the first run showed a re-binding refused for too few
cases and never retried, fixed before the second (retry at twice the moves, up
to 4,000). Then g50t and cd82 (same settings; `20261018_133244/`,
`20261018_133536/`): the button path works (g50t's ACTION4 no-op rule re-bound
on level 2; GbMap counted 802 no-op buttons; cd82's two rules re-bound after
928 and 1,056 moves). They showed two flaws in goal hints, fixed before
registration: a hint could be the background colour, which touches every
change and so is never dropped (cd82: a hint from a 9-move level 1 boosted
clicks all run, and the guide changed 37.7% of moves); and a boost was applied
on screens with no hint-coloured click to make (g50t), drawing again for
nothing. Now the first board's background is never a hint, and the boost
applies only where such a click is possible. Finally `tools/gb_sweep.py` was
smoke-tested on lp85 (2 arms x 2,000 actions; `results/glassbox/g2_smoke/`):
runs, scoring, manifests and the three verdicts work (the one against A0 is
BLOCKED there, as it should be: the action caps differ). Their levels are not
evidence of anything (one GPU run each).

**What it decides.** Nothing passes or fails. It is reported before G2, and
no S5 setting is changed after it (alpha, the 20-click rule, the wait and the
2,000-action window are fixed above).

**SETTLED (a): the book is `pool_g0b_g1`** (Matt, 18 Oct): G0b's (CP1's) and
G1's candidates pooled on their shared evidence, chosen and carried by CP1's
procedure (`docs/3-glassbox/pool-g0b-g1.md`; exploratory, not a gate). The plan's
text said "G1's books"; the pool holds every trusted rule G1 found and CP1's
too. Its no-op rules, per group the known no-op with the most moves on the
training level: dc22 #67 (red click) and #122 (blue click), ft09 #0 (blue) and
#5 (red), m0r0 #60 (orange), cd82 #30 (ACTION4) and #72 (orange), g50t #74
(ACTION4), vc33 #13 (blue); none on tu93 and tr87 (there, GB = GB-free).

**Settings.** `experiments/glassbox/glassbox.toml` config hash
`6c5d95983cf6fa97`; S5 reads from `[act]` (`f6ce02211dee2c82`) prune = on,
hints = on, alpha = 0.1, gamma = 1.0, hint_drop_moves = 2,000 (v1 does not read
its other keys), and from `[carry]` (`893dcab4f6e33df8`) noop_min_clicks = 20,
online_min_cases = 10, online_max_wait = 1,000; in the code (`s5_act.DEFAULTS`,
VERSION 1): noop_rule_min_cases = 10, waste window 2,000 actions, 2 hints, a
re-binding retried at twice the moves up to 4,000, the background never a hint.
The manifest is `[run]` (`203487a6ac045797`).

**Void.** If the code, the book or the settings differ from the above; if a
replay stops on an error (it is then rerun whole for that game, before the
summary is read).

### 5b. G2: the online dev gate

**Question.** Does Goose guided by S5 complete at least as many levels as A0
on the dev games?

**Arms.** A0: `mb_gated_att`'s 16 confirm runs on the 8 dev games, seeds 0-1,
reused (manifest above). GB: `EVAL_GB=1 EVAL_GB_BOOK=<BOOK>
--agent mb_gated_att`, 8 dev games x seeds 0-1 x 100,000 actions. GB-free:
the same with `EVAL_GB_BOOK` unset (no LLM anywhere: colour no-ops, goal hints).

**What runs.** `make gb-g2 BOOK=<BOOK> G2=1` (`tools/gb_sweep.py --run g2
--games dev --seeds 0,1 --cap 100000 --book <BOOK> --g2`): the 32 runs, 4 at a
time on the GPU, each scored by `compute_metrics.py` and added to
`results/glassbox/g2/online/manifest.tsv` as it finishes (pause and resume
keep finished runs). A0's 16 rows are copied from its confirm manifest into
`g2/online/a0.manifest.tsv` (same run folders), because `--expect 16` needs
exactly 16 runs on each side (A0's manifest holds all 75).

**Pass** (v2's dev rule, as the plan's §6.3): GB levels >= A0's; 16 of 16 GB
runs complete; paired wins >= losses; no game worse on both seeds.
`uv run python tools/paired_compare.py --base g2/online/a0.manifest.tsv:mb_gated_att
--new g2/online/manifest.tsv:GB --rule dev --expect 16` (`gb_sweep.py` runs it
at the end, with GB-free against A0 and GB against GB-free beside it).

**Reported, not gating.** GB against GB-free (a tie is reported plainly as "the
book adds nothing at this stage"); GB-free against A0; avoidable waste and
masking risk online (from `s5_stats.json` and `events.jsonl`); colour no-ops
dropped and rules demoted (rule survival); moves the guide changed; levels 2
and up; actions per second against A0's.

**Caveats, stated with the result.** A0's runs are from 27-28 Sep and GPU runs
are not reproducible past the first training step, so the comparison is of
distributions (CLAUDE.md); A0's recordings are the older kind (levels are
unaffected). The book was written from A0's seed-0 recordings of these same
games: G2 is a dev-game result, labelled as tuned.

**SETTLED (b): no LLM calls online in G2** (Matt, 18 Oct). H2 is not built;
the book is fixed before the run, built offline from A0's seed-0 recordings, so
GB runs share the GPU 4 at a time (about 6 h for GB and GB-free together).
Building H2 first (online S2 calls, plan §5.8; GB one at a time beside the
server, 12-20 h) would have moved G2 past 29 Oct.

**SETTLED (c): S5 as built** (Matt, 18 Oct): S5's online re-binding does not
use S4's role tie-break (`[carry] roles_in_bind` stays off). Its exploratory
test (`pool-g0b-g1.md`, tuned on dev games): m0r0's G1 rules 6.8% → 26.5% at
T1, the pooled book unchanged.

**SETTLED (d): G2 runs S5 v1 as built** (prune, hints, map; Matt, 18 Oct).
Probes, which the plan's cut line keeps with prune and hints, are added for
dev sweep 2, so G2 stays on its dates.

**Void.** If a GB run uses a different book, code or settings than registered;
if `cpu_check` with `EVAL_GB` unset does not match move for move; if a run
ends on an error and is not rerun before the verdict.

## 6. G3: templates on held-out games

*To be written before G3 runs, with the template catalog's registry entry and
hash.*

## 7. G4: the confirm

*To be written before G4 runs, with the frozen configuration's hash.*

## 8. Honesty rules

Kept from v1 and v2 (v2's section 8, registered 7 Oct 2026): pre-registration
before each run; anything tuned on dev games is labelled as such; the "off" path
is checked for isolation (`tools/cpu_check.py`, for `goose` and `mb_gated_att`);
the world model uses its own random stream; answer caches make offline reruns
exact and are never shared between scored runs; `paired_compare --expect`
blocks a verdict when runs are missing; cross-game artifacts come only from the
dev and seen games and are frozen and hashed in the registry before any
held-out use; the tiers of section 2, with the untouched tier run only inside
the confirm; sealed per-game held-out results in the confirm; **no learning
between scored runs** (frozen, declared artifacts built beforehand are allowed
and listed in the registry).

New in Glassbox: v2 is closed formally (section 1). CP0 and CP1 are final; the
replacement is motivated by CP1's diagnosis; every Glassbox gate is registered
here before any Glassbox run it judges.

## 9. Failure taxonomy

v1's flags stand (logging failure; unsupported action; no rules admitted; low
coverage; observation aliasing; no cleared level; completion test failed to fit;
completion test didn't transfer; search limit or watchdog; plan mismatch;
intervention budget exhausted), and so do v2's (level-bound rule; binding
ambiguous; binding failed; claim too narrow; schema mismatch; negative
transfer). Glassbox adds:

| Flag | Means |
|---|---|
| perception gap | S1 explains fewer than 50% of a game's changing cases (draft revised 16 Oct: the plan's "closure" cannot fail as written, so the flag uses S1's "explained"; plan §5.1 step 3) |
| guard overfit | a fitted guard fails its hold-out, or fails on the next level |
| imagined mismatch | an imagined step (S5 IMAGINE) differs from the real screen when walked |
| template coincidence | a template is trusted on a held-out game's training level but wrong at T1 |

## 10. Frontier showcase limits

Copied unchanged from v2's section 7.2 (registered 13 Oct 2026). Read its
"sections 3 to 6" as this file's sections 3 to 7, and "the playbook or CP5's
training data" as any Glassbox artifact (H1's vocabulary, H3's templates).

> The design (model, games, budget, the table to be reported) is added here
> before it runs. Whatever the design, these limits hold:
>
> - It is **descriptive only**. It is not a gate, it cannot change any verdict in
>   sections 3 to 6 or any adoption decision, and its levels are never added to
>   the totals of our own arms.
> - **Nothing flows back.** Frontier-written rules, helpers or lessons do not
>   enter any registered artifact, the playbook or CP5's training data, and no
>   setting of our own system is changed after seeing its results.
> - It uses **dev and seen games only**. The A/B-offline and untouched tiers are
>   not shown to a frontier model.
> - Only the rule writer changes: the same recordings, prompts, checker, trust
>   rules, measuring tools and budgets in calls and tokens as the local run it is
>   compared with. A hard spend cap is set before it starts.
> - It is reported apart from every other result, with the caveat that a frontier
>   model may have seen these public games.

## 11. Changes after registration

*(Each entry: date, section, what changed, why, and whether any run had
started.)*

1. **16 Oct 2026, section 3, after G0 ran (no change to its text or numbers).**
   The run's manifest records `341ebec+dirty`. The only difference from a
   clean tree was the `ARC-AGI-3-Agents` submodule's local edits
   (`agents/__init__.py`, `agents/structs.py`, the game harness), present
   before this session and not imported by anything G0 runs; G0's code was
   unchanged since `c051e50` as required. Recorded so "with a clean tree" is
   not read as met to the letter.
2. **16 Oct 2026, section 3, after G0 ran.** The run's summary has no T1 for
   tu93: `s3_guards.py` looked for the transfer level as the training level
   + 1 (tu93: 2), but CP1's transfer level for tu93 is 3. No rule of tu93's
   was in scope, so its book is the same before and after, and its T1 before
   and after is the import's (3.6% per move; equal to CP1's registered
   number). Fixed after the run (the code now takes the first later level
   with a fit split; `tests/test_gb_guards.py` uses level 2). G0 was not
   rerun: no G0 number depends on it.
3. **16 Oct 2026, section 3b, after G0b ran (no change to its text or
   numbers).** As in entry 1, the manifest records `763d253+dirty` only because
   of the `ARC-AGI-3-Agents` submodule's local edits; the control was rerun at
   `763d253` just before G0b and printed "ALL EQUAL TO CP1".
