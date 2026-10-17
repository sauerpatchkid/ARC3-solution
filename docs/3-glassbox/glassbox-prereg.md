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
| 4 | G1: the offline rule test | to be written before G1 runs (target Thu 22 Oct) |
| 5 | The offline counterfactual and G2 | to be written before they run (target Tue 27 Oct) |
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

## 4. G1: the offline rule test

*To be written before G1 runs. Content (plan §6.3): what is held equal to CP1;
primary (training-level cell coverage above CP1's on ≥ 5 of 8 dev games);
safety (T1 wrong ≤ 5% of the transfer level's changing cases on every game);
reported numbers; ablations; per-game expectations; the config hash.*

## 5. The offline counterfactual and G2

*To be written before they run.*

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
