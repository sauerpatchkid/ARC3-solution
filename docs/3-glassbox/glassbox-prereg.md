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
| 3 | G0: guard rescue (a diagnostic) | DRAFT (16 Oct), two items OPEN; must be registered before G0 runs (target Tue 20 Oct) |
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

**Question.** CP1 left rules that are right on almost every move but never
exact. Can S3's guard fitter, with S1's relation features and no LLM, learn the
condition under which such a rule fails, and so make it trusted?

**What runs.** No LLM, no GPU. On CP1's registered candidates
(`results/rulebook/v2/tier0a_v2/<game>.json`, read only) and its evidence
splits, for the 8 dev games:

1. S1 on the training level's cases (feature bits).
2. S3's guard fitter on every **admitted-but-not-trusted** rule: v2's admission
   bar met (≥ 20 moves, ≥ 95% claimed-exact, gain > 0, ≥ 20 changed cells
   right), not exact on every applicable case. Settings: `glassbox.toml` `[check]`
   as committed at registration (`guard_max_literals = 2`, `guard_holdout = 0.30`,
   `guard_min_wrong = 3`, `auto_exception = true`); the config hash goes here
   when this section is registered.
3. Every narrowed rule and automatic exception is re-graded on every case,
   exactly like any candidate, and re-tiered.

**Reported per game.** Rules that become trusted (with each rule's claimed
exactness before and after); the composed book's cell coverage and case coverage
on the training level, before and after; T1 on the transfer level, before and
after, with the registered interval; every guard found, in words.

**Expectation, written before running.** At least one of tu93 and m0r0 gains a
trusted rule (m0r0 has 11 admitted rules, the best at 99.8%).

**What it decides.** Nothing is passed or failed. It steers week 1: no rescues
means the features are the problem (S1 gets the remaining time); many rescues
means guards are central (the exception prompts get it).

**OPEN (Matt, before G0 runs):**

- **3a. tu93 is outside the scope as written.** CP1 admitted no rule on tu93
  (its best rule was 77% claimed-exact, below the 95% admission bar), so the
  guard fitter, which runs only on admitted rules, cannot rescue anything there.
  Either (i) keep the scope and change the expectation to "m0r0 gains a trusted
  rule", or (ii) widen G0's scope to near-misses below the admission bar (for
  example ≥ 70% claimed-exact and ≥ 20 moves), and say so here.
- **3b. The hold-out split's seed.** Proposed: the 70/30 split of a rule's
  cases is drawn with `random.Random(0)` over the cases sorted by key, so G0
  reruns exactly.

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
| perception gap | S1's closure is below 50% of a game's changing cases |
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

*(None yet. Each entry: date, section, what changed, why, and whether any run
had started.)*
