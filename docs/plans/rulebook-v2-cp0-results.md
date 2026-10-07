# Rulebook v2, CP0: results (7 Oct 2026)

CP0 closes v1, builds the measuring tools and spends v1's one allowed day of
other settings. Everything here was tuned on dev games. The gates and the
expectation below were registered in `docs/plans/rulebook-v2-prereg.md` (commit
`3601f18`) before the run started.

## 1. The near-miss refinement day

**Question.** Tier 0a left seven action groups whose best rule was 90–97% right
and never exact. Does showing the model a rule's own failures, and spending the
calls on the most promising rule, turn a near-miss into an exact rule?

**Result: 2 of 7 groups reached an exact rule.** The pre-registered expectation
was 2–4, and "exactly 2" means **CP1 goes ahead as written**.

| game | group | best rule before | best rule after | exact? | calls | share of the group's changing cases it covers | untouched on the next level: applies / right |
|---|---|---:|---:|---|---:|---:|---|
| tu93 | ACTION1 | 91.5% of 201 moves | 100% of 43 moves | **yes**, call 8 | 8 | 41% | 26 / 22 |
| tu93 | ACTION3 | 91.1% of 180 | 95.7% of 47 | no | 12 | 56% | 78 / 39 |
| m0r0 | ACTION1 | 95.7% of 555 | 98.6% of 569 | no | 12 | 88% | 530 / 29 |
| m0r0 | ACTION2 | 95.7% of 575 | 98.9% of 438 | no | 12 | 83% | 0 / 0 |
| m0r0 | ACTION3 | 91.4% of 546 | 96.2% of 343 | no | 12 | 50% | 207 / 0 |
| m0r0 | ACTION4 | 89.6% of 586 | 98.0% of 100 | no | 12 | 13% | 0 / 0 |
| dc22 | ACTION4 | 96.6% of 742 | 100% of 273 | **yes**, call 5 | 5 | 7% | 52 / 52 |

"Exact" is v1's plan-eligible, unchanged. "Best rule after" for a group that did
not reach exact is its most accurate rule that still applies to at least 20
moves. No group ended "exact up to conflicts".

**What it cost.** 73 refinement calls of the 84 allowed, 1,236,680 generated
tokens, about 2 hours on the GPU (09:45–11:43). 25 of the 73 answers (34%) ran
out of room before writing code, against 22% in Tier 0a; 28 calls needed the
code-only retry or a repair.

### Reading it

1. **Feedback moves accuracy, rarely to exact.** Every group's best rule got
   more accurate (for example m0r0 ACTION1 95.7% → 98.6% on more moves, m0r0
   ACTION4 89.6% → 98.0%). None of the four m0r0 groups reached exact in 12
   calls: their best rules are still wrong on 2 to 13 moves.
2. **Both exact rules got there partly by covering less.** tu93 ACTION1's exact
   rule applies to 43 of the group's 201 moves and covers 41% of its changing
   cases; dc22 ACTION4's covers 7%. The model made `applies()` narrower until
   the rule was right wherever it spoke. That is what v2's partial claims are
   for (plan section 4.5): under v1's whole-board grading a narrow exact rule
   and a wide 98.6% rule are both worth nothing to coverage.
3. **The rules still do not carry.** Untouched on the next level, m0r0 ACTION1's
   98.6% rule is right on 29 of the 530 cases it applies to, and m0r0 ACTION3's
   on 0 of 207. Only dc22's (52 of 52) and tu93 ACTION1's (22 of 26) travel.
   This is v1's diagnosis again: level-specific constants in the code. More
   refinement on one level does not fix it; separating the mechanism from the
   level's bindings (CP1) is aimed at exactly this.
4. **Hidden state is part of it, not all of it.** Six of the seven groups have
   cases with conflicting outcomes, and m0r0 ACTION1's best rule touches 6 of
   them, so it could not have become exact without leaving them out. But the
   other m0r0 groups' best rules touch 0 or 1 and still miss on a handful of
   ordinary cases.
5. **Answers running out of room got worse, not better** (34%). A refinement
   prompt carries a rule and six failures; the model thinks longer about it.
   CP2's lower answer limit will make this bite harder; shorter programs
   (CP3's helpers) and the code-only retry are the planned answers.

### How the run went

All 73 answers were generated in one run. The tool then stopped while writing
one group's summary (tu93 ACTION3, after its 12th call): a bookkeeping bug that
compared two rules with identical code. It was fixed and the run was re-scored
from its answer cache with no server running and new answers refused. The
re-score follows the original log step for step: the same rule refined at each
of the 73 calls, with the same result. Both logs are kept
(`run_original_2026-10-07.log`, `rescore.log`). Recorded in the
pre-registration's section 10.

Files: `results/rulebook/v2/nearmiss/` (per-group JSON with every step and
rule, `report.md`, the answer cache), registered as `rulebook_v2_cp0_nearmiss`.
Re-score without a GPU: `uv run python tools/wm_refine.py --from-cache`.

## 2. v1's rule books on v2's ruler

No LLM (`tools/wm_baseline.py`). v1's verdicts are not re-scored; this measures
its books the way CP1 will be measured, on the transfer level's **score split**
(the moves after the first 300), with memory also given the first 300.

| game | transfer level | cases | book (T0) | nothing | memory | beats both | wrong on changing cases |
|---|---:|---:|---:|---:|---:|---|---:|
| tu93 | 3 | 715 | 0.176 | 0.176 | 0.245 | no | 0.0% |
| tr87 | 2 | 1,386 | 0.000 | 0.000 | 0.051 | no | 0.0% |
| dc22 | 2 | 3,322 | 0.417 | 0.261 | 0.288 | **yes** | 0.0% |
| g50t | 2 | 7,784 | 0.199 | 0.199 | 0.207 | no | 0.0% |
| vc33 | 2 | 534 | 0.753 | 0.753 | 0.753 | no | 0.0% |
| ft09 | 2 | 17,530 | 0.032 | 0.032 | 0.147 | no | 58.8% |
| m0r0 | 2 | 5,338 | 0.371 | 0.371 | 0.386 | no | 0.2% |
| cd82 | 3 | 20,297 | 0.142 | 0.142 | 0.144 | no | 0.0% |

**1 of 8 beats both baselines (dc22); ft09 is wrong on 58.8% of changing
cases.** That is v1's result, now measured the way CP1's gate is written: the
gate asks for ≥ 3 of 8 with wrong ≤ 5% everywhere. Point estimates only; how to
compute the interval with one recording per game is still open
(pre-registration section 4.4).

One thing the new tools showed about v1: g50t's only book rule, the ACTION5
"does nothing" rule that v1 trusted, is right on 86.4% of the moves it applies
to. The same screen and action sometimes did change something. Since 6 Oct such
a rule is no longer trusted.

## 3. What was built

| Piece (plan section 3.3) | Where | State |
|---|---|---|
| Claimed cells: `predict()` may return `(board, claimed)` | `custom_agents/wm/check.py` | done; v1 rules and numbers unchanged |
| Claimed-cell exactness, cell coverage, case coverage; v2 admitted / trusted on one level | `custom_agents/wm/metrics.py` | done |
| Fit / score split of a transfer level | `LevelEvidence.split` | done |
| Transfer three ways | `metrics.transfer_t0` | T0 done; T1 and T2 need re-binding and repair, which are CP1's build |
| Held-out tiers | `custom_agents/wm/tiers.py` | done; the recorded draw reproduces |
| Sealed results | `tools/paired_compare.py --seal`, `--open-sealed` | done |
| Artifact registry | `artifacts/registry.json`, `tools/artifacts.py` | done; v1's results and this run are registered |
| Failure taxonomy additions | pre-registration section 9 | defined; flags are set by CP1's code |
| The refinement bandit | `tools/wm_refine.py` | done; CP1 reuses it (plan section 4.6) |

v1's results are frozen (registered, read-only, with a copy in
`results/rulebook_v1_frozen_2026-10-07.tar.gz`).

## 4. Open before CP1's run

Two items in the pre-registration are marked OPEN and are not binding yet:

- **4.4** how the interval in CP1's gate is resampled when there is one
  recording per game (recommended: resample attempts);
- **6.4** whether CP3's "no higher" is per game or pooled (not needed until
  28 Oct).
