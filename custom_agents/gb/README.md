# Glassbox code (`custom_agents/gb/`)

Goose explores. A local LLM writes small rules about the game from Goose's own
moves. Each rule is checked against every recorded move, and only checked rules
are allowed to steer Goose. Plan: [`docs/3-glassbox/glassbox-plan.md`](../../docs/3-glassbox/glassbox-plan.md).

**One stage per file, numbered in the order the data flows.** Each stage reads
files and writes files into its own folder of the run
(`results/glassbox/<run>/<stage>/`), so you can change one stage and re-run it
and the stages after it, without touching the ones before.

## The stages, and the problem each one fixes

| Step | File | The problem it fixes | What it does | LLM? | Judged by |
|---|---|---|---|---|---|
| **S1 SEE** | `s1_see.py` | The LLM must work out from pictures what moved and what blocked it, and we can't tell "couldn't see it" from "couldn't write it" | Finds objects, matches them before and after, lists relations (touching, what's ahead), and checks whether that description rebuilds the after-screen (**closure**) | no | G1 |
| **H1 VOCAB** | `h1_vocab.py` | Rules are long raw numpy with boundary bugs; 44% of CP1's answers ran out of room | Up to 16 tested grid primitives (`translate`, `is_blocked`, `recolour`…) the rules may call | no | G1 |
| **S2 WRITE** | `s2_write.py` | One rule tried to explain several mechanisms at once | Asks the LLM for one mechanism per rule, with S1's facts and H1's primitives in the prompt | **yes** | G1 |
| **S3 CHECK** | `s3_check.py`, `s3_guards.py` | Near-miss rules (90–99.8% right) only became exact by covering less | Replays every rule; a **guard fitter** learns when a rule fails; exceptions, priorities, three trust tiers | no | G0, G1 |
| **S4 CARRY** | `s4_carry.py` | Each new level needs its rules re-fitted, and at first it has little evidence (vc33) | Re-binds rules to the new level by search; waits for enough evidence online | no | G1 |
| **S5 ACT** | `s5_act.py` | Rules have never been used online: Goose still spends moves on actions known to do nothing | Soft-prunes known no-ops, probes what's unexplained, follows goal hints; later IMAGINE | no | G2, G4 |
| **H2 SCHEDULER** | `h2_schedule.py` | An LLM call takes 1–2 minutes; it must come at useful moments, within a budget, and keep runs reproducible | Event triggers (warm-up, level-up, surprise, stall) and a fixed budget | decides | G2 |
| **H3 TEMPLATES** | `h3_templates.py` | Every new game starts from zero rules | Tries rules learned on dev games on a new game, by re-binding, before asking the LLM | no | G3 |
| **S6 SHOW** | `s6_show.py` | The numbers are spread over many files, and you can't see a prediction next to what happened | One offline page per run: a funnel per game, a panel per stage, a move-by-move case viewer | no | (reads only) |

Inherited, not new: `core/` is Rulebook v2's checking machinery, copied
(see `core/__init__.py`). `artifacts.py` is the plumbing that keeps stages
separate. `cp1_import.py` re-runs Rulebook v2's CP1 with `core/` and writes it
as Glassbox artifacts: it proves the copy still gives CP1's numbers, and it is
S6's first data.

## Where things are

| | Path |
|---|---|
| Settings, one section per stage | `experiments/glassbox/glassbox.toml` |
| Commands, one per stage | `experiments/glassbox/gb.mk` (included by the root Makefile: `make gb-help`) |
| The offline driver | `tools/gb_offline.py` (`--import-cp1` now; `--stage <s>` as stages are built) |
| A run's files | `results/glassbox/<run>/` (offline), `results/runs/<ts>/<game>/glassbox/` (online) |
| Tests | `tests/test_gb_*.py` |
| Gates, registered before each run | `docs/3-glassbox/glassbox-prereg.md` |

## Status

| Piece | State |
|---|---|
| `core/` | copied from Rulebook v2 on 16 Oct; 51 tests |
| `artifacts.py` | built 16 Oct |
| `cp1_import.py` | built 16 Oct: all 437 recomputed numbers on CP1's 8 games equal the registered ones; `tests/test_gb_regression.py` |
| S6 SHOW v0 | built 16 Oct (`s6_show.py` + its page template `s6_show_page.html`): the funnel and the case viewer, `make gb-report RUN=<name>`; `tests/test_gb_show.py`. v1 (stage panels, online views, compare mode) comes with M2 |
| S1 SEE | built 16 Oct, all 8 steps (`s1_see.py`, `make gb-see`): objects, correspondence, closure and "explained"; relations, feature bits, roles, the exploration matrix and the facts text (pinned by a golden test); `tests/test_gb_see.py` (21 tests) |
| H1 VOCAB | started 16 Oct (`h1_vocab.py`): S1's objects, subject, relations and feature bits moved here (S1's outputs identical before and after on CP1's 8 games), plus `features()`, which works the same facts out inside the rule sandbox; `gb/core/contract.py` hands it to every rule. `tests/test_gb_vocab.py`. The other primitives (`translate`, `slide`, ...) come with H1 proper |
| S3 guard fitter | built 16 Oct (`s3_guards.py`, `make gb-guards`): near-miss rules split by a guard learned from S1's facts into a narrowed rule and an exception, replayed; the book and T1 before and after. Tested on synthetic games only (`tests/test_gb_guards.py`): its first run on CP1's candidates IS G0, so it refuses them without `G0=1` |
| S2, S3's grading, priorities and tiers, S4, S5, H2, H3 | not built yet (plan section 7) |
