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
| **S0 MASK** | `s0_mask.py` | The decoration mask misses the last cells of a bar (they change once per attempt, late), so the same screen seems to end two ways and every rule there is refused trust (G0, m0r0) | Rebuilds each level's evidence from the recording with each bar in the mask run to the screen's edges; off, it is CP1's evidence exactly | no | G0b |
| **S1 SEE** | `s1_see.py` | The LLM must work out from pictures what moved and what blocked it, and we can't tell "couldn't see it" from "couldn't write it" | Finds objects, matches them before and after, lists relations (touching, what's ahead), and checks whether that description rebuilds the after-screen (**closure**) | no | G1 |
| **H1 VOCAB** | `h1_vocab.py` | Rules are long raw numpy with boundary bugs; 44% of CP1's answers ran out of room | Up to 16 tested grid primitives (`translate`, `is_blocked`, `recolour`…) the rules may call | no | G1 |
| **S2 WRITE** | `s2_write.py` | One rule tried to explain several mechanisms at once | Asks the LLM for one mechanism per rule, with S1's facts and H1's primitives in the prompt | **yes** | G1 |
| S2 POOL | `s2_pool.py` | Each S2 run starts from scratch, so it loses rules an earlier run found (G1: ft09's red-click rule) | Pools finished runs' candidates on the same evidence into a new run; S3 and S4 then choose and carry the book by CP1's procedure | no | (exploratory) |
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
| S1 SEE | built 16 Oct, all 8 steps (VERSION 3 the same day: a tie in what an object sits on goes to the background) (`s1_see.py`, `make gb-see`): objects, correspondence, closure and "explained"; relations, feature bits, roles, the exploration matrix and the facts text (pinned by a golden test); `tests/test_gb_see.py` (21 tests) |
| H1 VOCAB | built 16-17 Oct (`h1_vocab.py`): S1's relation code, `features()`, ten primitives and `CARD`; registered as `glassbox_vocab_v1` (frozen). `tests/test_gb_vocab.py` |
| S3 guard fitter | built 16 Oct (`s3_guards.py`, `make gb-guards`): near-miss rules split by a guard learned from S1's facts into a narrowed rule and an exception, replayed; the book and T1 before and after. Tested on synthetic games only (`tests/test_gb_guards.py`): its first run on CP1's candidates IS G0, so it refuses them without `G0=1` |
| S0 MASK | built 16 Oct (`s0_mask.py`, `make gb-mask`), after G0: `[mask] complete_bars`. Off, it rebuilds CP1's evidence array for array (`tests/test_gb_mask.py`); on, m0r0's conflicting cases go 13 → 0 and tu93's 11 → 0 |
| S3 CHECK, grading v0 | built 16 Oct (`s3_check.py`, `make gb-check`): CP1's procedure (each candidate with its binding; the book as CP1 composed it). `cp1_import.py` uses the same functions |
| S4 CARRY v0 | built 16 Oct (`s4_carry.py`, `make gb-carry`): CP1's re-binding by search on the fit split; T0, T1. With S0 off, S0 → S3 → S4 gives CP1's book, T0, T1 and every re-binding on all 8 games (`make gb-g0b-control`; dc22 in `tests/test_gb_regression.py`) |
| S2 WRITE v1 | built 17 Oct; run in G1 (`docs/3-glassbox/g1-offline-rules.md`) |
| S2 POOL | built 18 Oct (`s2_pool.py`, `make gb-pool`): G0b's and G1's candidates pooled, `pool_report.md` beside both books; `tests/test_gb_pool.py` |
| S4 CARRY v1 | 18 Oct: the role tie-break between exact bindings (`[carry] roles_in_bind`, off by default so S4 is still CP1's) and the online wait (`ready`); `tests/test_gb_carry.py` |
| S5 ACT v1 | built 18 Oct (`s5_act.py`): prune (carried colour no-ops; the book's no-op rules, re-bound on each level), goal hints, `GbMap`; the agent hook `EVAL_GB=1` (`EVAL_GB_BOOK=<run>`), 16 lines tagged `# [gb]` in `action.py`; the replay (`make gb-replay`, the offline counterfactual on dev games, refused before prereg section 5a); `tests/test_gb_act.py`. Not built: probes, IMAGINE, contact goals |
| H3 TEMPLATES v0 | built 18 Oct (`h3_templates.py`, `--stage templates`): the plan's smoke test only (a throwaway catalog tried on lp85 and ls20; 0 of 21 templates adopted); refuses any game outside the seen tier; `tests/test_gb_templates.py` |
| S3's priorities and tiers, S4's repairs, H2 | not built yet (plan section 7) |
