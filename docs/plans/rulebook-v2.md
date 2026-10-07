# Rulebook v2: rules that carry over

*Revised plan, written 6 Oct 2026. It replaces the build order in `docs/plans/llm-rulebook.md` after Tier 0a's G1 fail. The v1 record is the 6 Oct write-up `rulebook-full-writeup.md` (kept outside the repo; `docs/plans/rulebook-stageA.md` and `rulebook-tier0a.md` hold the same results). v1's results and verdicts stand unchanged. Every v2 gate is pre-registered before the run it judges.*

---

## 0. The short version

**What v1 showed.** The local model (Qwen3.8-27B) often understands a game's mechanic, but the rules it writes keep level-specific facts inside the code. ft09's rule had level 1's colour baked in and was wrong on 59% of level 2's changes. The same mechanic, written as "the level's other main colour", was right 98.9% of the time. Rules phrased relationally (dc22) transferred with zero errors, and "this does nothing" rules transferred almost perfectly. Meanwhile, the "100% or not trusted" bar threw away the 90–97% movement rules, and because every rule had to reproduce the whole board, multi-part mechanics stalled (cd82 at 45% at best, tr87's glyph cycling at 0%).

**What v2 changes.**

1. **Rules split a mechanism from its level bindings.** Colours, positions and sizes live in a `PARAMS` block that is re-fitted automatically on each new level, with no LLM call.
2. **Rules may claim part of a change.** Several small rules can share an action group. Trust means "exact on every cell claimed, on every level seen so far".
3. **The rule book steers Goose before it plans.** Carried no-ops, goal hints and probes where hypotheses disagree feed into Goose's choices from the first move of a new level. Planning comes later, for games where a completion test fits.
4. **Across games, v2 carries ways of learning, never game rules.** That means a helper library, a catalog of mechanic schemas (adopted only when exact on the new game), a playbook of lessons for the rule writer, a pretrained prior for Goose and, last, a self-trained rule writer. All of them are built from the 8 dev games and the 2 games v1 already used, frozen, and tested on held-out games.

**The ladder.** Each checkpoint ends with a frozen configuration, a pre-registered verdict and a report section that stands on its own. From CP1 on, stopping at any rung leaves a complete method.

| Checkpoint | Adds | Standalone product if we stop here | Gate (summary) | Target |
|---|---|---|---|---|
| **CP0** Foundation | v1 closeout; transfer metrics; near-miss refinement day | v1's diagnosis plus measuring tools (analysis, not yet a method) | none; its result steers CP1 | 7–8 Oct |
| **CP1** Transferable rules (offline) | `PARAMS` and automatic re-binding; trust on every level; partial, composable rules; bandit refinement | an offline rule learner whose rules re-ground on new levels | the re-bound book beats both baselines on the next level on ≥ 3 of 8 dev games (v1: 1); wrong ≤ 5% everywhere | 8–14 Oct |
| **CP2** Rule-guided Goose (online) | carried no-ops, goal hints and probes steer Goose; no planner | **the first standalone agent**, with a 25-game confirm | dev sweep: levels ≥ reference, wins ≥ losses; confirm: v1's adoption rule | build 14–22 Oct; Confirm 1 23–27 Oct |
| **CP3** Cross-game layer | helper library, mechanic schemas, playbook, Goose prior; built from dev games and frozen | rule-guided Goose with learned cross-game priors | held-out offline A/B: more trusted rules and coverage on ≥ 5 of 8 held-out games, no rise in wrong predictions | 19–31 Oct |
| **CP4** Two-layer model and planner | finders → object state; completion test; planner; executor | rule-guided Goose that plans where its rules suffice | ≥ 2 dev games with a fitted, path-checked completion test; replay-to-level clears ≥ 1 level Goose didn't | 2–22 Nov; Confirm 2 23–28 Nov |
| **CP5** Self-trained rule writer | hindsight-relabelled fine-tune on Colab | a rule writer trained for this task, usable at any rung | beats the base writer on the held-out offline A/B at equal budget | 2–20 Nov, if CP1–CP3 look promising |

**The calendar this is built around.**
- 9 Oct: report table of contents and Meeting Log 1.
- 30 Oct: full report draft.
- 2 Nov: the idea must be concrete. Improving the same idea afterwards is allowed; switching to a new one is not.
- 7 Dec: final report.

v2 is therefore declared on 2 Nov as one idea in five stages (section 7). Work on CP4 and CP5 in November improves the declared idea rather than replacing it.

**The safety net.** CP2 is the first agent. Its 25-game confirm runs 23–27 Oct, before the draft, so the report has a complete, confirmed method whatever happens to the later rungs.

---

## 1. Why the plan changes

### 1.1 Transfer failed on the rules' form, not on the model's understanding

| Game | What the rule said | On the next level | Reading |
|---|---|---|---|
| ft09 (Tier 0a, one short run) | clicking a blue block turns it red, with red hard-coded | applied to 9,998 cases, right on 0; 59% of level 2's changes predicted wrongly | colour baked in |
| ft09 (Stage A, many runs) | the same mechanic, written as "the level's other main colour" | right on 98.9% of 2,221 cases | same mechanic, general form |
| vc33 | clicking the upper blue block shifts the boundaries 4 columns, in level-1 layout terms | applied to 0 of 134 changing cases | positions baked in |
| tr87 | ACTION4 moves the marker one slot right, wrapping (exact on 13,903 moves) | applied to 0 cases | layout baked in (most likely) |
| lp85 (Stage A) | the right plus rotates the ring one step | applied to 0 moves | positions baked in |
| dc22 | swaps written relative to neighbours ("the tile above", "two columns left", "only when…") | 517 cases, 0 errors; beat both baselines | relational form travels |
| no-op rules | clicks on a colour do nothing | vc33 621 of 621 right; m0r0 921 of 932 | the most reliable transfer we have |
| memory baseline | repeat what the same screen and action did | equal to "nothing changes" on every transfer level | raw experience doesn't travel |

The v1 writeup already names the cause for ft09: the evidence. Many runs pushed the model to a general rule; one short run let it bake in a colour. A scored run has only one run, so the variety has to come from somewhere else. The next level is the obvious source (CP1).

### 1.2 The exact-or-nothing bar threw away most of what the model learned

- **Near-misses counted as nothing.** Movement games produced 90–97% rules (tu93 92% and 91%, m0r0 96% and 96%, ls20 97%, dc22's fourth button 92%), none of which counted toward coverage.
- **Multi-part mechanics stalled.** A rule had to reproduce the whole board after each move, so a rule that got most parts right still scored as wrong. cd82 never passed 45%, tr87's glyph cycling stayed at 0%, and lp85's ring took four rounds to get one position-bound rule.
- **A settings quirk:** round 2 only ran when a group had nothing admitted, so m0r0's 96% rules never got a second try.
- **Overruns:** 22% of Tier 0a answers ran out of room before writing code.
- **Hidden state:** conflicting outcomes appear on g50t (93), ls20 (71), m0r0 (13) and tu93 (11).
- **Feedback rounds did most of the work.** Round 2 produced the exact rules for tr87, vc33, dc22 and half of ft09.

### 1.3 What can and can't carry between games

ARC-AGI-3 is built so that levels compose. Its technical report says difficulty is meant to come from combining what the player learned earlier in the same game, so later levels should need concepts from earlier ones [S1]. Rodionov's prompt tells his agent the same thing: later levels usually extend earlier mechanics [S3].

Games are the opposite. Each one is hand-crafted to test adaptation to novelty [S2], so a specific rule from one game says nothing reliable about another. What recurs is the *shape* of mechanics (move with collisions, toggle a colour, cycle a selector), not the rules themselves: tu93, m0r0 and ls20 are all movement games, but nothing guarantees the same step size, blockers or goal.

What people carry between games is priors. In Dubey et al.'s experiments, masking general priors such as "objects exist" and visual consistency slowed human players from about 2 minutes to over 20 [S14]. Priors can also mislead. The ARC Prize audit of 160 frontier-model traces found three recurring failures [S2]:

1. **True local effect, false world model.** The model saw what an action did but never turned it into a rule about the game.
2. **Wrong abstraction from training data.** Models mapped games onto Tetris, Sokoban, Frogger and others. A surface resemblance became a whole theory, and actions were wasted testing it.
3. **Solved the level, didn't learn the game.** A level-1 win hid a wrong primitive, which then confidently misled level 2. The audit's point: without a check on *why* a level was won, the misconception carries forward.

So v2 carries different things at the two scales:

- **Level to level:** the mechanisms themselves, re-bound to the new level and kept only while they stay exact.
- **Game to game:** the means of learning (perception helpers, mechanic templates, lessons for the rule writer, the rule writer itself). Each is accepted on a new game only when the evidence there confirms it.

### 1.4 What other systems do, and what we take from them

All the strong results below come from frontier models on the public games, and several authors warn that public-set scores may reflect saturation or contamination [S3][S8]. What carries over to our setting is the design, not the scores.

| System | How it handles transfer | What v2 takes |
|---|---|---|
| Tycho [S4] | One program per game: mechanics shared by all levels, plus a start state built for each level from its first frame. The renderer may abstain on cells it can't determine, and verification reports accuracy on claimed cells and coverage separately. The workspace persists across levels; nothing is shared across games. Automatic repair produced much more exact models but a lower score than building a model on request. | bindings vs mechanism (CP1); abstention (CP1); "exact is not the same as useful" (CP2) |
| Rodionov component study [S3] | The verification prompt forbids hard-coded layouts and asks for level-specific facts in a separate data structure. The verifier replays every attempt from level 1 onward. Scheduled simplification usually helped but hurt the weakest model setting. Every game starts from a fresh workspace. | every-level trust and checker-gated simplification (CP1); the held-out protocol (§2.4) |
| OPINE-World [S5] | Writes its own object extractor. A Bayesian "ontology error" sends exploration toward objects whose behaviour current types don't explain. Exact replay admits models. Solved 20 of 25 games. | finders (CP4); exploration toward the unexplained (CP2) |
| Kepler [S6] | A world model in two layers: named finders that turn pixels into objects, and a simulator written over those objects. | the two-layer model (CP4) |
| NVIDIA OO Agents [S7] | Encodes the grid into the few fields that drive the game, and predicts on those. | a compact planning state (CP4) |
| Agno ARC-AGI-Arcade [S8] | Per-game manuals of verified mechanics, hazards and corrections. Weaker models given GPT-5.6's manuals scored far higher (Gemini-3.7-Flash 37.33 → 96.42; open-weights GLM-5.2 7.92 → 83.46), but the manuals were for the same games. | weaker models can use written lessons (CP3 playbook); read as a ceiling, not a cross-game estimate |
| BDR-Pro agent [S9] | No LLM at runtime. Carries "goal colours" (colours that shrank when the score rose) into later levels. Its archetype solver activates only after the game confirms it. | goal hints (CP2); evidence-gated schemas (CP3) |
| Graph exploration, Rudakov et al. [S20] | Training-free: segments frames and ranks actions by visual salience. Placed 3rd on the preview's private leaderboard. | a cross-game prior for Goose (CP3) |
| WorldCoder [S10], LILO [S11], TheoryCoder and TheoryCoder-2 [S12], PoE-World [S13] | Code world models that transfer by reusing and editing program pieces; libraries learned by compressing and documenting solutions; general abstractions grounded per environment; world models built from many small causal programs. | helper library and schemas (CP3); composition (CP1) |

**Open-weights reference points, for calibration [S22]:**
- Tufa Labs' Duck harness with Qwen 3.6 27B on one GPU: a self-measured mean score of about 1.6.
- The Polyphony Agent (self-hosted Qwen 3.6) scored 19.8% on the community board.

### 1.5 Design principles

| # | Principle | Why |
|---|---|---|
| P1 | Write the mechanism once; bind level facts separately | ft09, vc33, tr87 and lp85 failed on baked-in constants; Tycho and Rodionov make the same split [S3][S4] |
| P2 | Trust means exact on everything claimed, on every level seen | variety across levels forces generality; Rodionov's verifier replays all levels [S3] |
| P3 | Partial and composable beats all-or-nothing | 90–97% rules counted for nothing, and whole-board grading sank multi-part mechanics; Tycho abstains, PoE-World composes small rules [S4][S13] |
| P4 | Spend the knowledge that already transfers, immediately | no-ops transferred at 99–100%; goal colours carry without an LLM [S9] |
| P5 | Use rules to choose what to try before using them to plan | a 95% rule is safe for guidance (no compounding error); in Tycho, extra exactness alone didn't buy score [S4] |
| P6 | Across games, carry ways of learning, not rules | each game is built to be novel, so its rules don't carry, though mechanic shapes recur [S1][S2]; code pieces and libraries transfer [S10][S11][S12] |
| P7 | Priors are hypotheses, adopted only when exact on the new game | familiar-game analogies derailed frontier models [S2]; evidence-gated archetypes [S9] |
| P8 | Measure transfer at both scales, with held-out games kept clean | the strongest published systems start every game fresh [S3][S4], so a cross-game effect only counts if the held-out protocol is clean |

---

## 2. How the ladder works

### 2.1 Spine and branches

```
CP0 foundation
 └── CP1 transferable rules (offline)
      ├── CP2 rule-guided Goose (online) ──► Confirm 1 (23–27 Oct)
      │    └── CP4 two-layer model + planner (Nov)
      ├── CP3 cross-game layer (dev-built, frozen) ──► slots into CP2 / CP4
      └── CP5 self-trained rule writer (Colab) ──────► slots into any rung
                                                  Confirm 2 (23–28 Nov): furthest passed configuration
```

The spine (CP1 → CP2 → CP4) is the agent. CP3 and CP5 are branches: each produces a frozen artifact that slots into whichever spine configuration exists at the time.

**Where each recommendation from the 6 Oct review lands.** All twelve are in the plan, plus v1's allowed follow-up day.

| # | Recommendation | Where |
|---|---|---|
| 1 | Separate the mechanism from level bindings; re-fit bindings automatically | CP1, §4.1–4.3 |
| 2 | Trust means exact on every level so far; two-level prompts; checker-gated simplification | CP1, §4.4 |
| 3 | Partial claims, with several small rules per group | CP1, §4.5 |
| 4 | Carry no-op masks and goal hints into Goose from the first move | CP2, §5.2–5.3 |
| 5 | A verified helper library built from the dev games | CP3, §6.2 |
| 6 | A catalog of mechanic schemas, adopted only when exact | CP3, §6.3 |
| 7 | A cross-game playbook for the rule writer | CP3, §6.4 |
| 8 | Fine-tune the rule writer on hindsight-relabelled rules | CP5, §9 |
| 9 | A cross-game prior for Goose's network | CP3, §6.5 |
| 10 | A two-layer world model: finders plus mechanisms | CP4, §8.1 |
| 11 | Let the rule book steer exploration first and plan second | CP2 (the whole rung); planning in CP4 |
| 12 | Measure transfer at both scales; reword the no-sharing rule | §2.4, §3.3 |
| — | v1's allowed day: near-miss refinement, done as a bandit | CP0, §3.2; CP1, §4.6 |

### 2.2 What every checkpoint must have

1. A build list with file paths and flags.
2. A standalone product: a frozen configuration, passing tests and a checked "off" path (v1 §3.10: with the flag off, a short CPU run matches the pre-change commit move for move).
3. A pre-registered gate, written and committed before the run it judges.
4. Expectations, including rough odds of passing (for planning only).
5. Pros, cons, risks and mitigations.
6. Cut lines: the minimum version if time is short.
7. The report paragraph if we stop there.

### 2.3 When a gate fails

| Fails | What still happens |
|---|---|
| CP1 | CP2 still runs, with v1-form rules plus the LLM-free parts (Goose's dead-click filter carried by colour, goal hints). CP4 narrows to games whose rules are exact. CP1's result is reported as a diagnosis with numbers. |
| CP2 dev gate | Confirm 1 is skipped. The report covers CP1 offline plus CP2's mechanism metrics. CP3 continues offline, and Confirm 2 decides the headline. |
| CP2 confirm | The method stands as built and measured. The headline is "not adopted", with the funnel showing where it stopped (as v1 planned). |
| CP3 | The layer isn't adopted. The held-out A/B is reported as a result in its own right. |
| CP4 | Report the funnel: finders, completion test, plan found, live success. |
| CP5 | Report it as an experiment; the base model stays. |

### 2.4 Honesty rules (v1's, plus four)

**Kept from v1:**
- pre-registration before each run;
- anything tuned on dev games is labelled as such;
- the "off" path is checked for isolation;
- the world model uses its own random stream;
- the response cache makes offline reruns exact;
- `paired_compare --expect` blocks verdicts when runs are missing.

**New:**

1. **Cross-game artifacts come only from the 8 dev games and the 2 seen games** (or external games). They are frozen and hashed in `artifacts/registry.json` before any held-out use.
2. **Held-out tiers**, drawn now and recorded in the pre-registration:

   | Tier | Games | Use |
   |---|---|---|
   | Dev (8) | tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82 | building, tuning, offline gates |
   | Seen (2) | ls20, lp85 | used in Stage A or the smoke test; reported separately |
   | A/B-offline (8) | ar25, lf52, re86, s5i5, sc25, sp80, su15, tn36 | offline held-out A/Bs for CP3 and CP5; otherwise only inside a confirm |
   | Untouched (7) | bp35, cn04, ka59, r11l, sb26, sk48, wa30 | only ever run inside a confirm |

   The draw is stratified so that both held-out tiers get 3 of the 6 games that play without ACTION7. Python 3.12, `random.Random(2026)`: first `sample` 3 of the sorted six (ar25, bp35, lf52, sb26, sk48, su15), then 5 of the sorted other nine (cn04, ka59, r11l, re86, s5i5, sc25, sp80, tn36, wa30). The recorded list is what counts, whatever another Python version would draw.
3. **Sealed per-game results.** In Confirm 1, per-game results on the A/B-offline and untouched tiers go to a sealed file that stays closed until CP3's artifacts are frozen. Before then, only the aggregate and the dev and seen tiers are read.
4. **Reworded constraint.** v1's "never share learned rules across scored runs" becomes: no learning between scored runs; frozen, declared artifacts built beforehand are allowed and listed in the registry.

---

## 3. CP0 — Foundation (7–8 Oct)

**Goal.** Close v1 cleanly, use v1's allowed follow-up day, and build the measuring tools every later rung needs.

### 3.1 Close v1

- Record Tier 0a (G1 FAIL, 2 of 8) and Stage A (NO-GO) as final. v2 metrics may be computed on v1 rules for comparison, but v1 verdicts are never re-scored.
- State plainly in the plan history that this plan replaces v1's fallback ("narrow the build to the games and groups that pass"). It is motivated by v1's diagnosis, and its gates were registered before any v2 model run.
- Commit this file and `docs/plans/rulebook-v2-prereg.md`, which holds the gates for CP1–CP3 and the tier lists. The CP4 and CP5 gates are added before their runs.

### 3.2 The allowed day: near-miss refinement with a bandit (v1 §9, option 1)

- **Groups (7):** tu93 ACTION1 (92%) and ACTION3 (91%); m0r0 ACTION1–4 (96%, 96%, 91%, 90%); dc22 ACTION4 (92%).
- **Method (REx-style [S15]):**
  - Every candidate rule is an arm.
  - Draw each arm's chance of success from a Beta distribution centred on its accuracy, penalised by how many times it has already been refined.
  - Refine the arm with the highest draw, showing up to 6 of its failing cases (counterexamples first).
  - The child joins the pool.
  - Stop a group at its first exact rule.
- **Budget:** 12 refinement calls per group, 84 calls in all, about 2 GPU hours at Tier 0a speeds.
- **Fixes the quirk:** admitted-but-inexact rules, such as m0r0's 96% pair, are now refined.
- **Pre-registered expectation:** 2–4 of the 7 groups reach exact.
  - 3 or more exact: feedback does convert near-misses, and CP1 keeps refinement central.
  - Exactly 2: CP1 goes ahead as written.
  - 1 or fewer: the misses are hidden state or edge cases, and CP1 prioritises partial claims and `api.t`.
- **Label:** tuned on dev games.

### 3.3 Measurement scaffolding (used by every later rung)

| Piece | What it does |
|---|---|
| Transfer, three ways | T0 is the untouched book (v1's L2). T1 is after automatic re-binding, with no LLM. T2 is after one repair call offline, or after k probe moves online. |
| Fit / score split | Re-binding and repair see only the first 300 moves of the transfer level; scores use the rest |
| Claimed-cell metrics | claimed-cell exactness; cell coverage (share of changed cells predicted right); case coverage (v1's strict metric, still reported) |
| Held-out tiers | drawn and recorded as in §2.4 |
| Sealed results | per-game held-out results from Confirm 1 written to a sealed file |
| Artifact registry | `artifacts/registry.json`: name, hash, build date and source games for every frozen cross-game artifact |
| Failure taxonomy | adds: level-bound rule, binding ambiguous, binding failed, claim too narrow, schema mismatch, negative transfer |

**Effort:** 1–1.5 days.

**If we stop here:** we have v1's diagnosis, the near-miss result and the tools. That is analysis, not yet a method; CP1 is the first rung that is one.

---

## 4. CP1 — Transferable rules, offline (8–14 Oct)

**Goal.** Rules whose mechanism survives a level change, with level facts re-fitted automatically. This rung covers recommendations 1–3.

### 4.1 The rule contract, v2

```python
RULE = "ACTION1 moves the agent block up one step if the cells above it are background"

PARAMS = {                     # every level-specific constant, named by its role
    "agent": Colour(),         # filled per level by bind() and/or the enumerator
    "step":  Int(1, 8),
}

def bind(board0, api):         # optional: a guess at PARAMS from the level's first board
    ...

def applies(board, act, api, p):
    ...

def predict(board, act, api, p):
    # returns (next_board, claimed): claimed is a 64x64 bool array of the cells
    # this rule vouches for. Every other cell is UNKNOWN as far as this rule goes.
    ...
```

- **Slot types:** `Colour()`, `Int(lo, hi)`, `Offset(max_abs)`, `Region()` (an index into `api.layout`), `Choice([...])`.
- **Backward compatible:** a v1 rule reads as `PARAMS = {}` with every cell claimed, so v1 rules and caches still score.
- **New `api` fields:**
  - `level`: the level index;
  - `colours`: the colours on the level's first board;
  - `t`: moves since the attempt started (a light hook for hidden state);
  - `attempt`.
- **System prompt addition:** levels share mechanics but differ in colours, positions and sizes. Write the mechanism once, put anything that could differ into `PARAMS`, and claim only the cells you can explain.

### 4.2 Literal lint

An AST pass over `applies` and `predict` flags:

- colour literals compared with board values (use a `PARAMS` slot or `api.background` instead);
- coordinate literals above 3 used in indices or slices;
- literal coordinate pairs.

Warnings go into the refinement feedback ("move these into `PARAMS`"). A rule with warnings can still be trusted on its own level, but it is marked *level-bound* and left out of T1 until it's rewritten. The lint is a heuristic; the checker remains the judge.

### 4.3 The re-binding enumerator (no LLM)

- **Domains:** colours on the new level's first board plus colours seen in changed cells; `Int` ranges; regions from `api.layout`; small offsets.
- **Search:** every combination, up to 4,096 (a rule with more must supply `bind()`). Combinations are filtered by successive halving on the fit split: 10 cases, then 50, then all. A binding survives only if it is exact on every claimed cell.
- **Start point:** `bind()`'s guess is tried first. The enumerator confirms or corrects it once moves arrive.
- **Ties:** every exact binding is kept as a competing hypothesis. Offline, T1 uses the one with the most applicable cases; online, CP2 probes them.
- **Sandbox change:** add a batch call so one spawned worker evaluates many bindings; spawning a process per binding would dominate the cost.
- **Cost:** seconds for a typical rule. The worst case is a few minutes: 4,096 bindings × 10 cases at v1's 5 ms-per-move limit is about 3.4 minutes for the first halving stage.

### 4.4 Trust on every level seen

- **Definition.** A rule is trusted at level L if, using each level's own binding, it is exact on every claimed cell of every case it applies to, on every level up to L, with no conflicting cases.
- **Regression.** Any repair or simplification is rechecked on all earlier levels.
- **Two-level prompts.** After a level-up and the fit window (≥ 300 moves or ≥ 30 changing cases), a group's prompt shows:
  - up to 3 cases from each level, chosen so the colours or layout differ;
  - the current rule, its binding results and its errors.

  The instruction is one mechanism for both levels, with whatever differs moved into `PARAMS`.
- **Checker-gated simplification.** Once a rule is trusted, one call asks for a shorter, more general version. The new version is accepted only if it stays trusted on every level and is shorter or has fewer lint warnings. Rodionov found simplification usually helped but hurt his weakest setting [S3]. A local 27B is very likely weaker than any model he tested, so the checker decides.

### 4.5 Partial claims and composition

- **Grading per case.** A case is right if every claimed cell matches. Gain is the number of changed cells claimed and right, minus the cells claimed as changing that didn't change.
- **Admitted:** applies to ≥ 20 moves, ≥ 95% claimed-exact, gain > 0, ≥ 20 changed cells predicted right in total, and ≤ 5 ms per move (as in v1). The cell floor blocks rules that claim almost nothing.
- **Trusted:** admitted, 100% claimed-exact on every level so far, and no conflicts.
- **The book:**
  - up to 4 rules per group;
  - a new rule joins only if it explains cells the others leave unclaimed, on ≥ 10 cases;
  - if two rules claim the same cell with different values, that cell is UNKNOWN and both rules are flagged for repair.
- **Residual prompting.** For a partly explained group, show the model the changed-but-unclaimed cells (boxed) and ask for a rule for those alone. This is PoE-World's idea of many small causal programs [S13], applied under our checker.
- **Metrics:** claimed-cell exactness, cell coverage, and v1's strict case coverage, all reported.
- **Hidden state.** `api.t` lets a rule depend on time within the attempt. Tycho's own example of a display hiding state is tu93's bar, which encodes a hidden move budget [S4]. A rule that reads `api.t` must be graded move by move, because v1's cases merge moves made at different times. Each game's report lists how many conflicting cases the rules using `api.t` resolve.

### 4.6 Refinement policy

The CP0 bandit replaces "round 2 only when nothing was admitted". The offline budget per group is 4 first-round candidates and up to 8 refinement calls, stopping at the first trusted rule.

### 4.7 Evaluation: Tier 0a v2

Everything else matches v1, so any differences come from the contract and policy, not the settings:

- the same 8 dev games and seed-0 recordings;
- the same training-level rule (the first level with ≥ 500 moves);
- the same transfer-level rule (the next level with ≥ 50 distinct cases);
- the same model, with thinking on and v1's 20,480-token answer limit.

The first 300 moves of the transfer level form the fit split for T1 and T2; scores use the rest. The baselines are scored on the same moves, and "memory" may also use the fit split, so every arm sees the same evidence.

**Pre-registered gate.**

| | Criterion | v1, for comparison |
|---|---|---|
| Primary | On the transfer level, the re-bound book (T1: one-step exact rate, unclaimed cells read as unchanged) beats both "nothing" and "memory", with the 95% CI above zero (computed as in Stage A's R1), on ≥ 3 of 8 games | 1 of 8 (dc22) |
| Safety | T1 wrong predictions ≤ 5% of the transfer level's changing cases, on every game | ft09: 59% |
| Reported, not gating | claimed-cell trusted coverage ≥ 80% (target: ≥ 3 of 8); strict coverage; T0 and T2; tokens; overrun rate | strict coverage: 2 of 8 |

**Pass = primary and safety.** Cost: about 5–8 GPU hours, run overnight.

### 4.8 Expectations

| Game | Expectation |
|---|---|
| ft09 | T1 ≥ 90% right on the blue rule's applicable cases, if the model uses a colour slot (Stage A's general form reached 98.9%) |
| dc22 | unchanged, since its rules are already relational (a regression check) |
| vc33, tr87 | positive T1 if the layout constants become regions or offsets; less certain |
| tu93, m0r0 | some movement rules trusted on the cells they claim (the agent block), so cell coverage rises; strict coverage may stay low |
| g50t, cd82 | probably still weak, because of hidden state and multi-part mechanics |

Rough odds of passing: about 60%.

### 4.9 Pros and cons

| Pros | Cons |
|---|---|
| Targets exactly the failure v1 measured | The 27B may not use `PARAMS` well; the lint nudges but can't force it |
| Re-binding costs no LLM calls | Re-binding can't fix a mechanism that genuinely changes between levels |
| Keeps v1's checker discipline and caches | Partial claims add grading complexity and a risk of trivially small claims (handled by the 20-cell floor and the gain rule) |
| Makes everything CP2 carries more useful | Every-level trust can reject a rule that is right on the new level but tied to a quirk of an older one |
| Offline and quick to test | Results are on dev games only, and labelled as tuned there |

### 4.10 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Several exact bindings fit the fit window | keep them all; offline T1 takes the most-applicable one; CP2 probes them |
| Enumerator blow-up | 4,096 cap; extra slots must be fixed by `bind()` |
| Fit window too small on short levels | fall back to `bind()` alone, flagged as such in T1 |
| Prompts grow with two-level examples | cap at 3 cases per level per group; keep the 12,288-token prompt limit |

### 4.11 Cut lines

- **Keep:** `PARAMS`, the lint, the enumerator, T0/T1/T2, claimed-cell grading.
- **Drop first:** the simplification pass, residual prompting, `api.t`.

### 4.12 If we stop here

> We show that LLM-written rules failed to transfer between levels because of their form: level-specific constants inside the code. Separating mechanisms from level bindings, re-fitting bindings without the LLM, and letting rules claim only what they can explain raises transfer from 1 of 8 dev games to N of 8, with wrong predictions held under 5%. The result is a standalone, offline world-model learner.

**Effort:** 4–5 days.

---

## 5. CP2 — Rule-guided Goose, online (14–22 Oct; Confirm 1 on 23–27 Oct)

**Goal.** The first standalone agent: Goose plus a rule book that carries across levels and steers what Goose tries. There is no planner yet. This rung covers recommendations 4 and 11.

### 5.1 What runs when

| Trigger | Fires when | What happens | LLM? |
|---|---|---|---|
| Warm-up | ≥ 2,000 moves on a level with no book | rules for the changing groups, most frequent first | yes |
| Level-up | the level counter rises | goal hints computed; book carried over; re-binding after the fit window; probes for unresolved bindings | no |
| Counterexamples | 5 in one group | a repair call for that group | yes |
| Stall | Goose's stall signal fires and cell coverage is below 80% | a residual call for unexplained groups | yes |

**Per-run budget:**
- 16 requests;
- 48 completions (k = 2 candidates per request, plus code-only retries);
- 400,000 generated tokens;
- a 12,288-token answer limit (v1 used 20,480).

v1's plan of 40 requests, 160 completions and 1.5M tokens doesn't fit beside Goose with the 27B (§5.4).

### 5.2 How guidance enters Goose

Goose still computes its own action distribution. A guide in `action.py` reweights it before sampling:

| Signal | Effect | Default |
|---|---|---|
| A trusted no-op (carried or current) | multiply by α. It is never zero, so a contradiction can still show up, and the first contradiction suspends the rule at once (v1's "trust follows evidence") | α = 0.1 |
| Disagreement: candidate rules or competing bindings predict different outcomes | multiply by (1 + β) | β = 2 |
| UNKNOWN: no rule applies | multiply by (1 + β/2) | |
| Goal hint: a click on a component of a hint colour | multiply by (1 + γ) | γ = 1 |

- **Probe caps:** the β and γ boosts switch off for the rest of a level once boosted actions make up 5% of its moves. At a level's start, at most 30 deliberate probe moves test the carried rules, one applicable case per rule. Goose still samples every move, as in v1's EXECUTE mode, so its random stream and training are unchanged; the guide replaces the sampled action only on these probe moves.
- **Unchanged:** Goose's training, novelty label, return map, masks and walk-back.
- **Off isolation:** with the flag off, the guide is never built, and a short CPU run must match the pre-change commit move for move (v1 §3.10).
- **Flags:** `EVAL_WM=guide`, `EVAL_WM_NOOP_ALPHA`, `EVAL_WM_PROBE_BETA`, `EVAL_WM_GOAL_GAMMA`, `EVAL_WM_PROBE_MOVES`, `EVAL_WM_FIT_MOVES`, plus v1's budget flags.

### 5.3 Goal hints, without the LLM

- At a level-up, compare the board before the winning move with the level's first board. Record up to 2 hints: the colours whose cell count fell to zero or shrank the most.
- On the next level, hints bias clicks on matching components. A hint is dropped after 2,000 moves with no hint-related change.
- This is adapted from the BDR-Pro agent's carried goal colours [S9]. Movement games get little from it; CP4's completion test is the richer version.

### 5.4 Compute beside Goose

The constraint is cache memory, not weights. The 19.6 GiB model, Goose's ~4 GB peak and Windows' 1.4–3.6 GB leave too little cache for v1's online design, which needed about 94,000 tokens of cache for 4 long candidates (v1 §6).

**How much fits (an estimate from v1's figures).** v1's 8-game run had cache for 86,016 tokens at 87% of the card and 97,621 at 90%, which works out to about 84 KB per token. Goose's 4 GB therefore costs about 47,000 tokens of cache. Beside one Goose run, the server can hold roughly 40,000 tokens when Windows holds 3.6 GB, and roughly 65,000 when it holds 1.4 GB. A k = 2 request at full limits needs up to 49,152 tokens (2 × (12,288 prompt + 12,288 answer)); a typical one, with v1-sized answers (about 10,400 tokens on average), needs roughly 30,000–40,000. So it fits when Windows is light and is marginal when Windows is heavy.

| Mitigation | Effect |
|---|---|
| k = 2 and a 12,288-token answer limit | per-request cache drops from about 94,000 tokens to at most 49,000 |
| Close GPU-heavy Windows apps during sweeps | worth about 25,000 tokens of cache |
| One Goose run at a time beside the server | v1 measured about 6 GB spare beside one run (short prompts); a second run would take 4 GB of it |
| Schema-first fitting (once CP3 exists) | many groups never reach the LLM |
| If still short, take turns on the card: the server sleeps during play (vLLM sleep mode, if your version supports it) and Goose's network moves to the CPU during each synchronous call | the server gets the whole card during calls. Without sleep mode the cache size is fixed when the server starts, so parking Goose alone frees nothing the server can use |

**Estimated time.** This assumes 80–150 generated tokens/s with two answers at a time beside Goose; v1 measured 110–290 with four, limited by cache.
- LLM time per run, if the budget is used up: about 30–60 minutes.
- Goose's 100,000 moves: about 14 minutes (117.7 moves/s beside the LLM, measured in v1).
- Dev sweep: the 16 C2 runs one at a time take about 12–20 hours; the 16 C2-free runs about 4 hours.
- Confirm (75 runs, one at a time): about 55–95 hours. That fits the 5-day window, only just at the top of the range; unused budgets and schema-first fitting shorten it. v1's plan estimated 35–45 hours of LLM time with the faster 35B.

### 5.5 Evaluation

**Step 1: offline counterfactual, before any online run.** On the mb_gated_att confirm recordings (seed 0, plus seeds 1–2 if logged), look at the first 2,000 moves of each level from level 2 up. Count:

- *avoidable waste*: moves on actions covered by the previous level's re-bound no-ops that indeed did nothing;
- *masking risk*: covered actions that did something.

Pre-registered expectation: avoidable waste ≥ 10% on the games with carried click no-ops (ft09, vc33, m0r0), and masking risk ≤ 1%.

**Step 2: dev sweep.** 8 dev games × seeds 0–1 × 100,000 moves.

| Arm | What it is |
|---|---|
| A0 | mb_gated_att (reusing the confirm runs) |
| C2 | full guidance |
| C2-free | Goose's own dead-click filter, re-keyed by colour and carried across levels, plus goal hints; no LLM. This isolates what the LLM adds |

**Gate (v1's G3 rule):**
- C2 levels ≥ A0;
- 16 of 16 runs complete;
- wins ≥ losses;
- no game worse on both seeds.

Mechanism check: avoidable waste in the first 2,000 moves of new levels is lower than A0's.

**Step 3: Confirm 1, if the gate passes.** 25 games × 3 seeds × 100,000 moves, judged by v1's adoption rule:
- strictly more than 112 levels;
- wins > losses;
- no game worse on every seed;
- 75 of 75 runs complete.

Report by tier. Per-game results on the A/B-offline and untouched tiers stay sealed until CP3 is frozen.

### 5.6 Expectations

- Gains should concentrate on click games and on the first few thousand moves of each new level.
- The level-2/3 ceiling found in the characterisation sweeps is at least partly about longer sequences in the right order (v1 §1.1). Guidance doesn't solve that; CP4 is aimed at it.
- Rough expectations:
  - Confirm 1: +0 to +6 levels;
  - dev gate: about 50% to pass;
  - adoption rule: about 35–45% to pass.
- If C2-free matches C2, the LLM isn't adding value at this rung. That gets reported plainly, and CP3 and CP4 have to carry the LLM's case.

### 5.7 Pros and cons

| Pros | Cons |
|---|---|
| The first complete agent. It uses imperfect rules safely: nothing runs open-loop, so the 1 − 0.95²⁰ argument doesn't apply | Modest expected gain; it may tie |
| Attacks "Goose resets everything at each level" directly | Soft masks can slow discovery when a new level revives a colour that used to do nothing |
| Cheap at runtime; most of the carry-over needs no LLM | The online LLM budget is tight beside Goose |
| The built-in ablation (C2-free) answers "is the LLM needed?" | Probe bonuses add knobs; they are kept at fixed defaults |

### 5.8 Risks and mitigations

| Risk | Mitigation |
|---|---|
| A wrongly carried no-op hides a new mechanic | α > 0; suspend on the first contradiction; drop the rule if re-binding fails |
| Probes waste moves | the 5% cap and the 30-move start budget |
| LLM latency stretches runs | budgets, the answer limit, one run at a time, taking turns on the card if needed |
| GPU memory spills into system RAM on WSL and silently slows everything | size the server to free memory at start (as `tier0a.sh` already does); log throughput per run |
| The lower answer limit raises overruns above v1's 22% | the code-only retry keeps the reasoning (v1's fix); CP3's helpers and CP5 aim to shorten answers; overruns are logged per run |

### 5.9 Cut lines

- **Keep:** carried no-ops, goal hints, warm-up rule writing, re-binding.
- **Drop first:** disagreement probes, then stall-triggered residual calls.

### 5.10 If we stop here

> An LLM-written, replay-checked rule book carries across levels and steers an exploring agent's choices. Against mb_gated_att on 25 games × 3 seeds × 100k moves, it completes X levels to 112 (adoption rule: pass or fail), and cuts avoidable waste on new levels by Y%. An LLM-free ablation shows what the LLM itself adds.

**Effort:** 5–7 days, plus compute.

---

## 6. CP3 — Cross-game layer (19–31 Oct)

**Goal.** Carry ways of learning across games: a helper library, mechanic schemas, a playbook and a Goose prior. All are built only from dev games, frozen, and tested on held-out games. This rung covers recommendations 5, 6, 7 and 9.

### 6.1 Protocol

- **Sources:** the Tier 0a v2 runs, the CP0 refinement traces and the CP2 dev runs, all on the 8 dev games. External grid games are allowed but optional.
- **Freeze:** by 28 Oct, before any sealed held-out result is opened. Hashes go into the registry.
- **Leave-one-dev-game-out:** rebuild the automatically built parts (library and playbook) 8 times, each without one dev game. This shows whether they help games they weren't built from.
- **Why this is worth testing:** every top system already carries cross-game knowledge, but it's written by hand. Tycho shipped ARC-AGI-3-specific prompts and a starting workspace, and Rodionov's verification variant ships fixed templates and helper programs [S3][S4]. Tycho describes its own cross-run improvement loop as human-mediated [S4]. CP3 asks whether that layer can be learned from dev games instead.

### 6.2 Helper library (recommendation 5)

**Candidate contents:** `objects(board, bg)`, `find(board, colour)`, `move_object(board, obj, dr, dc, blockers)`, `can_move(...)`, `recolour(board, obj, colour)`, `swap(board, a, b)`, `cycle_slots(...)`, `rotate_cells(cells, centre, k)`, `bar_cells(board, api)`, `diff(a, b)`.

**How it's built:**

1. Collect the trusted and best admitted rules from the dev runs.
2. One extraction pass finds repeated operations, writes helpers with docstrings and rewrites the rules to use them. The local 27B does this; optionally, one offline frontier pass, a small slice of the ~$100 budget.
3. You review it. Every helper gets unit tests on synthetic boards (extend `tests/test_wm_core.py`).
4. Rewritten rules must stay trusted (a regression check).

**Limits and placement:**
- **Prompt budget:** at most 30 helpers and 1,500 tokens of signatures and docstrings.
- **Sandbox:** helpers are pure numpy functions over arrays, exposed through `api`.

**Why:**
- Shorter programs should mean fewer edge-case bugs and fewer of the 22% overruns.
- Helpers take objects, not coordinates.
- LILO builds libraries by synthesising, compressing and documenting code, and found the documentation helped its synthesiser actually use the abstractions [S11].
- WorldCoder transferred knowledge between environments by reusing pieces of old programs [S10].
- TheoryCoder-2 learns reusable abstractions from experience and was markedly more sample-efficient than WorldCoder [S12].

### 6.3 Mechanic schemas (recommendation 6)

| Schema | Parameters | Example so far (dev or seen games) |
|---|---|---|
| translate | agent, step, direction per action, blockers, wrap | tu93, m0r0, ls20 |
| push | agent, box, step, blockers | none yet |
| swap with neighbour | agent, direction, condition | dc22 |
| click-recolour / toggle | target colour, new colour, region | ft09 |
| cycle | value sequence (colours or glyphs), target, trigger (button or click) | tr87's glyphs (ACTION1, ACTION2) |
| selector cycle | marker, slots, direction, wrap | tr87's marker |
| rotate group | cells, centre, step | lp85's ring, cd82's shape |
| boundary shift | boundaries, offset per click | vc33 |
| counter / bar | bar cells, change per move or per click | tu93's bar, ft09's bottom bar |
| gravity | movable colours, direction, blockers | none yet |
| paint / stamp | brush, canvas region | cd82 |
| mirror twin | any trusted rule, mirrored direction | tr87 right/left |

- **Fitter:** enumerates parameters for each action group on the current level. A schema is adopted only if it meets CP1's trust bar.
- **Schema-first loop:** try every schema first; this takes seconds to minutes, with no LLM.
  - Fully explained groups skip the LLM.
  - Partly explained groups get a residual prompt showing the best schema's errors.
  - Unexplained groups get the normal prompt, with helpers and playbook.
- **Mechanic-sized only:** no schema describes a whole game. That guards against the familiar-game failure the ARC Prize audit found [S2], in the spirit of BDR-Pro's evidence-gated archetypes [S9]. Dubey et al.'s results argue that general priors are worth having [S14]; the gate makes sure each one is earned on each game.
- **Provenance:** the catalog is hand-designed with knowledge of the 8 dev games and the 2 seen games, so its honest test is the held-out A/B, not leave-one-out.

### 6.4 Playbook (recommendation 7)

- After each dev game, one reflection call reads that game's report (best candidates per group, failure cases, lint warnings, transfer outcome). It proposes 1–3 lessons, each at most 2 lines and naming the games that support it.
- A curator merges entry by entry (add, merge, drop) and never rewrites the whole file. Dynamic Cheatsheet showed the value of keeping short, reusable snippets rather than transcripts [S17]. ACE, which builds on it, adds entry-by-entry updates because monolithic rewrites collapse into thin summaries [S18].
- Cap: about 25 entries, about 1,000 tokens. You review it, then it is frozen.
- **Seed entries** (the first four come from v1's results, the last from v2's contract):
  - never hard-code colours or positions (ft09, vc33);
  - handle edge cases explicitly: walls and other blockers, timers (v1's 90–97% movement rules missed on these);
  - if one direction is exact, write its mirror (tr87: right exact, left 0%);
  - leave bars and HUD cells unclaimed unless the rule really explains them (ft09's empty mask forced its rules to predict the bar);
  - claim only what you can explain.
- Agno's arcade shows weaker models using written manuals very well [S8]. Those were same-game manuals, though, so expect a much smaller effect across games.

### 6.5 Goose prior (recommendation 9)

- **Pretrain** Goose's change predictor on dev-game recordings: seed 0 at least (8 games × 100,000 moves, 0.8 million transitions), and seeds 1–2 if they were logged (2.4 million). The label is whether the masked screen changed.
- **Use:** initialise Goose from the prior at game start and at each level reset, instead of from random weights.
- **Check on dev games only,** leave-one-game-out: train on 7, test on 1, with 20,000-move runs. Measure the early effective-action rate (first 1,000 moves of each level) and levels. Adopt only if the prior is at least as good as a fresh start on ≥ 6 of 8 folds. Held-out games stay untouched.
- **Why:** a training-free agent that ranks actions by visual salience placed 3rd on the preview's private leaderboard [S20], and humans lean on "objects matter" priors [S14].

### 6.6 Evaluation and gate

| Test | Role | Pass |
|---|---|---|
| Offline held-out A/B: the CP1 pipeline with vs without the frozen layer, at the same budget, on the 8 A/B-offline games | **gate** | more trusted rules and higher cell coverage on ≥ 5 of 8; transfer-level wrong predictions and overrun rate no higher |
| Leave-one-dev-game-out (library, playbook) | reported | non-negative effect on ≥ 6 of 8 |
| Goose prior folds | adoption rule for the prior only | ≥ 6 of 8 folds (§6.5) |
| Online: CP2 + layer on dev games | regression check only (dev games built the layer) | no game worse on both seeds |
| The real held-out online test | headline | Confirm 2, untouched tier |

The A/B needs seed-0 recordings of the 8 A/B-offline games from the mb_gated_att confirm. Check before 28 Oct that they were logged like the dev games'.

### 6.7 Expectations

- Fewer overruns and fewer tokens per trusted rule.
- Schemas should settle symmetric pairs and simple movement on several held-out games; odd mechanics still need the LLM.
- The playbook's effect is smaller, but nearly free.
- The Goose prior is uncertain: it may help click games and hurt movement games.
- Rough odds of passing the held-out gate: about 55%.

### 6.8 Pros and cons

| Pros | Cons |
|---|---|
| The only part of the plan aimed directly at game-to-game transfer | Built from only 8 games, so the catalog may reflect their quirks |
| Learned rather than hand-written cross-game knowledge is a clean research question | Negative transfer is possible: a schema can fit by coincidence, or the prior can bias Goose |
| Cuts LLM calls (schema-first), which eases the online budget | More prompt text, which the 27B may ignore or misuse |
| The held-out protocol makes the claim credible | Uses up the A/B-offline tier for future offline tests |

### 6.9 Risks and mitigations

| Risk | Mitigation |
|---|---|
| A schema is exact by coincidence on a few cases | the same ≥ 20-move and every-level bar as any rule |
| Helpers hide bugs | unit tests, plus a regression check on rewritten rules |
| The playbook bloats or drifts | entry-by-entry curation, a 25-entry cap, review before freezing |
| The prior hurts some games | the leave-one-out adoption rule; fall back to a fresh start |

### 6.10 Cut lines

- **Keep:** the playbook, the six most-used schemas, the schema-first loop.
- **Drop first:** the Goose prior, then automated library extraction (hand-pick 10 helpers instead).

### 6.11 If we stop here

> On games never used to build it, a frozen cross-game layer learned from 8 dev games (helpers, schemas, lessons) changes the rule writer's results by Δ trusted rules and Δ coverage, at Δ tokens. Combined with CP2, it is tested on untouched games in the confirm.

**Effort:** 6–8 days.

---

## 7. The 2 Nov idea statement and the two confirms

### 7.1 The idea, as declared on 2 Nov

> **Rulebook v2.** An exploring agent (Goose) works with a local LLM that writes replay-checked, parameterised rules from Goose's transitions. Rules carry from level to level by automatic re-binding, and they steer what Goose tries. Knowledge carries from game to game as frozen ways of learning: helpers, mechanic schemas, lessons and a trained rule writer. Where a completion test fits, a planner uses the rules to finish levels. The build proceeds in five stages (CP1–CP5), each a refinement of this one idea.

Any work after 2 Nov is a stage of this statement, not a new idea.

### 7.2 Confirm 1 (23–27 Oct): CP2

As in §5.5, step 3. This is the safety net for both the draft and the final report.

### 7.3 Confirm 2 (23–28 Nov): the furthest passed configuration

- **Candidates, best first:**
  1. CP4, with CP3 and CP5 if they passed;
  2. CP2 + CP3;
  3. no Confirm 2, in which case Confirm 1 stands.
- The same adoption rule and tiers apply. The untouched tier is the headline held-out test of CP3.
- If CP4 isn't through its gate by 22 Nov, Confirm 2 runs CP2 + CP3, or is skipped if CP3 failed.

---

## 8. CP4 — Two-layer world model and planner (2–22 Nov)

**Goal.** Plan through levels that need long sequences in the right order: the level-2/3 ceiling that guidance alone won't break. This rung covers recommendation 10, plus v1's Tier 0b.

### 8.1 Finders: the perception layer

- The LLM writes per-game finders from library helpers, such as `find_agent(board)`, `find_targets(board)` and `find_walls(board)`. Each returns object records: name, colour, cells and bounding box.
- Bindings come from finders. For example, the agent colour is the colour of `find_agent(board0)`. A new level re-grounds itself by rerunning the finders on its first board.
- Object state is a dictionary of named objects. It serves as a compact key for the planner's visited set, in the spirit of NVIDIA's OO Agents [S7].
- Finders are built entirely from our own helpers, so the track doesn't depend on a teammate's object layer.
- **Evidence:**
  - Kepler structures its model this way [S6].
  - OPINE-World writes its own extractor and solved 20 of 25 games with it [S5].
  - TheoryCoder grounds general abstractions such as "move to" in each environment with a learned low-level model [S12].

### 8.2 Completion test (v1 §3.5, changed)

- `finishes_level(board, act, api, p)`, with `PARAMS` re-bound per level and a `GOAL` line.
- **"Why did we win" check before carrying anything over [S2].** At each level-up, replay the winning attempt through the book. Rules the winning path depends on that mispredict there lose their carried trust.
- It must fit every won level (as in v1).
- `progress()` only orders the search, and only if it passes v1's ≥ 0.6 hindsight check (as in v1).

### 8.3 Planner and executor (v1 §3.6–3.7, adapted)

- **Search** runs over object states. A move expands only if its predicted change claims every cell of the objects in the state; otherwise it is UNKNOWN and not expanded.
- **Order and limits** (v1): the known path first; breadth-first, then best-first on `progress()`; 200,000 expansions; a 60-second watchdog.
- **Executor:** compare after every move. A mismatch becomes a counterexample, the rule is suspended, and Goose resumes; deaths, surprise level-ups and finished plans are handled as in v1 §3.7. Re-plan from the observed state.
- **Caps** (v1): 10,000 executor moves per run; planning stops on a level after 3 failed plans in a row.

### 8.4 Evaluation and gate (v1's G2, adapted)

The gate passes when both hold:
- ≥ 2 dev games have a fitted, path-checked completion test that passes the in-model re-plan check;
- replay-to-level clears ≥ 1 level that the matching Goose run did not.

If it passes, run a dev sweep (C4 vs C2 vs A0) under v1's G3 rule, then Confirm 2 if time allows.

### 8.5 Expectations

- Success is most likely on click games (ft09, vc33) and on dc22.
- Rough odds: 30–40% to pass the gate, and about 20% that it adds levels in Confirm 2.
- It has the largest upside of any rung: levels past the ceiling.

### 8.6 Pros and cons

| Pros | Cons |
|---|---|
| The only rung aimed at the long-sequence ceiling | The most engineering, and the most points of failure |
| Object state keeps the search small | Finders are fragile on unusual layouts |
| Reuses v1's executor and safety caps | Completion tests may fit without generalising (a v1 risk) |

### 8.7 Cut lines

- **Keep:** finders and the completion test, as analysis (do they fit, and do they transfer?).
- **Drop first:** live planning.

### 8.8 If we stop here

> Where the rule book's completion test fits, planning on a two-layer model (finders over pixels, mechanisms over objects) finishes levels that Goose's matching runs did not. Elsewhere, the funnel shows where it breaks.

**Effort:** 8–12 days.

---

## 9. CP5 — Self-trained rule writer (2–20 Nov, on Colab, if CP1–CP3 look promising)

**Goal.** A rule writer trained on this exact task. It is also the natural home for the "small, possibly self-trained LLM" requirement. This rung covers recommendation 8.

### 9.1 Data

- **Hindsight relabelling [S16].** Every candidate that ran in the sandbox is executed on recorded before-boards from its own group: about 1,035 so far (728 checked in Stage A, 307 in Tier 0a), plus v2's candidates on dev and seen games. Tier 0a's per-game JSON files keep every candidate's code; check that Stage A's results do too. Its outputs become the observed after-boards, claimed cells only. By construction, the code is then an exact rule for that evidence. Prompts are built in the standard format, and the target is `RULE` + `PARAMS` + code.
- **Positives:** every trusted rule, with its real evidence.
- **Filters:** the rule must predict real changes; deduplicated; balanced across games and groups; capped per game.
- **Splits:** train on dev and seen games (Stage A's lp85 and ls20 candidates included), and optionally on external grid worlds such as AutumnBench's 43 environments [S19]. Never train on the A/B-offline or untouched games, including candidates written on them during Confirm 1.
- **Size:** low thousands of examples.

### 9.2 Model and training

| Option | Notes |
|---|---|
| A (first): a small open model, 2B–9B | fast enough to run beside Goose and easy to serve; the 27B stays as the fallback. Qwen3.8's open models start at 27B, so this means an earlier family; the small models used in Probes A and C were deleted on 3 Oct |
| B: QLoRA on the 27B | trains from the bf16 base, not the AWQ checkpoint. Check that vLLM can serve a LoRA on your quantised base; otherwise merge and re-quantise |

- Code-only targets (thinking off) keep sequences short.
- **Compute:** Colab GPU sessions from the 300 credits.
- **Feasibility signal:** someone has published a QLoRA adapter on Qwen3.6-27B for ARC-AGI-3, trained on synthetic simulations. Its own card says first-exposure generalisation is still unsolved [S21].

### 9.3 Evaluation and gate

Offline, on the 8 A/B-offline games, with the CP3 layer and an equal token budget: fine-tuned writer vs base writer. The base-writer arm reuses CP3's with-layer run, so only the fine-tuned arm needs GPU time.

**Pass:**
- more trusted rules on ≥ 5 of 8 games;
- an overrun rate at most half the base's;
- transfer-level wrong predictions no higher.

If it passes, it replaces the base writer in the CP2 and CP4 dev runs and in Confirm 2.

### 9.4 Expectations, pros and cons

- **Likely:** better format adherence and far fewer overruns.
- **Uncertain:** exactness gains (about 30–40%).
- **A real win either way:** a small model's speed eases the online budget.
- **Evidence:** SOAR improved substantially by fine-tuning on hindsight-relabelled successes and failures, and the gains carried into test time [S16]. It had far more data than we will.

| Pros | Cons |
|---|---|
| Makes the LLM genuinely self-trained | The smallest dataset in the plan; risk of learning the dev games' style |
| Could fix overruns and `PARAMS` discipline at the source | Synthetic evidence differs from real evidence |
| A small model eases online compute | LoRA serving on a quantised base is an engineering risk |

### 9.5 If we stop here

> A rule writer trained on hindsight-relabelled rules from the dev and seen games changes the yield of trusted rules on held-out games by Δ, at equal budget.

**Effort:** 5–8 days, mostly unattended training.

---

## 10. Calendar and compute

### 10.1 Calendar

| Dates | Work | GPU | Report milestone |
|---|---|---|---|
| Wed 7 Oct | CP0: close v1, pre-register CP1–CP3, metrics scaffolding | none | |
| Thu 8 Oct | CP0: near-miss refinement day | ~2 h | |
| Fri 9 Oct | | | table of contents and Meeting Log 1 (use §0's ladder) |
| 8–13 Oct | CP1 build | smoke tests | |
| 13–14 Oct (overnight) | Tier 0a v2 | 5–8 h | |
| Wed 14 Oct | CP1 gate | | |
| 14–19 Oct | CP2 build; offline counterfactual | smoke tests | |
| 19–21 Oct | CP2 dev sweep (C2 and C2-free) | ~16–24 h | |
| 19–22 Oct | CP3 extraction and reflection passes (before the confirm takes the GPU) | ~3–5 h | |
| Thu 22 Oct | CP2 gate | | |
| 23–27 Oct | Confirm 1 (CP2) | ~55–95 h | |
| 23–28 Oct | CP3 schemas, playbook curation and tests on the CPU; freeze on 28 Oct | none | |
| 28–31 Oct | CP3 held-out A/B; Goose prior folds; leave-one-out if time | ~12–20 h | |
| Fri 30 Oct | | | full draft: CP0–CP2 with Confirm 1 (per-game results unsealed after the 28 Oct freeze); CP3 status |
| Sat 31 Oct | CP3 gate | | |
| Mon 2 Nov | idea statement (§7.1) | | ideas concrete |
| 2–22 Nov | CP4 build and dev evaluation | daytime | |
| 2–20 Nov | CP5 data on the local CPU, training on Colab, if promising | Colab | |
| 23–28 Nov | Confirm 2 | ~55–95 h | |
| 29 Nov–7 Dec | write-up | none | final report, 7 Dec |

### 10.2 Slip rules

- **CP1 isn't through its gate by 16 Oct:** start CP2 with v1-form rules plus the LLM-free carry-over, and keep CP1 going offline.
- **The CP2 dev gate hasn't run by 24 Oct:** skip Confirm 1. The draft reports CP1 and the dev sweep, and Confirm 2 decides the headline.
- **Confirm 1 runs long:** time the first 10 runs. If the projection passes 27 Oct, the CP3 A/B moves to 29–31 Oct; the confirm itself is never shortened.
- **The CP3 freeze slips past 30 Oct:** drop the Goose prior and keep 6 schemas.
- **CP4 isn't through its gate by 22 Nov:** Confirm 2 runs CP2 + CP3.
- **CP5** starts only if CP1–CP3 results look promising, which is also when the Colab credits come into play.

### 10.3 Effort summary

| Rung | Person-days | GPU |
|---|---|---|
| CP0 | 1–1.5 | ~2 h |
| CP1 | 4–5 | 5–8 h |
| CP2 | 5–7 | dev ~16–24 h; Confirm 1 ~55–95 h |
| CP3 | 6–8 | ~15–25 h |
| CP4 | 8–12 | dev runs; Confirm 2 ~55–95 h |
| CP5 | 5–8 | Colab, plus ~5 h of local evaluation |

That is about 30–40 person-days in roughly 8 weeks, which is why CP4 and CP5 are the rungs most likely to be cut.

---

## 11. Evaluation summary

### 11.1 Arms

| Arm | Meaning |
|---|---|
| A0 | mb_gated_att, the reference (its 75 confirm runs reused) |
| C2 | rule-guided Goose |
| C2-free | LLM-free carry-over: Goose's dead-click filter re-keyed by colour, plus goal hints |
| C3 | C2 plus the frozen cross-game layer |
| C4 | C3 plus the planner on the two-layer model |
| C5 | any of the above with the fine-tuned writer |

### 11.2 Gates at a glance

| Rung | Gate | Where | Cost |
|---|---|---|---|
| CP1 | T1 beats both baselines on ≥ 3 of 8 dev games; wrong ≤ 5% everywhere | offline, dev | 5–8 h |
| CP2 | levels ≥ A0; wins ≥ losses; no game worse on both seeds; 16/16 runs | online, dev | 16–24 h |
| Confirm | > 112 levels; wins > losses; no game worse on every seed; 75/75 runs | online, all 25 games | ~55–95 h |
| CP3 | more trusted rules and cell coverage on ≥ 5 of 8; no rise in wrong predictions or overruns | offline, A/B-offline | 10–16 h |
| CP4 | ≥ 2 dev games with a fitted, path-checked completion test and re-plan; replay-to-level clears ≥ 1 level Goose didn't | offline and replay | varies |
| CP5 | more trusted rules on ≥ 5 of 8; overruns halved; no rise in wrong predictions | offline, A/B-offline | ~5 h (the base arm reuses CP3's run) |

### 11.3 Metrics reported everywhere

- **Levels:** total; paired better, same or worse; completions of level 2 and above counted separately.
- **Transfer:** T0, T1 and T2.
- **Rule quality:** claimed-cell exactness, cell coverage, strict case coverage.
- **New levels:** avoidable waste and masking risk.
- **LLM cost:** requests, tokens, wall clock, overruns.
- **Failures:** the failure taxonomy, with v2's additions.

### 11.4 Scope of claims

- Every result is on the 25 public games. "Held-out" means our own split, not ARC-AGI-3's private sets.
- Dev-game results are labelled as tuned.
- The cross-game claim rests on the untouched tier in Confirm 2, plus the A/B-offline tier.

---

## 12. Risk register

| Risk | Rungs | Likelihood | Mitigation |
|---|---|---|---|
| One person, eight weeks | all | high | the ladder, cut lines and slip rules |
| One GPU shared by everything | 2–5 | high | the calendar; one run at a time; taking turns on the card if needed (§5.4); Colab for CP5 |
| The 27B ignores `PARAMS` discipline | 1 | medium | lint feedback, schema-first fitting, a playbook entry, CP5 |
| Overruns rise under the lower answer limit | 2–4 | medium | code-only retry with the reasoning kept; helpers (CP3); CP5 |
| Hidden state on g50t and ls20 | 1, 4 | high | `api.t` and abstention; report it as a limit |
| Negative cross-game transfer | 3 | medium | the evidence gate, leave-one-out, fallback to no layer |
| Held-out contamination | 3, 5 | low, given the protocol | tiers, sealed results, the registry |
| Confirm compute overruns | 2, 4 | medium | budgets; time the first 10 runs and let the CP3 A/B slide (§10.2); book the GPU |
| Planner blow-up | 4 | medium | object state; v1's caps |
| LoRA serving issues | 5 | medium | try the small model first |

---

## 13. What carries over from v1, and what changes

**Kept as built:**
- the sandbox (plus a batch call);
- the evidence index and conflicts;
- the decoration mask with its edge guard;
- prompt sampling;
- the checker (extended);
- the LLM client and cache;
- the offline driver;
- the scoring fixes;
- the tests and launch scripts.

**Kept from v1's design, used later:**
- triggers and budgets (CP2, smaller);
- the completion test (CP4);
- the planner and executor (CP4);
- the exposure caps (CP4).

**Changed:**

| v1 | v2 |
|---|---|
| Trusted = 100% on every recorded case, one level | trusted = 100% on every claimed cell, on every level seen |
| One changing rule per group | up to 4 composable rules per group |
| Round 2 only if nothing was admitted | bandit refinement, including near-misses |
| Transfer = the untouched book only | T0 / T1 / T2 |
| W1 (guidance only) cut first | guidance is CP2's core |
| "Never share learned rules across scored runs" | no learning between scored runs; frozen, declared artifacts allowed |
| Out of scope: complementary rules, history-dependent models, fine-tuning a small model | in scope: composable rules (CP1), `api.t` (CP1), fine-tuning (CP5) |

**Still out of scope:**
- engine save/restore in scored runs;
- reading game code;
- the LLM choosing every move;
- adding ACTION7.

---

## 14. Combining with teammates' LLM additions

Since the team plans to compare and combine LLM additions at the end, v2 exposes what it knows through one small interface:

- `noop_mask(board)` → per-action multipliers;
- `probe_scores(board)` → per-action bonuses;
- `goal_hints()` → colours;
- `predict(board, act)` → (next board, claimed cells).

Another module can consume these, or supply its own versions through the same guide hook. Every comparison uses the same arms, tiers and adoption rule.

---

## 15. Open decisions (defaults in bold)

1. Replace v1's "narrow the build" fallback with this plan: **yes**, with v1's verdicts unchanged.
2. A frontier pass for helper extraction: **local first**; frontier only if the local pass yields fewer than 10 helpers that pass their tests.
3. Confirm 2 in late November: **only if a later rung passed its gate by 22 Nov**.
4. The held-out draw: **the stratified draw in §2.4**.
5. CP5's model: **the small model first**; 27B QLoRA only if Colab allows.
6. `api.t` in CP1: **yes**, as an optional field.
7. Using the Goose prior at level resets: **yes**, the prior instead of random weights, if it passes its folds.

---

## 16. Evidence base

Findings are paraphrased. Most ARC-AGI-3 results here come from frontier models on the public games, so treat their scores as context and their designs as the transferable part.

| ID | Source | What it found | Used in |
|---|---|---|---|
| S1 | ARC Prize Foundation, *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence*, technical report, 2026 (arXiv 2603.24621). https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf | The games are novel, abstract and turn-based. Difficulty is meant to come from composing concepts learned earlier in the same game, so later levels should build on earlier ones. | §1.3, P2, P6 |
| S2 | G. Kamradt, *Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3*, ARC Prize blog, 1 May 2026. https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis | Each environment is hand-crafted to test adaptation to novelty. An audit of 160 traces found three failures: local effects never turned into global rules; games mapped onto familiar ones, wasting actions; early levels won with wrong primitives that then misled later levels. Without a check on why a level was won, the misconception carries into the next one. | §1.3, P6, P7, CP3 schemas, CP4's "why did we win" check |
| S3 | S. Rodionov, *Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?*, arXiv 2607.15439. https://arxiv.org/abs/2607.15439 | The verification prompt bans hard-coded layouts and isolates level-specific data. The verifier replays all attempts from level 1. Simplification helped in 3 of 4 settings but not the weakest. Each game starts from a fresh workspace. The verification variant ships fixed templates and helper programs but no game rules. Public-set results may reflect saturation. | CP1, §2.4, CP3 |
| S4 | J. Lehmann, A. Aioanei, S. Vahdati, *Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3*, arXiv 2607.28287. https://arxiv.org/abs/2607.28287 | Models a game as shared mechanics plus a per-level start state from each level's first frame. The renderer may abstain; claimed-cell accuracy and coverage are reported separately. tu93's bar hides a move budget. The workspace persists across levels, with nothing shared across games. Automatic repair gave more exact models but lower RHAE (83.07 vs 88.49) than on-request modelling. The authors supplied ARC-specific prompts and a workspace, and their cross-run adaptation was human-mediated. | CP1, CP2, CP3, §2.4 |
| S5 | D. Courtis, W. Li, S. Sanner, *OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration for ARC-AGI-3*, arXiv 2607.01531. https://arxiv.org/abs/2607.01531 | Synthesises its own object extractor; a Bayesian ontology error steers exploration toward unexplained object types; solved 20 of 25 games (78.4 RHAE, Opus 4.8). | CP2 probes, CP4 finders |
| S6 | *Kepler: Auditable World Models for ARC-AGI-3*, arXiv 2610.00834. https://arxiv.org/abs/2610.00834 | A two-layer world model: named finders that turn pixels into objects, plus a simulator over those objects. | CP4 |
| S7 | *NVIDIA-labs OO Agents: Native Python Object-Oriented Agents*, arXiv 2607.20709. https://arxiv.org/abs/2607.20709 | Encodes the grid into the few fields that drive the game, and predicts on those. | CP4 |
| S8 | A. Bedi, *100% on the ARC-AGI-3 public set*, Agno, 25 Aug 2026. https://www.agno.com/articles/arc-agi-arcade | Per-game manuals of verified mechanics and corrections. Weaker models seeded with GPT-5.6's manuals improved sharply (Gemini-3.7-Flash 37.33 → 96.42; GLM-5.2 7.92 → 83.46). The manuals were for the same games, on the public set only. | CP3 playbook, as a ceiling |
| S9 | BDR-Pro, *arc-prize-2026-arc-agi-3*, GitHub. https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3 | No LLM at runtime. Colours that shrank when the score rose are carried as goal hypotheses into later levels. The archetype solver activates only after the game confirms it. | CP2 goal hints, CP3 gating |
| S10 | H. Tang, D. Key, K. Ellis, *WorldCoder, a Model-Based LLM Agent*, NeurIPS 2024 (arXiv 2402.12275) | Code world models that transfer across environments by editing and reusing pieces of old programs; optimism under model uncertainty. | CP3 library |
| S11 | G. Grand et al., *LILO: Learning Interpretable Libraries by Compressing and Documenting Code*, ICLR 2024 (arXiv 2310.19791) | Synthesises, compresses (with Stitch) and auto-documents code into a library. The documentation helped the synthesiser use its abstractions, and LILO beat DreamCoder. | CP3 library |
| S12 | Z. Ahmed et al., *Synthesizing world models for bilevel planning* (TheoryCoder), arXiv 2503.20124; *Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents* (TheoryCoder-2), arXiv 2602.00929 | General abstractions such as "move to" are grounded per environment with a learned low-level model. TheoryCoder-2 learns reusable abstractions and was more sample-efficient than WorldCoder. | CP3, CP4 |
| S13 | W. T. Piriyakulkij et al., *PoE-World: Compositional World Modeling with Products of Programmatic Experts*, NeurIPS 2025 (arXiv 2505.10819) | World models built from many small programs, each a simple causal rule, combined. Learned from few observations; generalised to unseen levels of Pong and Montezuma's Revenge. | CP1 composition |
| S14 | R. Dubey et al., *Investigating Human Priors for Playing Video Games*, ICML 2018 (arXiv 1802.10217) | Masking general priors such as objects and visual consistency slowed human players from about 2 minutes to over 20. | §1.3, CP3 |
| S15 | H. Tang et al., *Code Repair with LLMs gives an Exploration-Exploitation Tradeoff* (REx), NeurIPS 2024 (arXiv 2405.17503) | Refinement is an explore-exploit problem. Thompson sampling over which program to refine solved more problems with fewer LLM calls. | CP0, CP1 |
| S16 | J. Pourcel et al., *Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI* (SOAR), ICML 2025 (arXiv 2507.14172) | Hindsight learning turns successful and failed attempts into valid training pairs and fine-tunes sampling and refinement. The gains carried into test time (52% of the ARC-AGI public test set). | CP5 |
| S17 | M. Suzgun et al., *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory*, EACL 2026 (arXiv 2504.07952) | A self-curated memory of short, transferable snippets, rather than transcripts, improves later tasks. | CP3 playbook |
| S18 | *Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models*, ICLR 2026 (arXiv 2510.04618) | Evolving playbooks built by generation, reflection and curation. Warns that monolithic rewrites collapse into thin summaries. | CP3 playbook |
| S19 | A. Warrier et al., *Benchmarking World-Model Learning* (WorldTest / AutumnBench), arXiv 2510.19788 | 43 interactive grid-world environments for world-model learning. | CP5 external data |
| S20 | E. Rudakov et al., *Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks*, arXiv 2512.24156 | Training-free segmentation plus salience-ranked actions; 3rd on the preview's private leaderboard. | CP3 Goose prior |
| S21 | *Naestro-AGI3-27B*, Hugging Face model card (star-ga/naestro-agi3-27b) | A QLoRA adapter on Qwen3.6-27B for ARC-AGI-3, trained on synthetic simulations. The card flags first-exposure generalisation as unsolved. | CP5 feasibility |
| S22 | *arc3cb* README, GitHub (criticaldata/avo-qwen-arcagi3) | Best documented open-weights public-set results: the Duck harness at about 1.6 mean, self-measured (Qwen 3.6 27B, one GPU), and the Polyphony Agent at 19.8% (Qwen 3.6, self-hosted). | §1.4 calibration |

v1's own sources (baseline1, Polyphony, Duck and others) are listed in `rulebook-full-writeup.md` §3.13.

---

## 17. Glossary (new in v2)

| Term | Meaning |
|---|---|
| Binding / `PARAMS` | the level-specific values (colours, positions, sizes) a rule's mechanism uses |
| Re-binding | fitting a rule's `PARAMS` to a new level by search, with no LLM call |
| Claimed cells | the cells a rule vouches for; every other cell is UNKNOWN |
| Claimed-cell exactness | a case is right if every claimed cell is right |
| Cell coverage | the share of a level's changed cells predicted right |
| Trusted (v2) | exact on every claimed cell of every case it applies to, on every level so far |
| T0 / T1 / T2 | transfer untouched / after re-binding / after one repair or k probe moves |
| Fit window | the first 300 moves of a new level, used for re-binding and repair |
| Level-bound rule | a rule the lint flags for literal colours or coordinates |
| Residual prompt | a prompt asking for a rule for the changed cells the book leaves unclaimed |
| Schema | a parameterised template for a common mechanic, fitted by search |
| Playbook | a frozen list of lessons for the rule writer, learned from dev games |
| Goal hint | a colour the last win removed or shrank most, carried as a soft target |
| Avoidable waste | moves on a new level spent on actions the carried no-ops already covered |
| Masking risk | covered actions that did something on the new level |
| Tiers | dev, seen, A/B-offline, untouched (§2.4) |
| Sealed results | per-game held-out results written but not opened until CP3 is frozen |
| Registry | the list and hashes of frozen cross-game artifacts |
| Finder | a function that turns a board into named object records |
| C2-free | the LLM-free ablation of CP2 |

v1's glossary (`rulebook-full-writeup.md` §10) still applies.
