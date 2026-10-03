# Rulebook — Verified World Model (v2)
### Goose explores; a local LLM writes the game's rules and a level-completion test as code, checked against everything the run has recorded; a planner uses the checked rules to finish levels

Track: LLM integration · Owner: Matt · Repo: `sauerpatchkid/ARC3-solution` (branch `refactor/shared-baselines`, reviewed at 5da1508)
Base agent: **mb_gated_att** (`EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt`; documented 112 levels on the 25-game confirm test)
In-loop model: Qwen3.5-35B-A3B (GPTQ-Int4, vLLM, local RTX 5090)
Model note (3 Oct): only Qwen3.8-27B (`cyankiwi/Qwen3.8-27B-AWQ-INT4`) is kept locally; the Qwen3.5-35B and the other semester-1 models were deleted. If this plan is revived it uses the 27B, or re-downloads the 35B.
Version: **v2, 2 Oct 2026.** Folds in design review C01–C31; every change is listed in the changelog (§16). v1 and the annotated review are kept unchanged beside this file.
Idea lock: 2 Nov · Final report: 7 Dec
Name: renamed 3 Oct 2026 from "Plan C" to **Rulebook**; the stall-time advisor ("Plan A") is now **Coach** (`docs/plans/llm-coach.md`). Both are kept as options.

> **Freeze rule.** Numeric defaults marked *provisional* are calibrated in week 3 and frozen before the dev sweep. The thresholds in §6.4–§6.6 are frozen when the first Tier 0 run starts. The whole protocol (§6.7) is frozen before the confirm. Any later amendment is dated, explained, and kept beside the old version. Stage A's criteria are already frozen and are never edited.

---

## Plain-language summary (for the advisor)

Our best agent, mb_gated_att, is a fast explorer. It plays about 130 moves a second, remembers which screens it has seen, and walks back to promising places. Exploration upgrades took us from 54 to 112 levels on the 25 public games, but almost every game still stops at level 2–4. The agent learns *which* moves change the screen, never *what* they do, and it starts from scratch at every new level.

Rulebook adds a language model that runs on our own GPU:
- **Rules.** While Goose explores, the LLM reads what Goose's moves did and writes candidate rules as short Python functions ("clicking a red tile inside the grey frame turns it blue").
- **Checking.** A checker replays the run's recorded moves against each rule. A rule the planner may use must reproduce every recorded move it applies to exactly. We call that *replay-consistent*: strong evidence, but not a guarantee for situations the run hasn't seen yet.
- **Completion test.** When Goose finishes a level, the LLM writes a test that predicts, from a screen and a move, whether that move finishes the level. It is checked against the real winning move and every other recorded move.
- **Planning.** With replay-consistent rules and a completion test, a planner searches for a solution to the next level in simulation and plays it.
- **Safety.** After every real move the agent compares the screen, score and game state with the prediction, and at the first surprise it hands control back to Goose. A cap limits how many moves the planner can control over the whole run, so a wrong model costs a bounded number of moves. The confirm test measures whether the net effect is more levels.

The LLM never picks moves directly (far too slow for 100k-move runs) and never sees the game's code. That is the recipe behind the strongest ARC-AGI-3 systems published this summer. Our version gets its evidence from a fast explorer instead of from a slow LLM playing by itself.

---

## 0. Headline

### 0.1 The claim we are trying to prove by 7 Dec

> **Exploration finds the data, a local LLM writes rules that replay it exactly, and planning turns those rules into levels.**
>
> Given only the transitions mb_gated_att collects in the current run, a locally run open-weights LLM writes each game's mechanics and level-completion test as code that is replay-consistent with what the run has recorded. A planner that acts only on replay-consistent rules completes levels that exploration alone does not reach. On the same 25-game × 3-seed × 100k-action protocol, the hybrid completes **strictly more levels than the reconciled mb_gated_att total** (documented: 112; §6.1), with more paired wins than losses, no game worse on every seed, and all 75 runs complete.

### 0.2 The evidence chain: what the report shows even if the confirm test isn't a win

The claim is a chain of five links. Each is measured on its own (§6), so the report can say how far the idea got and where it broke.

| # | Link | What is measured | Where | Progress threshold (§6.4) |
|---|---|---|---|---|
| L1 | Rules | per-rule accuracy on the moves it applies to; separately, the assembled rule book's coverage of the level's changing transitions | Tier 0a, then online | assembled coverage ≥ 80% on ≥ 3 of 8 dev games |
| L2 | Transfer | before any repair: one-step accuracy on the next level's untouched transitions, and multi-step accuracy of an uninterrupted model rollout | Stage A (frozen), Tier 0a/0b | beats the "nothing" and "memory" baselines |
| L3 | Completion test | three tiers: *fitted* (passes the check on recorded moves), *path-checked* (fires exactly at the end of the recorded winning attempt replayed through the model), *live-validated* (a plan produced a real level-up) | Tier 0b, online | fitted and path-checked on ≥ 2 dev games |
| L4 | Planning | in-model re-planning (internal feasibility), reported separately from live success | Tier 0b, replay-to-level, online | ≥ 1 level cleared live that Goose's matching run did not, with complete terminal logging |
| L5 | Levels | the strict adoption rule, plus effect size, seed variation and uncertainty | dev sweep, confirm | strictly more than reconciled A0, wins > losses, no game worse on every seed, 75/75 runs |

The minimum defensible result is L1–L4 with the per-game failure taxonomy (§6.3).

The capability figure (9B → 35B → frontier) isolates model capability only where evidence, candidate counts and budgets are matched. So it is built offline at Tier 0a (§9), and the live frontier run is labelled a deployment demonstration.

### 0.3 Why we expect more levels, in five sentences

1. The exploration variants tested so far show diminishing returns at levels 2–4; persistent knowledge of mechanics and goals is one plausible way past that ceiling (action coverage and resource handling are others, §1.1).
2. ARC-AGI-3 levels reuse a game's mechanics in harder layouts, so rules learned on level k should be worth the most on level k+1, which is exactly where Goose resets to zero.
3. Executable world models with checking are behind the strongest published systems, and in the one controlled ablation the complete verification treatment ranked first (§1.2).
4. Goose collects far more evidence per level than an LLM-only agent can afford, and Stage A's smoke test showed the 35B fitting ft09's tile-flip rule to Goose's recordings (a training fit; transfer is what Tier 0 measures).
5. Goose stays the default controller; step checks and a cumulative cap on planner-controlled moves bound the exposure to wrong plans, and the confirm test measures whether the net effect is positive.

### 0.4 Terms used throughout

| Term | Meaning |
|---|---|
| replay-consistent | reproduces every retained recorded transition it applies to exactly (masked pixels), with no known contradiction. Evidence, not proof, for unseen states. |
| admitted rule | a candidate that applies to ≥ 20 recorded moves, is exact on ≥ 95% of them, and has gain > 0 over "nothing changes" (Stage A's rule). Admitted rules are hypotheses; they feed repair and diagnostics. |
| plan-eligible rule | an admitted rule that is replay-consistent (100% on its retained applicable evidence) and not suspended. Only these are used by the planner. |
| suspended rule | a rule contradicted by a new observation. It leaves planning immediately and waits for repair. |
| known no-op | a checked rule whose prediction is "nothing changes". Used to prune search, never confused with UNKNOWN. |
| UNKNOWN | what the rule book returns when no plan-eligible rule applies or applicable rules disagree. The planner never expands UNKNOWN moves. |
| completion test | `finishes_level(board, act, api)`: predicts whether this move on this observed board finishes the level. Tiers: fitted → path-checked → live-validated. |
| intervention | a move chosen by the WM executor instead of Goose. Capped cumulatively per run, failed plan prefixes included. |
| reconciled A0 | the mb_gated_att reference total after the scoring check in §6.1 (documented: 112). |

---

## 1. Why this design, and the evidence behind each claim

### 1.1 What limits mb_gated_att

- **The ceiling.** Three exploration upgrades took the 25-game test from 54 → 79 → 112 levels (change label → novelty label → mb_gated_att). But 112 is 20% of the 549 levels available over 3 seeds (183 per seed), and nearly every game stops at levels 2–4. Semester 1 found the same ceiling for Random, Goose and Blind Squirrel, and it persisted at 1M actions.
- **No model, no memory.** Goose learns which moves change the screen or reach a new screen; it never learns what a move does. Its network, buffer, seen-set and map all reset at each level by design. Keeping the network across levels only moved its screen from 14 to 18 levels.
- **Long purposeful sequences.** Later levels need more steps in the right order. `novelty_late_per_1k` shows Goose running out of new screens thousands of moves before the level counter confirms it is stuck.
- **Other possible causes.** Six games (ar25, bp35, lf52, sb26, sk48, su15) advertise ACTION7, which the harness drops, so they play a reduced action space (`run_local.py` says this must be disclosed). Missing actions, masked resource counters and reset behaviour may also contribute to plateaus; Rulebook does not claim every plateau comes from a lack of understanding.
- **Headroom on the Rulebook dev games.** mb_gated_att reaches 4 of 9 levels on tu93, 4 of 7 on vc33, 2 of 6 on ft09 and m0r0, 1–3 of 6 on dc22, 1–2 of 6 on cd82, and 1 of 7 on g50t.

**Conclusion:** the tested exploration variants show diminishing returns. That motivates adding persistent mechanics and goals, while leaving open whether better exploration, action coverage, resource management or reset behaviour could also raise the ceiling.

### 1.2 What the field found (to Sept 2026)

All results are on the 25 public games, mostly self-reported. arXiv versions are pinned in §15.

| System | Model | Approach | Result |
|---|---|---|---|
| baseline1 (Rodionov, arXiv 2605.05138) | GPT-5.5 | one coding agent; executable Python world model; replay verification; simplification; plans through the model | 15 games fully solved, mean RHAE 58.1% |
| Component study (Rodionov, arXiv 2607.15439) | GPT-5.4 / 5.5 / 5.6-sol | 4 nested variants: textual → executable → + simplification → + exact replay verification | the complete verification treatment ranks 1st in all 4 settings; an executable model *without* it is worse than plain text for GPT-5.5; with gpt-5.6-sol, every game solved at ~99% RHAE |
| OPINE-World (Courtis et al., arXiv 2607.01531, v2) | Claude Opus 4.8 | separate acting and model-writing agents; checked transition and reward models; plans only after a level is cleared, checking every step | 20/25 games, 160/183 levels (v2) |
| Tycho (Lehmann et al., arXiv 2607.28287) | Claude Opus 4.8 | when to build and use a model | model built when the actor asks: 88.5 RHAE; auto-repair after every failure fits transitions better but scores 83.1 |
| Polyphony Agent (community leaderboard, Jul 2026) | self-hosted Qwen3.6 | coding agent grows a verified per-game set of Python files (state, dynamics, planning, action choice) | 19.8%, the best documented open-weights result |
| Duck (Tufa Labs, Milestone 1 winner) | Qwen 3.6 27B, local | LLM writes and runs Python in a REPL every turn, nothing verified | ~1.6% public mean; a Qwen3.8-27B reproduction cleared ≥ 1 level on 11 of 25, no wins (90-min limit per game) |
| Reki / forge (Milestone 1, 2nd/3rd) | Gemma-4-31B, local | one JSON action per step from rendered frames, plus click heuristics | 0.87% semi-private (Reki) |

How we read it, carefully:

1. **Checking is part of the best package, not a lone cause.** Rodionov's top treatment bundles exact replay with fixed interfaces, templates and verifier tools, and uses more inference. We attribute his result to that package, and test our own trust gates with a single-factor ablation (W3, §6.2).
2. **Local models can do this when the work is decomposed and checked.** Polyphony and the Duck show that a verified-rules design is feasible with open weights. Their score gap is not causal evidence for verification: hardware, context, inference and time budgets differ.
3. **Event-driven model building is worth testing.** Tycho's comparison motivates calling the model builder on demand; it is not a universal scheduling result.
4. **Per-move LLM play is the wrong use of a local model** at 100k-move budgets. Even the Milestone 1 winners play only hundreds of moves per game.

On OPINE: the review reports that a later revision extends to history-recoverable hidden state and more flexible rule organization. We cite v2 as read on 1 Oct 2026 and recheck before the report. Outside ARC, coding agents that experiment with a game and then compile what they learned into a standalone player show the same "slow model, fast program" pattern (Xiao & Huang, arXiv 2609.18996).

### 1.3 Why the hybrid should beat both halves

- **Evidence volume and diversity.** LLM-only agents check against the few hundred moves they can afford to play, while Goose gives the LLM up to 100k transitions per run. Volume alone isn't the point, though: many moves repeat irrelevant effects and may still miss a rare interaction. So we measure evidence diversity (distinct effect signatures per action group) and changing-transition coverage, and prompts prioritize counterexamples and under-sampled contexts (§3.1). More data costs checking time, not tokens.
- **Memory across levels.** The rule book and completion test persist across the level boundary that resets Goose. On a new level the planner can try immediately from the first screen. Whether that pays off is an intervention the confirm test evaluates, not a guarantee of transfer.
- **Our own evidence, and its limit.** In Stage A's smoke test (`legacy/llm_track/README.md`), the 35B wrote "clicking a red block inside the dark-grey container turns it blue" from Goose's ft09 recordings.
  - That rule is exact on all 562 of its training moves.
  - The rule book predicted 99.9% of training moves, against 74.6% for "nothing changes".
  - The 35B needed 12 min against 43 for the dense 27B.

  That is a training fit on a simple mechanic. Generalization evidence must come from Stage A's frozen test and Tier 0 transfer.
- **Bounded intervention, not a guaranteed floor.** Goose stays the default controller. The planner uses only plan-eligible rules and a fitted, path-checked completion test, checks every move, and hands control back at the first surprise. A single wrong move can still ruin an attempt or change what Goose learns next, and fallback cannot restore the original trajectory. A cumulative cap on planner-controlled moves (§3.6) bounds that exposure; the confirm test measures the net effect.

### 1.4 What could still go wrong (honest priors)

| Risk | Evidence it's real | Mitigation |
|---|---|---|
| The 35B only manages simple mechanics | Stage A: with a 16k thinking budget, 23 of 24 ls20 answers never reached code; on a picture-only toy it misread colours and distances | exact changed cells as text; 20,480-token output budget with a code-only retry; one action group per task; the matched capability comparison (§9) shows how much is the model |
| A completion test fits but doesn't generalize | one positive per level against many negatives can still support memorization; Probe C's winning heuristic hard-coded a level-1 position | observed positives only; near-goal negatives; cross-level check after two wins; failed-plan negatives; path-checked and live-validated tiers |
| Wrong plans cost real progress | one move can end an attempt or burn a resource | plan-eligible rules only; UNKNOWN moves never expanded; pixel + score/state step checks; cumulative intervention cap; suspension and blocklists |
| Conflicting outcomes (observation aliasing) | can come from masked resources, mask errors, recoverable history or truly hidden state | outcome variants preserved; WM's own versioned mask; flagged in the taxonomy, never silently deduplicated |
| Search blows up, or the click abstraction is too coarse | branching = buttons + every clickable object; connected same-colour pixels need not click the same way | deterministic expansion limits; best-first on an optional progress score; click refinement deferred (§14) |
| LLM time, context or VRAM make sweeps too slow | prompts plus 20,480 output tokens must fit a 32,768 context; candidate latency is unmeasured | week-1 measurements; provisional budgets calibrated in week 3 (§8) |

---

## 2. Architecture

```
 run_local.py ─ every engine call → EVENT LOG (action + params, RESET, score/state before and after, frames),
                captured immediately, before the WIN / action-cap checks can stop the loop

            ┌───────────── fast loop: mb_gated_att, unchanged (~130 act/s) ─────────────┐
 frame ─► bars/ticker mask ─► novelty label ─► CNN sample ─► deadclick ─► return map ─► act
                                                    │                                    ▲
                                                    └─ if WM is EXECUTING, WM owns ──────┘
                                                       the move; deadclick and map are
                                                       not consulted (no route step used)
 ┌───────────────────────────── WM layer (custom_agents/wm/) ──────────────────────────────────────┐
 │ EVIDENCE: append-only raw log with attempt chronology; index (level, mask version, board, move)  │
 │           → outcome variants with counts and terminal labels (death, level-up)                   │
 │   ├─► RULES (warm-up, counterexamples, stall): LLM writes k candidates per action group          │
 │   │     → sandbox → check → admitted / plan-eligible / suspended; assembled book checked whole   │
 │   ├─► COMPLETION (level-up): LLM writes finishes_level(board, act, api), labelled by real wins   │
 │   │     → fitted → path-checked on the recorded winning attempt → live-validated after a win     │
 │   └─► PLAN (level start, stall, model update): search plan-eligible rules for a finishing move;  │
 │         UNKNOWN moves never expanded; deterministic expansion limits                             │
 │ EXECUTOR: owns the move; checks pixels + score + state after every move; on mismatch: record a   │
 │           counterexample, suspend the rule, invalidate dependent plans, return to Goose;         │
 │           cumulative intervention cap                                                            │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
 LLM: vLLM server in .venv-llm (separate process, OpenAI-compatible HTTP). The agent never imports vLLM.
```

Per level, the agent is in one of two modes, and ownership is explicit:

| Mode | Who picks the move | Lower-priority overrides | Entered when | Left when |
|---|---|---|---|---|
| EXPLORE | mb_gated_att, exactly as today (Goose sample → deadclick → map) | all active | default | a plan exists and the intervention cap has room |
| EXECUTE | the WM executor | not consulted; map routes cancelled on takeover and revalidated on handback | the planner returns a plan from the current screen | plan finished, mismatch (pixels, score or state), game over, level-up, or cap reached |

Goose still samples every move, even in EXECUTE, so its own RNG stream and training loop are unchanged. The executed move is the one recorded for CNN training, novelty memory and the map. LLM calls happen between moves, with Goose paused (§3.9).

---

## 3. Components

### 3.1 Evidence buffer

**Raw log.** Every transition reaches `wm.observe`, including deaths (delivered before `choose_action` returns RESET) and the final transition of a run (delivered from the event log). Raw frames are kept append-only per level, with chronology:
- attempt id (moves since the last RESET or level start) and move number;
- score and state before and after;
- animation frames, when the engine exposes them.

**Index.** A derived index maps (level, mask version, masked-board hash, move) to its outcome variants, with counts and terminal labels. A duplicate is compared with the stored outcomes before it is counted. A different outcome is kept as a new variant, never discarded. Keys that ever show more than one outcome are flagged as *conflicting outcomes*. That can mean observation aliasing (a masked resource, a mask error, recoverable history) or genuinely hidden state.

**Mask.** WM uses its own per-level mask: the bars detector and ticker detection run on that level's data after warm-up, with gaps inside ticker rows filled.
- Filling gaps is an inference (Stage A found rarely-changing bar cells that otherwise broke a correct rule), so the filled cells are logged.
- If the mask is re-estimated, its version number bumps, keys are rebuilt from raw frames, and any rules, completion tests and plans that depended on the old version are re-checked.
- WM never keys on the canonicalizer's rolling mask, whose keys are not comparable across refreshes.
- Raw values of masked bars stay in the log; encoding them into the model is deferred (§14).

**Check set and prompt sample.** The check set is every key in the index for the relevant levels, with all outcome variants. That includes keys from frozen stretches (moves that change nothing at all), which are excluded from prompt sampling only. Prompts show at most 6 moves per group, in Stage A's format: before/after pictures with the changes boxed, plus the exact changed cells as text, up to 24 changed areas. Moves are chosen to cover distinct effect signatures, counterexamples first, then under-sampled contexts.

**Why.** Exhaustive, version-consistent checking is what makes "replay-consistent" mean something; deduplicating before comparing outcomes would hide exactly the conflicts we need to see. The prompt stays small enough for a 35B, and Stage A's format is the only one that has worked here.

### 3.2 Rule synthesis

**Interface.** Unchanged from Stage A's `RULE_DOC`:

```python
RULE = "one line saying what happens"
def applies(board, act, api): ...   # True if this rule knows what this move does on this board
def predict(board, act, api): ...   # the 64x64 board right after the move
```

`act.action` is 1–5 for a button or 6 for a click, with `act.click = (row, col)`. `api` gives the level's first board, the background colour, the WM mask and layout regions, rebuilt for each level.

**Loop per action group.** Groups are each button, plus clicks grouped by the colour clicked; a game can have many click-colour groups.
1. Prompt with up to 6 recorded moves of the group (§3.1). The prompt is at most 12,288 tokens, so that it plus a 20,480-token output fits the 32,768-token context `rule_writer.py` runs at. Thinking is on; an answer that runs out of room is asked for its code alone, with thinking off.
2. Generate k candidates (*provisional* k = 4, set from the week-1 latency measurements).
3. Run each candidate through the sandbox, then check it on every key in the check set where `applies()` is True.
4. Classify each candidate:
   - **admitted** if it applies to ≥ 20 moves, is exact on ≥ 95% of them, and has gain > 0 over "nothing changes" (Stage A's rule);
   - **plan-eligible** if admitted and exact on 100% of its applicable evidence, with no conflicting outcomes;
   - **known no-op** if it predicts "nothing changes" exactly on ≥ 20 moves.
5. Round 2 runs only if nothing was admitted. The prompt shows the best candidates with their scores, the moves where the best one was wrong (before | prediction | real after), and one uncovered move.

**Rule book.**
- Each group keeps one admitted changing rule (the highest gain) plus its known no-op rule, if any.
- The assembled book is checked as a whole on the full check set, including moves where two rules' `applies()` overlap.
- For any move, the book returns the plan-eligible rule's prediction, "no change" for a known no-op, or **UNKNOWN** when nothing applies or applicable rules disagree.
- Several complementary changing rules per group are deferred (§14). With UNKNOWN, a group's uncovered cases are simply unknown rather than silently "nothing changes".

**Trust follows evidence.** The first new observation that contradicts a plan-eligible rule suspends it from planning immediately. The counterexample threshold (*provisional* 5) only batches repair calls; it never delays the loss of trust.

**Why.**
- *Rule books, one group per task:* this is the size of problem the 35B solved in Stage A. Rules are checked and repaired independently, and the book composes into a simulator. Shared helpers and joint effects stay allowed inside a rule.
- *Two trust levels:* 95% is Stage A's frozen threshold and is fine for admitting hypotheses, but errors compound in plans. A 20-step plan that leans on a 95% rule at every step fails about 64% of the time (1 − 0.95²⁰). Planning uses only exact rules.
- *UNKNOWN vs no-op:* an uncovered move is not evidence that the move does nothing. Treating it as a no-op would hide unmodelled actions from the planner and break W1.
- *Best-of-k with feedback:* the LLM proposes and the checker selects. Its cost is measured, not assumed (§8).

Stage A itself keeps its frozen rules, including "nothing changes" for uncovered moves. The changes above apply to the online layer and Tier 0.

### 3.3 Sandbox and integrity

All LLM-written code runs through a copy of `heur_sandbox.py`: AST checks (no imports, no `_` attributes, and a denylist of file, process and introspection names), restricted builtins, a spawned process and a timeout. Its own docstring says it is not a hardened security boundary; it protects against common mistakes in generated code. The worker receives only board arrays and the approved `api` object, never engine objects.

Two separate tests:
- generated code that tries to import `arc_agi` or `arcengine` fails;
- generated code that tries to open a file under the game directory fails.

`make check` also greps every prompt for game ids. That rules out one leakage path (game names in prompts), not all of them.

**What the report claims:** generated code had no imports, no file or engine API, and ran in a separate process; it was not hardened against deliberate escapes. A hardened worker with no filesystem access to the game sources is deferred (§14).

The code is copied, not imported, because the repo's isolation rule forbids baseline code importing `legacy/`. `check_repo.py` gets one new rule allowing `custom_agents/wm` to call the LLM over HTTP, while still forbidding vLLM imports in the agent's environment.

### 3.4 Completion test

**Trigger:** each level-up. A completion test no longer needs any rules to exist.

**Interface:**

```python
GOAL = "one line: what finishes a level"
def finishes_level(board, act, api): ...   # board BEFORE the move; True iff this move finishes the level
def progress(board, api): ...              # optional: higher = closer to finished (orders search only)
```

`api.predict(board, act)` returns the rule book's predicted next board, or None for UNKNOWN, so a test may reason about the successor. The labels are always the real level-up events; the observed score is never an input.

**Evidence in the prompt:**
- the winning move and its real before-board;
- the solved screen, if the animation frames genuinely expose one;
- the level's first screen;
- a few near-win, non-winning moves from the same attempt;
- a few earlier screens.

**Status tiers:**
1. **Fitted.** On every recorded move of every cleared level, `finishes_level` is True exactly on the level-completing moves. Every positive is an observed (board, move) pair; there are no simulator-made positives. Extra negatives: every move on the current level until it is won, and each new level's first screen with every move tried there.
2. **Path-checked.** The recorded winning attempt (the moves from the last RESET or level start up to the level-up) passes three checks:
   - each recorded step is predicted exactly by plan-eligible rules, or flagged UNKNOWN;
   - an uninterrupted model rollout from the attempt's first board, fed the recorded moves, matches the real boards at every step;
   - `finishes_level` is False at every earlier step and True on the winning move.
3. **Live-validated.** A plan using this test produced a real level-up.

**Keeping the test honest over time:**
- After the second win, the test must fit both levels.
- A failed plan (the test said True, but no level-up) adds that (board, move) as a negative and triggers a rewrite.
- An unexpected real level-up during a plan adds a positive.

**`progress()`** orders search only if its hindsight AUC on Probe C's referee is ≥ 0.6, just under the 0.64 a board-seeing rule scored in Probe A's hindsight check. We also report whether it actually reduces search expansions. Hindsight proximity on Goose's wandering trajectory isn't necessarily distance to the goal, which is why `progress()` never certifies completion.

**Unit test:** an "arrange, then submit" toy game, where the same board is a non-win before a submit move and a win on it. `finishes_level` must separate the two.

**Why:**
- In our corpus, the next frame after a winning move is usually the next level's first screen. A test on after-boards therefore needed a positive invented by the simulator, and re-finding that invented board in the same simulator was circular evidence.
- A (board, move) test is labelled by real events, handles submit-style wins, and can be written before any rule exists.

### 3.5 Planner

1. **Known path first.** For a cleared level, the recorded winning attempt is replayed through the model before any search (the §3.4 path check). A failure there is a model or completion-test problem, not a search problem.
2. **Search.** From the current masked board, search plan-eligible rules for a (board, move) where `finishes_level` is True.
   - Candidate moves: available buttons; one click per connected same-colour object (background and bars excluded, with the representative cell inside the object); and any cell where a click changed the screen this level.
   - Known no-ops are pruned and UNKNOWN moves are never expanded.
   - Recorded (board, move) pairs that caused a death are excluded.
3. **Deterministic limits.**
   - Search is breadth-first with a visited set, switching to best-first on `progress()` past a fixed number of expansions. Depth is tracked, and states later reached by a shorter path are reopened.
   - The node budget counts expansions in a fixed move order, so the same inputs give the same plan.
   - A wall-clock watchdog runs separately, and its firing is logged as its own outcome.
4. **Plan length.** At most the smallest of: remaining run actions, the remaining intervention budget, and the estimated moves left in this attempt. That last one is a heuristic: the median number of moves between game overs on this level, minus moves since the last RESET.
5. **In-model re-plan check.** From the first screen of each cleared level, the planner must find a completing plan inside the model. This shows internal feasibility, not live correctness, and is reported as such.

Click candidates are deliberately coarse, since connected same-colour pixels need not click identically. Refining them when search fails is deferred (§14).

### 3.6 Executor and counterexamples

The executor owns the move while a plan runs (§2). After each move it compares the observed masked board, score and game state with the prediction:

| Outcome | What happens next |
|---|---|
| Match | Play the next step. |
| Pixel mismatch | The transition becomes a counterexample, the rule that predicted it is suspended, plans that depended on it (by rule-book version) are invalidated, and Goose resumes. |
| Unexpected game over | Recorded as a death transition and a counterexample; the (board, move) joins the death blocklist; Goose resumes. |
| Unexpected level-up | Counted as a level and recorded as a completion positive; the plan ends. |
| Plan finished without a level-up | The final (board, move) becomes a completion negative, and the test is refit before more planning. |

**Exposure bounds:**
- a **cumulative intervention cap** per run (*provisional* 10,000 moves, 10% of the budget) counts every executor move, including failed prefixes, and repairs never reset it;
- a per-level gate stops planning after 3 failed plans in a row, until a rule or the test changes;
- a (board, move, model version) combination that already failed is not retried until relevant new evidence or a repair arrives.

**Reported per run:** executor moves, failed-prefix lengths, attempts lost to plan-induced game overs, and levels gained.

### 3.7 Triggers and budgets

| Trigger | Fires when | Calls |
|---|---|---|
| Warm-up | ≥ 2,000 transitions (*provisional*) accumulated, counting retained evidence from earlier levels, and no rule book for this level | rules for each changing group, most frequent first |
| Counterexample | the batch threshold (*provisional* 5) is reached in a group | repair that group (trust was already revoked at the first contradiction) |
| Stall | mb_gated_att's stall signal, and assembled coverage < 80% of the level's changing transitions | rules for uncovered groups |
| Level-up | the score changes | completion test; rules re-checked on the new level as transitions arrive |
| Plan | level start, any rule or test update, stall | planner only (no LLM call) |

A level that ends before warm-up keeps its evidence, and its rules are synthesized once enough evidence exists. Its completion test is written immediately, since the test needs no rules.

**Budget units are metered separately:** HTTP requests, candidate completions, retries, and generated tokens (thinking + answer).
- *Provisional* caps per run: 40 requests, 160 completions, 1.5M generated tokens.
- 25% of completions are reserved for completion tests and repairs, so first-pass rule synthesis can't exhaust them.
- A stall guard skips a group after two rounds without an accuracy gain, until a new counterexample arrives.
- An extra trigger for sparse evidence or competing hypotheses is deferred (§14).

**Why event-driven:** Tycho's comparison and OPINE's counterexample-triggered synthesis both motivate calling the LLM when the model is contradicted or missing. W2's logs let us check whether this schedule wastes calls.

### 3.8 Carry-forward across levels

On a level-up, Goose resets as it does today. The WM layer keeps its rules, completion test and evidence, but rebuilds the per-level parts: the `api` (first board, layout), the mask, the available actions and the attempt-length estimate.

Evidence is tagged with its cutoff, so the first predictions on the new level, made before any repair, are reported as transfer. Planning immediately from the new level's first screen is an intervention that the dev and confirm sweeps evaluate.

### 3.9 LLM serving, timing and reproducibility

**Server.** vLLM in `.venv-llm`, serving Qwen3.5-35B-A3B GPTQ-Int4. The agent talks to it over HTTP, so the baseline's pinned torch is never touched.
- **Context:** 32,768 tokens with thinking (`rule_writer.py`'s setting, the most that fits with images). With a 20,480-token output budget, prompts are capped at 12,288 tokens.
- **Memory**, reported in four parts: weights (21.1 GiB), vLLM's reserved fraction of GPU memory (`--gpu-mem`), KV capacity (about 105k tokens at 0.90), and Goose's peak.
- **Fit for k = 4:** the worst case is one shared 12,288-token prompt plus four 20,480-token outputs, about 94k tokens. That fits at 0.90; at lower settings, fewer candidates run concurrently. Week 1 measures this.

**Synchronous calls.** Goose pauses during each call, so every call has a fixed evidence cutoff (move number). That keeps comparisons fair across model speeds: with background calls, a slower model would let Goose explore more before its answer arrived. Wall-clock time still matters for one stop rule: the agent's `is_done` ends a run at about 8 hours. Every run logs its termination reason, and a W2 run that ends on the wall clock is a protocol failure to investigate, not a silent result.

**Reproducibility, as the repo actually supports it.** Same-seed runs are identical for only about 300 actions; after training starts, nondeterministic CUDA kernels make trajectories diverge (README, CLAUDE.md). So:
- Comparisons between fresh runs are distributional: paired by game and seed, never expected to match move for move.
- `EVAL_WM=off` is checked as **code-path isolation**. The layer is never constructed, and a short CPU-only run (deterministic algorithms, one thread, 2,000 actions) must match the pre-change commit move for move. If CPU training also diverges, the check falls back to the documented ~300-action identical window on GPU.
- A **mock-backend "unusable model" run** in full mode, where no candidate is ever admitted, must match off-mode in the same CPU check. This proves WM's bookkeeping doesn't perturb Goose.
- WM uses its own RNG (seeded from `EVAL_SEED` plus a fixed offset), never Goose's.
- Search uses deterministic expansion limits.

**Response cache.** Requests and responses are stored keyed by run identity, model and tokenizer revision, quantization, vLLM version, decoding settings, prompt hash, seed, and evidence/model versions. The cache makes Tier 0 reruns and debugging exact; it cannot make fresh live runs reproducible. Fresh scored runs never read another run's hypotheses or results.

**Backends behind one interface:** `mock` (tests), `vllm` (local), and `anthropic`/`openai` (the frontier runs, §9).

### 3.10 Logging

Everything is written to `results/runs/<ts>/<game>/wm/`, plus TensorBoard scalars `WM/*`, and summarized into the funnel table (§6.3).
- **Every engine call** (the event log): call number, action and parameters, RESET flag, raw observation, score and state before and after, and animation frames when exposed.
- **Every hypothesis:** evidence cutoff, applicability, accuracy, assembled coverage, overlap status, admitted / plan-eligible / suspended history, completion-test tier, and whether its positives were observed.
- **Every mask and plan:** the versions it depends on, revalidation events, expansions, watchdog firings, remaining resources, and who owned each move.
- **Every run:** code and configuration hashes, model/tokenizer/quantization/server versions, final engine score, scored call count, intervention total and termination reason.

---

## 4. Implementation plan (file by file)

These estimates are rough; they get re-estimated once the week-1 interfaces are prototyped. v1's ~15-line integration estimate was optimistic.

| File | New / changed | Lines (est.) | What | Reuses |
|---|---|---|---|---|
| `run_local.py` | changed | ~60 | event log: record every engine call's result right after step/reset, before the WIN/cap checks; flush on every exit path; termination reason | — |
| `eval_common.py` + corpus schema | changed | ~60 | post-action score/state and attempt chronology in the corpus; versioned schema that still reads historical runs | — |
| `compute_metrics.py` | changed | ~50 | scorer v2: levels from the event log's authoritative engine score, cross-checked against logged level-ups; the v1 path kept for historical runs | — |
| `tools/paired_compare.py` | changed | ~40 | `--strict` (new > base) and `--expect N` (exactly N valid pairs, no duplicates, matching configs); defaults unchanged so historical verdicts stand; bootstrap of the paired delta that resamples games | — |
| `custom_agents/action.py` | changed | ~50 | build from `EVAL_WM`; deliver terminal transitions (deaths, the final move) to WM; explicit EXECUTE ownership (deadclick and map skipped); executed move recorded for CNN, memory and map | — |
| `custom_agents/wm/__init__.py` | new | 60 | `WorldModel.from_env()`; hooks; private RNG | the return map's hook pattern |
| `custom_agents/wm/evidence.py` | new | 300 | raw log, chronology, outcome-variant index, versioned masks, prompt sampler | `rule_referee.group_of`, `_frozen`, `_fill_rows` |
| `custom_agents/wm/rules.py` | new | 260 | rule prompts, candidates, feedback, rule book with UNKNOWN and known no-ops | `rule_writer.group_prompt`, `feedback`, `RULE_DOC` |
| `custom_agents/wm/check.py` | new | 280 | per-rule and assembled checks, trust tiers, suspension, completion fit, path checks, rollouts | `rule_referee.check_rule`, `rule_book` |
| `custom_agents/wm/sandbox.py` | new | 160 | AST checks + spawned worker; import and file-access tests | `heur_sandbox.py` |
| `custom_agents/wm/goal.py` | new | 180 | `finishes_level` prompts, evidence, tiers, negatives | Probe C referee (for `progress`) |
| `custom_agents/wm/planner.py` | new | 240 | known-path replay, deterministic search, click candidates, plan-length bound | `upgrades.screen_objects` |
| `custom_agents/wm/executor.py` | new | 180 | ownership, pixel + event checks, suspension, invalidation, blocklists, intervention cap | the return map's route check |
| `custom_agents/wm/llm.py` | new | 180 | backends, namespaced cache, request/completion/token/dollar meters | Coach's backend sketch |
| `tools/wm_offline.py` | new | 220 | Tier 0: corpus → rules → completion test → path checks → re-plan; matched capability mode | the modules above |
| `tools/replay_to_level.py` | new | 150 | replay a run's event log to a level's first screen, checking length, parameters, RESET order and score/state checkpoints, then hand over to the planner (diagnostic only) | `run_local.make_env` |
| `tools/wm_funnel.py` | new | 140 | funnel and failure taxonomy (several flags, one primary bottleneck) | — |
| `tests/test_wm_*.py` | new | 450 | the pre-dev correctness suite (below) | Stage A's self-test |
| `experiments/wm_dev/`, `experiments/wm_confirm/` | new | — | runners, pre-registered READMEs, expected manifests | `experiments/upgrade_confirm/` |

That is roughly 3,000 lines, about 40% adapted from code that already runs. `inspect_corpus.py --verify-seed` compares only the shorter action prefix and sees neither RESET calls nor observations. The event-log replay in `replay_to_level.py` is therefore the chronology check.

**Pre-dev correctness suite.** Every check must pass before the dev sweep.

| Check | Acceptance criterion |
|---|---|
| Terminal observations | final win, death, RESET and last-budget-action outcomes are recorded; scorer v2 levels equal the engine score |
| Full-mode fallback | errors, timeouts, unavailable tests, unsupported actions and mismatches leave no stale plan and no crash |
| Map takeover | no route step is consumed for an overridden move; routes are cancelled or revalidated on handback |
| Evidence and masks | duplicate outcomes are compared before counting; differing terminal labels survive; a mask change re-keys and revalidates |
| Rule dispatcher | uncovered moves return UNKNOWN; overlaps are detected; assembled predictions are checked |
| Completion interface | identical boards with different moves can get different predictions (the submit toy) |
| Chronology | transfer uses only earlier evidence; recorded wins pass the one-step and rollout checks |
| Verdict | missing, duplicate, misconfigured or unfinished pairs block the verdict; strict improvement is implemented |
| Budgets and RNG | every executor move counts; repairs can't reset totals; WM has a private RNG; the off-mode and mock-backend CPU checks pass |

---

## 5. Flags

*Provisional* values are calibrated in week 3 and frozen before the dev sweep (§6.7).

| Flag | Default | Meaning |
|---|---|---|
| `EVAL_WM` | `off` | `off` \| `full` (W2) \| `nogate` (W3) \| `rules` (W1, stretch) |
| `EVAL_EVENT_LOG` | on | authoritative engine-call log; logging only; replaces v1's action-trace flag |
| `EVAL_WM_BACKEND` | `vllm` | `mock` \| `vllm` \| `anthropic` \| `openai` |
| `EVAL_WM_MODEL` | Qwen3.5-35B-A3B | model id or endpoint |
| `EVAL_WM_K` | 4 (provisional) | candidates per round |
| `EVAL_WM_ROUNDS` | 2 | rounds per group per trigger |
| `EVAL_WM_CTX` / `EVAL_WM_THINK` | 32768 / 20480 | context and output budget; prompts ≤ CTX − THINK |
| `EVAL_WM_WARMUP` | 2000 (provisional) | transitions before the first rules call |
| `EVAL_WM_ACCEPT` | 0.95 | admission threshold (not planning trust) |
| `EVAL_WM_PLAN_EXACT` | 1 | planning requires 100% on retained applicable evidence and no contradiction |
| `EVAL_WM_MIN_MOVES` | 20 | moves a rule must apply to |
| `EVAL_WM_CE` | 5 (provisional) | counterexamples per repair batch |
| `EVAL_WM_MAX_REQUESTS` / `_COMPLETIONS` / `_GEN` | 40 / 160 / 1.5M (provisional) | per-run caps on HTTP requests, candidate completions and generated tokens |
| `EVAL_WM_RESERVE` | 0.25 (provisional) | share of completions reserved for completion tests and repairs |
| `EVAL_WM_MAX_OVERRIDE_ACTIONS` | 10000 (provisional) | cumulative executor moves per run |
| `EVAL_WM_PLAN_NODES` | 200000 | deterministic search budget, counted in expansions |
| `EVAL_WM_PLAN_SECS` | 60 | wall-clock watchdog, logged as its own outcome |
| `EVAL_WM_PLAN_FAILS` | 3 | failed plans before the per-level gate closes |
| `EVAL_WM_CACHE` | `results/wm_cache` | namespaced response cache (offline work and debugging) |
| `EVAL_WM_BUDGET_USD` | — | hard spend cap (frontier backends only) |

With `EVAL_WM=off` the layer is never constructed. This is checked as code-path isolation (§3.9), not as move-for-move equality between fresh GPU runs.

---

## 6. Evaluation

### 6.1 Tiers, baseline and data

| Tier | What | Data | Cost | Feeds |
|---|---|---|---|---|
| Stage A (frozen) | the pre-registered R1/R2 test (ft09, lp85, ls20), run to the end under its original criteria | its own frozen data | ~4 h | L2 |
| 0a Rules (offline) | rule loop on recorded corpora: per-rule accuracy, assembled coverage, transfer to the next level before repair; the matched capability comparison (§9) | existing mb_gated_att corpora, 8 dev games, seed 0 | ~1–2 h LLM per game, overnight | G1, L1–L2 |
| 0b Completion + plan | completion tests (fitted, path-checked), in-model re-plan and rollouts; then replay-to-level, where the planner takes over at the first screen of a level Goose didn't clear | 8 new mb_gated_att runs with the event log | ~2 h of runs + LLM | G2, L2–L4 |
| 1 Dev (online) | W2 vs A0 | 8 dev games × seeds 0, 1 × 100k | ~10 h (estimate) | G3 |
| 2 Confirm (online) | W2 vs A0 under the frozen protocol | 25 games × seeds 0–2 × 100k | ~2 days (estimate) | L5 |

**Rulebook dev games:** tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82.
- Compared with the upgrade-screen set, this swaps out ar25 and su15, which play a reduced action space (§1.1), for two full-action games where mb_gated_att clears at least one level. That keeps tuning from being confounded by a missing action.
- All prompt and threshold tuning happens on these eight.
- The other 17 games are **held out from Rulebook tuning** — not from earlier project evaluation, and not necessarily from model pretraining — and are reported separately.
- Stage A's lp85 and ls20 are used only by its frozen test.

**A0, the reference.** The 75 mb_gated_att runs from the upgrade confirm (`results/confirm_upgrade/20260927_221915`), documented at 112 levels. Reusing them requires:
1. **Code path:** off-mode isolation passes (§3.9).
2. **Same setup:** the same engine and game versions, action space (both arms play the reduced space on the same six games), 100k action cap and stopping policy.
3. **Scoring reconciliation.** These runs predate the event log, and the historical scorer misses a level completed on a run's final move (and the last level of a won game).
   - As far as the confirm table shows, no A0 run finished a game; the best is lp85 at 7 of 8. So at most one final-move level-up per run can be missing.
   - If the A0 run logs were kept, compare each run's `scorecard score=` line with its counted levels. Any mismatch gives the corrected, reconciled total.
   - If the logs are gone, the primary verdict scores **both** arms with the historical scorer. That is conservative for W2, the only arm likely to finish games, and W2's scorer-v2 total is reported alongside.
4. **Reruns** of A0 happen only if reconciliation finds mismatches that can't be corrected from the logs.

**Transfer protocol.** Evidence is frozen at the level-up. Next-level predictions are scored before any repair: one step at a time on that level's transitions, and as an uninterrupted rollout along its first attempt.

**Replay-to-level** replays Goose's own recorded engine calls to reach a level's first screen, then lets the planner act. It is a diagnostic: its levels are never added to any score.

**Games without a cleared level.** mb_gated_att clears nothing on sc25, wa30 and sb26, so there is no positive example for a completion test, and W2 cannot gain from planning there. That doesn't make W2 identical to A0 on those games. Fresh runs differ distributionally, and full mode only behaves like A0 given a private RNG, no interfering overrides and the same stopping policy (§3.9). Only W1 (stretch) targets these games.

### 6.2 Arms

| Arm | Flags | Purpose | Where |
|---|---|---|---|
| A0 | mb_gated_att | reference | reused (§6.1) |
| W2 | `EVAL_WM=full` | the headline arm | dev + confirm |
| W3 | `EVAL_WM=nogate` | single-factor ablation of the **trust gates** | dev, 2 seeds |
| W1 | `EVAL_WM=rules` | stretch: plan-eligible rules guide exploration only (known no-ops avoided; UNKNOWN and predicted-unseen moves preferred); no completion test, no planner | dev, if time; cut first |
| W2-frontier | `EVAL_WM=full` + frontier backend | deployment demonstration | 1 seed (§9) |

**W3, specified before any W3 result.** W3 is identical to W2 in candidate generation, checking (still computed for ranking and feedback), feedback rounds, prompts, budgets, action space, planner, executor step checks and the intervention cap. The only difference is that the trust gates are off:
- the top-ranked rule candidate per group is used even if it fails admission or is contradicted (no suspension);
- plan eligibility ignores exactness;
- the top-ranked completion test is used even if it fails the fit or path checks;
- the in-model re-plan check is skipped.

A stronger "no checking at all" ablation (no ranking and no feedback) is deferred.

### 6.3 Metrics

**Levels:**
- levels completed (sum over game × seed), under the scorer named in §6.1;
- paired better / same / worse per (game, seed);
- the paired delta, with a bootstrap that resamples games (seeds nested within games);
- games with ≥ 1 level;
- levels ≥ 2 counted separately, because AERA (arXiv 2605.25931) reports that progress on every public game is reachable with trivial strategies, some in one blind move;
- actions-to-level, with the number of seeds that reached each level (k of n) and censoring kept.

**Mechanism (the funnel), per game and per run:**
1. evidence diversity (distinct effect signatures per action group) and assembled changing-transition coverage;
2. per-rule applicable accuracy;
3. transfer before repair: one-step and rollout accuracy;
4. completion test: fitted / path-checked / live-validated, and whether positives were observed;
5. plans: found in-model, executed, live success, mismatch step, expansions, watchdog firings;
6. intervention costs: executor moves, failed-prefix lengths, attempts lost, conflicting-outcome keys;
7. LLM cost: requests, completions, tokens, wall-clock.

**Failure taxonomy.** Every game in every run gets all applicable flags plus one primary bottleneck at its last level:
- logging failure
- unsupported action (reduced action space)
- no rules admitted
- low coverage
- observation aliasing
- no cleared level (no completion positive)
- completion test failed to fit
- completion test didn't transfer
- search limit or watchdog
- plan mismatch
- intervention budget exhausted
- infrastructure timeout
- plan succeeded

### 6.4 Gates

- **G1 — 9 Oct (Tier 0a).** Assembled changing-transition coverage ≥ 80% on level 1, for ≥ 3 of the 8 dev games.
  - Coverage is the share of the level's unique changing keys that the assembled book predicts exactly; UNKNOWN counts as a miss.
  - Untouched transfer is reported beside it, and Stage A's R1/R2 verdict is reported as it comes out.
  - If G1 fails, take one day to try the dense 27B and thinking-off, then narrow the online build to the groups and games that pass. The report then leads with the funnel and taxonomy.
- **G2 — 20 Oct (Tier 0b).** All three of:
  - on ≥ 2 dev games, a completion test that is fitted (on observed positives) and path-checked;
  - those games also pass the in-model re-plan check;
  - replay-to-level clears ≥ 1 level that the matching mb_gated_att run did not, with complete terminal logging.

  This is the first milestone: one independently grounded completion test, plus a transferred model that clears a live level. If G2 fails, online integration still goes ahead, and the dev sweep is read as a diagnostic.
- **G3 — 29 Oct (dev sweep).** Launch the confirm only if all of these hold on the dev games:
  - W2 levels ≥ A0 levels, under the same scorer;
  - all 16 pairs complete, each with an explicit termination reason;
  - paired wins ≥ losses;
  - no game worse on both seeds;
  - intervention costs reported.

### 6.5 Confirm rule

W2 vs A0 on 25 games × seeds 0–2 × 100k actions, verdict by `tools/paired_compare.py --strict --expect 75`:
1. Exactly the declared 75 (game, seed) pairs, one valid run each, with matching configurations and action/stopping budgets. Solved games may end early; every other early end needs an explicit reason.
2. W2's total levels are **strictly greater** than the reconciled A0 total, with paired wins > losses.
3. No game is worse on every seed.

Passing all three means the headline is met (L5). Results are reported for all 25 games and for the 17 held-out games, with the paired delta, its uncertainty and the failures. This is a descriptive adoption rule, not a significance test: a small passing margin is not, on its own, robust evidence of general improvement.

### 6.6 Ablation reading

- **W2 > W3** (matched, single-factor): the trust gates help *in this setup*. Report it with effect size, seeds, uncertainty and exactly what differed.
- **W2 ≈ W3:** this experiment did not show a material benefit from the gates under these conditions. That is not evidence that verification is unimportant in general.
- Neither result alone supports "verification is what makes it work".

### 6.7 Protocol freeze (before the confirm)

Frozen and recorded before the confirm:
- the source commit and the prompts;
- model, tokenizer and quantization revisions, and the vLLM version;
- decoding settings and all budgets and caps;
- engine and game versions;
- the action space, the scorer and the failure handling.

Changing the number of seeds creates a different protocol; it must be declared, not used as a runtime knob. Any amendment to a test that has already started is dated, explained and kept beside the old version. Stage A's criteria are never edited.

---

## 7. Timeline

| Week | Dates | Work | Deliverable / gate |
|---|---|---|---|
| 1 | Oct 2–8 | **Logging and scoring first:** event log, terminal delivery, termination reasons, scorer v2, `paired_compare --strict/--expect`, A0 reconciliation check. Action-space disclosure. vLLM context/KV, k = 1/2/4 latency and Goose co-residency measurements. Find and fix Stage A's model-load failure, then run Stage A. Tier 0a on the existing dev-game corpora (doesn't wait on the logging fix). Then 8 mb_gated_att runs with the event log. | **Oct 9:** report TOC + Meeting Log 1 · **G1** |
| 2 | Oct 9–15 | `finishes_level` synthesis and fitting; path checks and rollouts; planner with deterministic limits; in-model re-plan; `replay_to_level.py` with chronology checks | — |
| 3 | Oct 16–22 | Online layer: ownership, executor (pixel + event checks), suspension, cumulative cap, UNKNOWN, budgets. Pre-dev correctness suite; CPU off-mode and mock-backend checks. Smoke run (ft09, 1 seed, 10k). Calibrate and freeze the provisional defaults. | **G2** (Oct 20) |
| 4 | Oct 23–29 | Dev sweep (W2, 16 runs); funnel; draft writing | **Oct 30:** complete first draft · **G3** |
| — | Nov 2 | Idea lock (the idea is already fixed; tuning continues) | |
| 5–6 | Oct 30–Nov 12 | Protocol freeze (§6.7); **confirm sweep**; W3 on dev (2 seeds); W1 if time allows | confirm verdict |
| 7 | Nov 13–19 | Frontier calibration and live run (§9); matched Tier 0a capability comparison; combine with teammates' arms (§11) | |
| 8–9 | Nov 20–Dec 6 | Analysis, figures, final report; buffer for reruns (Thanksgiving week) | **Dec 7:** final report |

**Critical path:** logging and scoring → traced runs → completion test + planner → online layer → correctness suite → dev sweep.

**If time runs short, cut in this order:**
1. W1.
2. The optional frontier Tier 0a point.
3. Shrink the frontier live run to its pre-selected subset (§9).

W3 and the frontier showcase are both kept: W3 is cheap local compute, and the showcase is one of the project's stated goals. If week 2 slips, the dev sweep drops to 1 seed; the Oct 30 draft still has the Tier 0 results.

---

## 8. Compute budget (RTX 5090)

All numbers below are estimates, to be replaced by the week-1 and week-3 measurements.

- **Memory.** Report weights (21.1 GiB), vLLM's reserved fraction of GPU memory (`--gpu-mem`), KV capacity (about 105k tokens at 0.90) and Goose's peak separately. Week 1 measures how many Goose processes fit alongside; expect 1–2.
- **Context.** 32,768 tokens with thinking. Prompts are ≤ 12,288 tokens, measured with representative images.
- **Candidate cost.** Measure k = 1, 2 and 4 with long outputs, retries, images and Goose resident, as median and 90th-percentile latency. The ~250 tok/s aggregate over six streams doesn't, by itself, show that four candidates cost the same latency as one.
- **LLM time per run.**
  - A typical first rules synthesis is ~5 groups × 2 rounds × 4 candidates × ~6k tokens ≈ 240k tokens, about 16 min at 250 tok/s. Click-colour groups can push this higher.
  - Later levels are mostly repairs (~60k tokens) and completion tests (~40k).
  - Expect roughly 30–40 min of LLM time plus ~13 min of Goose per run. The cap scenario (1.5M generated tokens) alone would be about 100 min.
- **CPU time.** Rule checks, assembled-book checks, path checks, rollouts and search run on the CPU and are timed separately.
- **Sweeps.**

  | Sweep | Estimated time |
  |---|---|
  | Tier 0a | 8–16 h, overnight |
  | Dev | about 8–10 h |
  | Confirm | about 35–45 h of LLM time, pipelined so one run's Goose plays while another's LLM call runs |
  | W3 on dev | about 8 h |

  Measure median and 90th-percentile run time in week 3 before committing to dates.
- **Knobs before the freeze:**
  - k from 4 to 2;
  - output budget from 20,480 to 12,000 (the prompt cap rises to match);
  - warm-up from 2,000 to 4,000.

  Changing the number of seeds is a protocol change (§6.7), not a knob.

---

## 9. Frontier runs (~$100)

There are two different frontier runs, labelled differently:

1. **Matched capability comparison (Tier 0a, offline).** Qwen3.5-9B, Qwen3.5-35B-A3B and a frontier model get the same frozen evidence packs, prompts, k = 1, ~4k-token thinking cap and round limits. Only this comparison isolates model capability. The 35B's deployment setting (k = 4, 20,480 tokens) is reported beside it as a separate point.
2. **Live frontier run.** W2 with a frontier backend, k = 1, ~4k thinking, ≤ 25 requests per run. This is a **deployment demonstration**: its settings differ from local W2, it covers one seed, and newer frontier models may have seen the public games.

**Model.** Sonnet-class, the best price/quality for writing code. List prices as of Oct 2026, to re-check on the day:

| Class | Input ($/M tokens) | Output ($/M tokens) |
|---|---|---|
| Sonnet-class | 3 | 15 |
| Opus-class | 5 | 25 |
| Haiku 4.5 | 1 | 5 |

Cached input costs about 10% of the input price.

**Cost model.** About 19 requests per run × (~12k input + ~6k output tokens) ≈ $0.12 per request, so ≈ $2.3 per run, or ≈ $57 for 25 games × 1 seed. Billing is metered for input, reasoning-plus-output tokens, retries and cache hits.

**Budget split:**

| Item | Budget | Notes |
|---|---|---|
| Calibration | $5 | 2 dev games |
| Matched Tier 0a point | $10 | optional; cut right after W1 |
| Live run | ~$75 | seed 0; all 25 games if calibration comes in ≤ $3 per run |
| Buffer | $10 | |

**Pre-selected subset.** If calibration exceeds $3 per run, the live run covers the 8 dev games, then held-out games in an order fixed now: the 17 held-out games shuffled with `random.Random(20261002)`, run until the budget is spent. Choosing games after seeing local or frontier results would turn it into a selected showcase. `EVAL_WM_BUDGET_USD` stops each run at its share.

---

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation | Early signal |
|---|---|---|---|---|
| R1 The 35B admits too few rules | medium | high | Stage A format; output budget + code-only retry; one group per task; best-of-k + feedback; try the dense 27B at G1 | G1 coverage (Oct 9) |
| R2 Rules don't transfer, or drift over rollouts | medium | medium | re-check on new-level transitions; repair on counterexamples; "describe the mechanism, not positions" instruction; rollout checks | Tier 0a/0b transfer and rollout accuracy |
| R3 The completion test fits but doesn't generalize | high | high | observed positives; near-goal and failed-plan negatives; cross-level fit; path-checked and live-validated tiers | G2 (Oct 20) |
| R4 Circular completion evidence | low (by design) | high | positives are observed (board, move) pairs labelled by real level-ups; no simulator-made positives | fit report: positives observed |
| R5 Search exploits modelling gaps | medium | medium | plan-eligible rules only; UNKNOWN never expanded; pixel + event step checks | plan mismatch rate |
| R6 Wrong plans cost attempts | medium | material | cumulative cap; 3-fail gate; failed-combination and death blocklists; suspension | failed-prefix lengths, attempts lost |
| R7 Repairs reopen costly plans | medium | medium | the cumulative cap ignores repairs; failed (board, move, version) blocklist | executor moves after repairs |
| R8 Missing terminal outcomes | certain before the week-1 fix | high | event log; terminal delivery; scorer v2; same-scorer verdict | correctness suite |
| R9 Masked resources cause losses | medium | medium | attempt-length bound; raw bar values logged; encoding deferred | deaths during plans |
| R10 Controller bookkeeping corrupts map or deadclick state | low after the fix | medium | explicit ownership; map-takeover tests | correctness suite |
| R11 Incompatible baseline reuse | low–medium | high | code-path isolation; compatibility checklist; reconciliation; same-scorer verdict | week-1 reconciliation |
| R12 Reduced action space caps six games | certain | medium | disclosed; ar25 and su15 out of the dev set; ACTION7 not added within Rulebook | taxonomy flag |
| R13 Observation aliasing | medium | low–medium | outcome variants; versioned mask; flagged, not modelled | conflicting-outcome keys |
| R14 Context, VRAM or latency too tight | medium | medium | week-1 measurements; knobs (§8) | week-1 measurements |
| R15 Time (solo, 9 weeks) | high | high | dated gates; Tier 0 results stand on their own; cut order (§7) | weekly gate dates |

---

## 11. Working alongside teammates' LLM arms

- **One observation contract.** WM reads the event log and the transition stream defined by eval_common. It can sit on top of any explorer that honours that contract, including a teammate's modified Goose.
- **Explicit ownership.** While WM executes, it owns the move.
  - Lower-priority selectors (deadclick, the map, and any teammate override) are not consulted for that move and must not consume state for it.
  - Map routes are cancelled on takeover and revalidated on handback.
  - If a teammate's arm also overrides moves, agree on the ownership order first and add it to the correctness suite.
- **Combining at the end.** Run W2 on top of the best explorer arm (a teammate's, or mb_gated_att) under the same frozen protocol and scorer.
- **Shared GPU.** If a teammate also serves a model on the 5090, use one vLLM server and schedule sweeps; two ~20 GB models don't fit at once.
- **Shared reference.** Every arm compares against the same reconciled A0, using the same scorer.

---

## 12. What goes into the report

| Deliverable | Due | Contents |
|---|---|---|
| Report TOC | Oct 9 | Method: "Goose + replay-consistent world model" (architecture; rule / completion-test / plan loop; trust tiers). Evaluation: tiers, gates, protocol freeze. Results: Tier 0 funnel, dev, confirm, ablation, frontier. Discussion: failure taxonomy, capability comparison, threats to validity. |
| First draft | Oct 30 | Method complete; Stage A verdict; Tier 0a/0b funnel; scoring reconciliation; replay-to-level examples; dev sweep if finished. |
| Final report | Dec 7 | Confirm verdict; W3 ablation; matched capability comparison; frontier demonstration; combined arm. |

**Planned figures:**
1. **The funnel**, with every stage kept separate: games → admitted / plan-eligible rules → transfer before repair → completion test fitted → path-checked → in-model re-plan → live plan → new levels. This figure works even if L5 fails.
2. Paired levels per game, A0 vs W2, 3 seeds, with a confidence band from resampling games.
3. **Capability comparison** at matched settings (9B / 35B / frontier), with the deployment points marked as such.
4. A worked example: a rule book and completion test the 35B wrote, and the planned moves on the next level as a frame strip.
5. Levels-vs-actions curves for games where the planner took over, with k-of-n seeds and censoring shown.

**Wording.**
- Hypotheses are "replay-consistent", never "true rules".
- Gains are reported with intervention costs and any regressions, not as a guaranteed floor.
- The 17 games are "held out from Rulebook tuning".
- Integrity claims match the actual execution boundary (§3.3).

**Threats to validity:**
- public games only (the ARC authors call them an easier demo set);
- pretraining may include public discussion of the preview games (ft09, ls20, vc33);
- the first level of many public games is reachable by trivial strategies, so levels ≥ 2 are reported separately;
- six games play a reduced action space;
- historical A0 scoring (§6.1);
- seed variability (9.6× spread);
- intervention costs;
- the frontier live run is a one-seed deployment demonstration.

---

## 13. Decision log

| # | Decision | Chosen | Alternatives | Why |
|---|---|---|---|---|
| 1 | Job of the LLM | write and repair hypotheses about mechanics and level completion | pick moves; advise Goose; judge progress; write rewards | 100k-move runs rule out per-move calls; Probes A and C were NO-GO at 2–9B; checked executable models are behind the strongest results (§1.2) |
| 2 | Who acts | mb_gated_att by default; WM only while executing a plan, under a cumulative cap | LLM-led agent | keeps Goose's behaviour wherever the model has nothing to offer; exposure bounded and measured, not assumed away |
| 3 | Representation | Stage A rule books: one admitted changing rule per group, plus known no-ops; UNKNOWN otherwise | one `step()` program; object-state DSL; neural model | proven on ft09 with the 35B; small tasks; independent check and repair; complementary rules per group deferred |
| 4 | Trust | admission ≥ 95% (Stage A) vs plan eligibility (exact on retained evidence, uncontradicted); suspension on the first contradiction | one threshold; no check | errors compound over plan steps (1 − 0.95²⁰ ≈ 64%) |
| 5 | Evidence format | ≤ 6 moves per group; exact changed cells as text + pictures; diverse effect signatures | pictures only; full grids | Stage A: pictures alone led to misread colours and distances |
| 6 | Model | Qwen3.5-35B-A3B GPTQ-Int4 | dense 27B; 9B; newer 27Bs | largest that fits; same rule quality as the 27B at 3.6× the speed; 9B kept for the capability comparison |
| 7 | Sampling | best-of-k (provisional 4) + feedback; cost measured | single sample; long repair chains | generate-and-verify; k is set from week-1 latency, not assumed to be free |
| 8 | Context and thinking | 32,768 context, 20,480 output, prompts ≤ 12,288; code-only retry | 16k output | Stage A's limits; 16k ran out on ls20 and lp85 |
| 9 | When to call | warm-up, counterexample batches, stall, level-up; short levels keep their evidence | every N moves | Tycho and OPINE motivate on-demand synthesis; W2's logs check the schedule |
| 10 | Sync vs async | synchronous | background synthesis | fixed evidence cutoffs; fair across model speeds |
| 11 | Completion form | `finishes_level(board, act, api)`, labelled by real level-ups | a test on after-boards | no invented positives; handles submit-style wins; doesn't need rules first |
| 12 | Completion trust | fitted → path-checked → live-validated; cross-level, near-goal and failed-plan negatives | trust after fitting | one positive per level can still be memorized |
| 13 | Planner | known path first; UNKNOWN never expanded; deterministic expansions; BFS → best-first with depth bookkeeping | MCTS; LLM-scored nodes; wall-clock limits | separates model errors from search errors; reproducible search |
| 14 | Exposure | pixel + event step checks; suspension; invalidation; blocklists; 3-fail gate (local); cumulative cap (global) | the 3-fail gate alone | the gate alone doesn't bound total waste across repairs |
| 15 | Across levels | carry rules and test; rebuild per-level api, mask, actions; transfer measured before repair | reset everything; persist Goose's network | ARC-AGI-3 reuses mechanics across levels; persisting the network only moved 14 → 18 |
| 16 | Evaluation | 25 × 3 × 100k; A0 reused only under the §6.1 conditions; same scorer for both arms; strict > and 75/75 runs | new protocol; the tool's defaults (≥, intersection of pairs) | comparability with semester 2, without hidden ties or dropped runs |
| 17 | Dev / held-out | 8 dev games, with ar25 and su15 swapped for m0r0 and cd82; 17 held out from Rulebook tuning | the upgrade-screen set | a reduced action space would confound tuning; honest "held-out" wording |
| 18 | Ablation | W3 = trust gates off, everything else identical | several simultaneous changes | a single-factor ablation is interpretable |
| 19 | Fine-tuning | not now | LoRA or RL on public games | no time before Nov 2; cross-game generalization unproven; the RL effort found used 2×H100 |
| 20 | Frontier | matched offline comparison + live deployment demonstration; pre-selected subset; billing metered | one unmatched live run | only matched settings isolate the model |
| 21 | Integrity | AST checks, restricted builtins, spawned worker without engine objects; separate import and file-access tests; bounded claim | claim full source exclusion | the sandbox describes itself as non-hardened |
| 22 | Reproducibility | code-path isolation + CPU off-mode and mock-backend checks; distributional comparisons; cache for offline work and debugging | byte-identical fresh runs | the repo documents CUDA divergence after ~300 actions |
| 23 | Scoring | event log + scorer v2; A0 reconciled via scorecard lines; same-scorer fallback | trust the historical scorer | final-move and final-win levels were not recorded |
| 24 | Ownership | WM owns the move while executing; lower overrides not consulted | WM override last in the chain | the map consumes route steps and RNG before a later override |
| 25 | Action space | reduced space disclosed; ACTION7 not added inside Rulebook | add ACTION7 now | adding it needs a matched expanded-action baseline (a separate experiment) |

---

## 14. Out of scope (and why)

**Deferred engineering** (good ideas, not before the lock):
- several complementary changing rules per action group;
- click refinement when search fails;
- encoding resource-bar values into the model;
- extra synthesis triggered by sparse evidence or competing hypotheses;
- a hardened worker without filesystem access to the game sources;
- a fully deterministic training mode;
- a "no checking at all" ablation;
- adding ACTION7, as a separate experiment with its own matched baseline.

**Scope choices:**
- Fine-tuning or distilling a small model: revisit after December.
- Goal-directed execution before the first level is won: a frontier demonstration alone doesn't decide whether it's feasible.
- History-dependent models: out of scope, though raw chronology is still kept to diagnose aliasing.

**Never:** engine save/restore in scored runs, reading game code, or sharing learned hypotheses across scored runs. Cache replay only reproduces a recorded run; fresh scored runs never inherit another attempt's hypotheses or results.

---

## 15. References

arXiv versions are pinned where known; re-verify every version before the report.

- ARC Prize Foundation. *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence.* arXiv 2603.24621.
- S. Rodionov. *Executable World Models for ARC-AGI-3 in the Era of Coding Agents.* arXiv 2605.05138 (pin version). Code: github.com/astroseger/arc-3-agents-baseline1.
- S. Rodionov. *Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?* arXiv 2607.15439 (pin version).
- D. Courtis, W. Li, S. Sanner. *OPINE-World.* arXiv 2607.01531 (v2, read 1 Oct 2026; the review reports a later revision).
- J. Lehmann, A. Aioanei, S. Vahdati. *Tycho.* arXiv 2607.28287 (pin version). Code: github.com/NIMI-research/Tycho.
- ARC Prize 2026 Milestone 1 results: arcprize.org/blog/arc-prize-2026-milestone-1. Tufa Labs Duck write-up: tufalabs.ai/research/duck-harness.
- Polyphony Agent: arcprize.org/leaderboard/community; github.com/Mininglamp-AI/polyphony-arc-3.
- *Explore Before You Solve* (AERA). arXiv 2605.25931.
- J. Xiao, H. Huang. *Compiled Agency.* arXiv 2609.18996.
- H. Tang, D. Key, K. Ellis. *WorldCoder.* arXiv 2402.12275.
- N. Dainese et al. *Generating Code World Models with LLMs Guided by MCTS.* arXiv 2405.15383.
- Corrêa et al. 2025. *Classical planning with LLM-generated heuristics* (as cited by OPINE-World).
- A. Ecoffet et al. *First return, then explore* (Go-Explore). Nature, 2021.
- vLLM engine arguments: docs.vllm.ai/en/latest/configuration/engine_args/.
- This repo at 5da1508:
  - `README.md` (CUDA divergence);
  - `run_local.py` (WIN/cap loop, ACTION7 filter);
  - `custom_agents/action.py`, `custom_agents/return_map.py`, `custom_agents/canon.py`;
  - `tools/paired_compare.py`, `compute_metrics.py`;
  - `legacy/llm_track/` (Probes A and C, Stage A, `rule_writer.py`, `heur_sandbox.py`);
  - `docs/plans/upgrade-confirm-results.md`.
- Design review annotations C01–C31 and sources R1–R16: `plan-C-verified-world-model-annotated.md`.

---

## 16. Changelog (v1 → v2)

| Review | Status | What changed | Where |
|---|---|---|---|
| C01 | Adopted | the summary uses "replay-consistent", separates admission from planning trust, and states that completion positives are observed | Summary, §0.4 |
| C02 | Adopted (modified) | headline made empirical; strict > reconciled A0, with 112 kept as the documented figure pending §6.1; wording tightened rather than taken verbatim | §0.1 |
| C03 | Adopted | links split: applicability vs coverage, untouched transfer + rollouts, completion tiers, in-model vs live planning, uncertainty; matched vs deployment capability points | §0.2, §9 |
| C04 | Adopted | the "floor" replaced by bounded intervention; sentences 1, 3 and 5 rewritten | §0.3, §1.3, §13 (2, 14) |
| C05 | Adopted | ceiling claim qualified; the Stage A smoke test described as a training fit | §1.1, §1.3 |
| C06 | Adopted (modified) | Rodionov's result attributed to the whole package; Tycho and Polyphony/Duck readings softened; OPINE cited as v2 with the later revision noted, since its details weren't independently confirmed | §1.2, §15 |
| C07 | Adopted | evidence-diversity metric; prompts prioritize counterexamples and under-sampled contexts | §1.3, §3.1, §6.3 |
| C08 | Adopted | risks revised: aliasing, material bad-plan impact, fit vs generalization, week-1 calibration | §1.4, §10 |
| C09 | Adopted | immediate event capture; explicit ownership; executed-move bookkeeping; versioned plan dependencies; UNKNOWN vs no-op; 8-hour stop disclosed | §2, §3.6, §3.9, §3.10, §4 |
| C10 | Adopted, part deferred | raw log + outcome variants + versioned per-level mask; frozen stretches kept in the check set; resource-bar encoding deferred | §3.1, §14 |
| C11 | Adopted, part deferred | admission vs plan eligibility; suspension on the first contradiction; UNKNOWN; assembled-book check; known no-ops; complementary rules per group deferred | §3.2, §14 |
| C12 | Adopted (claim + tests) | bounded integrity claim; separate import and file-access tests; hardened worker deferred | §3.3, §14 |
| C13 | Adopted | `finishes_level(board, act, api)`; observed positives only; fitted / path-checked / live-validated tiers; submit toy; `progress` treated as a diagnostic | §3.4 |
| C14 | Adopted, part deferred | known path first; UNKNOWN moves never expanded; deterministic expansions; reopening; remaining-budget bound; click refinement deferred | §3.5, §14 |
| C15 | Adopted | cumulative cap; suspension; invalidation; blocklists; takeover handling; terminal capture; pixel + event checks | §3.6 |
| C16 | Adopted, part deferred | budget units metered separately; reserve; click-colour groups; short levels; cutoff-tagged transfer; information trigger deferred | §3.7, §3.8 |
| C17 | Adopted | byte-identity claims replaced with code-path isolation, CPU and mock-backend checks; namespaced cache; private RNG; deterministic search | §3.9, §5 |
| C18 | Adopted (modified) | event log + scorer v2; A0 reconciled via scorecard lines, rerun only on an uncorrectable mismatch, with a same-scorer fallback | §3.10, §4, §6.1 |
| C19 | Adopted | file plan expanded; pre-dev correctness suite; estimates reopened | §4 |
| C20 | Adopted (modified) | new controls added with provisional defaults (calibrated in week 3, frozen before dev) instead of being left undecided | §5 |
| C21 | Adopted (modified) | compatibility conditions; transfer protocol; "W2 = A0 by construction" removed; held-out wording; reruns replaced by reconciliation unless it fails | §6.1 |
| C22 | Adopted (modified) | W3 specified as single-factor (trust gates off); matched capability comparison by running local models at frontier settings; uncertainty from resampling games; expanded taxonomy | §6.2, §6.3, §9 |
| C23 | Adopted | gates clarified; full protocol freeze | §6.4, §6.7 |
| C24 | Adopted | strict >; 75/75 runs; `--strict/--expect`; uncertainty; historical verdicts preserved | §4, §6.5 |
| C25 | Adopted | ablation interpretation wording | §6.6 |
| C26 | Adopted (modified) | logging and scoring moved onto the week-1 critical path; G2 moved to Oct 20; cut order puts W1 first, keeping both the frontier showcase and W3 (the showcase is a stated project goal) | §7 |
| C27 | Adopted | context/KV accounting; prompt cap of 12,288; measured candidate latency; CPU time; cap scenario; seed changes declared as protocol changes | §3.9, §8 |
| C28 | Adopted | deployment-demonstration label; billing metering; seeded pre-selection of any subset | §9 |
| C29 | Adopted + addition | new risks; ownership wording; ACTION7 disclosure; the dev set swaps ar25 and su15 for m0r0 and cd82 | §1.1, §6.1, §10, §11, §12 |
| C30 | Adopted | report wording and decision log reconciled; decisions 22–25 added | §12, §13 |
| C31 | Adopted | scope boundaries; G2 framed as the first milestone | §6.4, §14 |
