# LLM Track — Detailed Design: "Judge → Head → Heuristics" (v1)

**Owner:** Matt · **Companion to:** `295B-llm-track-plan.md` (the option survey) · **Status:** design, pre-implementation, no code
**Scope:** everything needed to build, evaluate, and write up the recommended LLM design on top of StochasticGoose (`custom_agents/action.py`) and the existing eval contract (`eval_common.py`, `metrics_common.py`, `compute_metrics.py`).

---

## 1. Design goal and the one-paragraph summary

**Goal.** Give StochasticGoose a sense of *progress* — not just "did the screen change" — sourced from a small language model's common-sense priors, without ever putting the LLM on the per-action path, and in a form where every piece can be switched off for a clean ablation.

**Summary.** An LLM ("the Judge") looks at pairs of logged transitions, described in a compact text form, and says which one looks more like progress. Those preferences train a tiny transition scorer ("g") that can rate any transition in microseconds. A new output head on the existing Goose network ("the V-head") learns to *predict* g's rating for each possible action before acting, and the sampler is biased toward high-predicted-progress actions. Separately, every few thousand actions (or when the run stalls), the LLM reads a digest of the run and *writes small scoring heuristics* that become additional label sources for the V-head. The LLM runs offline and periodically; the agent runs at full speed.

Three artifacts leave the LLM and run for free: **a label set** (offline), **a distilled scorer g** (every step), and **heuristic functions** (every step). That is the whole trick.

---

## 2. Why this design and not the others (reasoning)

The option survey listed eight ways to use an LLM. This design is P1 as the backbone, P2 as the periodic mechanism, with P6/P3 as stretch. The reasons, in order of weight:

1. **It matches the measured failure.** Semester 1 showed the learning currency is wrong: "frame changed" is satisfied by tickers and animations; the signal is degenerate on ~80% of games; agents jiggle instead of discover. A *progress* prior attacks that directly. Memory (graph) and modeling (dynamics) attack other failures — the teammates own those.
2. **It has the closest published analogue.** Motif (LLM preference labels → intrinsic reward) and ONI (the online version) are the one line of work where an LLM's priors were turned into an RL-speed reward on a sparse, instruction-free game (NetHack) and beat count-based exploration. Nothing else in the survey has evidence that close to our setting.
3. **It respects the throughput budget by construction.** The per-step path is a CNN head plus a small MLP — the same order of cost as the current change head. The LLM never blocks the agent.
4. **Every stage is separable and has a zero-setting that reproduces the baseline bit-for-bit.** With the bias coefficient at zero the sampler is unchanged; with no heuristics loaded the label pipeline reduces to the Judge alone. That is exactly what an ablation study needs.
5. **It is self-sufficient.** The serializer is computed from `(frame, action, next_frame, level)` — the exact fields already in the `.npz` corpus — so nothing waits on the object-perception teammate. When their layer arrives, it enriches the serializer behind the same interface.
6. **It produces defensible report material even if the answer is "no."** Label-agreement studies, distillation accuracy, held-out pairwise accuracy, and multi-seed ablations are all results regardless of whether the level ceiling moves.

What it deliberately is *not*: a planner, a world model, or a rule-inducer. Those are higher-ceiling and higher-risk; P3 (verified rule induction) is the designated stretch if the probes make it look reachable.

---

## 3. System overview

### 3.1 Components

| # | Component | Runs | Cost per call | Input → Output |
|---|---|---|---|---|
| C1 | **Serializer** | every logged transition (and offline over the corpus) | microseconds (numpy) | `(frame, action, next_frame, level Δ, score Δ)` → structured feature record + ~100–200-token text |
| C2 | **Judge** (LLM, Qwen3.5-9B class; 27B-class for a gold subset) | offline; optional async refresh during runs | ~0.1–1 s per pair | two serialized transitions → preference (A / B / tie) |
| C3 | **Transition scorer g** (tiny MLP, optional small conv branch) | every step (post-hoc, once `next_frame` is known) | microseconds | serialized feature record → scalar progress score |
| C4 | **V-head** (new output head on `ActionModel`) | every step (pre-hoc) | same as existing head | state → 5 + 4096 predicted progress values, one per candidate action |
| C5 | **Sampler bias** (modification of `_sample_from_combined_output`) | every step | negligible | change-probabilities × exp(β · V) → action |
| C6 | **Run digest** | every K actions or on event | milliseconds | rolling stats + top/bottom transitions → text |
| C7 | **Heuristic writer** (LLM, same model, thinking mode) | periodic | ~10–60 s | digest + state-API spec → candidate scoring functions |
| C8 | **Heuristic validator / sandbox** | on each candidate | seconds | candidate → accepted / rejected + agreement score |
| C9 | **Label mixer** | every step | negligible | g score + accepted-heuristic scores → training label for C4 |
| C10 | **Event/stall trigger** | every step (cheap counters) | negligible | online `novelty_late`, level change → fires C6/C7 |

### 3.2 Data flow (three time-scales)

```
OFFLINE (once per corpus version)
  corpus .npz ──C1──► feature records ──dedupe/stratify──► pair sampler ──C2──► preferences
                                                                                  │
                                                          Bradley–Terry fit ◄─────┘
                                                                 │
                                                          train C3 (g)  ──► g.pt (frozen artifact)

PERIODIC (every K≈5k actions, or on level change / stall)
  run state ──C6──► digest ──C7──► candidate heuristics ──C8──► accepted set (hot-loaded)

PER STEP (inside choose_action, ~120 act/s)
  prev transition ──C1──► record ──C3 (g) + accepted heuristics ──C9──► label
                                                                          │
  experience buffer entry gains a `progress` field ◄──────────────────────┘
                                                                          │
  train step (existing cadence) ──► change loss (unchanged) + V-head loss ◄┘
  current frame ──ActionModel──► change logits + V values ──C5──► action
```

---

## 4. Component designs

### 4.1 C1 — Serializer (the foundation)

**Purpose.** Turn a raw transition into (a) a fixed-length numeric feature record for g and the validator, and (b) a short text rendering for the Judge and the heuristic writer. Same function for both; text is rendered from the record.

**Inputs** (all already logged by `TransitionLogger`): `frame` (64×64 uint8), unified `action_idx` (0–4 = ACTION1–5; 5 + 64·y + x = click), `next_frame`, `changed`, `level` before/after, `action_num`.

**Feature record contents.**
- *Action:* type (1–6); for clicks, coordinates and the color under the click point before and after.
- *Global change:* changed-cell count; fraction of grid; whether change is confined to a known indicator/ticker region (from the canonicalizer in `metrics_common.py`: fixed tickers = cells changing ≥95% of transitions; rotating tickers = compact set covering tiny ≤2-cell transitions).
- *Change components:* connected components of the changed-cell mask (8-connectivity), up to N=8 largest, each with: size, bounding box, dominant color before → after, centroid; plus a *displacement guess* (if a same-shape same-color component appears shifted by (dy, dx) between frames, record the shift — this captures "object moved" without a real object layer).
- *Per-color deltas:* count of cells per color before and after (16 numbers each) and their difference — "4 cells of color 9 vanished" falls out of this.
- *Outcome flags:* level/score changed; `changed` flag; distance in actions to the next level-up if known (hindsight only; offline).
- *Context (for the Judge's benefit, cheap):* number of distinct colors on screen; count of non-background components before/after (using the most frequent color as background).

**Text rendering.** Deterministic template, no free text: `ACTION3 | changed 6 cells (0.1%) | comp1: 2×2 color 4→4 shifted (+0,+1) | comp2: 4 cells color 9→0 at (12,40) | color 9: 8→4 | ticker: no | level: same`. Aim for ≤200 tokens; truncate components beyond N.

**Design notes and reasoning.**
- Everything is computable from the corpus, so the Judge can be run over historical runs (millions of transitions from 25 games) with no new play — the labeling study starts on day one.
- The ticker flag is deliberately included as an input: the whole point is to teach the scorer that "ticker-only change" is not progress, and the harness already knows which cells are tickers.
- Displacement guess is the cheapest possible "object moved" detector; it will be wrong for overlapping shapes, which is acceptable at this stage and is precisely what the teammate's object layer later fixes.
- *Interface stability:* the record is a named schema (versioned); the object layer, when ready, adds fields and richer component descriptors without changing consumers.

**Pros:** trivial cost; self-sufficient; reusable by the validator, the digest, and the report. **Cons:** hand-designed features carry designer bias (mitigated by the pairwise Judge, which sees the text and can weigh features differently than the designer expected); ambiguity on overlapping objects.

### 4.2 C2 — The Judge (pairwise LLM preference labeling)

**Purpose.** Convert the LLM's priors into labels without trusting any free-form claim.

**Model choice.** Default **Qwen3.5-9B** (dense, Apache 2.0, ~18 GB bf16 or ~6 GB 4-bit; thinking mode off for labeling to keep throughput ~5–10 pairs/s). **Gold subset** labeled by a 27–31B model (Qwen3.6-27B or Gemma 4 31B, quantized, offline only, ~1 pair/s) to measure how much the 9B loses. **Fallback/ablation** Qwen3.5-4B (~25 pairs/s) — if Probe A shows it matches the 9B, use it for volume.

**Why pairwise rather than absolute scores.** Absolute 1–10 scores from small models drift with phrasing and are inconsistent across batches; pairwise "which is more like progress" is the format Motif validated, and Bradley–Terry turns preferences into a consistent scalar scale. Ties are allowed and informative (two ticker-only transitions should tie).

**Pair sampling policy** (this decides label efficiency more than model size):
- *Deduplicate by signature.* Many thousands of transitions are identical in the feature record (the same ticker tick, the same no-op). Collapse to unique signatures with multiplicity; label unique signatures only. Expect 10–100× reduction.
- *Stratify.* Buckets: unchanged; ticker-only; small non-ticker change; large change; level-completing; click vs. button; per game. Sample pairs within and across buckets so the scale covers the full range.
- *Same-game, same-level pairs.* Comparing across games is meaningless to a model that doesn't know either game; keep pairs within a game and, where possible, within a level.
- *Hindsight anchors.* Include pairs where one side is within the last k (e.g., 20) non-trivial transitions before a level completion — the closest thing to ground truth. These are the calibration set for Probe A and a held-out test for g.
- *Volume.* Target 20–50k labeled pairs across games for the first g; at 5–10 pairs/s that is 1–3 hours of 9B time. Gold subset: 2–5k pairs overnight.

**Prompting.** System prompt states the setting (a grid puzzle game with an unknown goal; the player is trying to make progress toward completing the level), defines the text format, and asks for `A`, `B`, or `TIE` plus one short reason (the reason is *logged for the report, never consumed*). Both orderings of each pair are queried to cancel position bias; disagreements between orderings count as ties. Temperature 0.

**Outputs.** A labels table: pair id, game, level, signature ids, verdict (both orderings), model, reason. Bradley–Terry (or simple logistic) fit yields a scalar `progress` per signature, per game (with a pooled fit as a second variant).

**Pros:** robust to hallucination (only the choice is consumed); parallel and offline; the labels themselves are a publishable artifact. **Cons:** small-model priors may be wrong about ARC mechanics (measured by Probe A against hindsight anchors); labeling cost scales with signature diversity, which is highest on exactly the games that matter (ft09 has ~181k unique canonical states) — mitigated by stratified caps per bucket.

### 4.3 C3 — Transition scorer g (distilled judge)

**Purpose.** Make the Judge's taste available every step, on every transition, including ones the Judge never saw.

**Form.** A small MLP over the numeric feature record (order of 10³–10⁴ parameters). Optional second branch: a 3-layer conv over the *changed-cell mask and color-delta channels* (not the full frame), pooled and concatenated. Start with the MLP; add the conv branch only if held-out pairwise accuracy is clearly limited by features.

**Training.** Pairwise loss on Judge labels (predict the preferred side; ties as soft targets), i.e., the same Bradley–Terry objective — g *is* the learned BT scale generalized to unseen records. Two variants: per-game g (one per game, trained on that game's labels) and pooled g (all games, with a game-id embedding *withheld* — forces transferable features). Report both; the pooled one is the "generalization" claim.

**Validation.** Held-out pairwise accuracy vs. the 9B Judge; accuracy vs. the 27B gold subset; agreement with hindsight anchors; calibration plot of g against BT scores. Acceptance bar for wiring into the agent: beats a "changed-cell-count" baseline scorer on held-out pairs by a clear margin, and rates ticker-only transitions below non-ticker changes essentially always.

**Runtime.** Called once per logged transition in `choose_action`, right where `frame_changed` is computed today (the transition is fully known there). Output stored on the experience entry as `progress` alongside the existing 0/1 `reward`.

**Pros:** negligible cost; deterministic; can be retrained in minutes when labels improve; per-game vs pooled is a clean transfer experiment. **Cons:** only as good as the serializer's features; a scorer that has only seen level-1/2 transitions may extrapolate poorly to later levels (mitigated by the periodic refresh and by hindsight anchors from the few level-3+ completions in the corpus).

### 4.4 C4 — V-head (predicting progress before acting)

**Purpose.** g scores a transition after the fact; the sampler needs a per-action estimate *before* acting. The V-head learns to predict g's score for each candidate action from the current frame — exactly the relationship the existing change head has to the `changed` label.

**Form.** A second output head on `ActionModel` sharing the conv backbone: 5 + 4096 outputs, same layout as the change head. Regression targets are g scores (standardized per game) for the *taken* action only, as with the change label; other actions get no gradient. Loss: Huber, weighted by a coefficient that is a new, logged hyperparameter (the existing change loss and its confidence-bonus coefficients are left untouched, per the repo's freeze rule).

**Training cadence.** Same `train_frequency` and `batch_size` as today; the batch simply carries one more target. No new optimizer, no new schedule — the ablation stays clean.

**Reset semantics.** The V-head is part of `ActionModel`, so under `EVAL_RESET_ON_LEVEL=1` it resets with the model at each level boundary, and under the persistence arm it carries over. g and the accepted heuristics are *not* part of the model and persist regardless. This gives a new, cheap factor: what persists (nothing / the head / the head + heuristics).

**Pros:** reuses the entire existing training path; no per-step cost beyond a larger final layer; ablation-clean. **Cons:** for click actions the head must learn a 4096-way progress map from sparse labels — the same difficulty the change head already faces, and the reason Goose works best on click games; the head inherits that.

### 4.5 C5 — Sampler bias

**Purpose.** Use the V-head without breaking what already works.

**Rule.** Today: sigmoid change-probabilities per action (clicks scaled by 1/4096), normalized, sampled. New: multiply each action's probability by `exp(β · V(a))` before normalization, with V standardized over the available actions. β is a new logged hyperparameter with a schedule: 0 for the first M actions of a level (let the change head warm up, as it does now), ramping to β_max. Masking of unavailable actions is unchanged.

**Properties.** β = 0 reproduces the baseline sampler exactly. The change head still gates — an action predicted not to change the frame is still unlikely — so the bias reorders *among* live actions rather than overriding the one thing Goose learns well. Guard: cap the multiplicative factor (e.g., ≤ 20×) so a badly calibrated head cannot collapse exploration.

**Pros:** minimal, interpretable, reversible. **Cons:** β is a real tuning knob; report a small sweep (3 values) rather than a single chosen value.

### 4.6 C6 — Run digest

**Purpose.** Give the periodic LLM a compact, honest picture of the run so far.

**Contents (~500–1000 tokens).** Game id, level, actions so far; per-1k-action series of `novelty_late`, meaningful-change rate, redundancy (the harness already computes these per 1k); action-type mix; the top-10 and bottom-10 transitions by g (rendered by C1); the most frequent 10 signatures with counts (this is where tickers show up); currently accepted heuristics and their measured agreement/uplift; the last few level events with the transitions that preceded them.

### 4.7 C7 — Heuristic writer (periodic LLM)

**Purpose.** Let the LLM adapt to *this* game as it is being played, at a cadence the budget allows, by producing code that runs for free.

**What it writes.** Small pure functions over the feature record (never the raw frame; never the network) returning a scalar. Two allowed kinds: *transition heuristics* (score a completed transition — become extra label sources for the V-head through C9) and *action priors* (score a candidate action from the current record's static context, e.g. "prefer clicks on cells whose color has never been clicked" — applied in C5 as a bounded multiplicative prior). The state-API spec the LLM sees is the record schema plus a short list of helper accessors; nothing else is importable.

**Cadence.** Every K = 5,000 actions (≈40 s of play at 120 act/s; the LLM has that long), plus on level change and on stall (C10). The LLM runs in a *separate process* (vLLM or HF) with a fixed GPU memory reservation so the agent's compute is unaffected; requests and results go through a small file/queue handoff; the agent hot-loads accepted heuristics at the next boundary and never waits.

**Prompt.** Digest + schema + the currently accepted heuristics with their scores + instruction to propose 3–5 *diverse* candidates with a one-line rationale each (rationale logged for the report). Thinking mode on. Temperature moderate for diversity.

### 4.8 C8 — Validator / sandbox

**Purpose.** The Judge's hallucination defense was "only consume the choice"; the writer's defense is "only consume code that passes tests and helps on data."

**Gates, in order.** (1) Static: parses; imports nothing; only allowed accessors; bounded runtime on a synthetic worst-case record. (2) Dynamic: runs on the last 5k logged records without exception, finite output, non-constant. (3) Agreement: rank-correlation with g and with hindsight anchors on the same records; reject if it anti-correlates with anchors. (4) Online trial: accepted candidates enter a small bandit — each window of W actions uses one candidate set; keep those whose windows show better `novelty_late`/coverage per action; retire losers. Every accepted heuristic is logged with source, rationale, gate scores, and window results — this table is a report section by itself.

### 4.9 C9 — Label mixer

Training label for the V-head = standardized g score + Σ (weights × accepted transition-heuristic scores), with heuristic weights bounded and decaying unless refreshed by C8's online results. With no heuristics accepted, the label is g alone (the pure P1 arm).

### 4.10 C10 — Event and stall triggers

Maintain online the canonical-state set (the harness's canonicalizer applied incrementally) and compute new-states-per-1k in a rolling window — the online form of `novelty_late_per_1k`. Fire C6/C7 when the rolling rate drops below a threshold relative to the run's own peak, on level change, and at the fixed cadence. Log every trigger with its cause.

---

## 5. Integration with the existing codebase (where each piece lands)

No code here — this is the map of touch points so the change surface is known and reviewable in advance.

| Touch point | Change | Why here |
|---|---|---|
| `eval_common.py` | New env flags: `EVAL_LLM_HEAD` (adds the V-head and its loss), `EVAL_LLM_SCORER` (use g as a label source) + `EVAL_LLM_SCORER_PATH` (which g artifact), `EVAL_LLM_BETA` (float, 0 = baseline sampler), `EVAL_LLM_HEURISTICS` (periodic writer on/off), `EVAL_LLM_CADENCE` (K actions). All recorded by `write_run_config` | The contract already governs every behavioral switch; ablations must be reproducible from `run_config.json` alone |
| `custom_agents/action.py` — `ActionModel` | Second output head (5 + 4096) on the shared backbone | Reuses the backbone; V-head resets with the model, as designed |
| `custom_agents/action.py` — `choose_action`, transition block | After `frame_changed` is computed: build the C1 record, call g and accepted heuristics, attach `progress` to the experience entry; feed C10 counters | This is the one place where `(prev_frame_raw, prev_action_idx, current_frame_raw)` are all in hand |
| `custom_agents/action.py` — `_train_action_model` | Add the V-head Huber term with its own coefficient; leave `main_loss` and the confidence-bonus terms byte-identical | Freeze rule: existing hyperparameters unchanged |
| `custom_agents/action.py` — `_sample_from_combined_output` | Multiply by `exp(β·V)` with cap, before normalization; apply accepted action-prior heuristics the same way | Single choke point for action selection |
| `custom_agents/action.py` — level-change block | On score change: flush digest, fire C7 trigger (non-blocking), record which artifacts persist per reset arm | Level events are already handled here (flush, reset) |
| New module `llm_track/serializer` | C1 | Shared by agent, offline labeler, validator, digest |
| New module `llm_track/judge` | C2 offline labeling CLI (pair sampler, prompts, BT fit) | Offline; reads corpus shards directly |
| New module `llm_track/scorer` | C3 train/eval CLI + runtime loader | Produces the frozen `g` artifact under `results/` |
| New module `llm_track/heuristics` | C6–C9: digest, writer client, sandbox, registry | Separate process boundary for the LLM |
| `compute_metrics.py` / `analyze_curves.py` | New per-run scalars: mean g of taken actions, V-head/g correlation, count of accepted heuristics, trigger counts; per-1k series of mean g | So the existing summary/curve tooling reports the new arm without new scripts |
| `Makefile` | Targets: `label` (C2), `scorer` (C3), `llm-probe` (Section 7 probes) | Matches the existing `make sweep / random / curves` style |
| `results/` layout | `results/labels/<corpus_version>/`, `results/scorers/<id>/`, per-run `llm/` folder with digests, candidates, gate results | Consistent with "all output under results/" |

**Throughput guardrails.** Per-step additions are the record build (numpy on a 64×64 diff), one MLP call, and at most a handful of tiny Python heuristic calls. Budget: keep `model_ms` within +15% of baseline; Probe B measures it. The LLM process gets a fixed VRAM reservation (e.g., 12 GB for a 4-bit 9B with KV cache) leaving ≥15 GB for the agent and engine.

---

## 6. Evaluation design

### 6.1 Arms

| Arm | Head | g labels | β | Heuristics | What it isolates |
|---|---|---|---|---|---|
| A0 baseline | off | — | 0 | off | Reproduces semester-1 Goose exactly |
| A1 head-only, β=0 | on | on | 0 | off | Does adding the head/loss alone perturb the change head? (must ≈ A0) |
| A2 distilled prior | on | on | β_max | off | Value of the LLM prior (P1) |
| A3 heuristics only | on | off | β_max | on | Value of periodic LLM adaptation without the offline judge (labels come only from accepted heuristics) |
| A4 full | on | on | β_max | on | Composition |
| A5 pooled-g vs per-game-g | on | on (two artifacts) | β_max | off | Transfer: does a judge trained on other games help this one? |
| A6 scrambled-label control | on | on (g trained on shuffled Judge labels) | β_max | off | Rules out "any extra head/bias helps" — the effect must come from the labels |

Crossed with the reset factor (`EVAL_RESET_ON_LEVEL` on/off) on the focus games; β at three values (small/medium/large) in A2 on ft09 only.

### 6.2 Games, seeds, budgets

- **Stage 1 (development):** ft09 (learnable), ls20 (null contrast), tu93 (rare level-2), dc22 and g50t (mid-novelty) — 5 seeds, 100k actions. This is the set where semester-1 medians exist for `novelty_late`.
- **Stage 2 (final matrix):** all 25 public games, 5 seeds, 100k actions for A0/A2/A4; 1M-action long runs on ft09 + tu93 for A0 vs A4 only (censoring discipline: report k/n seeds reached level k).
- Random floors from `make random` for uplift ratios, as the harness requires.

### 6.3 Metrics

*Agent-level (existing):* levels reached (k/n), actions-to-level medians with censoring, `unique_states_per_action`, `meaningful_change_rate`, redundancy, `novelty_late_per_1k`, AULC/RHAE from `analyze_curves.py`, uplift vs random; `model_ms` for throughput. **Never report an exploration metric alone** (the harness rule).

*LLM-track-specific (new):*
- Judge quality: agreement 4B/9B/27B pairwise; both-orderings consistency; agreement with hindsight anchors.
- Scorer quality: held-out pairwise accuracy vs Judge and vs gold; ticker-only vs non-ticker separation; per-game vs pooled.
- Head quality: V-head vs g correlation over a run (per 1k); calibration.
- Heuristic pipeline: candidates generated / passed each gate / retained; window-level uplift of retained heuristics; example accepted heuristics with rationale.
- Behavioral: mean g of taken actions per 1k (does the agent actually take "progress-like" actions more?), and its relationship to `novelty_late`.

### 6.4 Hypotheses stated in advance

H1: A2 raises `unique_states_per_action` and lowers redundancy vs A0 on ft09/dc22/g50t with no throughput regression (>90% of baseline act/s). H2: A2 does *not* change ls20 (null contrast holds). H3: A4 ≥ A2 on `novelty_late` in the final 20% of long runs (periodic adaptation delays stalls). H4: A6 ≈ A0 (the effect is the labels, not the head). H5 (the ceiling question, stated as a test not a promise): the fraction of seeds reaching level 2 on tu93 and level 3 on ft09 is higher in A4 than A0 at 1M actions. A negative H5 with positive H1–H4 is a complete, publishable story: "progress priors improve exploration efficiency but do not by themselves break the composition ceiling."

---

## 7. Feasibility probes (weeks 1–2; go/no-go gates)

| Probe | What | Pass condition | If it fails |
|---|---|---|---|
| **A — Judge validity** | Serialize 500 ft09 + 500 ls20 + 200 tu93 transitions (stratified, deduped); label ~1,500 pairs with 4B, 9B, 27B; include hindsight anchors | 9B beats chance on anchors by a wide margin; both-orderings consistency high; 4B close to 9B | Switch the Judge to image input (Qwen3.5 native vision) at low frequency; if that also fails, pivot the track to P3 (rules + verifier) with the 27B offline |
| **B — Co-residency** | Run Goose at full speed with a 4B/9B model resident and scoring sampled transitions asynchronously | act/s ≥ 90% of baseline; VRAM headroom ≥ 4 GB | Move periodic LLM calls to *between* runs only (still P2, lower cadence) |
| **C — Heuristic writing** | Digest + schema to the 9B (thinking on); 5 candidates | ≥ 1 of 5 passes static + dynamic gates and correlates positively with anchors | Use the 27B for writing between runs; reduce to single-shot generation |
| **D — Distillation** | Train g on Probe-A labels; held-out pairwise accuracy | Clearly above a changed-cell-count baseline scorer | Add the conv branch; if still flat, the serializer is the bottleneck — invest there (or pull the object layer forward) |

---

## 8. Work breakdown and schedule (LLM track only)

| Week | Deliverable | Notes |
|---|---|---|
| Sep 1–7 | C1 serializer + schema doc; corpus scan (signature dedupe stats per game); LLM serving set up (vLLM or HF, 4-bit 9B, fixed memory reservation) | Corpus stats tell us the labeling budget |
| Sep 8–14 | Probes A–D; decision memo (judge size, cadence, go/no-go) | Advisor review point; approval for the head/loss/sampler changes |
| Sep 15–28 | C2 labeling run (20–50k pairs) + gold subset; C3 g (per-game + pooled) with validation report | Runs overnight; first report table (judge/scorer quality) |
| Sep 29–Oct 10 | C4/C5/C9 wired behind flags; A0/A1/A2/A6 on Stage-1 games × 5 seeds × both reset arms | First ablation table; H1/H2/H4 answered |
| Oct 11–24 | C6–C8, C10 (digest, writer, sandbox, triggers); A3/A4 on Stage-1 | Mid-semester checkpoint material |
| Oct 25–Nov 7 | β sweep; pooled-vs-per-game (A5); long runs on ft09/tu93 | H3/H5 evidence |
| Nov 8–21 | Stretch: P6 (GRPO-tune the judge on outcome-derived rewards; ColabPro if local is saturated) or P3 (verified rules) | Only if Stage-1 results justify it |
| Nov 22–30 | Stage-2 final matrix (25 games × 5 seeds, A0/A2/A4); integrate with teammates' layers | Freeze |
| Dec 1–15 | Analysis, chapters, defense | Reuse the semester-1 report pipeline |

Cut order if time runs short: P6/P3 → online bandit (keep offline gates only) → action-prior heuristics (keep transition heuristics) → β sweep (keep one β). The floor deliverable is A0/A2/A6 with the judge/scorer quality study.

---

## 9. Pros and cons of the design as a whole

**Pros**
- Targets the measured failure (wrong learning currency) with the best-evidenced mechanism (LLM preference → distilled reward).
- Zero LLM cost on the per-action path; throughput protected by construction and verified by a probe.
- Every stage has an off switch and a zero-setting that reproduces the baseline; the scrambled-label control separates "labels help" from "extra head helps."
- Self-sufficient serializer; upgrades cleanly when the object layer lands.
- Rich, honest report material even on a null result: labeling study, distillation accuracy, heuristic gate statistics, multi-seed ablations.
- Small models are used only in hallucination-safe roles (choices, code that passes tests, never claims).
- Natural path to a "self-trained" LLM (P6) without redesign.

**Cons / risks**
- **Serializer ceiling.** Hand-built features may miss what matters on some games; the Judge and the writer see only what the serializer shows. Mitigation: Probe D; object layer upgrade path.
- **Judge priors may be wrong for ARC.** Human-like intuitions ("collect the thing") could be systematically misleading on some mechanics. Mitigation: hindsight anchors in the label set; per-game g; A6 control.
- **Reward hacking through the back door.** A progress prior can create a new decorative attractor (e.g., always move the movable block). Mitigation: the change head still gates; capped multiplier; redundancy and coverage reported alongside; the online bandit retires heuristics that raise "activity" without raising coverage.
- **Sparse signal for later levels.** The corpus is dominated by level-1/2 transitions; g may not know what level-3 progress looks like. Mitigation: hindsight anchors from every completion in the corpus; periodic refresh of labels from new runs; the heuristic writer adapts per level.
- **Engineering surface.** Ten components is a lot for one person in one semester. Mitigation: the cut order above; C6–C10 are a single module with a process boundary; the floor deliverable needs only C1–C5.
- **Hyperparameter-freeze rule.** New loss coefficient, β, and cadence are new knobs. Mitigation: log everything in `run_config.json`; advisor approval at the Sep 8–14 review; small, pre-declared sweeps rather than tuning.

---

## 10. Alternatives considered at the design level (and why not)

- **Score with the LLM online instead of distilling.** Even a 0.5B model at ~250 req/s would be marginal and low quality; a 4B at ~25 req/s covers 20% of transitions — enough for *refreshing* labels (ONI-style, kept as an option) but not for replacing g.
- **Feed frames as images to a VLM.** Kept as the Probe-A fallback only; 64×64 abstract grids are far from VLM training data, and the frontier-LRM results on ARC-AGI-3 (<0.4%) say raw-frame reasoning is the weak axis.
- **Have the LLM write a whole policy.** That is the executable-world-model paper's regime with GPT-5.5; small models writing full policies would fail the sandbox gates constantly. Scoring functions are the largest code unit a 9B reliably produces.
- **Train the head on hindsight labels alone (no LLM).** A legitimate non-LLM control worth one extra arm if time allows (it isolates "LLM priors" from "hindsight credit assignment"), but too sparse to be the main signal — most runs have zero or one completion.

---

## 11. Deliverables checklist (what exists at the end)

1. Serializer + schema doc, with dedupe/stratification statistics per game.
2. Labels dataset (9B primary, 27B gold, 4B ablation) with prompts and reasons.
3. Scorer artifacts (per-game and pooled) with validation report.
4. Modified agent behind flags, byte-identical to baseline at β = 0 / flags off.
5. Heuristic pipeline with registry of every candidate and its gate results.
6. Ablation matrix (Stage 1 and Stage 2), curves, and censored level-reach tables.
7. Report chapters: serializer; labeling study; distillation; ablations; heuristic case studies; limitations.
