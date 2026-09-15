# Plan A — "Stall-Triggered LLM Advisor"
### A local 9B model ranks *what to try next* for StochasticGoose, only when Goose is stuck, and only keeps its advice if the numbers say it helped

Track: LLM integration. Owner: Matt. Repo: `sauerpatchkid/ARC3-solution`. Model class: Qwen3.6/3.8-9B-instruct
(or whatever 9B-class instruct model you already have serving). Status: proposed 15 Sep 2026. Idea-lock: Nov 2.

---

## 1. One-paragraph summary

Goose keeps playing at ~136 act/s exactly as before. A stall detector (no new canonical state for W actions)
fires rarely — tens of times per 200k-action run. When it fires, the current frame and the level's
interaction history are serialized as a **short object-level text summary** (≈1–2k tokens: connected
components with color/size/position, what clicking each of them did so far, what ACTION1–5 did so far,
and — on later levels — what the previous level's win looked like). A local 9B instruct model returns a
small JSON: a one-line hypothesis, a ranked list of objects to click, weights for ACTION1–5, and things to
avoid. That JSON becomes a **bias vector over Goose's 4101 logits** for the next K actions. A gate compares
the novelty rate during the advised window with the window before it; if advice did not help, it is dropped
and the trigger backs off. Two control arms — **random advice** (placebo) and **hand-written heuristic
advice** (Reki-style "small, rare-colored, untried objects first") — run on the same schedule and gate, so
the report can say exactly how much the *language model* contributed beyond "poke the agent when it stalls".

## 2. Why this shape, and why not the shape you were trying

What you were trying: have the LLM describe what each input does, derive the rules, write code. The
literature this month says that is the hardest version of the problem:

| Finding | Source | Consequence |
|---|---|---|
| "Every variant scores higher as model capability and reasoning effort increase. These gains often exceed variant differences." Textual reasoning beats executable-model variants at some settings even for GPT-5.5. | Rodionov component study (arXiv 2607.15439) | At 9B, harness cleverness will not substitute for capability; ask the model the smallest question that helps |
| Auto-repaired models match transitions *better* yet play *worse*: transition match "shows whether a simulator reproduces observed dynamics, not whether it has identified the objective" | Tycho (arXiv 2607.28287) | "Describe what each input does" is the part that *does* work; turning it into a winning strategy is the part that does not — even for Opus |
| Best open-weights code agent (Qwen3.6-27B in a REPL) averages ~1.6% on public games; "hand-crafted tools actually hurt" | Tufa "Duck" (Milestone 1) | A 9B writing a game model will be worse than that; do not build tools for it, build a *question* for it |
| 2nd/3rd place used "look at the board, return one JSON action"; the winning forge profile turned all extra machinery off | Reki, forge (Milestone 1) | JSON ranking/selection is what small models do reliably |
| Small LLMs have a token bias against ACTION6/coordinates; "probe first, then let the LLM interpret the probe" works, blind LLM choice does not | AERA (arXiv 2605.25931, with evaluator caveats) | Never let the LLM emit coordinates; let it pick *objects*; feed it probe results, not raw frames |
| "Programmatic agent drives; LLM only on stall; the LLM can only add to the floor" | BDR-Pro (Kaggle 0.26–0.27 floor) | This is the only pattern compatible with 100k+ action runs and with not losing what Goose already does |
| Frontier models "notice local effects but fail to assemble them into a coherent world model" | ARC Prize failure analysis (May 2026) | Do not ask the 9B for the world model; ask it for the next *experiment* |

So the design question changes from "can a 9B understand this game?" to "can a 9B, shown a compact record
of what has been tried, pick a better *next experiment* than random and better than a hand rule?" That is
a question you can answer offline in a week on corpora you already have (§7.1), before spending any GPU
on online runs.

## 3. Hypotheses (pre-registered)

**H1 (offline).** On stall frames drawn from existing corpora, the 9B's top-3 ranked objects contain the
cell that eventually produced the next novel state / level-up more often than a random top-3 and at least
as often as the heuristic top-3.

**H2 (online).** B3 (LLM advice) reaches more levels / fewer actions-to-level than B0 (no advice) on ≥2 of 6
dev games, with no game worse than seed noise.

**H3 (attribution).** B3 > B1 (random advice). If B3 ≈ B2 (heuristic advice), the honest conclusion is that
the LLM's contribution is a heuristic it could have been replaced by — still a completed LLM arm, still a
result.

**Expected non-result:** on ft09 the stall detector may never fire (novelty never flattens), so B3 = B0
there by construction. On keyboard games the click ranking is irrelevant and only `action_weights` matter.

## 4. Architecture

```
                 ┌──────────────────── fast loop (unchanged Goose, ~136 act/s) ────────────────────┐
frame ─► canon key ─► novelty label ─► CNN ─► sigmoid ─► tried-mask ─► [advice bias] ─► sample ─► act
                                           ▲                                   │
                                           │                                   │ every step: update object stats
                 ┌────────── slow loop ─────┴────────────────────────────────────┴──────────────┐
                 │ stall detector ─► serializer ─► 9B (vLLM, JSON) ─► validator ─► bias builder │
                 │                     ▲                                             │          │
                 │        level-win summariser (on level-up)          uplift gate ◄──┘          │
                 └────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Trigger

Per level, maintain `last_novel_action` (action index of the most recent novel canonical state).
- **Stall:** `action_counter − last_novel_action ≥ W` (default `W = 1500`) and no advice currently active.
- **Periodic fallback:** every `P = 10000` actions in a level regardless (covers ft09-type games where
  novelty never flattens but progress does).
- **Level-up hook:** on score change, write a *level-win summary* (§4.3) — no LLM call needed at that
  moment; it is context for the next call.
- **Back-off:** after a rejected advice window, `W ← min(2W, 12000)`; after an accepted one, `W ← 1500`.

Budget: at W=1500 with rejection back-off, a 200k-action run makes ≤ ~40 calls; at ~5 s each that is ≤ 4 min
of wall time per run (≈ 2–3% at 136 act/s). Calls are synchronous for simplicity (Goose pauses); an async
variant is a later optimization.

### 4.2 Serializer (the part that decides everything)

Object extraction (numpy, no ML): connected components of the raw frame, 4-connectivity, per color,
excluding indicator-masked cells; background = most frequent color. Keep the top `N_OBJ = 30` objects by a
salience score (small size, rare color, touching ≥2 colors) plus any object with click history. For each
object: `id`, `color`, `bbox (x0,y0,x1,y1)`, `size`, `centroid`, `n_same_shape` (identical-mask siblings).

Object click history (per level, updated each step from the tried map): matched to objects by
`(color, bbox)` at serialization time. For each object: `clicks_tried`, `clicks_changed_frame`,
`clicks_novel_state`, `last_effect` ("no-op" | "changed ≤3 cells" | "changed N cells; colors X→Y").

Discrete-action history (per level, last 200 executions of each): `n`, `changed%`, `novel%`,
`dominant_effect` = the color whose cells changed most and its mean centroid shift (a two-line "mover"
detector; BDR-Pro's avatar logic without the votes).

Level context: level index, actions in level, unique states seen, actions since last novel state, actions
since last level-up, available actions.

Previous level(s): the level-win summary — objects present in the frame before the winning action, the
winning action (type; for clicks, the object clicked), colors whose cell counts changed at the win.

Everything is plain text tables, ≤ 2k tokens. No images, no ASCII grid (a 9B does not read 64×64 grids
reliably, and Duck's gains came from the *27B* reading images).

### 4.3 Prompt and output schema

System prompt (fixed, versioned in the repo):
> You are helping an exploration agent that is stuck in an unknown 64×64 grid game. You see a summary of
> the objects on screen and what the agent has already tried. Your job is to pick the most promising things
> to try next. Prefer untried objects and actions; prefer small, distinctive, button-like objects; prefer
> actions whose effects are unknown over ones proven useless; use the previous level's win, if given, to
> guess what "progress" looks like. Answer with JSON only.

Output schema (validated; anything else is discarded):
```json
{
  "hypothesis": "one or two sentences",
  "click_targets": [{"id": 12, "weight": 5}, {"id": 3, "weight": 2}],
  "action_weights": {"ACTION1": 1, "ACTION2": 1, "ACTION3": 0, "ACTION4": 3, "ACTION5": 5},
  "avoid": {"objects": [7], "actions": []}
}
```
Weights are integers 0–5. Use vLLM guided-JSON (or `outlines`) so the schema is enforced at decode time;
keep a regex-repair fallback. Temperature 0.3, `max_tokens` 400.

### 4.4 Bias builder and injection

`bias` is a float vector of length 4101 initialised to 1.0.
- For each `click_targets` entry with weight `w`: cells of that object get `× (1 + β·w)`, `β = 1.0`
  (weight 5 → ×6). Cells of `avoid.objects` get `× 0.2`.
- `action_weights[ACTIONk] = w` → index `k−1` gets `× (1 + β·w)`; 0 → `× 0.2`.
- Applied in `_sample_from_combined_output` after the tried-mask and before the 1/4096 coordinate scaling
  and normalisation; then clipped so no single cell exceeds `p_max = 0.25` of the total mass (prevents the
  advice from turning Goose into a deterministic clicker).
- Active for `K = 1500` actions or until level-up; then removed. Bias is multiplicative, so masked/unavailable
  actions stay impossible and the CNN's own preferences still matter.

### 4.5 Uplift gate (Eureka-style selection at n = 1)

Measure `nov_pre` = novel states per 1k actions in the `K` actions before the trigger (by definition low)
and `nov_adv` = same during the advised window. Early check at `K/3`: if `nov_adv ≤ nov_pre + margin`
(margin = 1 state/1k), drop the advice, mark **rejected**, back off. Otherwise keep to `K`, mark **accepted**.
Optionally request two candidate JSONs (temperature 0.7) and A/B them for `K/3` each, keeping the better — this
is closer to Eureka's population selection and costs one extra call.

The gate does not make the LLM smarter; it makes the arm **safe** (it cannot be much worse than B0) and it
produces the acceptance-rate statistic that tells you whether the advice is informative.

### 4.6 Serving

vLLM serving the 9B in bf16 (~18–20 GB) or FP8 on the 5090, co-resident with Goose (Goose's CNN is small;
its 200k buffer lives in host RAM). Backends behind one interface: `mock` (canned JSON, for tests and CI),
`openai` (vLLM server), `hf` (transformers, fallback). Same design BDR-Pro landed on after their OOM.
For the final "capability curve" run (§10), point the same interface at a 27B on Colab.

### 4.7 Config flags

| Flag | Default | Meaning |
|---|---|---|
| `EVAL_ADVISOR` | `off` | `off` \| `random` \| `heuristic` \| `llm` |
| `EVAL_ADVISOR_W` | 1500 | stall window |
| `EVAL_ADVISOR_P` | 10000 | periodic fallback |
| `EVAL_ADVISOR_K` | 1500 | advice window |
| `EVAL_ADVISOR_BETA` | 1.0 | bias strength |
| `EVAL_ADVISOR_GATE` | 1 | uplift gate on/off |
| `EVAL_ADVISOR_MODEL` | — | model id / endpoint |
| `EVAL_ADVISOR_LEVELCTX` | 1 | include previous-level win summary |

Arms:

| Arm | `EVAL_ADVISOR` | What it tests |
|---|---|---|
| B0 | off | base (Plan B's A3, or A0 if Plan B is not adopted) |
| B1 | random | placebo: random object ids and weights on the same schedule/gate |
| B2 | heuristic | Reki-style rule: rank untried small rare-colored objects; action weights ∝ novel% + untried bonus |
| B3 | llm | the 9B |
| B4 | llm, `LEVELCTX=0` | ablates the previous-level context (differs from B3 only on levels ≥2) |

## 5. Implementation plan (file by file)

1. **`custom_agents/objects.py`** (new, ~120 lines): connected components (scipy.ndimage.label per color, or
   a pure-numpy BFS), salience score, top-N selection, `(color,bbox)` matching, object-history table.
2. **`custom_agents/serializer.py`** (new, ~120 lines): builds the text summary from objects + tried map +
   action history + level summaries. Deterministic; token-capped (truncate lowest-salience objects first).
3. **`custom_agents/advisor.py`** (new, ~200 lines): `Advisor` with `maybe_trigger()`, `request()`,
   `validate()`, `build_bias()`, `gate()`; backends `mock` / `openai` / `hf`; the `random` and `heuristic`
   advisors implement the same `request()` interface so the arms share every other line.
4. **`custom_agents/level_summary.py`** (new, ~60 lines): on level-up, diff the last decision frame against
   the frame at the previous level-up and record the winning action + object + color-count deltas.
5. **`custom_agents/action.py`**: instantiate the advisor from flags; per step feed
   `(key, action_idx, changed, novel, frame_raw)` to the object-history updater; call `maybe_trigger()`
   before sampling; pass `advisor.bias` into `_sample_from_combined_output`; call `advisor.observe(novel)`
   after each step for the gate; on level-up call `level_summary.record(...)`. TensorBoard scalars:
   `Advisor/calls`, `Advisor/accepted`, `Advisor/parse_fail`, `Advisor/nov_pre`, `Advisor/nov_adv`,
   `Advisor/llm_seconds`.
6. **`tools/advice_probe.py`** (new): the offline probe (§7.1) over existing corpora.
7. **`prompts/advisor_v1.txt`**: the system prompt, versioned; the prompt hash goes into `run_config.json`.
8. **`tests/test_objects.py`, `test_serializer.py`, `test_advisor.py`** with the `mock` backend.
9. Sweep tooling: `ADVISOR=` override in `sweep.sh`; arm tag `adv_<x>` in the manifest.
10. Corpus: unchanged. Advice events go to a sidecar `advice.jsonl` per run (trigger action, prompt hash,
    raw JSON, accepted/rejected, nov_pre/nov_adv) — this is what the report's attribution analysis reads.

Estimated code: ~600 lines + tests. New dependencies: `vllm` (or `llama-cpp-python`), `scipy` (optional).

## 6. Metrics and success criteria

Primary: same as Plan B — levels reached, k/n seeds, actions-to-level with censoring, AULC, paired vs B0.

Advisor-specific:
- calls per run, parse-failure rate (target < 5% with guided JSON), acceptance rate at the gate;
- `nov_adv − nov_pre` distribution (accepted vs rejected);
- **advice hit rate**: fraction of level-ups that occur inside an advised window, versus the fraction of
  actions that are advised (if 5% of actions are advised and 30% of level-ups happen there, the advice is
  doing something);
- LLM seconds per 1k actions (cost line for the report).

Attribution rule: report B3 − B0, B3 − B1, B3 − B2 as three separate paired deltas. Only B3 − B2 is "the
language model's contribution". Pre-registered success bar: B3 > B1 on the primary metrics on ≥2 dev games
with no regression, **and** H1 holds offline. B3 ≈ B2 is an acceptable, publishable outcome with the framing
"a 9B reproduces a hand heuristic; capability, not harness, is the limit" — which is exactly what
Rodionov found at the frontier and would be a nice echo in your report.

## 7. Testing plan

### 7.1 Offline advice-quality probe (week 1, no agent changes, uses corpora you already have)

This replaces the earlier "probe A/B/C" plan with one measurable question.

1. From existing corpora (`results/runs/*/*/transitions/`), for each level segment, locate **stall points**
   (≥1500 actions without a new canonical state, computed offline with `metrics_common`) and **pre-win
   points** (the frame 1 decision before each level-up).
2. At each point, build the serializer output (objects + history up to that action).
3. Ask each ranker — random, heuristic, 9B — for `click_targets`.
4. Score: for pre-win points on click games, precision@3 = "is the winning click's object in the top 3?"
   For stall points, "did the next novel state in the corpus come from a top-3 object?" (only defined when
   Goose happened to try one; report coverage).
5. Also record: parse-fail rate, latency, prompt length, and whether the `hypothesis` line is even about the
   right objects (spot-check 20 by hand — a table of 20 hypotheses is a great appendix).

Go/no-go: proceed to online arms if the 9B beats random on precision@3 by a margin that survives a sign
test over ≥15 points and is not clearly below the heuristic. If it is clearly below the heuristic, ship the
heuristic arm as the "advisor" and use the LLM for the extension in §10.1 instead.

### 7.2 Unit tests (with `mock` backend)
- Object extraction on synthetic frames (two same-color objects separated by background; masked ticker
  excluded; background detection).
- Serializer determinism and token cap; `(color,bbox)` matching after an object moves.
- Validator rejects unknown ids, clamps weights, survives malformed JSON.
- Bias math: multipliers, `p_max` clipping, unavailable actions remain impossible.
- Gate: accept/reject on scripted novelty streams; back-off doubling.
- End-to-end 2k-action smoke with `mock` returning fixed advice: throughput within 10% of B0 excluding LLM
  wall time.

### 7.3 Online dev tier (weeks 2–3)
- 6 dev games × 5 seeds × 200k, arms B0–B3 (B0 shared with Plan B's A3). ≈ 4 × 12 h plus LLM time.
- Run B1 and B2 *before* B3 — if B2 already helps, the LLM's bar is set correctly.

### 7.4 Confirm tier (week 4)
- Best of B2/B3 vs B0, all runnable public games × 3 seeds × 100k.

### 7.5 Capability-curve run (optional, late Oct, Colab credits)
- B3 with a 27B instead of the 9B on the 6 dev games × 3 seeds × 200k. One figure: uplift vs model size
  (heuristic / 9B / 27B). This is the most report-worthy single experiment in the LLM track.

## 8. Timeline

| Week | Dates | Work | Output |
|---|---|---|---|
| 1 | Sep 30–Oct 6 (after Plan B week 2) | `objects.py`, `serializer.py`, offline probe with random/heuristic/9B | Go/no-go table; 20 hypotheses appendix |
| 2 | Oct 7–13 | `advisor.py` (mock/heuristic/random), bias injection, gate, tests, smoke | B0–B2 runnable |
| 3 | Oct 14–20 | vLLM backend, B3; dev sweep B0–B3 | First paired results |
| 4 | Oct 21–27 | Confirm sweep; B4 ablation if levels ≥2 are reached | Adopt/reject |
| 5 | Oct 28–Nov 2 | Freeze; optional 27B run on Colab; methods section | **Idea locked** |
| Nov | | Combine with teammates; final figures | Report draft |

If Plan B slips, Plan A's week 1 (offline probe) can start any time — it needs only corpora and the model.

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 9B ranks no better than random | Medium | Offline probe answers this in week 1 for ~zero GPU; heuristic arm is the fallback deliverable |
| 9B ≈ heuristic | Medium-high | Pre-registered as an acceptable result; 27B capability-curve run turns it into a finding |
| JSON/format failures | Medium | Guided decoding; validator; failures are logged and skipped (never crash the run) |
| Serialization loses what matters (relative positions, shapes) | Medium | Add `bbox`/centroid/shape-siblings; spot-check hypotheses; keep the ASCII grid *out* unless the 27B run |
| Stall detector never fires (ft09) or fires constantly (tu93) | High on specific games | Periodic fallback + back-off; report calls per game |
| Advice makes Goose deterministic and it self-traps | Low | `p_max` clip, multiplicative bias, K-window expiry, gate |
| vLLM + Goose VRAM contention on the 5090 | Low | 9B bf16 ≈ 20 GB + Goose < 2 GB; FP8 if needed; measured in smoke |
| Wall-time overhead hides the result | Low | ≤ 4 min per 200k run; reported separately |
| Public-set overfitting via prompt tuning | Medium | Freeze the prompt after week 1 (hash in `run_config`); holdout games never used for prompt iteration |

## 10. Extensions (only after B3 is measured)

1. **Eureka-lite: LLM writes a scoring function.** Instead of a ranking, the 9B returns
   `def score(obj) -> float` over the serializer's object fields, executed in a sandbox (no imports,
   timeouts); two candidates are A/B-tested for `K/3` each; the winner biases the next `K`. Same gate, same
   arms. Only worth it if B3 > B2 — otherwise the code will be a heuristic with extra failure modes.
2. **Distill accepted advice into a reward head.** Every accepted advised window is a labelled dataset
   (states, actions, novelty); train a tiny extra head on the CNN to predict "advisor would boost this".
   This is your original P1 (LLM-labelled reward → distilled head) but with labels that have *already been
   selected for uplift*.
3. **Level manual.** Carry the level-win summaries as a growing "manual" across levels (Agno-style); on
   level ≥2 the advisor sees every prior win. Cheap; already half-built by §4.3.
4. **Capability curve.** Same harness, 9B → 27B (Colab), ideally one frontier API model on 2 games as an
   upper reference (not Kaggle-eligible, but Kaggle compliance is low priority).

## 11. Pros and cons

**Pros**
- Satisfies the 295B LLM requirement with an arm that cannot make Goose materially worse (gate + rare
  invocation + multiplicative bias).
- Turns the LLM question into a measurable one *before* GPU runs (offline probe on existing corpora).
- Built-in attribution: placebo and heuristic controls make the LLM's contribution a number, not a story.
- Consistent with the pattern the best offline no-LLM agent chose when it added an LLM (drive/rescue), and
  with the small-model results from Milestone 1 (JSON selection, not code).
- Composes with Plan B: uses its canonical key, novelty signal, tried map and sampler hook; B0 is A3.
- The capability-curve extension gives the report a figure nobody else in the class will have.

**Cons**
- The most likely outcome is "helps a little on some click games, indistinguishable from the heuristic" —
  a modest headline.
- Serializer quality caps everything; it is hand-designed and therefore the one place public-set overfitting
  can creep in (mitigated by freezing it early and by the holdout).
- Extra moving parts: vLLM server, JSON handling, sidecar logs; ~600 lines vs Plan B's ~250.
- Rarely fires on games where Goose keeps discovering states (ft09), so it cannot address the ft09 L2 wall
  by design.
- It is an exploration nudge, not goal understanding; on rules-not-search games (tu93, g50t) the ceiling may
  not move even with good advice.
- The strongest version (frontier model as advisor) is out of scope for Kaggle and expensive; the 9B version
  is what you can afford, and the literature predicts capability is the limit.

## 12. How this appears in the report

Section: "Intervention 2 — an LLM as a stall-time experiment designer." Figure: offline precision@3
(random / heuristic / 9B). Figure: paired deltas B1/B2/B3 vs B0 on the dev games. Table: acceptance rate,
calls, LLM seconds. Appendix: 20 advisor hypotheses with the frames they were about. Optional figure: uplift
vs model size.
