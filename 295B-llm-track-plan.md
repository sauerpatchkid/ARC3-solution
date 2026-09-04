# LLM Track — Research & Plan (v1)

**Owner:** Matt · **Platform:** StochasticGoose + local eval harness · **Hardware:** RTX 5090 (32 GB), ColabPro credits later
**Constraints (locked):** pretrained open-weights models are fine; the track must be self-sufficient (own minimal state serialization); runs are 100k–1M+ actions at ~120 act/s, so the LLM cannot sit in the per-action loop.

---

## 0. The constraint, quantified — what "can't run every move" actually means

Before choosing *how* to use an LLM, pin down what the 5090 can do at each call frequency. Measured vLLM numbers on an RTX 5090 (Runpod, 128 tokens in / 128 out, batched):

| Model size | Throughput | Requests/s | Notes |
|---|---|---|---|
| ~0.5B (Qwen2-0.5B) | ~65,000 tok/s | ~250 req/s | 1024 concurrent prompts, 18 GB used |
| ~3.8B (Phi-3-mini) | ~6,400 tok/s | ~25 req/s | same setup |
| 27–35B (Qwen3.6-27B Q8, Qwen3.5-35B-A3B Int4) | 45–200 tok/s single-stream | ~1 req/s | fills the card; no room for the agent alongside |

Our agent produces ~120 transitions/s. So the honest picture is:

- **Every action, synchronous:** infeasible for anything that *generates* text. Marginal for a ≤1B model answering with one token (prefill-only, async) — and a 0.5B model's judgment on abstract grid games is not worth the plumbing. **Don't design for this.**
- **Every action, but via a distilled head:** entirely feasible — the LLM's *knowledge* runs every step, the LLM itself doesn't. A small MLP/CNN head trained on LLM labels costs microseconds. This is how Motif/ONI make LLM guidance work at RL speeds.
- **Sampled, asynchronous (1–10% of transitions):** a 2–4B model can score 10–25 transitions/s in the background while the agent runs — enough to keep a distilled head *fresh* during a run (ONI's online variant).
- **Periodic (every 1k–10k actions, or on events):** at 10k actions ≈ every 80 s of play. A 9B–12B model in thinking mode has all the time it needs, and a 27–31B model works if the agent pauses or the call is batched between runs. **This is your instinct, and the numbers support it.**
- **Offline (before / between runs):** anything that fits 32 GB — corpus labeling, fine-tuning (LoRA up to ~9B, full FT up to ~2B), training distilled models. Unlimited wall-clock, so this is where the heaviest lifting should go.

**Design principle that falls out:** *slow LLM, fast agent.* Every LLM option below is defined by (a) what runs offline, (b) what runs periodically, and (c) what tiny artifact the LLM leaves behind that runs every step for free.

---

## 1. Model landscape you can actually run (Sept 2026)

Everything below is Apache-2.0 or MIT unless noted, so it's usable in the open-source competition path too.

| Tier | Candidates | Inference VRAM | LoRA fine-tune VRAM | Best role in this project |
|---|---|---|---|---|
| **Tiny (≤1B)** | Qwen3.5-0.8B (Mar 2026, native vision), SmolLM2-135M/360M, Pythia 70–410M; *from-scratch* 7M–100M models (TRM lineage) | <2 GB | ~3 GB (0.8B) | Distillation target; per-step or high-rate scoring after fine-tuning; the "self-trained" story |
| **Small (2–4B)** | **Qwen3.5-2B / 4B** (native vision, 262K ctx), **Gemma 4 E2B / E4B**, Phi-4-mini 3.8B (MIT), SmolLM3-3B, Jamba Reasoning 3B | 2–8 GB | 5 GB (2B) / 10 GB (4B) | **Sweet spot:** async sampled scorer during runs; GRPO/RL fine-tune target; can co-reside with the agent |
| **Mid (8–12B)** | **Qwen3.5-9B**, Gemma 4 12B, Qwen3-8B | ~18 GB bf16 / ~6 GB 4-bit | 22 GB (9B, bf16 LoRA) | Periodic reflection/hypothesis generation (thinking mode); offline corpus labeler; fits *alongside* the agent in 4-bit |
| **Large (20–35B)** | Qwen3.6-27B (Q8 ~30 GB), Qwen3.5/3.6-35B-A3B MoE (Q6 ~28 GB), Gemma 4 31B (Q6 ~25 GB) / 26B-A4B, gpt-oss-20B | 22–30 GB | not feasible | Offline only: highest-quality labels and rule proposals between runs; the "teacher" for distillation |

Notes that matter for choices:
- **Qwen3.5 small models** (0.8B/2B/4B/9B, dense, Apache 2.0, released Mar 5 2026) are the strongest per-parameter option and have **native vision** — frames can be shown as images at low frequency without any serialization. Caveat from Artificial Analysis: high hallucination rates on the 4B/9B (~80% on their omniscience benchmark), so never trust an unverified claim from them. Unsloth explicitly advises against 4-bit QLoRA on Qwen3.5 (use bf16 LoRA); GRPO/GSPO RL fine-tuning is supported.
- **Gemma 4 E2B/E4B** are the best "designed for local" alternatives, also Apache 2.0, 128K context.
- Budget reality: the Goose CNN + engine + corpus logger use ~2–3 GB. A 4B judge in bf16 (~8 GB) or a 9B in 4-bit (~6 GB) co-resides comfortably; a 9B in bf16 (~18 GB) fits but leaves little headroom; 27B+ is offline-only.

---

## 2. The ways to use an LLM — option list

Each option is tagged with its frequency regime (Offline / Periodic / Async / Distilled-per-step), the evidence behind it, and how it survives without the teammate's object layer.

### P1. LLM-labeled intrinsic reward, distilled into a per-step head ("the judge")
- **How:** Serialize transitions minimally (see §3). Offline, a mid/large model labels tens of thousands of sampled transitions — *pairwise* ("which of these two looks more like progress?") rather than absolute scores, which is far more robust. Train a tiny reward head on the labels; add its output to Goose's sampling bias every step. Optionally refresh labels asynchronously during runs.
- **Regime:** Offline → Distilled-per-step (+ Async refresh).
- **Evidence:** Motif (LLM preference labels → intrinsic reward) beat every count-based exploration baseline on NetHack, a sparse, instruction-free game, and its online successor ONI (arXiv 2410.23022) removed the offline-dataset requirement. This is the closest published analogue to our exact problem.
- **Pros:** zero throughput cost at runtime; the strongest evidence base; graceful failure (bad labels ≈ noise on a sampler that already works); every artifact (label set, head, agreement stats) is a report table.
- **Cons:** label quality hinges on serialization; LLM judgments on abstract color-grid mechanics are unproven (→ week-2 probe); pairwise labeling needs ~10⁴–10⁵ comparisons (hours of offline GPU time — fine).

### P2. LLM-written scoring code, evolved periodically ("Eureka for exploration")
- **How:** Every N thousand actions (or at a stall / level event), a mid-size model reads a run digest and *writes a Python heuristic* over the serialized state — "reward transitions where a small object moves toward a differently-colored region," "penalize transitions that only change cells inside this bounding box (it's a ticker)." The heuristic runs every step at zero cost. Keep several candidates, select by measured `novelty_late` / level progress, and let the LLM refine the winners.
- **Regime:** Periodic (LLM) → per-step (its code).
- **Evidence:** Eureka (arXiv 2310.12931) showed LLM-written reward code beats expert human reward design across 29 RL tasks using exactly this evolve-and-reflect loop; a 2026 successor (Reward Design Agent, arXiv 2606.01672) improves reward-alignment 49% over Eureka on HumanoidBench. Both run the LLM *offline between training iterations* — the same regime as "every 10k moves."
- **Pros:** the most natural fit for your "run it every 1k–10k moves" instinct; artifacts are human-readable code — ideal for the report and defense; no fine-tuning needed; a 9B–12B model writes competent numpy.
- **Cons:** needs a small, stable state API for the code to call; generated code can crash or reward-hack (mitigate: sandbox, unit tests, measured-uplift selection); small models write worse code than GPT-5-class, so expect more candidates per useful one.

### P3. Periodic reflection with a mechanical verifier ("theory-making")
- **How:** At stall/level events, the LLM proposes *rules* about the game ("pressing UP moves the red 2×2 block one cell; blocked by gray") in a fixed template. Each rule is checked against the logged transition corpus — exact, free, instant. Verified rules bias exploration toward their untested predictions and give the planner (teammates' N2/N5) operators.
- **Regime:** Periodic → per-step (verified rules are cheap lookups).
- **Evidence:** WorldLLM (arXiv 2506.06725) closes exactly this loop (curiosity-driven theory-making with a verifier); Executable World Models for ARC-AGI-3 (arXiv 2605.05138) is the maximal version — GPT-5.5 + Codex solving 15/25 games — which we can't afford but *can* scope down: the verifier does the quality control, so a small model only has to beat random hypothesis search.
- **Pros:** highest ceiling — it targets *modeling* and *goal-setting*, the two capabilities the benchmark isolates and no baseline has; extremely reportable (table of induced, verified rules per game).
- **Cons:** the largest build (rule language, verifier, integration); small-model proposal quality is the open risk; scope creeps toward program synthesis unless fenced to a fixed rule template.

### P4. Periodic subgoal / plan conditioning
- **How:** Every ~1k actions the LLM picks a target from the state graph (once N2 exists) or writes a short plan; the policy is conditioned on an embedding of it.
- **Regime:** Periodic.
- **Evidence:** SGA-ACR (arXiv 2511.20993) uses Qwen3-8B every 100 steps in Crafter to beat prior LLM-guided baselines (17.6 vs 14.3); LLM-augmented observations (arXiv 2510.08779) hit +71% success on the hardest BabyAI task — but with Llama-3-70B every 5 steps, and the authors name query cost as the main limitation.
- **Pros:** proven in Crafter/BabyAI, which are structurally similar (grid, sparse, compositional).
- **Cons:** Goose has no goal-conditioned policy — this needs a new head and a graph to point at; mostly duplicates P1/P2's value with more plumbing. **Keep as a later add-on, not a primary.**

### P5. Self-trained tiny model on our own corpus
- **How:** Tokenize the transition corpus (action + serialized before/after) and train a 7M–100M transformer from scratch as a forward-dynamics model and/or a hindsight-labeled progress predictor; run it every step.
- **Regime:** Offline train → per-step.
- **Evidence:** Tiny Recursive Models (arXiv 2510.04871; ~7M params, ~45% ARC-AGI-1) and follow-ups (2511.02886, 2512.11847) show tiny models handle ARC-style grid abstraction when the representation is right; nobody else has millions of in-domain ARC-AGI-3 transitions to pretrain on.
- **Pros:** the purest "self-trained" story; hours of training on the 5090; directly tests *transfer* across games (the project's title).
- **Cons:** ~80% of the corpus is from degenerate-signal games (curation needed); not a language model — pair with P1 or P6 so the LLM requirement is unambiguous; it's really the teammates' N4 predictor in different clothes.

### P6. RL fine-tune the judge on downstream outcomes (self-improving judge)
- **How:** Start from P1's 2–4B judge. Use GRPO (Unsloth supports it for Qwen3.5) with a reward derived from *what actually happened* — did transitions the judge rated highly precede level progress / sustained novelty? The judge learns ARC-specific taste from the harness's own outcomes.
- **Regime:** Offline between runs.
- **Evidence:** Unsloth's Qwen3.5 GRPO support; the "capacity-headroom" findings for RL on 70–500M LMs (arXiv 2607.25091) say small-model PPO/GRPO works when the reward is discriminative — which ours is only after P1 exists.
- **Pros:** the strongest form of "self-trained LLM" that is still a *language* model; a clean before/after ablation (frozen judge vs tuned judge).
- **Cons:** reward for the judge is noisy and delayed; needs P1 working first; a late-semester item by construction.

### P7. LLM as run controller (stall doctor)
- **How:** The harness already has a stall detector (`novelty_late_per_1k` flattens thousands of actions before the level counter confirms it). On stall, hand the LLM a digest and let it switch exploration mode, adjust temperature, or pick a different P2 heuristic.
- **Regime:** Event-triggered.
- **Pros:** trivial to build once P2 exists; makes the "every 1k–10k moves" cadence adaptive instead of fixed.
- **Cons:** low ceiling alone; it's a feature of P2, not a separate option.

### P8. Vision-LLM on frames at level boundaries (hindsight explainer)
- **How:** Qwen3.5 / Gemma 4 take images natively. When a level completes (rare, so cheap), show the last k frames and ask what changed — turn a completion into a *hypothesis* ("the level ended when all blue tiles matched the pattern"), which feeds P1 labels and P3 rules.
- **Regime:** Event-triggered, very low frequency.
- **Pros:** needs no serialization at all — the fully self-sufficient fallback; unique to 2026 small VLMs.
- **Cons:** 64×64 abstract grids are far from VLM training distribution; the technical report's frontier-LRM results (<0.4%) warn that raw-frame reasoning is the weak axis. Use as a *supplement*, never the backbone.

### Considered and rejected
- **LLM chooses every action** (the naive agent): infeasible at our budgets and, per the technical report, the approach that scores <0.4% even with frontier models.
- **Replicating Executable World Models with a frontier API:** violates the local/small constraint; already published; our niche is the low-compute regime it leaves open.

---

## 3. Recommended design: "Judge → Head → Heuristics"

**Backbone = P1, cadence mechanism = P2, stretch = P6 (self-trained) or P3 (rules).** This combination covers the frequency spectrum without ever putting an LLM on the per-action path, and every stage is a separable ablation.

```
OFFLINE (before/between runs)                PERIODIC (every ~5–10k actions / on stall)
  corpus ──► minimal serializer ──► 9B–27B    run digest ──► 9B–12B LLM ──► heuristic code
              (§3.1)                  judge                                  + rule proposals
                                        │                                         │
                                  pairwise labels                          verify vs corpus /
                                        │                                    measured uplift
                                        ▼                                         ▼
                          tiny reward head (MLP/CNN)  ◄──────────────────  weights/rules
                                        │
PER STEP (free)                         ▼
  goose sampler ← change-pred logits + reward-head bonus + heuristic score
```

### 3.1 The minimal serialization (self-sufficient; no object layer required)
Everything the LLM sees is computed from `(frame_before, action, frame_after, score/level deltas)` with numpy in microseconds:
- action taken (type, click coords) and whether the frame changed at all;
- changed-cell count, and connected components of changed cells with their color-before/color-after, bounding box, centroid displacement;
- per-color cell-count deltas (a "key vanished" shows up as −4 cells of color 9);
- whether the changed region matches a *known ticker* (the canonicalizer already identifies indicator cells);
- score/level delta.

That is enough for a text prompt of ~100–200 tokens ("ACTION3: 4 cells of color 9 disappeared at (12,40); a 2×2 color-4 block moved +1 column; ticker region unchanged"). When the teammate's object layer lands, swap it in behind the same interface — the prompt gets richer, nothing else changes.

### 3.2 Why this order
1. P1 has the best evidence and the least coupling — it's the arm most likely to produce a clean positive or negative result by mid-semester.
2. P2 is the mechanism that gives you the periodic cadence *and* the report-friendly artifacts, and it reuses P1's serializer and state API.
3. P6 turns the frozen judge into a self-trained one late in the semester when there's outcome data to train on; P3 is the alternative stretch if rule induction looks more promising in the probes.

### 3.3 Feasibility probes (weeks 1–2; each is one script, one evening)
- **Probe A — can small models judge ARC transitions?** Sample 500 ft09 and 500 ls20 transitions; label with Qwen3.5-4B, Qwen3.5-9B, and a 27–31B model. Report inter-model agreement and agreement with hindsight labels (transitions within the last k steps before a level completion vs. random transitions). *Go if the 9B beats chance by a wide margin and the 4B is close to the 9B; otherwise the design shifts to P8+P3 with the larger model offline.*
- **Probe B — throughput/co-residency.** Run Goose at full speed with a 4B judge scoring sampled transitions asynchronously; confirm act/s stays >100 and VRAM stays within budget.
- **Probe C — can a 9B model write a working heuristic?** Give it a serialized digest and the state API; check that ≥1 of 5 generated heuristics runs without error and changes `novelty_late` on ft09 in a 20k-action smoke test.

---

## 4. Phased plan for the LLM track

| Window | Build | Ablation question answered |
|---|---|---|
| **Sep 1–14** | Minimal serializer + state API; Probes A/B/C; pick judge size | Is small-LLM judgment on our transitions better than chance? |
| **Sep 15–Oct 10** | P1: offline pairwise labeling (Qwen3.5-9B or 27B), train reward head, wire into Goose sampling behind `EVAL_LLM_REWARD` flag; 5-seed sweep on focus games + a 5-game subset | Does a distilled LLM prior change levels reached / `unique_states_per_action` / redundancy vs. Goose-baseline? |
| **Oct 11–Nov 7** | P2: periodic heuristic generation every ~5k actions + stall trigger (P7); selection by measured uplift; **mid-semester checkpoint (late Oct)** with P1 results + P2 first pass | Does periodic LLM adaptation beat a fixed distilled prior? |
| **Nov 8–30** | Stretch: P6 GRPO-tune the judge on outcome data (ColabPro if local is saturated) **or** P3 verified rules; full 25-game multi-seed sweep for the final matrix; integrate with teammates' layers | Does self-training the judge add anything over the frozen one? Do layers compose? |
| **Dec 1–15** | Freeze; analysis; report chapters (serializer, labeling study, ablations); defense | — |

Droppable in order if time runs short: P6/P3 → P7 → P2's evolutionary refinement (keep single-shot generation). P1 is the floor: even a null result ("small-LLM priors do not help exploration on ARC-AGI-3, with N seeds, here's why") is a legitimate chapter.

---

## 5. Risks and their mitigations

- **Serialization is the whole game.** Bad descriptions → bad labels → bad head. Mitigation: Probe A measures it before anything is built on it; the interface is designed to accept the teammate's object layer later.
- **Small-model hallucination (Qwen3.5 ~80% on omniscience-style tests).** Mitigation: never consume free-form claims — only pairwise preferences (P1), executable code judged by measured uplift (P2), or rules checked against the corpus (P3).
- **Reward hacking / decorative-change attraction returns through the back door.** Mitigation: hold out the ticker/canonicalizer features so the judge sees "this is a known ticker"; report `meaningful_change_rate`, redundancy, and coverage together as the harness already requires.
- **Throughput regression.** Mitigation: the per-step path is a tiny head + numpy; the LLM only ever runs async or periodically (Probe B guards this).
- **Hyperparameter-freeze rule.** Adding a bonus term to sampling changes the agent's behavior by design — get explicit advisor approval and log the exact change in `run_config.json` as the repo already does.

---

## Sources

- [RTX 5090 vLLM throughput (Runpod)](https://www.runpod.io/blog/rtx-5090-launch-runpod) · [RTX 5090 local-model guide](https://openclawdc.com/blog/best-local-llm-rtx-5090/) · [Qwen3.5-35B-A3B on a 5090 (HF discussion)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/discussions/3)
- [Qwen3.5 small models (Artificial Analysis)](https://artificialanalysis.ai/articles/qwen3-5-small-models) · [Qwen3.5 fine-tuning guide (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/fine-tune) · [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Small LMs in 2026 (TinyWeights)](https://tinyweights.dev/posts/best-small-language-models-2026/)
- [Motif: Intrinsic Motivation from AI Feedback (arXiv 2310.00166)](https://arxiv.org/abs/2310.00166) · [ONI: Online Intrinsic Rewards from LLM Feedback (arXiv 2410.23022)](https://arxiv.org/abs/2410.23022)
- [Eureka: Human-Level Reward Design via Coding LLMs (arXiv 2310.12931)](https://arxiv.org/abs/2310.12931) · [Reward Design Agent (arXiv 2606.01672)](https://arxiv.org/html/2606.01672v1)
- [WorldLLM (arXiv 2506.06725)](https://arxiv.org/abs/2506.06725) · [Executable World Models for ARC-AGI-3 (arXiv 2605.05138)](https://arxiv.org/abs/2605.05138)
- [SGA-ACR: Subgoal Graph-Augmented Planning (arXiv 2511.20993)](https://arxiv.org/html/2511.20993v1) · [LLM-Augmented Observations for RL Exploration (arXiv 2510.08779)](https://arxiv.org/html/2510.08779v1) · [LLM-Driven Intrinsic Motivation for Sparse Rewards (arXiv 2508.18420)](https://arxiv.org/abs/2508.18420)
- [Tiny Recursive Models (arXiv 2510.04871)](https://arxiv.org/abs/2510.04871) · [Robust RL for Small-Scale LM Agents (arXiv 2607.25091)](https://arxiv.org/abs/2607.25091)
- [ARC-AGI-3 Technical Report (arXiv 2603.24621)](https://arxiv.org/abs/2603.24621)
