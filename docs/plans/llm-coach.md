# Coach (v2) — the stall-triggered LLM advisor
### A local LLM ranks *what to try next* for mb_gated_att, only when it is stuck, and the advice is kept only if the numbers say it helped

Track: LLM integration. Owner: Matt. Repo: `sauerpatchkid/ARC3-solution` (branch `refactor/shared-baselines`, at 5da1508).
Base agent: **mb_gated_att** (`EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt`; 112 levels on the 25-game confirm).
In-loop model: **Qwen3.8-27B** (`cyankiwi/Qwen3.8-27B-AWQ-INT4`), the only local model kept (decision 3 Oct; the others were deleted).
Version: **v2, 3 Oct 2026.** v1 (`plan-A-llm-advisor.md`, 15 Sep, written as "Plan A") is kept unchanged. Most changes come from Rulebook
(`llm-rulebook.md`, written as "Plan C"). Both plans are kept as options; renamed 3 Oct 2026. §13 lists every change and where it came from, plus the parts of Rulebook not taken.
Dates: report TOC Oct 9 · first draft Oct 30 · idea lock Nov 2 · final report Dec 7.

> **Freeze rule (from Rulebook).** The prompt, serializer version, model revision, vLLM version and decoding settings are
> frozen after the offline probe (G1). Gate thresholds are frozen before the first dev run. The whole protocol (§6.4) is
> frozen before the confirm. Any later change is dated, explained and kept beside the old version.

---

## 1. One-paragraph summary

mb_gated_att keeps playing at ~118–130 act/s exactly as now. A stall detector (no new screen for W moves) fires rarely.
When it fires, the current screen and the level's history are turned into a **short text summary** (≈1–2k tokens): the
objects on screen with colour, size and position, what clicking each one has done so far, what ACTION1–5 have done so
far, and, on later levels, what the previous level's win looked like. The LLM returns a small JSON: a one-line
hypothesis, a ranked list of objects to click, weights for ACTION1–5, and things to avoid. That JSON becomes a **bias on
Goose's 4101 action probabilities** for the next K moves. A gate compares how many new screens Goose finds during the
advised window with the window before; advice that doesn't help is dropped. Two control arms, **random advice**
(placebo) and **hand-written heuristic advice**, use the same schedule and gate. The report can then say how much the
*language model* added beyond "poke the agent when it stalls".

## 2. Why this shape (updated)

| Finding | Source | Consequence |
|---|---|---|
| The complete treatment (executable model + simplification + exact replay verification) ranks first in every setting, but uses substantially more resources. An executable model *without* verification is worse than plain text for GPT-5.5. | Rodionov component study (arXiv 2607.15439), checked 3 Oct | v1 quoted only the second half. Checked world models are the strongest *frontier* recipe. Rulebook bets on that, at roughly 3,000 lines. This plan takes the smaller, measurable question first; Rulebook stays the follow-on (§11). |
| Best open-weights code agent (Qwen3.6-27B in a REPL) averages ~1.6% on public games | Tufa "Duck" (Milestone 1) | Don't build a coding harness for a small model; ask it a *question* |
| 2nd/3rd place used "look at the board, return one JSON action" | Reki, forge (Milestone 1) | JSON ranking is what small models do reliably |
| Small LLMs are biased against coordinates; "probe first, then interpret" works | AERA (arXiv 2605.25931) | The LLM picks *objects*, never coordinates |
| "Programmatic agent drives; LLM only on stall" | BDR-Pro | The only pattern that fits 100k-move runs without losing what Goose does |
| Qwen3.5-35B-A3B wrote ft09's tile-flip rule, exact on all 562 training moves, in 12 min (the dense 27B: 43 min); pictures alone made it misread colours and distances | our Stage A smoke test (`legacy/llm_track/README.md`) | The 35B is the strongest model that runs here, and exact changed cells must be given as text |

The question stays: "can a local LLM, shown a compact record of what has been tried, pick a better *next experiment*
than random and better than a hand rule?" It is answered offline first (§7.1), on corpora we already have.

## 3. Hypotheses (pre-registered)

**H1 (offline).** On stall and pre-win screens from the mb_gated_att confirm corpora, the LLM's top-3 objects contain
the object that produced the next new screen or the level-up more often than a random top-3, and at least as often as the
heuristic top-3. The same frozen inputs go to every model (§7.1), so model capability is measured, not assumed.

**H2 (online).** B3 (LLM advice) completes more levels than B0 on ≥ 2 of the 8 dev games, with no game worse on both seeds.

**H3 (attribution).** B3 > B1 (random advice). If B3 ≈ B2 (heuristic), the honest result is that the LLM's contribution
is a heuristic it could be replaced by. That is still a completed LLM arm and still a result.

**Expected non-results:** on ft09 the stall detector may rarely fire (new screens keep coming), so B3 ≈ B0 there. On
keyboard-only games, only `action_weights` matter.

## 4. Architecture

```
                ┌──────────── fast loop: mb_gated_att, unchanged (~125 act/s) ────────────┐
frame ─► bars mask ─► novelty label ─► CNN ─► sigmoid ─► [advice bias] ─► sample ─► deadclick ─► map ─► act
                                                  ▲                                                   │
                ┌───────────── slow loop ──────────┴───────────────────────────────────────────────────┤
                │ stall detector ─► serializer ─► LLM (vLLM, JSON) ─► validator ─► bias builder       │
                │      ▲                                                       │                       │
                │ level-win summary (on level-up)                 uplift gate ◄┘  (per move: object    │
                └──────────────────────────────────────────────────────────────── stats, novelty) ◄────┘
```

### 4.1 Trigger

Per level, track the move number of the last new screen (`LevelMemory.observe` already says whether a screen is new).
- **Stall:** no new screen for `W` moves (default `W = 1500`) and no advice active. The map's own stall routing starts
  at 200, so by the time the advisor fires, the map has already tried its routes.
- **Periodic fallback:** every `P = 10000` moves in a level regardless (ft09-type games where new screens never stop
  but progress does).
- **Level-up:** write the level-win summary (§4.2). No LLM call at that moment.
- **Back-off:** after rejected advice, `W ← min(2W, 12000)`; after accepted advice, `W ← 1500`.

Calls are **synchronous**: Goose pauses, so every call has a fixed evidence cutoff (move number). A slower model therefore
cannot get more exploration "for free". Every run logs why it ended (§5), including the 8-hour `is_done` stop.

### 4.2 Serializer (the part that decides everything)

- **Objects:** reuse `upgrades.screen_objects` (connected same-colour regions, background and bars excluded, salience =
  small size × rare colour; already used by mb_gated_att). Keep the top `N_OBJ = 30` plus any object with click
  history. Per object: `id`, `color`, `bbox`, `size`, `centroid`, `n_same_shape`.
- **Click history** per object, matched by `(color, bbox)`: `clicks_tried`, `clicks_changed`, `clicks_new_screen`,
  `last_effect` ("no-op", "changed ≤ 3 cells" or "changed N cells; colours X→Y"). The exact changed cells go in as text,
  as Stage A learned.
- **Button history** (last 200 uses of each): `n`, `changed%`, `new%`, and the dominant effect (which colour moved and
  by how much). The motion detector `_detect_moves` in `legacy/llm_track/serializer.py` does this already. It is
  **copied, not imported**, because `legacy/` is a leaf.
- **Level context:** level index, moves in level, screens seen, moves since the last new screen, available actions.
- **Previous level(s):** objects present before the winning move, the winning move (object for clicks), colour-count
  changes at the win.

Plain text tables, ≤ 2k tokens, versioned (the version and hash go into `run_config.json`). No images and no ASCII grid.

### 4.3 Prompt and output schema

The system prompt is fixed and versioned in `prompts/advisor_v2.txt`. Same text as v1: prefer untried objects and
actions, small distinctive button-like objects, and unknown effects over proven-useless ones; use the previous win to
guess what progress looks like; JSON only.

```json
{
  "hypothesis": "one or two sentences",
  "click_targets": [{"id": 12, "weight": 5}, {"id": 3, "weight": 2}],
  "action_weights": {"ACTION1": 1, "ACTION2": 1, "ACTION3": 0, "ACTION4": 3, "ACTION5": 5},
  "avoid": {"objects": [7], "actions": []}
}
```

Integer weights 0–5. vLLM guided-JSON enforces the schema; a regex-repair fallback stays. Temperature 0.3,
`max_tokens` 400, **thinking off** by default (the probe measures thinking on as one extra column).

### 4.4 Bias, injection and who owns the move

- `bias` is a length-4101 vector, initialised to 1.0.
  - A click target with weight `w`: its cells get `× (1 + β·w)`, with `β = 1.0`.
  - An avoided object's cells get `× 0.2`.
  - The action weights set the multiplier for each button the same way.
- It is applied in `_sample_from_combined_output` after the sigmoid and before the 1/4096 coordinate scaling. (v1's
  tried-mask step no longer exists; Plan B dropped the mask.)
- The result is clipped so that no single cell holds more than `p_max = 0.25` of the total probability.
- Advice stays active for `K = 1500` moves or until the level ends.
- **Ownership (from Rulebook).** The advice biases Goose's *sample* only. Deadclick and the map run afterwards, exactly as
  in mb_gated_att, and may still replace the move. This keeps the base agent intact, and the placebo arm sees the same
  overrides, so attribution stays clean.
  - The map chose a median 9% of moves in the confirm runs, and 50% on su15.
  - Each advised window therefore logs its **advice reach**: the share of its moves the sampler actually chose.
  - If the median reach in the dev sweep is below 50%, a "map pauses its stall routes during advice" variant becomes
    the ablation B5. It is not a mid-sweep retune.
- The advisor has its **own RNG** (seeded from `EVAL_SEED` plus a fixed offset). It never draws from Goose's numpy or
  torch streams, so the random and heuristic arms can't shift Goose's sampling by consuming random numbers.

### 4.5 Uplift gate (unchanged)

`nov_pre` is the number of new screens per 1k moves in the K moves before the trigger; `nov_adv` is the same during the
advice.
- Early check at `K/3`: if `nov_adv ≤ nov_pre + 1`, the advice is dropped, marked **rejected**, and the trigger backs off.
- Otherwise the advice is kept for all K moves and marked **accepted**.
- Optional: request two candidates (temperature 0.7) and A/B them for `K/3` each.

The gate doesn't make the LLM smarter. It keeps the arm close to B0 at worst, and it produces the acceptance rate.

### 4.6 Serving and GPU memory (new measurements)

vLLM runs in `.venv-llm` as a separate process with an OpenAI-compatible HTTP API, so the agent never imports vLLM.
Backends behind one interface: `mock` (tests), `vllm` (local), `anthropic`/`openai` (only the offline frontier column, §7.1).

Measured on this card (32,607 MiB total):

| Item | Memory | Source |
|---|---|---|
| Windows desktop, idle | ~2.5–3.0 GB | `nvidia-smi`, 1 and 3 Oct |
| One mb_gated_att run (peak) | **~4.0 GB** | 3,000-move ft09 run, 3 Oct (v1 assumed < 2 GB) |
| Qwen3.5-35B-A3B weights | 21.1 GiB | Stage A README |
| 35B at `--gpu-mem 0.90` | leaves 2.9 GiB of KV (~105k tokens) and **no room for Goose** | Stage A README + the two rows above |

So the 35B must run with a short context and a lower memory fraction. This plan's prompts are ≤ 2k tokens with 400
output tokens, so a few thousand tokens of KV is enough. That is the opposite of Rulebook, which needs ~94k.

**Step 2 result (3 Oct): the 35B fits beside one Goose run.**
- **Server:** `experiments/coach/serve.sh` with text only (`--language-model-only`), a fixed 0.5 GiB KV cache
  (14k tokens), 2,048 batched tokens and forced compact JSON.
- **Measurement:** `experiments/coach/fit_check.py`, a tu93 10k-move Goose run beside the server, then 20
  Coach-sized requests (1.7k tokens in, schema-forced JSON out, thinking off).
  Results are in `results/coach/fit_*.json`.

| Measure | Result |
|---|---|
| Server alone | 22.5 GB |
| Peak, server + Goose + Windows | 28.9 of 32.6 GB (**3.6 GB spare**) |
| Goose speed beside the idle server | 117–124 act/s (solo: 118–130) |
| Call latency, Goose paused (how Coach runs) | median **1.4 s**, 90th percentile 1.9 s; ~220 tokens out |
| Valid JSON | 20 of 20 |
| Calls while Goose trains | median 2.8 s, and Goose drops to ~53 act/s, so **never overlap them**: B3 runs go one at a time |

- **Two lessons.**
  - Pretty-printed JSON ran into the 400-token cap and was cut off (11 of 40 in the first run), so the server now
    forces compact JSON.
  - The desktop's own GPU use varies. One run started with 3 GB extra held by Windows apps: the peak hit 31.9 GB
    and Goose slowed to 83 act/s, because WSL spills to system RAM instead of failing. Keep GPU-heavy Windows apps
    closed during sweeps, and check `peak_mib_by_phase`.
- **The 9B fallback is not needed.** At 60 calls per run, LLM time is about 1.5 minutes per 100k-move run.
- **Newer models (checked 3 Oct).** There is no newer 35B-A3B than Qwen3.6-35B-A3B (Apr 2026); Qwen3.8 has no
  model at that size. Its open 27B dense model (Aug 2026) leads Qwen3.6-35B-A3B on public reasoning and coding
  benchmarks, at roughly a third of the speed.
- **Decision (Matt, 3 Oct): use Qwen3.8-27B only.** `cyankiwi/Qwen3.8-27B-AWQ-INT4` (19.6 GiB on disk) is the
  in-loop model and the only local model in the offline probe. Every other cached model was deleted (~118 GB),
  including the Qwen3.5-35B measured above. `experiments/coach/serve.sh` now defaults to it, with a 1 GiB KV
  cache: it needs 0.81 GiB for one 8k-token request.
  - **Measured so far:** the server alone uses 21.8 GB (card reading before and after loading), against 22.5 GB
    for the 35B. Beside a 3.9 GB Goose run and a ~2.3 GB idle desktop, that leaves about 4.5 GB spare.
  - **Still to measure:** the full `fit_check.py` (Goose speed and call latency). The first attempt was taken while
    a game was running on the same GPU, so it doesn't count. Expect calls to take about 3× as long as the 35B's,
    roughly 4–5 s each and under 5 minutes of LLM time per run.

### 4.7 Flags and arms

| Flag | Default | Meaning |
|---|---|---|
| `EVAL_ADVISOR` | `off` | `off` \| `random` \| `heuristic` \| `llm` |
| `EVAL_ADVISOR_W` / `_P` / `_K` | 1500 / 10000 / 1500 | stall window / periodic fallback / advice window |
| `EVAL_ADVISOR_BETA` | 1.0 | bias strength |
| `EVAL_ADVISOR_GATE` | 1 | uplift gate on/off |
| `EVAL_ADVISOR_BACKEND` / `_MODEL` | `vllm` / Qwen3.8-27B | backend and model id |
| `EVAL_ADVISOR_LEVELCTX` | 1 | include the previous-level win summary |
| `EVAL_ADVISOR_MAX_CALLS` | 60 | per-run cap on LLM requests (metered with tokens and seconds) |
| `EVAL_ADVISOR_CACHE` | `results/advisor_cache` | response cache for offline reruns and debugging only; scored runs never read another run's answers |

With `EVAL_ADVISOR=off` the advisor is never constructed. Every hook line in `action.py` ends in `# [advisor]`, so
deleting those lines restores the file byte for byte (the same convention as the map and the upgrades).

| Arm | `EVAL_ADVISOR` | What it tests |
|---|---|---|
| B0 | off | mb_gated_att. The 75 confirm runs (`results/confirm_upgrade/20260927_221915`) are reused (§6.1). |
| B1 | random | placebo: random object ids and weights, same schedule and gate |
| B2 | heuristic | rank untried, small, rare-coloured objects; button weights ∝ new% + an untried bonus |
| B3 | llm | the in-loop model |
| B4 | llm, `LEVELCTX=0` | ablates the previous-level summary (differs from B3 only on levels ≥ 2) |

## 5. Implementation plan (file by file)

| File | New / changed | Lines (est.) | What |
|---|---|---|---|
| `run_local.py` | changed | ~25 | **done 3 Oct:** writes `run_end.json` beside `run_config.json`: the engine's final level count, the final state and why the run stopped (`cap`, `win`, `is_done`, `error`, `interrupted`) |
| `compute_metrics.py` | changed | ~15 | **done 3 Oct:** uses the engine count when `run_end.json` exists and warns on a mismatch; a level finished on the final move is added at the last action. Runs without the file score exactly as before |
| `tools/paired_compare.py` | changed | ~40 | **done 3 Oct:** `--expect N` (exactly N pairs, no duplicates, one action cap, no run ended on an error; else BLOCKED, exit 2) and a game-resampling bootstrap interval. The saved upgrade-confirm verdict reproduces line for line |
| `custom_agents/objects.py` | new | ~100 | object history and `(color, bbox)` matching on top of `upgrades.screen_objects` |
| `custom_agents/serializer.py` | new | ~140 | the text summary; motion detector copied from the legacy serializer; deterministic and token-capped |
| `custom_agents/advisor.py` | new | ~220 | `Advisor`: trigger, request, validate, bias, gate, private RNG, meters; backends `mock`/`vllm`; random and heuristic advisors share the interface |
| `custom_agents/level_summary.py` | new | ~60 | the level-win summary |
| `custom_agents/action.py` | changed | ~20 | `# [advisor]` hooks: build from flags, feed per-move stats, bias the sampler, log reach |
| `tools/advice_probe.py` | new | ~200 | the offline probe (§7.1), with the matched multi-model mode |
| `prompts/advisor_v2.txt` | new | — | system prompt, hashed into `run_config.json` |
| `tests/test_objects.py`, `test_serializer.py`, `test_advisor.py` | new | ~250 | §7.2 |
| `experiments/advisor_dev/`, `experiments/advisor_confirm/` | new | — | runners with pre-registered READMEs, reusing `experiments/upgrade_confirm/` |

About 1,000 lines plus tests. Advice events go to `advice.jsonl` per run: trigger move, prompt hash, raw JSON,
accepted/rejected, `nov_pre`/`nov_adv`, reach and latency. The attribution analysis reads that file.

## 6. Metrics and success criteria

### 6.1 The reference (B0)

The 75 mb_gated_att confirm runs are reused. **Reconciliation is already done (3 Oct).** The last progress line of every
run log ("100000 actions score=N") is the engine's own level count, and it equals the scorer's count on all 75 runs. So
112 stands. A final-move level-up could be missed by the old scorer, but none occurred in B0. (The `scorecard score=`
line is an efficiency score, not a level count, so it can't be used for this.) Reuse also requires an identical code
path with the advisor off (§7.2) and the same engine, games, action space and 100k cap.

### 6.2 Metrics

- **Levels:**
  - total over game × seed, with paired better / same / worse;
  - the paired delta with a bootstrap over games;
  - games with ≥ 1 level;
  - **levels ≥ 2 counted separately**, because AERA reports that level 1 of many public games falls to trivial
    strategies;
  - actions-to-level with k-of-n seeds and censoring.
- **The advice funnel**, per game:
  1. stall triggers;
  2. valid JSON;
  3. advice accepted by the gate;
  4. advice reach;
  5. level-ups inside advised windows, compared with the share of moves that were advised (if 5% of moves are advised
     and 30% of level-ups land there, the advice is doing something).
- **Cost:** requests, generated tokens, LLM seconds per run, latency median and 90th percentile.
- **Disclosure:** six games (ar25, bp35, lf52, sb26, sk48, su15) play without ACTION7, for every arm.

### 6.3 Attribution and success bar

- Report B3 − B0, B3 − B1 and B3 − B2 as three separate paired deltas. Only B3 − B2 is "the language model's
  contribution".
- **Success bar:** B3 > B1 on levels on ≥ 2 dev games with no game worse on both seeds, **and** H1 holds offline.
- B3 ≈ B2 is an acceptable, reportable outcome.

### 6.4 Confirm rule and protocol freeze

- **Confirm:** the best of B2/B3 vs B0 on 25 games × seeds 0–2 × 100k. The verdict comes from
  `tools/paired_compare.py --expect 75`, using the same rule as every earlier confirm:
  - total levels ≥ B0 and paired wins > losses;
  - no game worse on every seed;
  - exactly 75 valid pairs, each run with a termination reason.

  Report the bootstrap interval beside the verdict. A small passing margin is an adoption decision, not proof of a
  general improvement.
- **Frozen before the confirm:** source commit, prompt and serializer versions, model / tokenizer / quantization
  revisions, vLLM version, decoding settings, all caps, engine and game versions, the scorer. Changing the number of
  seeds is a protocol change, not a knob.

## 7. Testing plan

### 7.0 Week-1 housekeeping (small, blocks nothing else)

- The `run_local.py` / `compute_metrics.py` / `paired_compare.py` changes in §5.
- The memory and latency measurements in §4.6.

### 7.1 Offline probe with a matched model comparison (week 2, no agent changes)

1. From the 75 B0 confirm corpora, find **stall points** (≥ 1500 moves without a new screen) and **pre-win points** (the
   decision before each of the 112 level-ups), restricted to the 8 dev games for tuning. The other 17 games are scored
   once, after the prompt is frozen.
2. At each point, build the serializer output from history up to that move. These are **frozen evidence packs**: every
   ranker sees identical input.
3. Rankers: random, heuristic, Qwen3.8-27B, and one frontier model on the same packs (~$5). (Decision 3 Oct:
   Qwen3.8-27B is the only local model. The 9B/27B/35B size ladder is dropped, so the capability figure becomes
   "local 27B vs frontier" at matched inputs.) It replaces v1's Colab 27B run.
4. Score:
   - pre-win points on click games: precision@3, "is the winning click's object in the top 3?";
   - stall points: "did the next new screen come from a top-3 object?" This is only defined when Goose happened to try
     one, so report coverage.
5. Also record parse failures, latency and prompt length, and hand-check 20 `hypothesis` lines. That table is the
   appendix.

**G1 (Oct 16), go/no-go:**
- **Go:** the in-loop model beats random on precision@3 with a sign test over ≥ 15 points, and is not clearly below the
  heuristic.
- **Clearly below the heuristic:** ship the heuristic as the advisor arm. The LLM result is then the capability figure.

### 7.2 Unit and isolation tests (`mock` backend)

- Objects, serializer, validator, bias and gate: as in v1 (synthetic frames, determinism, token cap, malformed JSON,
  `p_max`, unavailable actions stay impossible, scripted gate decisions).
- **Code-path isolation (from Rulebook):** with `EVAL_ADVISOR=off` the advisor is never built. Then a short CPU run
  (`CUDA_VISIBLE_DEVICES=`, deterministic algorithms, fixed thread count, 2,000 moves) must match the pre-change commit
  move for move.
- **Neutral advice:** a mock that always returns equal weights and no targets (bias exactly 1.0) must match off-mode
  move for move in the same CPU check. This proves the bookkeeping and the private RNG don't perturb Goose. If CPU
  training also diverges, fall back to the documented ~300-move identical window on the GPU.
- 2k-move smoke with fixed mock advice: throughput within 10% of B0, not counting LLM time.

### 7.3 Dev tier (weeks 3–4)

- **Games:** tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82. This is Rulebook's set, chosen because they play the full
  action space and mb_gated_att clears ≥ 1 level on each.
- **Protocol:** seeds 0–1 × 100k (the repo's dev protocol). B0 is free: it comes from the confirm runs.
- **Order:**
  1. B1 and B2 first, 4 runs at a time (no LLM on the GPU).
  2. Then B3 one run at a time beside vLLM.
  3. B4 only if B3 reaches levels ≥ 2 differently from B0.
- **G2 (Oct 28):** the success bar in §6.3, all 16 B3 pairs complete with termination reasons, and the funnel reported.

### 7.4 Confirm tier (weeks 5–6)

The best of B2/B3 vs B0 under the frozen protocol (§6.4). It is reported for all 25 games and separately for the 17
games held out from prompt tuning.

## 8. Timeline

| Week | Dates | Work | Deliverable / gate |
|---|---|---|---|
| 1 | Oct 3–9 | §7.0 housekeeping; `objects.py`, `serializer.py`; vLLM memory and latency beside Goose | **Oct 9:** report TOC + Meeting Log 1 |
| 2 | Oct 10–16 | `advice_probe.py`; matched probe on the dev games; freeze the prompt | **G1 (Oct 16)** |
| 3 | Oct 17–23 | `advisor.py`, bias, gate, ownership and reach logging; §7.2 tests incl. CPU checks; smoke; B1/B2 dev (overnight) | B0–B2 runnable |
| 4 | Oct 24–30 | B3 dev; funnel; draft writing | **G2 (Oct 28)** · **Oct 30:** first draft |
| — | Nov 2 | idea lock | |
| 5–6 | Oct 31–Nov 13 | protocol freeze; confirm; B4 if warranted | confirm verdict |
| 7–9 | Nov 14–Dec 6 | extensions if time (§11); figures; report; Thanksgiving buffer | **Dec 7:** final report |

**If time runs short, cut in this order:**
1. B4.
2. The B5 ownership ablation.
3. The frontier column.
4. The confirm shrinks to the 8 dev games + 3 seeds, labelled as such.

The offline probe and the B1/B2 arms are never cut. They are the minimum defensible result.

## 9. Compute budget (estimates, replaced by week-1 measurements)

| Item | Estimate |
|---|---|
| Offline probe | a few hundred points × 5 models, each ≤ 2k in / 400 out tokens; one evening per local model; ~$5 frontier |
| B1 + B2 dev (32 runs, 4 at a time) | ~6.5 h (the confirm ran ~5 runs/h at 4 at a time) |
| B3 dev (16 runs, 1 at a time beside vLLM) | ~16 × (13 min Goose + ≤ 60 calls × a few seconds) ≈ 4–5 h |
| Confirm with B3 (75 runs, 1 at a time) | ~20–22 h; with B2 instead (no LLM, 4 at a time) ~14 h |

Goose's training step alone keeps the GPU most of the way busy, so running Goose and the LLM side by side mainly helps
VRAM planning, not speed. Measured runs replace these figures before any date is committed.

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation | Early signal |
|---|---|---|---|
| The LLM ranks no better than random | medium | offline probe in week 2 at no online cost; the heuristic arm is the fallback deliverable | G1 |
| LLM ≈ heuristic | medium–high | pre-registered as acceptable; the matched capability figure turns it into a finding | G1 |
| The in-loop model doesn't fit beside Goose | low (server measured at 21.8 GB; ~4.5 GB spare) | fixed small KV cache, text only; keep GPU-heavy Windows apps (games, video) closed during runs | `fit_check.py` peak |
| The map overrides most advised moves | medium on map-heavy games (su15, tn36, r11l) | advice-reach metric; B5 ablation if median reach < 50% | dev funnel |
| JSON failures | medium | guided decoding; validator; failures logged and skipped, never a crash | probe parse-fail rate |
| Serialization loses what matters | medium | exact changed cells as text; hypotheses hand-checked; version-locked | probe hypotheses table |
| Stall never fires (ft09) or fires constantly (tu93) | high on specific games | periodic fallback + back-off; calls per game reported | dev funnel |
| Advice traps Goose in a loop | low | `p_max` clip, multiplicative bias, K-window expiry, gate | `nov_adv` |
| Missed terminal level-ups or silently dropped runs | low after week 1 | engine final count + termination reason; `--expect 75` | §7.0 |
| Prompt overfits the public games | medium | prompt and serializer frozen at G1; 17 games held out from tuning | held-out vs dev gap |
| Time (solo, alongside other coursework) | high | dated gates; offline probe stands on its own; cut order (§8) | weekly gates |

## 11. Extensions (only after B3 is measured)

1. **Level manual.** Carry every level-win summary forward, so on level ≥ 3 the advisor sees all earlier wins. Cheap;
   half-built by §4.2.
2. **Eureka-lite.** The LLM writes `def score(obj) -> float` over the serializer fields, sandboxed (a copy of
   `legacy/llm_track/heur_sandbox.py`). Two candidates are A/B-tested, same gate. Only worth it if B3 > B2.
3. **Distill accepted advice into a head.** Accepted windows are labelled data for a small extra CNN head.
4. **The door to Rulebook (verified world model).** If there is GPU time after the lock:
   - fix Stage A's model-load failure and run its frozen R1/R2 test (~4 h, unattended);
   - then Rulebook's Tier 0a: rule-book coverage on the dev-game corpora.

   Either result is reportable on its own, and together they say whether Rulebook is worth building after December.

## 12. How this appears in the report

- **Section:** "Intervention 2 — an LLM as a stall-time experiment designer."
- **Figures:**
  1. offline precision@3 at matched settings (random / heuristic / Qwen3.8-27B / frontier);
  2. the advice funnel per game;
  3. paired deltas B1/B2/B3 vs B0 on the dev games, with seeds shown;
  4. confirm levels per game with a bootstrap band.
- **Table:** acceptance rate, reach, calls, LLM seconds.
- **Appendix:** 20 advisor hypotheses with their screens.
- **Threats to validity:**
  - public games only;
  - pretraining may include public discussion of the preview games (ft09, ls20, vc33);
  - level 1 of many games is reachable trivially (hence levels ≥ 2 reported separately);
  - six games play without ACTION7;
  - seed variability;
  - the frontier column is an offline comparison, not a live run.

## 13. Changelog v1 → v2

| # | Change | Source |
|---|---|---|
| 1 | Base agent is mb_gated_att; B0 reuses its 75 confirm runs (v1 said Plan B's A3, whose mask was dropped) | repo state |
| 2 | B0 reconciled against the engine's own level count: 75/75 match, 112 stands | Rulebook §6.1 idea; checked 3 Oct with the log progress line instead of the scorecard line |
| 3 | In-loop model is the 35B-A3B if it fits, 9B as fallback; thinking off for JSON | Rulebook §13.6, Stage A smoke test |
| 4 | Memory budget rewritten: Goose peaks at ~4 GB (not < 2 GB), Windows holds ~2.5–3 GB, so vLLM can't use 0.90 | measured 3 Oct; Rulebook §8 asked for it |
| 5 | Matched capability comparison moved into the offline probe (same frozen packs to every model); Colab 27B run dropped | Rulebook §9 |
| 6 | Explicit ownership: advice biases the sample, deadclick and map still override; advice reach logged; B5 ablation if reach is low | Rulebook §2, §11 |
| 7 | Private advisor RNG; code-path isolation and neutral-advice CPU checks | Rulebook §3.9 |
| 8 | Engine final level count, termination reason and last-transition delivery in `run_local.py` (the lightweight version of Rulebook's event log) | Rulebook §3.10, §4 |
| 9 | `paired_compare --expect` and a game-level bootstrap; old verdicts unchanged | Rulebook §4, §6.5 |
| 10 | Dev set = Rulebook's 8 full-action games; 2 seeds × 100k; 17 games held out from prompt tuning | Rulebook §6.1 |
| 11 | Levels ≥ 2 reported separately; advice funnel; ACTION7 disclosure; threats-to-validity list | Rulebook §6.3, §12 |
| 12 | Dated gates G1/G2, cut order, protocol freeze list, per-run call cap, response cache | Rulebook §6.4, §6.7, §7, §3.9 |
| 13 | §2 corrected: Rodionov's study puts the full verification treatment first; v1 quoted only the half that suited it | Rulebook §1.2, checked on arXiv 3 Oct |
| 14 | `upgrades.screen_objects` and the legacy motion detector reused (copied, not imported) | Rulebook §4 (reuse pattern) |
| 15 | Rulebook's Stage A + Tier 0a added as the post-lock extension | Rulebook §6.1 |
| 16 | **After v2 (3 Oct):** in-loop and probe model set to Qwen3.8-27B only; rows 3 and 5 superseded (no 9B fallback, no local size ladder) | Matt's decision after the step-2 model check |

**Not taken from Rulebook, and why:**
- **The core design: an LLM-written world model, completion test, planner and executor (~3,000 lines).** It is the
  stronger idea on paper, but it is too large for one person before the Nov 2 lock. Its first gate (G1, Oct 9) is six
  days away, and its whole result depends on a chain where every link must work. It stays as extension 4.
- **Its 32k context with 4 × 20k-token candidates.** That needs ~94k tokens of KV cache, which cannot sit beside a
  4 GB Goose run on this card.
- **The full event log and scorer v2.** The B0 check above showed the existing scorer agrees with the engine on all 75
  reference runs. The two-line fix in `run_local.py` covers future wins and final-move level-ups.
- **A live frontier run (~$75).** The matched offline column answers the capability question for ~$5.
- **The strict `>` adoption rule.** Earlier confirms used `≥` plus wins > losses. Keeping it keeps this verdict
  comparable with them.
