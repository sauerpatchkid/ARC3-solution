# Plan B — "Memory-Aware Goose"
### Replace the frame-change label with canonical-state novelty, and stop retrying what already failed

Track: algorithm edit, no LLM. Owner: Matt. Repo: `sauerpatchkid/ARC3-solution`, `custom_agents/action.py`.
Status: proposed 15 Sep 2026. Idea-lock: Nov 2.

---

## 1. One-paragraph summary

StochasticGoose learns one thing: "will this action change the frame?" Your 25-game sweeps showed that
signal is dead on ~80% of games (decorative animation makes every action look successful) and that the
agent re-executes the same (state, action) pairs at a 94% repeat rate on ft09. Plan B changes two things
and nothing else: (1) the training label becomes **"did this action lead to a state I have not seen before
in this level?"**, where "state" is the frame with decorative indicator cells masked out (the same
canonicalizer your metrics already use, run online); and (2) at sampling time, actions already executed
from the current canonical state are **soft-masked** so the sampler prefers untried actions. The CNN,
optimizer, buffer, hyperparameters, reset semantics and eval contract stay identical. The agent then
optimizes exactly the quantity your metric stack already reports (`unique_states_per_action`) and its
redundancy metric becomes a direct diagnostic of whether the mask is working.

## 2. Hypothesis and predictions (pre-registered)

**H1.** On games where the raw change signal is degenerate (change rate ≥ 0.8), the novelty label restores a
non-trivial positive/negative balance in the buffer (positive rate moves from ≈1.0 toward 0.05–0.5) and the
CNN's predictions become informative (training accuracy > majority-class rate).

**H2.** Tried-action masking reduces `redundancy` and raises `novelty_late_per_1k` on keyboard/long-horizon
games (tu93, g50t, tr87, wa30, re86) where the same few discrete actions are hammered.

**H3.** Together they increase the number of seeds reaching each level and reduce median actions-to-level
on at least 2 of the 6 dev games, with no game worse than seed noise (paired comparison).

**Expected non-result, stated up front:** on ft09 the unique-state count already explodes (181k states in
200k actions), so novelty ≈ change and H1 predicts *no* label-side gain there; any ft09 gain must come
from masking. If ft09 improves under the label alone, that is a surprise worth investigating, not a win
to bank.

## 3. Why this, in one table

| Evidence | Implication |
|---|---|
| Your sweeps: 13/19 games change rate ≥0.80; learning only on click games; vc33 meaningful change 0.05 | The label is the bottleneck, not the network |
| Your ft09 finding: 94% exact-repeat fraction, click entropy 0.47→0.13, unique frames saturating in first 10% | The agent needs memory of where it has been |
| Your persistence ablation: ~18× redundancy under naive persistence | Redundancy is the failure signature; a mechanism that attacks it directly is the right first move |
| BDR-Pro (no-LLM Kaggle 0.26–0.27): volatility-masked hashing is architecture step #1; no-op skipping in step #2; contextual dead-click rules gave **+68% mean** | Three independent teams converged on "canonical state + don't repeat" as the foundation |
| Blind Squirrel / Rudakov prune loop and no-change actions | Same idea, graph form |
| ARC designer: games were hardened against semi-random exploration | Goose's *sampling* is fine; its *knowledge of what it has tried* is what's missing |

## 4. Design

### 4.1 Online canonicalizer (state identity)

Reuse `metrics_common.find_indicator_cells(freq, tiny_frac, tiny_cell_counts)` with **running** aggregates
instead of end-of-run aggregates.

Per game (indicator cells are a property of the game's HUD, not the level):
- `n_trans`: transitions seen
- `cell_change_count[64,64]`: how many transitions changed each cell
- `tiny_count`: transitions with 1–2 changed cells
- `tiny_cell_counts[64,64]`: per-cell changes summed over tiny transitions only

Every `MASK_REFRESH = 250` actions, once `n_trans ≥ MASK_WARMUP = 200`:
```
freq = cell_change_count / n_trans
mask = find_indicator_cells(freq=freq, tiny_frac=tiny_count/n_trans, tiny_cell_counts=tiny_cell_counts)
```
Before warm-up the mask is all-False (canonical == raw). The mask self-heals because it is recomputed
from running counts; a cell that stops changing drops out.

Canonical key of a raw uint8 frame `f`:
```
g = f.copy(); g[mask] = 0
key = xxhash.xxh64(g.tobytes()).intdigest()
```
Cost: one 4 KB copy and hash per step (you already hash the one-hot tensor per step; this replaces nothing
and adds ~5 µs).

Why not BDR-Pro's 20% per-cell threshold? Their threshold is tuned for their agent; yours is validated on
your own corpora (vc33 rotating bar, fixed tickers) and the offline/online agreement is testable (§7.1).
Keep the offline thresholds; make them config so the ablation is reproducible.

### 4.2 Novelty label

Per level (cleared on level change, exactly where the buffer is cleared today; **not** cleared on
GAME_OVER — after a death the level restarts, and states seen before the death are still "seen"):
- `seen_keys: set[int]`

On every decision frame:
```
key = canon(current_frame_raw)
novel = key not in seen_keys
seen_keys.add(key)
```
The experience built from `(prev_frame, prev_action)` gets:
```
reward = 1.0 if (LABEL == "novel" and novel) else (1.0 if (LABEL == "change" and frame_changed) else 0.0)
```
Everything downstream (hash-dedup into the buffer, BCE on the selected logit, confidence bonus, training
cadence) is untouched.

Properties worth writing down for the report:
- A no-op transition is never novel (identical frame ⇒ identical key). So novelty ⊆ change.
- A change to an already-seen state (undo, toggling a ticker, walking back) is 1 under `change`, 0 under
  `novel`. This is the whole difference.
- The label is non-stationary in principle (the same (s,a) executed later would be labelled 0), but the
  buffer stores each (frame, action) hash once, so each pair keeps its **first-execution** label. The CNN is
  therefore trained on "when this (s,a) was first tried, did it lead somewhere new" — a stable target.
- Under `novel`, the buffer will contain far more negatives on decorative games. Class balance is the
  point: today those games have ~100% positives and nothing to learn.

### 4.3 Tried-action soft mask

Per level:
- `tried: dict[int, dict[int, int]]` mapping canonical key → {unified_action_idx → count}

At sampling time, after masking unavailable actions and applying sigmoid, before the 1/4096 coordinate
scaling:
```
counts = tried.get(key, {})
for a, n in counts.items():
    probs[a] *= MASK_DECAY ** n          # MASK_DECAY = 0.1, so 1 try → ×0.1, 2 → ×0.01
probs = np.maximum(probs, MASK_FLOOR)   # MASK_FLOOR = 1e-4, keeps every action possible
```
After choosing action `a` from state `key`: `tried[key][a] += 1`.

Soft, not hard, for two reasons: stateful games sometimes require repeating an action from an identical-
looking frame (hidden counters, tu93's budget bar is masked out by design), and a hard mask would
eventually leave a state with nothing to do. Decay by count means "try something else first, but come back
if you must."

Memory: bounded by (unique canonical states) × (actions tried per state). ft09's 181k states × ~1–2 actions
each ≈ 300k small ints — fine. Add an LRU cap (`TRIED_MAX_STATES = 500k`) for safety.

### 4.4 What is deliberately unchanged

CNN architecture, `lr=1e-4`, `train_frequency=5`, `batch_size=64`, buffer 200k, confidence coefficients,
`EVAL_RESET_ON_LEVEL` semantics, the experience hash (still on the raw one-hot frame + action), the
transition corpus schema, and the reward being a scalar on the selected logit. `CLAUDE.md` forbids touching
these without approval; this plan does not need to.

### 4.5 Config flags (eval_common contract style)

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `EVAL_LABEL` | `change` \| `novel` | `change` | training label |
| `EVAL_MASK_TRIED` | `0` \| `1` | `0` | tried-action soft mask on/off |
| `EVAL_MASK_DECAY` | float | `0.1` | per-try multiplier |
| `EVAL_MASK_FLOOR` | float | `1e-4` | minimum probability |
| `EVAL_CANON_WARMUP` | int | `200` | transitions before masking indicator cells |
| `EVAL_CANON_REFRESH` | int | `250` | recompute cadence |

All recorded in `run_config.json` by `write_run_config`. Arms:

| Arm | `EVAL_LABEL` | `EVAL_MASK_TRIED` |
|---|---|---|
| A0 baseline | change | 0 |
| A1 novelty | novel | 0 |
| A2 mask | change | 1 |
| A3 both | novel | 1 |

Stage 2 (after A3 is adopted) adds `EVAL_DEAD_CLICK=1` (contextual dead-click rules) and, if time allows,
`EVAL_CLICK_PRIOR=objects` (connected-component click prior). Each is its own arm on top of A3.

## 5. Implementation plan (file by file)

1. **`custom_agents/canon.py` (new, ~80 lines).** `OnlineCanonicalizer` class: `update(prev_raw, cur_raw)`,
   `key(raw) -> int`, `mask` property, `stats()` for logging. Imports `find_indicator_cells` from
   `metrics_common`. Pure numpy; no torch.
2. **`custom_agents/action.py`.**
   - `__init__`: read the flags; create `self.canon = OnlineCanonicalizer(...)`, `self.seen_keys = set()`,
     `self.tried = {}`; record flags in `write_run_config`.
   - In `choose_action`, right after `current_frame_raw` is computed and **before** the experience is built:
     `self.canon.update(self.prev_frame_raw, current_frame_raw)` (only when `prev` exists), then
     `key = self.canon.key(current_frame_raw)`, `novel = key not in self.seen_keys`, `self.seen_keys.add(key)`.
   - Experience reward: replace `1.0 if frame_changed else 0.0` with the label switch.
   - On level change (the block that clears the buffer): also `self.seen_keys.clear()`, `self.tried.clear()`.
     Do **not** clear on `GAME_OVER`/`RESET`.
   - `_sample_from_combined_output`: add optional `tried_counts` argument; apply the soft mask after sigmoid,
     before coordinate scaling. Return unchanged tuple.
   - After the action is chosen: `self.tried.setdefault(key, {})[unified_idx] = ... + 1`.
   - TensorBoard scalars every 100 actions: `Canon/masked_cells`, `Label/positive_rate_buffer`,
     `Novelty/seen_states`, `Mask/mean_multiplier_applied`.
3. **`eval_common.py`.** Add the six flags to the documented env-var table and to `write_run_config`
   kwargs. Corpus schema unchanged (novelty is recomputable offline by `compute_metrics.py`, which already
   canonicalizes; the whole point is that the agent and the scorer agree).
4. **`sweep.sh`.** Add `LABEL` and `MASK` overrides that export the two flags; tag arms in the manifest as
   `label_<x>_mask_<y>` so `summarize_overnight.py` and `analyze_curves.py` group by arm. (They already
   group by `reset_<arm>`; generalize the arm string.)
5. **`tests/test_canon.py`, `tests/test_label.py`** (new; pytest). See §7.1.
6. **`tools/label_diagnostic.py`** (new). Replays existing corpora in `results/runs/*/*/transitions/` and
   reports, per game and seed, the positive-label rate under `change` and under `novel`, and the
   end-of-run mask size. This runs on data you already have and is the first deliverable (§8, week 1).
7. `README.md`/`CLAUDE.md`: document the flags and the arms.

Estimated code: ~250 lines including tests. No new dependencies.

## 6. Metrics and success criteria

Primary (what the report leads with):
- `max_level`, "k/n seeds reached level j", median `actions_to_level_j` with censoring, levels-vs-budget
  curve with AULC at the tier's `T_max`. Paired per game against A0 on the same seeds.

Mechanism (why it worked, or why it didn't):
- `unique_states_per_action` uplift over the matched random floor; `redundancy`; `novelty_late_per_1k`;
  `meaningful_change_rate`; buffer positive rate; CNN training accuracy vs. majority-class rate.

Guardrails:
- Throughput within 10% of A0 (both extra structures are O(1) per step).
- Buffer/host memory within limits (watch `tried` size on ft09).

Adoption rule (borrowed from BDR-Pro): a change is adopted only if it wins on the Confirm tier
(≥19 games × 3 seeds) on the primary metrics **and** is not worse than A0 on any game beyond seed noise.
If A3 wins on Dev but not Confirm, report that honestly — it is the public-set-overfit story the field is
already telling, and a capstone that measures it is more valuable than one that hides it.

Pre-registered success bar for the semester: A3 (or A1/A2) yields at least one of
(a) a new level reached on any dev game that A0 never reached, or
(b) ≥30% reduction in median actions-to-L1 on ≥2 games, or
(c) ≥2× `unique_states_per_action` uplift on the decorative-change games (vc33, ls20) with redundancy down —
and no game regresses.

## 7. Testing plan

### 7.1 Unit and offline tests (day 1–3, no GPU)
- **Canonicalizer agreement:** run `OnlineCanonicalizer` over a saved vc33 corpus and a saved ft09 corpus;
  the final online mask must equal `compute_metrics.py`'s offline mask on the same corpus (bit-exact for
  fixed tickers; ≥95% Jaccard for the rotating detector, because online counts lag).
- **Synthetic frames:** a static 64×64 grid + one blinking cell + a 60-cell rotating bar + one "real" object
  that moves on demand. Assert: the blinking cell and the bar are masked after warm-up; the object is not;
  moving the object produces a novel key; blinking does not.
- **Label semantics:** scripted sequence (no-op, change-to-new, change-back, change-to-new) → labels under
  `change` = [0,1,1,1], under `novel` = [0,1,0,1].
- **Mask math:** a state with actions {0: 1 try, 3: 2 tries} → multipliers [0.1, 1, 1, 0.01, 1, …]; floor
  respected; unavailable actions still −inf.
- **Reset semantics:** level change clears `seen_keys`/`tried`; GAME_OVER does not.
- **Label diagnostic on existing corpora:** table of positive rate (change vs novel) per game/seed. Expected:
  change ≈ 1.0 on ≥13 games; novel well below on the decorative games; both high on ft09. This table goes
  in the report as the motivation figure.

### 7.2 Smoke (day 3–4)
- 2k actions each on ls20, ft09, vc33, tu93 with A3; assert throughput ≥ 0.9 × A0; scalars sane; no
  exceptions on the six previously crashing games (`sk48 su15 lf52 ar25 sb26 bp35`) — if any still crash,
  exclude and note.

### 7.3 Dev tier (week 2)
- 6 dev games × 5 seeds × 200k × 4 arms ≈ 4 × 12 h. Run A0 first (it is also the reference for Plan A).
- Deliverable: paired table + curves; decide which arm goes to Confirm.

### 7.4 Confirm tier (week 3)
- Winning arm vs A0 on all runnable public games × 3 seeds × 100k. Directly comparable to the existing
  25-game characterization sweep (same cap, same scorer).

### 7.5 Long-horizon and holdout (late Oct)
- ft09 + tu93 × 3 seeds × 1M for the winning arm and A0.
- Holdout: 6–8 games never opened during Plan B design, 3 seeds × 100k, run once.

## 8. Timeline (against Nov 2 idea-lock, early-Dec report)

| Week | Dates | Work | Output |
|---|---|---|---|
| 1 | Sep 16–22 | Label diagnostic on existing corpora; `canon.py`; label switch; mask; unit tests; smoke | Motivation table; A0–A3 runnable |
| 2 | Sep 23–29 | Dev sweep, 4 arms | First paired results; pick arm |
| 3 | Sep 30–Oct 6 | Confirm sweep on the pick; start Stage 2 (dead-click rules) | Adopt/reject decision; Stage-2 arm |
| 4–5 | Oct 7–20 | Stage 2 dev sweep; optional object click prior | Second adopt/reject |
| 6–7 | Oct 21–Nov 2 | Long-horizon + holdout; freeze mechanism; write methods section | **Idea locked** |
| Nov | | Combine with teammates' arms (switchable layers); final sweeps; report figures | Report draft |

GPU budget: ~4 days total for Plan B through Confirm; fits alongside Plan A if the two share the A0 runs.

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Novelty ≈ change on games with huge state spaces (ft09) → no label-side gain there | High | Pre-registered as expected; the mask is the ft09 lever; report per-game, not pooled |
| Canonical key merges genuinely different states (over-masking, e.g. a game where a HUD cell *is* the goal) | Low–medium | Warm-up + refresh; log masked-cell count per level; a per-game spike is the alarm; offline agreement test |
| Soft mask suppresses a required repeated action (hold-direction games) | Medium | Decay-by-count with floor; A2 vs A0 on keyboard games measures it directly |
| `tried` memory growth | Low | LRU cap; ft09 is the stress test |
| Label becomes too sparse late in a level (everything seen) → CNN drifts toward "nothing is novel" and sampling flattens toward uniform | Medium | That is the correct behaviour when the level is exhausted; it is also exactly the stall signal Plan A triggers on; log positive rate over time |
| Wins on Dev, not on Confirm/Holdout | Medium | Adoption rule; report both; this is a finding |
| Throughput regression from Python dict work | Low | Measured in smoke; xxhash + dict are µs-scale |

## 10. Stage-2 and Stage-3 extensions (same arm, same flags pattern)

- **Contextual dead-click rules** (`EVAL_DEAD_CLICK`): per level, `dead[(y,x,color)] += 1` when a click at
  (y,x) on color c produced no change; ban (multiplier 0) once count ≥ 4. Keyed by appearance so a button
  that arms itself escapes the ban. BDR-Pro reports +68% mean from this alone.
- **Object click prior** (`EVAL_CLICK_PRIOR=objects`): connected components (4-connectivity, per color);
  prior weight ∝ 1/(size) × rarity(color) for components with 2 ≤ size ≤ 64; multiply into coordinate probs.
  Overlaps teammate arm (2) — coordinate with them; if they deliver a segmenter, consume it here.
- **Winning-prefix replay** after GAME_OVER (Stage 3): store the action prefix from level start to the
  furthest novel state; after a death, replay it before sampling. Addresses the "re-earn progress" tax.
- **Frontier BFS over the tried map** (Stage 3, overlaps arm (3)): the `tried` structure is already a
  partial transition graph; adding `next_key` per (key, action) turns it into one.

## 11. Pros and cons

**Pros**
- Attacks the two failure modes your own data identified (dead label, redundancy) with the two mechanisms
  the strongest no-LLM agents converged on.
- Smallest possible change: ~250 lines, two flags, nothing else moves. Cleanest ablation story in the
  project; easy to explain to a committee in one slide ("the goose now remembers where it has been").
- A first number in two weeks; a Confirm-tier result before Milestone 2.
- The agent optimizes what your scorer measures; the mechanism metrics become diagnostics, not decorations.
- Zero new dependencies, zero throughput risk, runs on the existing sweep tooling.
- Composes with Plan A (shares the stall signal and the sampler hook) and with every teammate arm.

**Cons**
- No new *knowledge* about the game: it is still search. It raises coverage; it cannot create goal
  understanding. On rules-not-search games (tu93, g50t) the ceiling may not move.
- On ft09-like games with combinatorial state spaces the label side is expected to be a no-op.
- Coverage is Goodhart-prone: more unique states is not more levels unless levels live in the unexplored
  region. The primary metric must stay level completions.
- Not the LLM-bearing arm the 295B plan requires — Plan A still has to exist.
- Overlaps lightly with the canonical-state graph-memory arm (3); agree on the hash so the two arms can be
  combined in November without a schism.

## 12. How this appears in the report

Section: "Intervention 1 — state identity and novelty." Figure 1: positive-label rate per game under the
two labels (the motivation). Figure 2: levels-vs-budget curves A0 vs A3 on the 6 dev games with seed bands.
Table: paired deltas, Confirm tier, with sign counts. One paragraph on the holdout result. One paragraph on
what did not change (ft09 label-side) and why that was predicted.
