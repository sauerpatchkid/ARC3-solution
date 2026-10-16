# Claude Code Handoff — StochasticGoose Repo Cleanup & Bugfixes

**Repo:** `ARC3-solution` (Team B, ARC-AGI-3 capstone). Open this in a Claude Code
**WSL** session pointed at the repo on the Linux filesystem.

**What I want from this session:** fix the confirmed bugs, address the oversights,
apply the small safe improvements, and clean up the repo. Work incrementally —
one logical change at a time, verify it, then move on. Do **not** implement any
new "learning" / generalization capability (see §5, Out of scope). Preserve the
`eval_common` contract, reproducibility, and the untouched API path.

---

## 1. Background — what this repo is

This is the StochasticGoose track of an ARC-AGI-3 agent project. The pipeline has
three cleanly separated stages:

1. **The agent** (`custom_agents/action.py`) — a CNN "change predictor." It
   one-hot-encodes the 64×64 grid into 16 color channels, runs a shared conv
   backbone, and splits into two heads: 5 logits for ACTION1–5 and a 4096-logit
   spatial head for ACTION6 clicks. It is trained online with per-action binary
   cross-entropy on the label "did this action change the frame?" and samples its
   next action from the sigmoid outputs (biased toward actions predicted to cause
   change). On each level-up it optionally resets model+optimizer+buffer (the
   persistence ablation, controlled by `EVAL_RESET_ON_LEVEL`).
2. **The corpus logger** (`TransitionLogger` in `eval_common.py`) — writes every
   `(frame, action, next_frame, changed, level, action_num, wall_ms, model_ms)`
   transition to compressed `.npz` shards. It stores the **full next frame**, not
   just the 0/1 changed label.
3. **The scorers** — `compute_metrics.py` (lean per-run scorer used by the local
   pipeline) and `summarize_runs.py` (fuller analyzer for the legacy API suite)
   turn corpora into the metric stack: level completions + actions-to-each-level
   (headline); unique-state coverage / discovery AUC / meaningful-change-rate /
   redundancy / entropy drift (mechanism); timing (wall vs. model ms, model-bound
   act/s = the offline Kaggle ceiling).

**Runner:** `run_local.py` drives the agent against the in-process `arc_agi`
engine (fast path, ~60 act/s) instead of the hosted API. `run_overnight.sh`
sweeps games × seeds × reset-arms and calls `compute_metrics.py`, then
`summarize_overnight.py` aggregates.

**The `eval_common` contract (env vars):**
- `EVAL_SEED` — base seed; each game adds a stable md5 offset. Unset ⇒ time-based.
- `EVAL_MAX_ACTIONS` — per-game action cap (0/unset = unlimited).
- `EVAL_LOG_METRICS` (default on) — TensorBoard scalars.
- `EVAL_LOG_TRANSITIONS` (default on) — the on-disk corpus.
- `EVAL_SAVE_VIS` (default off) — expensive debug heatmaps.
- `EVAL_RESET_ON_LEVEL` (default on) — reset model/optimizer/buffer at each level
  boundary (the persistence ablation). **Do not change this semantics.**

**Current focus games:** `ft09` (learnable, reliable completions) and `ls20`
(null contrast, completes nothing). The older quartet (`vc33/g50t/dc22`) is legacy.

**Env / how to run:** Windows 11 + RTX 5090, WSL2/Ubuntu 24.04, repo on the Linux
filesystem, `uv` venv (Python 3.12, torch 2.8/cu128). `ARC-AGI-3-Agents` is a git
submodule. Smoke test + score:
```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ls20
uv run python compute_metrics.py runs/<ts>/ls20/transitions \
    --game ls20 --agent goose --seed 0
```

**Note on an existing patch:** three of the bug fixes below (B1, B2, B3) may
already be applied via a file `action_fixes.patch` in the repo root. **Before
touching `custom_agents/action.py`, check whether each fix is already present**
(grep hints given per-task) and skip anything already done — do not double-apply.

---

## 2. Bugs — must fix (`custom_agents/action.py` unless noted)

### B1 — Remove per-step `torch.cuda.empty_cache()`
At the end of `_train_action_model()`. It fires every training step (every 5
actions), hands cached memory back to the driver, and serializes the CUDA stream
for no benefit (the model is tiny; no OOM risk). Delete these two lines:
```python
        # Clean up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```
*Already applied if:* `grep -n empty_cache custom_agents/action.py` returns nothing.

### B2 — Fix `experience_hashes` eviction starvation
In `choose_action`, the "Only store if unique" block. `experience_buffer` is a
`deque(maxlen=200000)` but `experience_hashes` is never pruned when the deque
evicts. Consequences: the set grows unbounded, an evicted `(frame, action)` can
never be re-stored, and the `replay_unique_hashes` scalar diverges from the real
buffer size past 200K. Fix by storing the hash on each experience and discarding
the evicted one so the set mirrors the deque:
```python
            # Only store if unique
            if experience_hash not in self.experience_hashes:
                # If the buffer is at capacity, the leftmost (oldest) item is
                # about to be evicted by the append below — drop its hash too so
                # experience_hashes stays a mirror of what's actually buffered.
                if len(self.experience_buffer) == self.experience_buffer.maxlen:
                    evicted = self.experience_buffer[0]
                    self.experience_hashes.discard(evicted['hash'])
                experience = {
                    'state': self.prev_frame,            # numpy bool
                    'action_idx': self.prev_action_idx,  # unified action index
                    'reward': 1.0 if frame_changed else 0.0,
                    'hash': experience_hash,             # kept so eviction syncs the set
                }
                self.experience_buffer.append(experience)
                self.experience_hashes.add(experience_hash)
```
*Already applied if:* `grep -n "'hash': experience_hash" custom_agents/action.py` matches.

### B3 — Off-by-5 in the ACTION6 reasoning string
In the ACTION6 branch of `choose_action`, the human-readable log printed the wrong
probability (`all_probs` has 5 action slots then 4096 coord slots; `coord_idx`
indexes within the coord block only). Cosmetic — behavior is unaffected — but fix:
```python
selected_action.reasoning = f"ACTION6 at ({x}, {y}) (prob: {all_probs[5 + coord_idx]:.3f})"
```
*Already applied if:* `grep -n "all_probs\[5 + coord_idx\]" custom_agents/action.py` matches.

### B4 — `suite_summary.csv` schema collision (already corrupting data)
Two pipelines write the **same filename** with **different schemas**:
- `summarize_runs.py` (legacy API path) writes a 50-column table.
- `compute_metrics.py` (local pipeline) *appends* a 22-column row (its
  `SUITE_COLUMNS`) whenever `--suite suite_summary.csv` is passed.

The committed `suite_summary.csv` is already broken: 13 rows of the 50-col format
plus 5 rows of the 22-col format jammed under the 50-col header. Fix:
1. In `run_overnight.sh`, change `SUITE="suite_summary.csv"` → `SUITE="local_suite.csv"`.
2. Update the example command in the `run_local.py` module docstring (and any
   README reference) that says `--suite suite_summary.csv` → `--suite local_suite.csv`.
3. Repair the committed file: create `legacy/` and move the **50-column** portion
   (header + the 13 API rows) to `legacy/suite_summary_api.csv`. Discard the 5
   stray 22-column rows (they are reproducible 10K pilot runs). Then `git rm` the
   old top-level `suite_summary.csv`.
4. Keep the local pipeline's file (`local_suite.csv`) **untracked** (see C5).

---

## 3. Oversights — should fix (`custom_agents/action.py` unless noted)

### O1 — Dead "weird frame" fallback branch
`choose_action` has `if current_frame is None:` (returns a random action), but
`_frame_to_tensor` never returns `None` — it `assert`s the shape and would crash
on a malformed frame instead. Make the fallback reachable by replacing the bare
assert in `_frame_to_tensor` with a guarded return:
```python
        if frame.shape != (self.grid_size, self.grid_size):
            self.logger.warning(f"Unexpected frame shape {frame.shape}; skipping")
            return None
```
(Leave the existing `if current_frame is None:` handler as-is; it now actually runs.)

### O2 — Mislabeled "entropy" regularization (rename only, do NOT change math)
In `_train_action_model`, the "entropy" terms are actually the **mean sigmoid
confidence** (`action_probs.mean()` etc.), and subtracting them rewards *higher*
predicted-change probabilities. The name is misleading and the TensorBoard scalars
`Training/action_entropy` / `Training/coord_entropy` do not measure entropy. Fix
the naming for honesty **without changing behavior** (keep the same coefficients
and the same arithmetic so ablations aren't confounded):
- Rename locals `action_entropy`→`action_confidence`, `coord_entropy`→`coord_confidence`.
- Rename coeffs `action_coeff`→`action_conf_coeff`, `coord_coeff`→`coord_conf_coeff`
  (values unchanged: `0.0001`, `0.00001`).
- Rename the scalar tags accordingly: `Training/action_confidence`,
  `Training/coord_confidence`, `Training/action_confidence_coeff`,
  `Training/coord_confidence_coeff`.
- Update the surrounding comments (they currently say "entropy bonus" / "entropy
  regularization") to state plainly: *this is a mean-sigmoid confidence bonus, not
  entropy.*
- Leave the selection-time `Agent/coord_entropy` scalar alone — that one **is** a
  real Shannon entropy and is correctly named.

### O3 — Unify the decorative/indicator-cell canonicalizer (medium effort; the one non-trivial task)
`compute_metrics.py` masks "decorative" cells with a simple frequency threshold
(`DECOR_THRESHOLD = 0.95`), while `summarize_runs.py` uses a smarter
`find_indicator_cells` that catches both fixed and **rotating** tickers. Because of
this, `meaningful_change_rate` and `redundancy` are **not comparable across the two
pipelines** — a problem, since these metrics get frozen for the writeup.

Goal: one canonicalizer, used everywhere.
1. Create `metrics_common.py` and move `find_indicator_cells` (the fixed +
   rotating-ticker detector, currently in `summarize_runs.py`) into it, plus any
   shared small helpers. Have `summarize_runs.py` import it (pure dedup — behavior
   unchanged there).
2. Make `compute_metrics.py` use the same logic. **Preserve its streaming /
   low-memory design** — it must not load the whole diff array into memory. The
   rotating-ticker detector only needs streaming-accumulable aggregates:
   per-cell change frequency (already have `diff_count`), the count of
   "tiny" transitions (≤2 changed cells), and a per-cell coverage accumulator
   summed over just those tiny transitions. Accumulate those in `pass1`, then
   reproduce the `find_indicator_cells` decision in `compute()` from the
   aggregates. Keep `DECOR_THRESHOLD` as the fixed-ticker threshold.
3. After the change, re-score one existing `ft09` and one `ls20` corpus with both
   pipelines and confirm `meaningful_change_rate` now agrees (within rounding).

### O5 — Missing CUDA seeding
In `Action.__init__`, right after `torch.manual_seed(seed % (2**32 - 1))`, add:
```python
        torch.cuda.manual_seed_all(seed % (2**32 - 1))
```
(No-op on CPU.) **Do not** enable `torch.use_deterministic_algorithms(True)` or
force cuDNN determinism — some conv ops lack deterministic kernels and would raise,
and the plan already relies on multi-seed to handle residual GPU nondeterminism.

### O4 — (doc note, no code change needed)
`discovery_auc` is normalized by the final unique-state count, so it isn't directly
comparable across runs with different coverage. `compute_metrics.py` already
reports `unique_states` alongside it — just make sure any writeup reports them
together, never AUC alone. Add a one-line clarifying comment near the AUC
computation.

---

## 4. Small improvements & repo cleanup

### C1 — Stop tracking generated output artifacts
These are run outputs and should not be in version control:
`suite_summary.csv`, `suite_summary.json`, `suite_manifest.json`,
`probe_results.csv`, `probe_manifest.json`. Move genuine legacy reference data you
want to keep into `legacy/` (see B4 for `suite_summary.csv`); `git rm --cached`
the rest. Keep the *scripts* (`probe_games.py`, etc.), just not their outputs.

### C2 — Mark legacy API-path scripts
`run_suite.py`, `summarize_runs.py`, and `probe_games.py` are the old API path /
one-time game-selection tooling, superseded by
`run_local.py` + `compute_metrics.py` + `summarize_overnight.py`. Keep them for
provenance, but add a one-line banner at the top of each module docstring, e.g.:
`LEGACY (API path): superseded by run_local.py + compute_metrics.py. Kept for provenance.`
After O3, `summarize_runs.py` shares the canonicalizer, so it stays functional.

### C3 — Tidy stray dead code in `action.py`
Remove the commented-out `add_scalar` / `logger.info` lines that are clearly
abandoned (several `# self.writer.add_scalar(...)` and one commented
`# self.logger.info(...)`). Resolve the `# TODO: Update this to a smaller value?`
next to `self.train_frequency = 5` by either deleting the TODO or replacing it
with a short note that 5 is intentional. **Do not change the value of
`train_frequency`** — that would alter agent behavior and confound results.

### C4 — (investigate + document, low priority) Local-engine dependency capture
`run_local.py` imports `arc_agi` / `arcengine`, which are not in
`requirements.txt`. Confirm where they come from (likely the submodule's
`uv sync --all-extras` in the Makefile `install` target). If so, add a one-line
note in the README that the local engine is provided by the submodule extras. If
they are a separate package, pin them. Do not guess — verify from the Makefile /
submodule before writing anything.

### C5 — `.gitignore` additions
Append entries so future outputs stay untracked:
```
local_suite.csv
overnight_*.manifest
overnight_*_summary.*
overnight.log
metrics.json
suite_summary.csv
suite_summary.json
suite_manifest.json
probe_results.csv
probe_manifest.json
```

### O6 — (OPTIONAL, only if straightforward and behavior-preserving) Buffer memory
The training buffer stores each state as the one-hot `bool` tensor
(16×64×64 ≈ 64 KB/experience). Storing the uint8 index frame (64×64) and
one-hot-ing at train time would cut buffer RAM ~16×. This touches the hot path
(`_compute_experience_hash`, the stored `state`, and the `torch.from_numpy(...)`
reconstruction in `_train_action_model`), so **only do this if you can guarantee
identical behavior** — the reconstructed one-hot tensor fed to the model must be
bit-identical to today's. Verify with a same-seed reproducibility check (below).
If there's any doubt, **skip it** and just leave a `# NOTE:` comment describing
the optimization for later.

---

## 5. Out of scope — do NOT implement in this session

Explicitly leave these for later (they are the "learning" work I'm deferring):
- No checkpoint saving / warm-start / cross-game weight loading.
- No count-based novelty or untried-action bonus in the sampler.
- No forward-dynamics / next-frame predictor (Phase 2).
- No prediction-error curiosity / ICM / RND.
- No changes to `EVAL_RESET_ON_LEVEL` persistence semantics.
- No hyperparameter changes (learning rate, `train_frequency`, batch size,
  buffer capacity, entropy/confidence coefficients).

If you think a change edges into any of the above, stop and leave a note instead.

---

## 6. Verification (run after each logical change, and again at the end)

1. **Compiles:** `python -m py_compile custom_agents/action.py compute_metrics.py summarize_runs.py summarize_overnight.py run_local.py eval_common.py metrics_common.py` (skip files that don't exist yet).
2. **Smoke run** (if the engine is available):
   ```bash
   EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \
       uv run python run_local.py --game ls20
   uv run python compute_metrics.py runs/<ts>/ls20/transitions \
       --game ls20 --agent goose --seed 0
   ```
   Confirm it runs clean and writes `metrics.json`.
3. **Seed reproducibility:** run the same seed twice on `ls20` at a small cap and
   compare action sequences:
   ```bash
   uv run python inspect_corpus.py runs/<A>/ls20/transitions \
       --verify-seed runs/<B>/ls20/transitions
   ```
   Identical is expected; divergence *may* be GPU nondeterminism rather than a
   seeding bug (check frames at the divergence point) — do not treat a single
   divergence as a regression.
4. **O3 consistency:** re-score one `ft09` + one `ls20` corpus with both scorers
   and confirm `meaningful_change_rate` agrees within rounding.
5. **API path intact:** the `make action` path and `custom_agent.py` import chain
   should be unchanged.
6. **Clean tree:** `git status` shows no tracked run outputs; `suite_summary.csv`
   is moved/untracked; `local_suite.csv` is ignored.

---

## 7. Suggested commit sequence

1. `fix: remove per-step empty_cache; sync experience_hashes with buffer eviction; correct ACTION6 log index` (B1–B3)
2. `fix: resolve suite_summary.csv schema collision; rename local suite file; archive legacy API rows` (B4)
3. `fix: make weird-frame fallback reachable; add cuda seeding` (O1, O5)
4. `refactor: rename mislabeled confidence bonus (was "entropy"); no behavior change` (O2)
5. `refactor: unify indicator-cell canonicalizer into metrics_common` (O3)
6. `chore: mark legacy scripts, untrack outputs, tidy dead code, update .gitignore` (C1–C5)
7. (optional) `perf: store index frames in buffer, one-hot at train time` (O6, only if verified identical)

---

## 8. Optional final step

Once the above is green, write a short `CLAUDE.md` at the repo root capturing the
project conventions so future sessions have context: the `eval_common` contract
and env vars, the ft09/ls20 focus, the three-stage pipeline, the (now unified)
metric definitions, and the local-vs-API distinction. Keep it concise. (This is
project memory, not a "learning" feature — it's fine to include.)
