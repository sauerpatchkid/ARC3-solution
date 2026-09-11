# CLAUDE.md — StochasticGoose (Team B, ARC-AGI-3)

## What this repo is

ARC-AGI-3 capstone agent. Three stages: the **agent** (`custom_agents/action.py`,
CNN change predictor), the **corpus logger** (`TransitionLogger` in
`eval_common.py`, writes `.npz` shards), and the **scorers** (`compute_metrics.py`
for the local pipeline, `legacy/summarize_runs.py` for the legacy API path — both
share the indicator-cell canonicalizer from `metrics_common.py`).

Two multi-game drivers sit on top of `run_local.py`: `run_curriculum.py` (ONE
persistent brain across a game list, for cross-game transfer) and `sweep.sh`
(games × seeds × reset-arms, each a fresh process, for baseline tables).

**It is a shared repo.** Teammates clone it and add their own agents, so three
things are load-bearing and must not be casually changed:

- `custom_agents/__init__.py` — the agent REGISTRY. Every agent lives in
  `custom_agents/` and is one line here; `TEMPLATE.py` is the starting point.
  Nothing else in the repo is agent-specific.
- `benchmark.py` — the FROZEN test set (`smoke`/`quick`/`standard`/`full`).
  Changing a suite invalidates every comparison already made with it; add a new
  suite instead. `make bench SUITE=x AGENT=y` runs one; `make compare` puts two
  side by side and refuses mismatched suites.
- `check_repo.py` (`make check`) — enforces that no baseline file imports
  `llm_track` and no baseline tooling runs it, that every registered agent
  imports and has the runner's surface, and that every third-party import is
  declared. Run it before pushing.

## Running

The frozen benchmark (what goes in any comparison):
```bash
make suites                              # smoke / quick / standard / full
make bench SUITE=standard AGENT=goose    # backgrounded, ~3.8 h
make compare M1=<manifest> M2=<manifest>
```

Local engine (fast path, ~120 act/s):
```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ls20
uv run python compute_metrics.py results/runs/<ts>/ls20/transitions \
    --game ls20 --agent goose --seed 0 --suite results/local_suite.csv
```

Curriculum (persistent brain across several games, for transfer):
```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=200000 PYTHONHASHSEED=0 \
    uv run python run_curriculum.py --games ft09,dc22,ls20
```

Ad-hoc sweep — your own games/seeds, NOT comparable between people
(`DRY_RUN=1` prints the plan and ETA without running):
```bash
make sweep
```

Long-horizon probe (ft09 + tu93, 1 seed, 2M actions each, ~4h/game):

```bash
make long
```

Matched random-policy floor (needed to turn any game-dependent metric into an
uplift ratio) and the curve/AULC/RHAE analysis:

```bash
make random
make curves MANIFEST=results/sweeps/sweep_<stamp>.manifest
```

API path (unchanged, slower): `make action`

## eval_common contract (env vars)

| Variable | Default | Notes |
|---|---|---|
| `EVAL_SEED` | time-based | base seed; each game adds `stable_game_offset(game_id)` |
| `EVAL_MAX_ACTIONS` | unlimited | per-game action cap (0 = unlimited) |
| `EVAL_LOG_METRICS` | on | TensorBoard scalars |
| `EVAL_LOG_TRANSITIONS` | on | `.npz` transition corpus |
| `EVAL_SAVE_VIS` | off | expensive PNG heatmaps |
| `EVAL_RESET_ON_LEVEL` | on | reset model/optimizer/buffer at level boundary |
| `EVAL_RESULTS_DIR` | `results` | root for ALL run output |

Always run with `PYTHONHASHSEED=0`.

## Focus games

Updated after the 25-game sweep (`results/sweeps/sweep_20260727_225828_summary.md`):
**12 of 25 games complete at least one level**, not two. At 100k actions, 1 seed:
ar25, cd82, ft09, lp85 reach L2; cn04, m0r0, r11l, sk48, sp80, tr87, vc33 reach
L1; tu93 reached 0 in that sweep.

- **ft09** — learnable, reliable level completions. Test both reset arms.
  Caveat: 100% ACTION6 clicks and a 99.1% raw change rate, so the change head's
  label is nearly constant here and does not gate exploration the way it does
  elsewhere.
- **ls20** — null contrast, completes nothing. Exploration-only.
- **ar25 / cd82 / lp85** — the other L2 games. Prefer these over tu93 for
  anchor-hungry work; they have no semester-1 medians yet, so they need a
  baseline sweep before use in a comparison table.
- **tu93** — 1 of 4 seeds reached L2 at ~126k actions in the earlier sweep but 0
  in the 25-game sweep. Weak; use only for long-horizon runs paired with ft09.

## LLM track (`llm_track/`) — isolated from the baseline

Matt's LLM work lives entirely in `llm_track/`: code, commands
(`make -C llm_track help`), serving venv (`.venv-llm`), design docs
(`llm_track/docs/`), and status and results (`llm_track/README.md`). Nothing in
the baseline imports or runs it. LLM-track status and results are recorded only
in `llm_track/README.md`, so LLM work never needs to change this shared file.

## Key conventions

- ALL run output goes under `results/` (gitignored, override with
  `EVAL_RESULTS_DIR`). Nothing writes results to the repo root:
    - `results/runs/<ts>/<game>/` — corpus, `run_config.json`, tensorboard,
      `metrics.json`
    - `results/sweeps/` — manifests, `sweep_<stamp>_summary.{md,csv}`, logs
    - `results/local_suite.csv` — append-only per-run metric table
    - `results/curriculum_suite.csv` — same, for `run_curriculum.py`
    - `results/recordings/` — API-path replays (`RECORDINGS_DIR` in the
      submodule's `.env`)
- Legacy API-path outputs (`suite_summary_api.csv`) and the superseded
  `sweep_all25.sh` (now `make bench SUITE=full`) are archived in `legacy/`.
- ALL agents live in `custom_agents/` and are registered in its `__init__.py`
  (`random_agent.py` moved there from the repo root).
- `llm_track/` must stay a leaf: it may import from the baseline, nothing in the
  baseline may import it, and baseline tooling (root Makefile, `sweep.sh`) must
  never invoke it or `.venv-llm`. `make check` enforces both. Its commands live
  in `llm_track/Makefile`, its docs and results in `llm_track/README.md`.
- The `arc-agi` package (provides `arcengine`) is needed for the local engine
  but not declared in `requirements.txt` — install separately.
- Do NOT change `EVAL_RESET_ON_LEVEL` semantics or any hyperparameters
  (learning rate, `train_frequency`, batch size, buffer capacity, confidence
  coefficients) without explicit approval — they'd confound ablation results.

## Metric definitions

All metrics use the unified indicator-cell canonicalizer (`metrics_common.py`):
fixed tickers (cell changing in >=95% of transitions) plus rotating tickers
(compact cell set covering >=95% of tiny <=2-cell transitions, if those make up
>=30% of the run). Both pipelines now produce comparable `meaningful_change_rate`
and `redundancy`.

**Never report an exploration metric alone.** A high `meaningful_change_rate`
only means the frame keeps changing — an agent jiggling a decorative animation
scores 100%. Read change rate, redundancy and coverage together.

- `unique_states_per_action` — the coverage number. Use this, not `discovery_auc`.
- `discovery_auc` — DEPRECATED for cross-run comparison. Normalized by FINAL
  unique-state count, so it measures curve shape only. On the 4-seed baseline it
  ranks tu93 (145 unique states) at 0.95 and ft09 (181k states) at 0.49 — exactly
  backwards. Never report it without `unique_states_per_action` beside it.
- `novelty_late_per_1k` — new canonical states per 1k actions over the final 10%.
  The stall detector, and a leading indicator: it flattens thousands of actions
  before the level counter confirms the agent is stuck. Baseline medians: tu93
  0.0 (dead), dc22 2.4, ls20 7.4, g50t 9.9, ft09 955 (still discovering).
- `series` in `metrics.json` — per-1000-action novelty / meaningful / redundancy.
  Run-level scalars average away the collapse on long runs; plot the series.
- Levels vs. budget, AULC, RHAE — `analyze_curves.py` (post-processing over the
  `metrics.json` files, so it re-runs on historical runs for free). AULC is
  defined in that file's docstring; always report it next to its `T_max`.
- Uplift — every game-dependent metric above needs the matched random floor
  (`make random`) to be comparable across games. Random is ~2700 act/s, so a
  full 5-seed sweep is cheap.

## Known measurement caveats (found 2026-09-04, not yet fixed in the pipeline)

Two facts that affect how existing numbers should be read. Neither is fixed in
`metrics_common.py` / `compute_metrics.py` — changing the canonicalizer moves
every published baseline number, so it is an advisor decision, not a side effect.

- **The rotating-ticker detector misses most tickers.** Its unit is the whole
  transition (`<=2` cells changed *in total*), so a ticker is only visible when
  it ticks *alone*. On ft09 row 63 changes in 85.4% of transitions, always
  exactly 2 cells, sweeping the full width — a progress bar — but it always
  co-occurs with the 36-cell tile toggle, so `tiny_frac` is 0.000 and the
  branch never fires. ft09, ls20, ar25, lp85, dc22 and g50t are all scored with
  an EMPTY decorative mask. Masking ft09's row 63 cuts unique canonical states
  from 81,916 to 59,999 over 96,411 actions — a **1.4x inflation** in
  `unique_states_per_action`, and `novelty_late_per_1k` is inflated the same way.
  `llm_track/tickers.py` has a fixed version (same thresholds, unit changed from
  transition to connected component); it reproduces the old detector exactly on
  the two games where the old one fired (cd82 60 cells, tu93 60 cells).
- **Runs are not reproducible past the first training step.** Seeding is
  correct — two runs with the same `EVAL_SEED` are identical for ~300 actions —
  but once `_train_action_model` starts, nondeterministic CUDA kernels diverge
  the weights and the action sequences part company (measured: first mismatch at
  action 350; level-up at 1110 vs 1423 on the same seed). `EVAL_SEED` fixes the
  initial conditions, not the trajectory. **Never claim a byte-identical
  comparison between two arms**; compare distributions across seeds.

Censoring discipline: most runs never reach level k. Always report
"k/n seeds reached" next to any actions-to-level median (both
`summarize_overnight.py` and `analyze_curves.py` do).
