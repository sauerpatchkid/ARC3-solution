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

## Running

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

Overnight sweep (games × seeds × reset-arms → aggregate summary):
```bash
make sweep
```

Long-horizon probe (ft09 + tu93, 1 seed, 1M actions each, ~2h/game):

```bash
make long
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

- **ft09** — learnable, reliable level completions. Test both reset arms.
- **ls20** — null contrast, completes nothing. Exploration-only.
- **tu93** — the only other game that has completed a level (1 of 4 seeds
  reached L2, at ~126k actions). Pair with ft09 for long-horizon runs.

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
- Legacy API-path outputs (`suite_summary_api.csv`) are archived in `legacy/`.
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

`discovery_auc` is normalized by final unique-state count — always report
`unique_states` alongside it.
