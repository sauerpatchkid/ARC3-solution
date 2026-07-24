# ARC3-solution — Team B (StochasticGoose track)

Team B's fork of the StochasticGoose agent for the ARC-AGI-3 capstone (ARC Prize
2026). This README covers setup, the shared evaluation contract, running the
agent locally or against the hosted API, scoring runs, and the overnight sweep.

See `CLAUDE.md` for repo conventions, the focus games, and the metric definitions.

---

## 1. Repository layout

```
ARC3-solution/
├── ARC-AGI-3-Agents/          # git SUBMODULE (arcprize harness) - do not edit
├── custom_agents/
│   ├── action.py              # StochasticGoose agent (the "brain")
│   └── view_utils.py          # action-probability heatmap rendering
├── eval_common.py             # shared evaluation contract (seeds, caps, corpus)
├── utils.py                   # experiment-directory + logging helpers
├── run_local.py               # run the agent vs the LOCAL engine (one game)
├── run_curriculum.py          # play several games in a row, ONE persistent brain
├── compute_metrics.py         # score a run's transition corpus
├── metrics_common.py          # shared indicator-cell canonicalizer (both scorers)
├── summarize_overnight.py     # aggregate a sweep into one report
├── sweep.sh                   # unified overnight sweep orchestrator
├── inspect_corpus.py          # corpus schema validator (contract authority)
├── legacy/                    # archived API-path scripts + results
│   ├── suite_summary_api.csv  # 50-column per-run table from the API path
│   └── run_suite.py, summarize_runs.py, probe_games.py
├── runs/                      # run outputs (gitignored)
└── environment_files/         # locally cached game code (gitignored)
```

## 2. Setup

Prereqs: WSL2 + Ubuntu, an NVIDIA GPU with the Windows driver, `uv`, and `make`.

```bash
git clone --recurse-submodules <your-fork-url> ARC3-solution
cd ARC3-solution
# if you forgot --recurse-submodules:
git submodule update --init --recursive
make install                      # uv sync -> Python 3.12 venv at .venv
```

Put an ARC API key in `ARC-AGI-3-Agents/.env` (needed for the hosted API and for
the local engine's one-time game download). Copy `.env.example` (dot, not hyphen).

## 3. The evaluation contract (`eval_common.py`)

Every agent in the Team B comparison imports these so the protocol can't drift.
Behaviour is controlled by environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `EVAL_SEED` | base seed; each game adds a stable offset | time-based (not reproducible) |
| `EVAL_MAX_ACTIONS` | hard per-game action cap (0 = unlimited) | unlimited |
| `EVAL_LOG_METRICS` | TensorBoard scalars on/off | on |
| `EVAL_LOG_TRANSITIONS` | write the transition corpus | on |
| `EVAL_SAVE_VIS` | expensive PNG heatmaps | off |
| `EVAL_RESET_ON_LEVEL` | reset model+optimizer+buffer at each level (StochasticGoose only) | on |

Always run with `PYTHONHASHSEED=0` for reproducibility.

A run writes to `runs/<timestamp>/<game>/`: `transitions/` (the `.npz` corpus),
`run_config.json` (exact configuration), and `tensorboard/`.

## 4. Running the agent

### 4a. Local engine (fast, recommended for dev)
`run_local.py` runs the game **in-process** via the `arc-agi` engine, removing the
~50 ms/action HTTP round trip. Throughput goes from ~15 act/s (API) to ~120 act/s
(local, compute-bound on an RTX 5090 after the torch.compile / xxhash /
batched-GPU-transfer optimizations).

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ft09
```

Flags: `--offline` (fully airgapped; needs a previously cached game), `--render
terminal`. Note: `run_local.py` builds a minimal in-memory `agents` package so it
does not trigger the harness's fragile package init (LangGraph/Pillow); this is
why no submodule patch is required. The local engine requires the `arc-agi`
package (which provides `arcengine`); these are not in `requirements.txt` and
must be installed separately (`uv pip install arc-agi`).

### 4b. Hosted API (original path, unchanged)
```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run ARC-AGI-3-Agents/main.py --agent=action --game=ft09
```

**Do not mix API and local numbers in one comparison** — the game seed differs
(the server picks it; local sets it from `EVAL_SEED`), so the two play different
level instances. Keep all agents in a comparison on the same path.

### 4c. Cross-game curriculum (persistent brain)
`run_curriculum.py` plays a list of games back-to-back with **one** agent brain —
model, optimizer, and experience buffer carry across every game boundary — so
learning on earlier games can transfer to later ones. It reuses `run_local.py`'s
engine glue and scores each leg in-process with `compute_metrics.py`.

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=200000 PYTHONHASHSEED=0 \
    uv run python run_curriculum.py --games ft09,dc22,ls20
```

Edit the `GAMES` list at the top of the file or pass `--games`. It writes one
`runs/<ts>/` tree (a `<game>/` subdir per leg) plus `curriculum_summary.{md,csv}`.
The transfer signal is `first_levelup_action` falling down the sequence — read it
against the same games played cold (the solo sweeps), since order/difficulty
confound the raw curve.

## 5. Scoring a run

`compute_metrics.py` reads a run's corpus and reports the Team B metric set:
level completions (+ action index of each level-up), unique canonical states and
discovery-curve AUC, meaningful (decorative-corrected) change rate, redundancy,
early-vs-late action entropy, and timing/throughput. It writes `metrics.json`
next to the corpus and can append a row to a shared CSV (e.g. `local_suite.csv`).

```bash
uv run python compute_metrics.py runs/<ts>/ft09/transitions \
    --game ft09 --agent goose --seed 0 --suite local_suite.csv
```

## 6. Overnight sweep

`sweep.sh` runs a games × seeds × reset-arms sweep, scores each run, and
aggregates everything into one report. Configure it with the CONFIG block at the
top or by overriding `GAMES` / `SEEDS` / `CAP` on the command line. A game tagged
`:both` runs both reset arms (the persistence ablation); untagged games run
reset-on only.

```bash
# default sweep, backgrounded for a real overnight:
nohup bash sweep.sh > sweep.log 2>&1 &

# a quicker characterization sweep via env overrides:
GAMES="ka59 tn36 r11l wa30" SEEDS="0 1 2" CAP=50000 bash sweep.sh
```

It prints an ETA and per-run summaries, appends rows to `local_suite.csv`, and
at the end calls `summarize_overnight.py` to produce
`sweep_<stamp>_summary.{md,csv}` — aggregated per (game, arm) with
actions-to-each-level and a persistence-ablation verdict.

## 7. Baseline comparison

Three baselines share the contract and metric set: **random**, **Blind Squirrel**,
and **StochasticGoose**. Run each locally with the same `EVAL_SEED` set so they
face identical game instances, score them all with `compute_metrics.py`, and
compare via `local_suite.csv`. A corpus is valid iff `inspect_corpus.py` loads
it without error.
