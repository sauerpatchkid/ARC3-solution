# ARC3-solution — Team B (StochasticGoose track)

Team B's fork of the StochasticGoose agent for the ARC-AGI-3 capstone (ARC Prize
2026). This README covers setup, the shared evaluation contract, running the
agent locally or against the hosted API, scoring runs, and the overnight sweep.

The companion `arc3_project_plan.md` holds the research plan and roadmap.

---

## 1. Repository layout

```
ARC3-solution/
├── ARC-AGI-3-Agents/          # git SUBMODULE (arcprize harness) - do not edit
├── custom_agents/
│   ├── action.py              # StochasticGoose agent (the "brain")
│   ├── utils.py, view_utils.py
├── eval_common.py             # shared evaluation contract (seeds, caps, corpus)
├── run_local.py               # NEW: run the agent against the LOCAL engine
├── compute_metrics.py         # NEW: score a run's transition corpus
├── run_overnight.sh           # NEW: overnight sweep orchestrator
├── summarize_overnight.py     # NEW: aggregate a sweep into one report
├── suite_summary.csv          # per-run metric rows (append-only)
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
~50 ms/action HTTP round trip. Throughput goes from ~15 act/s (API) to ~60 act/s
(local, compute-bound on an RTX 5090).

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ft09
```

Flags: `--offline` (fully airgapped; needs a previously cached game), `--render
terminal`. Note: `run_local.py` builds a minimal in-memory `agents` package so it
does not trigger the harness's fragile package init (LangGraph/Pillow); this is
why no submodule patch is required.

### 4b. Hosted API (original path, unchanged)
```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run ARC-AGI-3-Agents/main.py --agent=action --game=ft09
```

**Do not mix API and local numbers in one comparison** — the game seed differs
(the server picks it; local sets it from `EVAL_SEED`), so the two play different
level instances. Keep all agents in a comparison on the same path.

## 5. Scoring a run

`compute_metrics.py` reads a run's corpus and reports the Team B metric set:
level completions (+ action index of each level-up), unique canonical states and
discovery-curve AUC, meaningful (decorative-corrected) change rate, redundancy,
early-vs-late action entropy, and timing/throughput. It writes `metrics.json`
next to the corpus and can append a row to `suite_summary.csv`.

```bash
uv run python compute_metrics.py runs/<ts>/ft09/transitions \
    --game ft09 --agent goose --seed 0 --suite suite_summary.csv
```

## 6. Overnight sweep

`run_overnight.sh` runs a games x seeds x reset-arms sweep, scores each run, and
aggregates everything into one report. Edit the CONFIG block at the top to change
seeds/cap/games.

```bash
# foreground:
bash run_overnight.sh
# background (real overnight):
nohup bash run_overnight.sh > overnight.log 2>&1 &
```

It prints an ETA and per-run summaries, appends rows to `suite_summary.csv`, and
at the end calls `summarize_overnight.py` to produce
`overnight_<stamp>_summary.{md,csv}` — aggregated per (game, arm) with
actions-to-each-level and a persistence-ablation verdict.

## 7. Baseline comparison

Three baselines share the contract and metric set: **random**, **Blind Squirrel**,
and **StochasticGoose**. Run each locally with the same `EVAL_SEED` set so they
face identical game instances, score them all with `compute_metrics.py`, and
compare via `suite_summary.csv`. A corpus is valid iff `inspect_corpus.py` loads
it without error.
