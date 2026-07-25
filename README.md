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
├── analyze_curves.py          # levels-vs-budget curves, AULC, RHAE
├── random_agent.py            # matched uniform-random floor (uplift baseline)
├── metrics_common.py          # shared indicator-cell canonicalizer (both scorers)
├── summarize_overnight.py     # aggregate a sweep into one report
├── sweep.sh                   # unified overnight sweep orchestrator
├── inspect_corpus.py          # corpus schema validator (contract authority)
├── legacy/                    # archived API-path scripts + results
│   ├── suite_summary_api.csv  # 50-column per-run table from the API path
│   └── run_suite.py, summarize_runs.py, probe_games.py
├── results/                   # ALL run output (gitignored)
│   ├── runs/<ts>/<game>/      #   per-run trees (corpus, tensorboard, metrics)
│   ├── sweeps/                #   sweep manifests, summaries, logs
│   ├── recordings/            #   API-path replay recordings
│   └── local_suite.csv        #   append-only per-run metric table
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

A run writes to `results/runs/<timestamp>/<game>/`: `transitions/` (the `.npz`
corpus), `run_config.json` (exact configuration), and `tensorboard/`. Everything
any driver produces lands under `results/`; set `EVAL_RESULTS_DIR` to relocate
that root (e.g. onto a scratch disk).

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
`results/runs/<ts>/` tree (a `<game>/` subdir per leg) plus `curriculum_summary.{md,csv}`.
The transfer signal is `first_levelup_action` falling down the sequence — read it
against the same games played cold (the solo sweeps), since order/difficulty
confound the raw curve.

## 5. Scoring a run

`compute_metrics.py` reads a run's corpus and reports the Team B metric set:
level completions (+ action index of each level-up), unique canonical states with
per-action coverage and late-run novelty, meaningful (decorative-corrected) change
rate, redundancy, early-vs-late action entropy, timing/throughput, and per-1000-
action series for the exploration metrics. It writes `metrics.json` next to the
corpus and can append a row to a shared CSV (e.g. `results/local_suite.csv`).

Read the exploration metrics together, never alone — a high change rate only
means the frame keeps moving. Use `unique_states_per_action` for coverage;
`discovery_auc` is normalized by final unique count and measures curve shape
only (it ranks a 145-state run above a 181k-state one). `novelty_late_per_1k`
is the stall detector.

```bash
uv run python compute_metrics.py results/runs/<ts>/ft09/transitions \
    --game ft09 --agent goose --seed 0 --suite results/local_suite.csv
```

## 6. Overnight sweep

`sweep.sh` runs a games × seeds × reset-arms sweep, scores each run, and
aggregates everything into one report. Configure it with the CONFIG block at the
top or by overriding `GAMES` / `SEEDS` / `CAP` on the command line. A game tagged
`:both` runs both reset arms (the persistence ablation); untagged games run
reset-on only.

```bash
# default sweep, backgrounded for a real overnight:
make sweep        # == nohup bash sweep.sh > results/sweeps/sweep.log 2>&1 &

# a quicker characterization sweep via env overrides:
GAMES="ka59 tn36 r11l wa30" SEEDS="0 1 2" CAP=50000 bash sweep.sh
```

It prints an ETA and per-run summaries, appends rows to `results/local_suite.csv`,
and at the end calls `summarize_overnight.py` to produce
`results/sweeps/sweep_<stamp>_summary.{md,csv}` — aggregated per (game, arm) with
actions-to-each-level and a persistence-ablation verdict.

Long-horizon probe on the two games that actually complete levels (ft09, tu93),
one seed, 2M actions each — ~4 h/game on a 5090:

```bash
make long         # == GAMES="ft09 tu93" SEEDS="0" CAP=2000000 bash sweep.sh
```

Set `AGENT=random` for the matched random-policy floor (`make random`). It has
no model, so it runs at ~2700 act/s and a full 5-seed sweep costs minutes.

Each sweep also calls `analyze_curves.py`, which writes
`results/sweeps/curves_<stamp>.{md,csv,png}`: max-level-vs-action-budget on a
log axis (median across seeds, min–max band), AULC per (game, arm), and
actions-to-level-k with seed-censoring counts. Pass `--human-baselines` a JSON
of `{game: {level: human_actions}}` to add per-level RHAE.

## 7. Baseline comparison

Three baselines share the contract and metric set: **random** (`random_agent.py`,
`--agent random`), **Blind Squirrel**, and **StochasticGoose**. Run each locally
with the same `EVAL_SEED` so they face identical game instances, score them all
with `compute_metrics.py`, and compare via `results/local_suite.csv`. A corpus is
valid iff `inspect_corpus.py` loads it without error.

The random agent samples uniformly over the *same* masked 5 + 64×64 combined
action space StochasticGoose samples from, so ACTION6 contributes all 4096 click
coordinates individually. That is what makes it a matched floor for a click-heavy
agent; uniform over `{ACTION1..ACTION6}` would be a different and much stronger
prior. Its uplift ratios are what make change rate, redundancy and coverage
comparable across heterogeneous games.

**Cross-agent requirement:** every agent's corpus must be scored by *this*
`compute_metrics.py`, not by its own tooling, or the canonicalizer differs and the
comparison is meaningless. Since the scorer is corpus-only, that reduces to
emitting the `.npz` schema `inspect_corpus.py` validates.
