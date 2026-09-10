# ARC3-solution — Team B (StochasticGoose track)

Team B's fork of the StochasticGoose agent for the ARC-AGI-3 capstone (ARC Prize
2026). Clone it, drop your agent in, run the frozen benchmark, compare.

See `CLAUDE.md` for repo conventions, focus games, metric definitions, and known
measurement caveats.

---

## 1. Quickstart

Prereqs: WSL2 + Ubuntu, an NVIDIA GPU with the Windows driver, `uv`, and `make`.

```bash
git clone --recurse-submodules <your-fork-url> ARC3-solution
cd ARC3-solution
git submodule update --init --recursive   # if you forgot --recurse-submodules
make install
uv pip install arc-agi                    # the local engine; not on PyPI mirrors
make check                                # verifies deps, agents, isolation
make bench-fg SUITE=smoke                 # ~30 s end-to-end proof it works
```

Put an ARC API key in `ARC-AGI-3-Agents/.env` (needed for the hosted API and for
the local engine's one-time game download). Copy `.env.example` (dot, not hyphen).

`make help` lists everything.

## 2. Adding your own agent

Three steps. No file outside `custom_agents/` needs to change.

```bash
cp custom_agents/TEMPLATE.py custom_agents/my_agent.py
```

`TEMPLATE.py` is a working random agent with the whole contract already wired up,
and its docstring documents the surface the runner drives. Then add one line to
`REGISTRY` in `custom_agents/__init__.py`:

```python
REGISTRY = {
    "goose":  ("action", "Action"),
    "random": ("random_agent", "RandomAgent"),
    "mine":   ("my_agent", "MyAgent"),        # <- yours
}
```

and run it:

```bash
make check                                 # confirms it imports and has the surface
uv run python run_local.py --game ft09 --agent mine
make bench SUITE=standard AGENT=mine
```

Seeding, the action cap, the transition corpus, metrics, the benchmark suites and
the comparison table all work for your agent the moment it is registered, because
none of them know anything agent-specific — they go through `eval_common.py` and
the corpus.

**The one hard rule:** log every transition, before any filtering of your own. All
metrics are computed from the corpus, so an agent that filters what it logs is
scoring itself on a different dataset. `inspect_corpus.py` is the schema authority
— if it loads your shards, you are in contract.

## 3. The frozen benchmark

`benchmark.py` is the single source of truth for what "the benchmark" means:
which games, which seeds, how many actions. Two people who run the same suite
have provably run the same thing, so their numbers can be put side by side.

```bash
make suites                                # list them
make bench SUITE=standard AGENT=goose      # backgrounded; hours
make bench SUITE=standard AGENT=mine
make compare M1=<goose manifest> M2=<your manifest>
```

| Suite | What | Cost |
|---|---|---|
| `smoke` | ft09, 1 seed, 2k actions | ~30 s — does it run at all |
| `quick` | 3 games × 2 seeds × 20k | ~15 min — for iterating, **not for reporting** |
| `standard` | 6 games × 3 seeds × 100k | ~3.8 h — **report this one** |
| `full` | all 25 games × 1 seed × 100k | ~5.3 h — breadth, not statistical power |

`standard` is ft09, ar25, cd82, lp85 (the four games that reach level 2, so there
is room to show improvement), ls20 (the null contrast — it completes nothing, so
any "improvement" there is a measurement artefact), and dc22 (mid-novelty, has
semester-1 medians for continuity).

**Changing a suite invalidates every comparison already made with it.** Add a new
suite instead; suites are versioned and the version is in the manifest filename.

`make compare` refuses to compare manifests that did not run the same games and
seeds — that mismatch is the exact mistake this setup exists to prevent.

### Reading the comparison

Never read one exploration metric alone. A high `meaningful` change rate only
means the frame keeps changing; an agent jiggling a decorative animation scores
1.00. Judge by `levels` and `uniq/act`, with `redundancy` and `act/s` as the cost
side, and always report `k/n seeds reached` next to any level median.

**Runs are not reproducible seed-for-seed.** Seeding is correct — two runs with
the same `EVAL_SEED` are identical for ~300 actions — but once training starts,
nondeterministic CUDA kernels diverge the weights and the trajectories part
(measured: level-up at 1110 vs 1423 on the same seed). `EVAL_SEED` fixes the
initial conditions, not the trajectory. Compare distributions across seeds; never
claim a byte-identical comparison between two arms.

## 4. Repository layout

```
ARC3-solution/
├── ARC-AGI-3-Agents/          # git SUBMODULE (arcprize harness) - do not edit
├── custom_agents/             # ALL AGENTS LIVE HERE
│   ├── __init__.py            #   the REGISTRY - add your agent here
│   ├── TEMPLATE.py            #   copy this to start a new agent
│   ├── action.py              #   StochasticGoose (the "brain")
│   ├── random_agent.py        #   matched uniform-random floor (uplift baseline)
│   └── view_utils.py          #   action-probability heatmap rendering
├── benchmark.py               # THE FROZEN TEST SET (suites: smoke/quick/standard/full)
├── check_repo.py              # health check: isolation, registry, deps, suites
├── compare.py                 # side-by-side table from two+ bench manifests
├── eval_common.py             # shared evaluation contract (seeds, caps, corpus)
├── run_local.py               # run one agent vs the LOCAL engine
├── run_curriculum.py          # several games in a row, ONE persistent brain
├── compute_metrics.py         # score a run's transition corpus
├── metrics_common.py          # shared indicator-cell canonicalizer
├── analyze_curves.py          # levels-vs-budget curves, AULC, RHAE
├── summarize_overnight.py     # aggregate a sweep into one report
├── inspect_corpus.py          # corpus schema validator (contract authority)
├── sweep.sh                   # benchmark + ad-hoc sweep orchestrator
├── utils.py                   # experiment-directory + logging helpers
├── llm_track/                 # Matt's LLM work - ISOLATED, see section 8
├── legacy/                    # archived API-path scripts + results
├── results/                   # ALL run output (gitignored)
│   ├── runs/<ts>/<game>/      #   per-run trees (corpus, tensorboard, metrics)
│   ├── sweeps/                #   manifests, summaries, logs
│   └── local_suite.csv        #   append-only per-run metric table
└── environment_files/         # locally cached game code (gitignored)
```

## 5. The evaluation contract (`eval_common.py`)

Every agent imports these so the protocol cannot drift.

| Variable | Meaning | Default |
|---|---|---|
| `EVAL_SEED` | base seed; each game adds a stable offset | time-based (not reproducible) |
| `EVAL_MAX_ACTIONS` | hard per-game action cap (0 = unlimited) | unlimited |
| `EVAL_LOG_METRICS` | TensorBoard scalars on/off | on |
| `EVAL_LOG_TRANSITIONS` | write the transition corpus | on |
| `EVAL_SAVE_VIS` | expensive PNG heatmaps | off |
| `EVAL_RESET_ON_LEVEL` | reset model+optimizer+buffer at each level (StochasticGoose only) | on |
| `EVAL_RESULTS_DIR` | root for all output | `results` |

Always run with `PYTHONHASHSEED=0`.

A run writes `results/runs/<timestamp>/<game>/`: `transitions/` (the `.npz`
corpus), `run_config.json` (exact configuration), `tensorboard/`, and after
scoring, `metrics.json`.

The **unified action index** names every action with one integer, identically in
your agent, the corpus, and the metrics: `0-4` = ACTION1–5, `5 + (64*y + x)` =
ACTION6 clicking column x, row y.

## 6. Running one game

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ft09 --agent goose
```

`run_local.py` runs the game in-process via the `arc-agi` engine, removing the
~50 ms/action HTTP round trip: ~120 act/s versus ~15 on the API. Flags:
`--offline` (airgapped; needs a cached game), `--render terminal`.

It builds a minimal in-memory `agents` package so it never triggers the harness's
fragile package init — which is why no submodule patch is required, and why agent
modules are imported lazily through the registry.

**Hosted API path** (unchanged): `make action`, or
`uv run ARC-AGI-3-Agents/main.py --agent=action --game=ft09`. It needs two small
edits to the harness submodule, applied once after cloning (the local path does
not need them):

```bash
git -C ARC-AGI-3-Agents apply ../harness_patches.patch
```
**Do not mix API and local numbers in one comparison** — the game seed differs, so
the two play different level instances.

**Cross-game curriculum** (one persistent brain across a game list, for transfer):

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=200000 PYTHONHASHSEED=0 \
    uv run python run_curriculum.py --games ft09,dc22,ls20
```

## 7. Scoring, sweeps and analysis

`compute_metrics.py` reads a corpus and writes `metrics.json`: level completions
and the action index of each level-up, unique canonical states with per-action
coverage and late-run novelty, meaningful (decorative-corrected) change rate,
redundancy, early-vs-late action entropy, timing, and per-1000-action series.

```bash
uv run python compute_metrics.py results/runs/<ts>/ft09/transitions \
    --game ft09 --agent goose --seed 0 --suite results/local_suite.csv
```

Use `unique_states_per_action` for coverage. `discovery_auc` is normalized by
final unique count and measures curve *shape* only — it ranks a 145-state run
above a 181k-state one, so never report it alone.

For work you are **not** going to report, `sweep.sh` also runs ad-hoc sweeps with
your own games/seeds (`make sweep`, `make long`, `make random`). Anything you
intend to compare against a teammate must go through `make bench`.

Every sweep calls `analyze_curves.py` for levels-vs-budget on a log axis, AULC per
(game, arm), and actions-to-level-k with censoring counts.

## 8. LLM track — and why it cannot affect your baselines

`llm_track/` is Matt's LLM work (see `295B-llm-track-plan.md`). It is isolated by
three mechanisms, so you can ignore it entirely:

1. **No baseline file imports it.** `make check` fails if one ever does. The
   dependency arrow points one way: `llm_track/` reads the corpus and the
   canonicalizer, and nothing reads `llm_track/`.
2. **Separate virtualenv.** Serving runs in `.venv-llm` (vLLM + its own torch
   2.13/cu130) so it can never bump the baseline's pinned torch 2.8.0.
3. **Offline only.** Nothing in the package is on any agent's per-action path.

When LLM features do reach the agent, they will be behind `EVAL_LLM_*` flags that
default to off and are recorded in `run_config.json`, so a baseline run is a
baseline run.

## 9. Baseline comparison notes

The random agent samples uniformly over the *same* masked 5 + 64×64 combined
action space StochasticGoose samples from, so ACTION6 contributes all 4096 click
coordinates individually. That is what makes it a matched floor for a click-heavy
agent; uniform over `{ACTION1..ACTION6}` would be a much stronger prior. Its
uplift ratios are what make change rate, redundancy and coverage comparable
across heterogeneous games.

**Every agent's corpus must be scored by *this* `compute_metrics.py`**, not by its
own tooling, or the canonicalizer differs and the comparison is meaningless. Since
the scorer is corpus-only, that reduces to emitting the `.npz` schema
`inspect_corpus.py` validates.
