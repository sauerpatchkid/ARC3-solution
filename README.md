# ARC3-solution — Team B (StochasticGoose track)

Team B's fork of the StochasticGoose agent for the ARC-AGI-3 capstone (ARC Prize
2026). Clone it, drop your agent in, run the frozen benchmark, compare.
Conventions for contributors (and for Claude Code) are in `CLAUDE.md`.

## The project in three parts

```
 1. GOOSE UPGRADES (no LLM)            2. LLM HISTORY                     3. GLASSBOX (current)
 Goose, semester 1      54 levels      Judge, heuristics   NO-GO          Goose + an LLM that writes
 + novelty label        79 levels      Coach               NO-GO          small rules, each checked
 + map, bars, gated,                   Rulebook v1         FAIL 2 of 8    against every recorded move,
   attempt = mb_gated_att  112 levels  Rulebook v2         FAIL 2 of 8,   in separate stages that can
                                                           0.0% wrong     be tested one at a time
```

Levels are on all 25 public games × 3 seeds × 100,000 actions each.

| Part | What it is | Code | Write-ups |
|---|---|---|---|
| **1. Goose upgrades** | The exploring agent and the semester-2 improvements to it. Each upgrade is its own file and switch, off by default; `--agent mb_gated_att` is the adopted bundle and the reference (A0) for everything after it | `custom_agents/action.py` (Goose), `canon.py` (novelty label), `return_map.py`, `upgrades.py`, `presets.py` | [`docs/1-goose-upgrades/`](docs/1-goose-upgrades/) |
| **2. LLM history** | Earlier attempts to add an LLM: a judge, the Coach, Rulebook v1 and v2. What each tried, what its test showed, and why it stopped | frozen in `legacy/` | [`docs/2-llm-history/`](docs/2-llm-history/) |
| **3. Glassbox** | The current plan. Rulebook v2 fixed carrying rules between levels but got too few exact rules; Glassbox splits the pipeline into stages that each fix one problem | `custom_agents/gb/` (one file per stage; see its README) | [`docs/3-glassbox/`](docs/3-glassbox/) |

**[`docs/README.md`](docs/README.md) is the full map**: every step with its
result, and the problem each Glassbox stage fixes.

Everything new is **off by default**: an unflagged `--agent goose` run is the
semester-1 baseline, verified action for action. Each feature's hook lines in
`custom_agents/action.py` carry a tag (`# [return-map]`, `# [upgrades]`, and
later `# [gb]`); deleting a tag's lines (`sed -i '/\[return-map\]/d' ...`)
removes that feature.

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

No file outside `custom_agents/` needs to change.

```bash
cp custom_agents/TEMPLATE.py custom_agents/my_agent.py
```

`TEMPLATE.py` is a working random agent with the whole contract wired up; its
docstring documents the surface the runner drives. Add one line to `REGISTRY`
in `custom_agents/__init__.py`:

```python
REGISTRY = {
    "goose":        ("action", "Action"),
    "mb_gated_att": ("presets", "MbGatedAtt"),
    "random":       ("random_agent", "RandomAgent"),
    "mine":         ("my_agent", "MyAgent"),        # <- yours
}
```

and run it:

```bash
make check                                 # confirms it imports and has the surface
uv run python run_local.py --game ft09 --agent mine
make bench SUITE=standard AGENT=mine
```

Seeding, the action cap, the transition corpus, metrics, the benchmark suites
and the comparison table all work for your agent once it is registered, because
they go through `eval_common.py` and the corpus.

**The one hard rule:** log every transition, before any filtering of your own.
All metrics are computed from the corpus. `inspect_corpus.py` is the schema
authority: if it loads your shards, you are in contract.

## 3. The frozen benchmark

`benchmark.py` defines what "the benchmark" means: which games, which seeds, how
many actions. Two people who run the same suite have run the same thing.

```bash
make suites                                # list them
make bench SUITE=standard AGENT=mb_gated_att   # backgrounded; hours
make bench SUITE=standard AGENT=mine
make compare M1=<reference manifest> M2=<your manifest>
```

| Suite | What | Cost |
|---|---|---|
| `smoke` | ft09, 1 seed, 2k actions | ~30 s: does it run at all |
| `quick` | 3 games × 2 seeds × 20k | ~15 min: for iterating, **not for reporting** |
| `standard` | 6 games × 3 seeds × 100k | ~3.8 h: **report this one** |
| `full` | all 25 games × 1 seed × 100k | ~5.3 h: breadth, not statistical power |

**Changing a suite invalidates every comparison already made with it.** Add a
new suite instead. `make compare` refuses manifests that did not run the same
games and seeds.

**Reading a comparison.** Never read one exploration metric alone: a high
`meaningful` change rate only means the frame keeps changing. Judge by `levels`
and `uniq/act`, with `redundancy` and `act/s` as the cost side, and report
`k/n seeds reached` next to any level median. **Runs are not reproducible
seed-for-seed on the GPU** (identical for ~300 actions, then CUDA kernels
diverge), so compare distributions across seeds.

## 4. Running, scoring, comparing

```bash
EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ft09 --agent mb_gated_att     # ~120 act/s
uv run python compute_metrics.py results/runs/<ts>/ft09/transitions \
    --game ft09 --agent mb_gated_att --seed 0 --suite results/local_suite.csv
```

- `compute_metrics.py` writes `metrics.json`: levels and level-up times, unique
  canonical states per action, late-run novelty, meaningful change rate,
  redundancy, series. Use `unique_states_per_action` for coverage, never
  `discovery_auc` alone.
- `analyze_curves.py` gives levels against budget, AULC and censoring counts;
  every sweep calls it.
- `tools/paired_compare.py --base <manifest>:<arm> --new <manifest>:<arm> --rule adopt|dev|confirm`
  gives a verdict against a pre-registered rule.
- **Did my change alter the agent?** GPU runs can't answer that. Record 2,000
  moves on the CPU before and after and compare them move for move:
  `tools/cpu_check.py`.
- Ad-hoc sweeps you won't report: `make sweep`, `make long`, `make random`.
  Anything compared against a teammate goes through `make bench`.
- Cross-game curriculum (one brain across a game list):
  `uv run python run_curriculum.py --games ft09,dc22,ls20`.
- Hosted API path (unchanged, slower): `make action`. It needs
  `git -C ARC-AGI-3-Agents apply ../harness_patches.patch` once. Don't mix API
  and local numbers: the game seed differs.

## 5. The evaluation contract (`eval_common.py`)

| Variable | Meaning | Default |
|---|---|---|
| `EVAL_SEED` | base seed; each game adds a stable offset | time-based |
| `EVAL_MAX_ACTIONS` | per-game action cap (0 = unlimited) | unlimited |
| `EVAL_LOG_METRICS` / `EVAL_LOG_TRANSITIONS` / `EVAL_SAVE_VIS` | TensorBoard / the corpus / PNG heatmaps | on / on / off |
| `EVAL_RESET_ON_LEVEL` | reset model, optimizer and buffer at each level (Goose) | on |
| `EVAL_RESULTS_DIR` | root for all output | `results` |
| `EVAL_LABEL` | Goose's training label: `change` or `novel` (part 1, Plan B) | `change` |
| `EVAL_CANON_WARMUP` / `EVAL_CANON_REFRESH` | the online decoration mask: warm-up / recompute cadence | `200` / `250` |
| `EVAL_RETURN_MAP` | the return map (part 1, Option 1) | off |
| `EVAL_MAP_STALL` / `EVAL_MAP_CLICK_TRIES` / `EVAL_MAP_MAX_ROUTE` / `EVAL_MAP_RETURN` | map tuning | `200` / `20` / `100` / `1` |
| `EVAL_UPGRADES` | comma list from `bars`, `attempt`, `map_gated`, `deadclick` (part 1, upgrade screen) | unset |

`--agent mb_gated_att` fixes `EVAL_LABEL=novel EVAL_RETURN_MAP=1
EVAL_UPGRADES=bars,map_gated,attempt`. Always run with `PYTHONHASHSEED=0`.
Removed settings (`EVAL_MASK_TRIED`, upgrades `graded`, `map_objects`,
`map_diverse`) stop the run with a pointer to their git tag.

A run writes `results/runs/<timestamp>/<game>/`: `transitions/` (the `.npz`
corpus), `run_config.json`, `tensorboard/`, and after scoring `metrics.json`.
One row per move; a reset is an action but not a row, so `n_actions` (rows) is a
little smaller than `actions_taken` (what the cap counts). The unified action
index: `0-4` = ACTION1–5, `5 + 64*y + x` = ACTION6 at column x, row y.

## 6. Repository layout

```
ARC3-solution/
├── custom_agents/             # ALL AGENTS
│   ├── __init__.py            #   the agent REGISTRY - add yours here
│   ├── TEMPLATE.py            #   copy this to start a new agent
│   ├── action.py              #   [1] Goose (the CNN agent); features hook in on tagged lines
│   ├── canon.py               #   [1] screen fingerprints + per-level memory (novelty label)
│   ├── return_map.py          #   [1] the return map
│   ├── upgrades.py            #   [1] bars / attempt / map_gated / deadclick
│   ├── presets.py             #   [1] --agent mb_gated_att, the adopted bundle
│   ├── gridtools.py           #   shared 64x64 screen helpers (bars, blobs, objects)
│   ├── random_agent.py        #   matched uniform-random floor
│   ├── view_utils.py          #   heatmap rendering
│   └── gb/                    #   [3] Glassbox: one file per stage, core/ copied from Rulebook v2
├── run_local.py               # run one agent against the LOCAL engine
├── run_curriculum.py          # several games in a row, one persistent brain
├── eval_common.py             # the evaluation contract (seeds, caps, corpus)
├── compute_metrics.py         # score a run's corpus
├── metrics_common.py          # the shared decoration-cell canonicalizer
├── analyze_curves.py          # levels-vs-budget curves, AULC, RHAE
├── benchmark.py               # THE FROZEN TEST SET
├── compare.py                 # side-by-side table from bench manifests
├── sweep.sh                   # benchmark and ad-hoc sweep runner
├── summarize_overnight.py     # one report per sweep
├── inspect_corpus.py          # corpus schema validator
├── manifest.py                # the one reader for sweep manifests
├── tiers.py, registry.py      # game tiers (dev / seen / held-out) and the frozen-artifact registry
├── check_repo.py              # make check: isolation, registry, deps, suites
├── tools/                     # paired_compare.py (verdicts), cpu_check.py (did it change?), artifacts.py
├── experiments/glassbox/      # [3] glassbox.toml (every knob) and gb.mk (make gb-help)
├── experiments/llm/           # [3] the local LLM server and its environment
├── tests/                     # pytest
├── docs/                      # README.md (the map), 1-goose-upgrades/, 2-llm-history/, 3-glassbox/, reports/
├── artifacts/registry.json    # hashes of frozen results and artifacts
├── legacy/                    # FROZEN: earlier experiments and LLM attempts (see legacy/README.md)
├── ARC-AGI-3-Agents/          # git submodule (arcprize harness) - do not edit
└── results/                   # ALL run output (gitignored)
```

`[1]`, `[3]` mark which part of the project a file belongs to.

## 7. `legacy/`: archived, and why it cannot affect your baselines

`legacy/` holds finished work kept for the record (index:
`legacy/README.md`): the semester-1 API-path scripts, the semester-1 LLM track,
the launchers for the Goose upgrade experiments, the Coach, and Rulebook v1/v2.
You can ignore it. It is isolated by three mechanisms, all enforced by
`make check`:

1. **No live file imports it.** The arrow points one way: archived code may
   read the baseline, nothing live reads archived code. Glassbox *copied* what
   it reuses from Rulebook v2 into `custom_agents/gb/core/`.
2. **No baseline tooling runs it.** The root Makefile and `sweep.sh` never call
   into `legacy/` or the LLM's environment.
3. **Separate virtualenv.** The LLM server runs in `.venv-llm` (vLLM and its
   own torch), built by `experiments/llm/make_env.sh`, so it never touches the
   baseline's pinned torch.

## 8. Comparison notes

The random agent samples uniformly over the *same* masked 5 + 64×64 action
space Goose samples from, so it is a matched floor for a click-heavy agent.
Its uplift ratios make change rate, redundancy and coverage comparable across
games. **Every agent's corpus must be scored by this `compute_metrics.py`**, not
by its own tooling, or the canonicalizer differs and the comparison means
nothing.
