# ARC3-solution — Team B (StochasticGoose track)

Team B's fork of the StochasticGoose agent for the ARC-AGI-3 capstone (ARC Prize
2026). Clone it, drop your agent in, run the frozen benchmark, compare.

See `CLAUDE.md` for repo conventions, focus games, metric definitions, and known
measurement caveats.

## Semester 2 at a glance (September 2026)

Everything below is **off by default**: an unflagged run is the original
baseline, verified action-for-action. Plain-language summary for non-technical
readers: `docs/reports/Goose_Semester2_Progress.docx`.

| Change | Switch | Status | Details |
|---|---|---|---|
| **Novelty label.** Goose learns from "did this move reach a screen not seen before in this level?" instead of "did the screen change?", with decorations (blinking cells, progress bars) masked out before each screen is fingerprinted | `EVAL_LABEL=novel` | **Adopted.** 25 games × 3 seeds × 100k: 79 levels vs 54, 18 of 25 games reach a level vs 11, 23 paired wins / 1 loss | `docs/plans/plan-B-*.md` |
| Tried-action mask ("don't repeat yourself") | `EVAL_MASK_TRIED=1` | Dropped: helped nowhere, hurt games that need repeated presses | `docs/plans/plan-B-dev-sweep-results.md` |
| **Return map.** A map of the level (which move leads from which screen to which); walks back to untried places when stuck and after a game over | `EVAL_RETURN_MAP=1` | **Not adopted on its own** (adopted in the combination below). Passed its 8-game dev test (45 vs 32 levels) but not the 25-game Confirm after two seeds: helps games Goose was stuck on (tu93 level 5, vc33 level 4, first levels on bp35 and lf52), hurts games it already solved (ar25, tr87) | `docs/plans/option-1-*.md` |
| LLM advisor (Plan A) | — | Deferred | `docs/plans/plan-A-llm-advisor.md` |
| **Upgrade screen + 25-game confirm.** Nine candidate improvements screened in two rounds; the winner combines the map with progress bars masked, walking back only when it pays, and half credit within an attempt | `EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt` (with `EVAL_LABEL=novel`) | **Adopted.** All 25 games × 3 seeds × 100k: 112 levels vs 79 for the novelty label alone, 22 of 25 games reach a level vs 18, 31 paired wins / 8 losses, no game worse on every seed, no speed cost | `docs/plans/upgrade-*.md` |

The semester-1 LLM track is archived in `legacy/llm_track/` (section 8).

**Designed to be removable.** Each change lives in its own file
(`custom_agents/canon.py`, `custom_agents/return_map.py`) behind its own switch.
Every line the return map adds to `custom_agents/action.py` ends in
`# [return-map]`; `sed -i '/\[return-map\]/d' custom_agents/action.py` restores
the previous agent byte for byte. The upgrade screen works the same way with the tag
`[upgrades]` (`custom_agents/upgrades.py`, `experiments/upgrade_screen/`), and
can be removed independently of the map.

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
│   ├── canon.py               #   screen fingerprints + per-level memory (novelty label)
│   ├── return_map.py          #   the return map (EVAL_RETURN_MAP; not adopted)
│   ├── upgrades.py            #   upgrade-screen candidates (EVAL_UPGRADES; experimental)
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
├── docs/plans/                # semester-2 plans, pre-registered rules and results
├── docs/reports/              # plain-language reports (Word) + the script that builds them
├── tools/                     # label_diagnostic.py (Plan B step 1), paired_compare.py (verdicts)
├── experiments/upgrade_screen/ # the upgrade screen: runner, leaderboard, make targets
├── tests/                     # pytest: fingerprints, memory, sampler guard, return map
├── legacy/                    # ARCHIVED: API-path scripts + semester-1 LLM track, see section 8
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
| `EVAL_LABEL` | Plan B training label: `change` (frame changed) or `novel` (new canonical state this level) (StochasticGoose only) | `change` |
| `EVAL_MASK_TRIED` | Plan B: soft-mask actions already tried from the current canonical state (StochasticGoose only) | off |
| `EVAL_MASK_DECAY` / `EVAL_MASK_FLOOR` | per-try multiplier / minimum probability for that mask | `0.1` / `1e-4` |
| `EVAL_CANON_WARMUP` / `EVAL_CANON_REFRESH` | online indicator-cell mask: warm-up transitions / recompute cadence | `200` / `250` |
| `EVAL_RETURN_MAP` | the return map (StochasticGoose only; not adopted) | off |
| `EVAL_MAP_STALL` / `EVAL_MAP_CLICK_TRIES` / `EVAL_MAP_MAX_ROUTE` / `EVAL_MAP_RETURN` | map tuning: decisions without a new screen before routing / clicks before a click screen counts as tried / longest route / walk back after game over | `200` / `20` / `100` / `1` |
| `EVAL_UPGRADES` | comma list of upgrade-screen candidates: `bars`, `attempt`, `graded`, `deadclick`, `map_gated`, `map_objects`, `map_diverse` (experimental) | unset |

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

### Semester-2 experiment sweeps

All detached (they survive closing the terminal, not the machine sleeping), all
logging to `results/sweeps/planb_dev.log`, all resumable:

```bash
make planb-dev        # Plan B dev tier: arms A0-A3 on 6 games (done)
make planb-confirm    # Plan B Confirm: A0 vs A1 on all 25 games (done)
make map-dev          # return map dev test: A1 vs A4 on 8 games (done)
make map-confirm      # return map Confirm: A4 on all 25 games (stopped after 2 seeds)
make planb-status     # progress
make planb-pause      # stop after the current run finishes - loses nothing
make planb-stop       # stop now - loses only the run in progress
make map-confirm RESUME=results/sweeps/<manifest>   # pick up where it stopped
```

The upgrade screen has its own targets and results folder (`results/screen/`):
`make upgrade-screen`, `make upgrade-screen-status`, `make upgrade-screen-pause`,
`make upgrade-screen RESUME=<manifest.tsv>`. Parallel runs share the GPU: four
at once give about 1.15x the speed of one (measured), not 4x.

Arms: A0 = baseline, A1 = novelty label, A2 = tried mask, A3 = both, A4 = novelty
label + return map. Verdicts against a pre-registered rule, across manifests:

```bash
uv run python tools/paired_compare.py --base <manifest>:A1 --new <manifest>:A4
make label-diag       # replay old corpora: how often each label says "good"
uv run python -m pytest tests/ -q
```

## 8. `legacy/` — archived, and why it cannot affect your baselines

`legacy/` holds code kept for the record only: the old API-path scripts, and
in `legacy/llm_track/` the semester-1 LLM track (Probe A, Probe C, rule-finding;
all NO-GO or unfinished, summarised at the top of `legacy/llm_track/README.md`).
Semester-2 work starts from the baseline, following `docs/plans/`. You can
ignore `legacy/` entirely. It is isolated by four mechanisms:

1. **No baseline file imports it.** `make check` fails if one ever does. The
   dependency arrow points one way: archived code may read the corpus and the
   canonicalizer, and nothing reads archived code.
2. **No baseline tooling runs it.** Its make targets live in
   `legacy/llm_track/Makefile`, and `make check` also fails if the root Makefile
   or `sweep.sh` ever invokes it or its environment.
3. **Separate virtualenv.** LLM serving runs in `.venv-llm` (vLLM + its own
   torch) so it can never bump the baseline's pinned torch 2.8.0.
4. **Offline only.** Nothing in the archive is on any agent's per-action path.

If LLM features ever reach an agent (Plan A), they will be behind `EVAL_*` flags
that default to off and are recorded in `run_config.json`, so a baseline run is
a baseline run.

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
