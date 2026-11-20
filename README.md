# ARC3-solution — Team B (StochasticGoose track)

Team B's work on ARC-AGI-3 (ARC Prize 2026), built on the StochasticGoose
agent. Teammates can clone it, drop an agent in, run the frozen benchmark and
compare. Rules for contributors (and for Claude Code) are in `CLAUDE.md`.

## The project in four parts

| Part | What it is | Code | Write-ups |
|---|---|---|---|
| **[StochasticGoose improvements](#stochasticgoose-improvements)** | The exploring agent and the upgrades to it. No LLM. **54 → 79 → 112 levels** | [`custom_agents/`](custom_agents/README.md) | [`docs/1-goose-upgrades/`](docs/1-goose-upgrades/) |
| **[Glassbox](#glassbox)** | The current method: Goose plus a local LLM that writes small rules, each checked against every recorded move. **110 vs 101 levels, within noise** | [`custom_agents/gb/`](custom_agents/gb/README.md) | [`docs/3-glassbox/`](docs/3-glassbox/) |
| **[Previously tried methods](#previously-tried-methods-legacy)** | A judge, the Coach, Rulebook v1 and v2. Each stopped at its test. Frozen | [`legacy/`](legacy/README.md) | [`docs/2-llm-history/`](docs/2-llm-history/) |
| **[Future methods](#future-methods-future)** | Ideas we may or may not add. Nothing there is built | [`future/`](future/README.md) | [`future/README.md`](future/README.md) |

Levels are counted on all 25 public games × 3 seeds × 100,000 actions each
(the games hold 183 levels, so 549 over the three seeds). The `docs/` folders
are numbered in the order things happened; [`docs/README.md`](docs/README.md)
is the full map, with every step and its result.

**Also here, since 19 Nov: the [Knowledge Planner](#knowledge-planner).** A
groupmate's planner, run unedited behind Goose. Its registered test passed:
**62 levels against 44 and 47** by 5,000 actions, all of the gain on two
games. Not adopted. Whether it becomes a fifth part, and under what name, is
not decided.

**Everything new is off by default.** An unflagged `--agent goose` run is the
semester-1 baseline, verified move for move. Each feature's lines in
`custom_agents/action.py` carry a tag (`# [return-map]`, `# [upgrades]`,
`# [train]`, `# [gb]`, `# [kp]`); deleting a tag's lines
(`sed -i '/\[return-map\]/d' ...`) removes that feature.

---

## StochasticGoose improvements

Goose is a small CNN that learns, while it plays, which moves do something,
and samples its moves from that prediction. Each improvement is one file and
one switch.

| Step | What changed | File (switch) | Levels | Outcome |
|---|---|---|---|---|
| Goose, semester 1 | Learns "did this move change the screen?" | `action.py` (`--agent goose`) | 54 | the baseline |
| Novelty label | Learns "did this move reach a screen not seen before on this level?", with blinking cells masked out | `canon.py` (`EVAL_LABEL=novel`) | 79 | adopted 22 Sep |
| Return map | Remembers which move leads where, and walks back to untried places when stuck | `return_map.py` (`EVAL_RETURN_MAP=1`) | 45 vs 32 on dev games | not adopted alone |
| Upgrade screens | Nine small candidates in two rounds. Winners: `bars` (progress bars masked), `map_gated` (walk back only while it pays), `attempt` (half credit for a screen new this attempt) | `upgrades.py` (`EVAL_UPGRADES=...`) | | winners bundled |
| **`mb_gated_att`** | Novelty label + return map + the three winners, as one agent | `presets.py` (`--agent mb_gated_att`) | **112** | **adopted 28 Sep** |
| Carry | Colours that do nothing when clicked, and what shrank before a win, carried to the next level. No LLM | `carry.py` (`EVAL_UPGRADES=...,carry_noops,carry_hints`) | 87 vs 81 on 7 games | **not adopted** (13 Nov): [its registered test](docs/1-goose-upgrades/carry-results.md) failed. Later levels came no sooner (23 level races against 22); no levels lost |

| Upgrade screen 3 | Three more candidates, alone and combined, at a short budget: `map_sched` (a bandit over five map schedules), `objclick` (clicks chosen by shape), train more on fresher data | `upgrades.py` (`EVAL_UPGRADES=...,map_sched` / `objclick`), `train_more.py` (`EVAL_TRAIN_EVERY`, `EVAL_TRAIN_RECENT`) | 22 vs 15 by 5,000 actions, 25 games x 1 seed (all three together) | **not adopted**: an exploratory screen, 16-17 Nov ([plan](docs/1-goose-upgrades/upgrade-screen3.md), [results](docs/1-goose-upgrades/upgrade-screen3-results.md)). All three together ranked first and clear the noise on one seed; a candidate for a confirm, not a result |

`mb_gated_att` is **A0**, the reference every later result is compared with.
File by file: [`custom_agents/README.md`](custom_agents/README.md). Three
[probes](docs/1-goose-upgrades/input-probes.md) of 10 Nov sized further ideas
before building: a reset watchdog has nothing to act on, and ACTION7 is an
undo that an explorer has little use for.

## Glassbox

Goose explores. A local LLM (Qwen3.8-27B) writes small rules about a game from
Goose's own recordings. Every rule is replayed against every recorded move,
and only rules that pass may steer Goose. The pipeline is split into stages
that each fix one problem and can be tested alone, one file per stage:

| Stage | File in `custom_agents/gb/` | What it does | LLM? |
|---|---|---|---|
| S0 MASK | `s0_mask.py` | Masks decorations, with each progress bar run to the screen's edge | no |
| S1 SEE | `s1_see.py` | Describes each move as objects and what happened to them | no |
| H1 VOCAB | `h1_vocab.py` | Ten tested grid primitives the rules may call (frozen) | no |
| S2 WRITE | `s2_write.py`, `s2_pool.py` | The LLM writes one mechanism per rule; finished runs can be pooled | **yes** |
| S3 CHECK | `s3_check.py`, `s3_guards.py` | Replays every rule on every recorded move; picks the book | no |
| S4 CARRY | `s4_carry.py` | Re-fits the book's rules to each new level by search | no |
| S5 ACT | `s5_act.py` | Steers Goose online: no-op moves pruned, goal hints (`EVAL_GB=1`) | no |
| H3 TEMPLATES | `h3_templates.py` | Tries rules from known games on a new game | no |
| S6 SHOW | `s6_show.py` | One report page per run | no |

The problem each stage fixes, and each one's status:
[`custom_agents/gb/README.md`](custom_agents/gb/README.md).

| Test (date) | Question | Result |
|---|---|---|
| G0, G0b (16 Oct) | Can near-miss rules be rescued? | The misses were a gap in the mask. With bars masked to the edge, m0r0's book went from 0% to 53.5% of changed cells |
| G1 (17 Oct) | Does the staged pipeline write better rules offline? | **FAIL**: 3 of 8 games above G0b (5 needed) |
| G2 (19 Oct) | Does the guide add levels on the 8 dev games? | **FAIL**: 34 vs 37 levels, within seed noise |
| G3 (3 Nov) | Do rule templates carry to held-out games? | **FAIL**: 0 of 792 tries adopted |
| Dev sweep 2 (5 Nov) | S5 v2 against a same-day A0, 8 dev games × 4 seeds | **PASS**: 72 vs 67 |
| G4 (8 Nov) | The confirm on all 25 games × 3 seeds | **ADOPT**: 110 vs 101 (12 better, 57 same, 6 worse) |
| Time to level (10 Nov, exploratory) | Does the guide finish levels sooner? | After level 1 it finished a level in fewer actions on 36 against 17 in G4 (two unguided arms: 24 against 27). **It did not repeat:** the registered [carry test](docs/1-goose-upgrades/carry-results.md) of 13 Nov found 23 against 22 |
| The funnel (12 Nov) | Where are a game's cases lost? | Not at seeing: S1 describes 83% of changing cases, an admitted rule gets 57%, the book gets the whole screen on 28% (5% on held-out games) |

**How to read G4.** The gain is about one standard deviation of chance, and
the guided agent is level with A0's own 27 Sep runs (110 vs 112). Outside the
8 dev games the guide used no LLM output, only colour no-ops carried between
levels and goal hints, and 5 of the 9 extra levels came from there. G4 had no
LLM-free arm; in G2 that arm did as well as the guided one (35 vs 34). Plan,
gates and every write-up: [`docs/3-glassbox/`](docs/3-glassbox/); commands:
`make gb-help`.

## Knowledge Planner

Goose plays level 1. A planner then takes the later levels. At each level
clear it learns three things: what is clickable, what a click does, and which
rule is true only on the solved board. It plans the next level to that rule,
and hands the level back to Goose when it has no rule, no plan or too many
wrong predictions. No LLM.

The planner is a groupmate's agent (their E20). It runs unedited through a
small bridge, `custom_agents/kp_bridge.py` (`EVAL_KP=1`, off by default); its
four files are kept byte for byte in `custom_agents/kp_ref/`. Every run here
is Goose with upgrade screen 3's three upgrades (`SOT`), at 5,000 actions.

| Step (19 Nov) | Question | Result |
|---|---|---|
| Phase 0 (CPU) | Does the planner work from Goose's own level 1? | Yes: from recordings it won ft09 20 times of 20 and reached level 4 of lp85 15 times of 15 |
| The bridge (CPU) | Does it leave Goose alone when off, and make the planner's own clicks when on? | Off: the same moves as before, on every move. On: the same 178 planned clicks as the planner in its own harness |
| Offline, the games it may be built on | Which games does it reach? | ft09 and lp85. On cd82, dc22, m0r0 and vc33 it never starts: pieces move |
| Dev sweep (exploratory) | With and without the planner, 7 games x 2 seeds | 36 vs 20 levels |
| **The test (registered)** | The same on all 25 games x 2 new seeds, with the baseline run twice to measure chance | **Clears the noise: 62 vs 44 and 47** (+18 against a bar of 10) |

**How to read the test.** All of the gain is on ft09 (+9) and lp85 (+10), the
two games the planner was built on, so it is a tuned-tier result at 5,000
actions and two seeds. There it turns later levels from thousands of actions
into tens: ft09 is won in under 800 actions. On the other 23 games it made no
click, and the 15 held-out games ended 18 against 17 and 18. Nothing is
adopted and there is no preset; the base it ran on is not adopted either.
Results: [`kp-results.md`](docs/4-knowledge-planner/kp-results.md); plan and
every write-up: [`docs/4-knowledge-planner/`](docs/4-knowledge-planner/).

## Previously tried methods (`legacy/`)

Finished work, kept for the record and frozen. Nothing live imports it
([§7](#7-legacy-archived-and-why-it-cannot-affect-your-baselines)).

| Method | The idea | Result | Code | Write-up |
|---|---|---|---|---|
| LLM judge, heuristic writer (semester 1) | An LLM ranks which screen is more promising; an LLM writes scoring code | NO-GO: judge at chance; heuristics never beat a one-line rule | `legacy/llm_track/` | its `README.md` |
| Coach (3-4 Oct) | When Goose stalls, the LLM reads a summary and suggests what to try | NO-GO: 21 hits vs 21.2 expected by chance | `legacy/coach_track/` | [`docs/2-llm-history/`](docs/2-llm-history/) |
| Rulebook v1 (4-6 Oct) | The LLM writes the game's rules as code, replayed against recorded moves | FAIL: 2 of 8 games; rules did not carry to the next level | `legacy/rulebook/` | [`docs/2-llm-history/`](docs/2-llm-history/) |
| Rulebook v2 (7-15 Oct) | Rules take level facts as parameters and are re-fitted per level | FAIL: 2 of 8 games; 0.0% wrong after re-fitting, but too few rules | `legacy/rulebook/` | [`docs/2-llm-history/`](docs/2-llm-history/) |
| Goose candidates that lost | Tried-action mask; `graded`, `map_objects`, `map_diverse` | dropped by their sweeps | git tags `pre-cleanup-2026-10-06`, `upgrade-screen-round1` | [`docs/1-goose-upgrades/`](docs/1-goose-upgrades/) |

Also there: the semester-1 API-path scripts (`legacy/semester1/`) and the
launchers of the Goose upgrade experiments (`legacy/goose_experiments/`; the
upgrades themselves are live code). Index: [`legacy/README.md`](legacy/README.md).

## Future methods (`future/`)

Ideas that are not built. [`future/README.md`](future/README.md) lists this
week's items, the nine ideas from the 10 Nov review with their cost, and how an
idea moves from there into the live code. An idea is built in its real home
(`custom_agents/` or `custom_agents/gb/`), off by default behind its own flag,
and only then tested.

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
- `tools/levels_by_budget.py --base <manifest>:<arm> --new <manifest>:<arm>`
  compares two arms on *when* levels were completed: levels by action budget,
  AULC, and which arm finished each level in fewer actions. No verdict.
- **Did my change alter the agent?** GPU runs can't answer that. Record 2,000
  moves on the CPU before and after and compare them move for move:
  `tools/cpu_check.py`.
- Ad-hoc sweeps you won't report: `make sweep`, `make long`, `make random`.
  Anything compared against a teammate goes through `make bench`.
- Cross-game curriculum (one brain across a game list):
  `uv run python run_curriculum.py --games ft09,dc22,ls20`.
- Glassbox has its own commands, one per stage: `make gb-help`. The guided
  agent is `mb_gated_att` with `EVAL_GB=1 EVAL_GB_BOOK=<an offline run>`; the
  book lives under `results/glassbox/`, outside git.
- The Knowledge Planner: `EVAL_KP=1` on a `--agent goose` run. Its test:
  `uv run python experiments/kp/kp_test.py plan` (also `status`, `verdict`);
  what it does on recordings, on the CPU: `uv run python tools/kp_offline.py takeover`.
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
| `EVAL_LABEL` | Goose's training label: `change` or `novel` | `change` |
| `EVAL_CANON_WARMUP` / `EVAL_CANON_REFRESH` | the online decoration mask: warm-up / recompute cadence | `200` / `250` |
| `EVAL_RETURN_MAP` | the return map | off |
| `EVAL_MAP_STALL` / `EVAL_MAP_CLICK_TRIES` / `EVAL_MAP_MAX_ROUTE` / `EVAL_MAP_RETURN` | map tuning | `200` / `20` / `100` / `1` |
| `EVAL_UPGRADES` | comma list from `bars`, `attempt`, `map_gated`, `deadclick`, `carry_noops`, `carry_hints`, `map_sched` (in `map_gated`'s place), `objclick` | unset |
| `EVAL_TRAIN_EVERY` / `EVAL_TRAIN_RECENT` | train every N actions / this many of the newest stored moves in every batch (`train_more.py`) | `5` / `0` |
| `EVAL_GB` / `EVAL_GB_BOOK` | Glassbox's guide in the agent; the offline run whose book it carries (unset: no LLM rules) | off / unset |
| `EVAL_KP` / `EVAL_KP_PLAN` / `EVAL_KP_CONFIG` | the Knowledge Planner behind Goose (`kp_bridge.py`); `EVAL_KP_PLAN=0` = watch and learn only; its settings file. Refused together with `EVAL_GB` | off / on / `experiments/kp/kp.toml` |

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
├── README.md, CLAUDE.md       # this map; the rules for contributors
│
├── custom_agents/             # ALL LIVE AGENTS (README.md inside)
│   ├── __init__.py            #   the agent REGISTRY - add yours here
│   ├── TEMPLATE.py            #   copy this to start a new agent
│   ├── action.py              #   Goose (the CNN agent); features hook in on tagged lines
│   ├── canon.py               #   Goose upgrade: screen fingerprints, the novelty label
│   ├── return_map.py          #   Goose upgrade: the return map
│   ├── upgrades.py            #   Goose upgrades: bars / attempt / map_gated / deadclick / map_sched / objclick
│   ├── carry.py               #   Goose upgrade: colour no-ops and goal hints carried between levels
│   ├── train_more.py          #   Goose upgrade: train more often, on fresher data
│   ├── presets.py             #   --agent mb_gated_att, the adopted Goose
│   ├── kp_bridge.py, kp_ref/  #   KNOWLEDGE PLANNER: the bridge, and the groupmate's four files (never edited)
│   ├── gridtools.py, random_agent.py, view_utils.py   # shared helpers, the random floor
│   └── gb/                    #   GLASSBOX: one file per stage (README.md inside)
├── legacy/                    # PREVIOUSLY TRIED METHODS, frozen (README.md inside)
├── future/                    # FUTURE METHODS, not built (README.md inside)
├── docs/                      # every write-up: README.md (the map), 1-goose-upgrades/,
│                              #   2-llm-history/, 3-glassbox/, 4-knowledge-planner/, reports/
│
├── run_local.py               # run one agent on one game (the local engine)
├── run_curriculum.py          # several games in a row, one persistent brain
├── sweep.sh                   # benchmark and ad-hoc sweep runner
├── benchmark.py               # THE FROZEN TEST SET
├── eval_common.py             # the evaluation contract (seeds, caps, the corpus)
├── utils.py                   # run folders and the git stamp
│
├── compute_metrics.py         # score one run's corpus
├── metrics_common.py          # the shared decoration-cell detector
├── analyze_curves.py          # levels against budget, AULC, RHAE
├── summarize_overnight.py     # one report per sweep
├── compare.py                 # side-by-side table from bench manifests
├── inspect_corpus.py          # corpus schema check
├── manifest.py                # the one reader for sweep manifests
│
├── check_repo.py              # make check: isolation, registry, deps, suites
├── tiers.py, registry.py      # game tiers (dev / seen / held-out); the frozen-artifact registry
├── artifacts/                 # registry.json and the frozen artifacts it names
├── tools/                     # paired_compare.py (verdicts), levels_by_budget.py (when levels came),
│                              #   cpu_check.py (did it change?), frozen_moves.py, probe_engine.py
│                              #   (probes), artifacts.py, gb_offline.py, gb_sweep.py, gb_gate.py,
│                              #   gb_funnel.py (Glassbox), kp_offline.py (Knowledge Planner)
├── experiments/goose/         # carry_test.sh (the carry test), screen3.py (upgrade screen 3)
├── experiments/kp/            # the Knowledge Planner: kp.toml, kp_test.py (its test), kp_verdict.py, run_reference.py
├── experiments/glassbox/      # glassbox.toml (every setting) and gb.mk (make gb-help)
├── experiments/llm/           # the local LLM server and its environment
├── tests/                     # pytest
│
├── custom_agent.py, harness_patches.patch   # the hosted-API path only (make action)
├── ARC-AGI-3-Agents/          # git submodule (arcprize harness) - do not edit
└── results/                   # ALL run output (gitignored)
```

## 7. `legacy/`: archived, and why it cannot affect your baselines

`legacy/` holds the previously tried methods (index: `legacy/README.md`). You
can ignore it. It is isolated by three mechanisms, all enforced by
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
