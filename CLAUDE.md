# CLAUDE.md — StochasticGoose (Team B, ARC-AGI-3)

What is true now and what must not be broken. History lives in `docs/` (map:
`docs/README.md`); finished code lives in `legacy/` (index: `legacy/README.md`).

## The project in three parts

The repo, the docs and the archive all follow this order, so keep new work in it:

1. **Goose upgrades (no LLM)**: the exploring agent, Goose (a CNN that learns
   online which actions change the screen), and the semester-2 improvements to
   it. 54 → 79 → 112 levels on 25 games × 3 seeds × 100k.
   Code: `custom_agents/action.py`, `canon.py`, `return_map.py`, `upgrades.py`,
   `presets.py`. Docs: `docs/1-goose-upgrades/`. Launchers: `legacy/goose_experiments/`.
2. **LLM history**: the judge, the Coach, Rulebook v1 and v2. All stopped at
   their gates. Docs: `docs/2-llm-history/`. Code: frozen in `legacy/`.
3. **Glassbox (current)**: Goose plus a local LLM that writes small rules,
   each replay-checked, in separate stages that each fix one problem.
   Code: `custom_agents/gb/` (one file per stage; `gb/README.md` has the
   problem each fixes). Docs: `docs/3-glassbox/`.

**Simplicity and readability come first** (Matt, 16 Oct). A reader should be
able to see what each step changed and why: one problem per stage, one stage
per file, one write-up per result, numbers next to every claim.

## It is a shared repo

Teammates clone it and add their own agents, so these are load-bearing and must
not be casually changed:

- `custom_agents/__init__.py`: the agent REGISTRY (one line per agent;
  `TEMPLATE.py` is the starting point).
- `benchmark.py`: the FROZEN test set (`smoke`/`quick`/`standard`/`full`).
  Changing a suite invalidates every comparison made with it; add a new suite.
  `make bench SUITE=x AGENT=y`, `make compare M1= M2=`.
- `check_repo.py` (`make check`): no live file imports `legacy/` (or `wm`); the
  baseline imports `gb` only on lines tagged `# [gb]`; the root Makefile,
  `sweep.sh` and `gb.mk` never run legacy code or `.venv-llm`; every registered
  agent imports; every third-party import is declared. Run it before pushing.
- Also kept for teammates: `run_curriculum.py`, `compare.py`, `sweep.sh`.

Branch: `glassbox` (cut from `refactor/shared-baselines` on 16 Oct). `main` is
deliberately far behind; do not merge or open a PR into it unless asked.

## The agents

| `--agent` | What it is |
|---|---|
| `goose` | Semester-1 baseline. Defaults unchanged, so teammates' comparisons hold |
| `mb_gated_att` | **The adopted Goose, A0** (28 Sep): `EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt`, fixed in `presets.py`. 112 vs 79 levels |
| `random` | Matched uniform-random floor, for uplift ratios |

A preset name means the same thing forever: a new adopted configuration gets a
new preset (Glassbox's will be `glassbox`), never an edit to an old one.

## Running

```bash
make help                                     # everything; make gb-help for Glassbox
make bench SUITE=standard AGENT=mb_gated_att  # backgrounded, ~3.8 h
EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ls20 --agent mb_gated_att   # ~120 act/s
uv run python compute_metrics.py results/runs/<ts>/ls20/transitions \
    --game ls20 --agent mb_gated_att --seed 0 --suite results/local_suite.csv
uv run python tools/paired_compare.py --base <manifest>:<arm> --new <manifest>:<arm> \
    [--rule adopt|dev|confirm] [--expect N]
uv run python -m pytest tests -q              # live tests (~85 s)
uv run python -m pytest legacy/rulebook/tests -q   # frozen Rulebook tests (~40 s)
make gb-config                                # Glassbox's settings and their hashes
```

**Did a change alter the agent?** GPU runs cannot say (see caveats). Record
2,000 moves on the CPU before and after and compare move for move, for both
`goose` and `mb_gated_att` (~20 min each run, so ask Matt first):

```bash
uv run python tools/cpu_check.py run --out /tmp/chk/before --game ft09
uv run python tools/cpu_check.py compare /tmp/chk/before /tmp/chk/after
```

Finished Goose experiments keep their launchers:
`make -f legacy/goose_experiments/experiments.mk <target>` (planb-*, map-*,
upgrade-screen*, upgrade-confirm). `make clean` needs `CONFIRM=yes`.

## eval_common contract (env vars)

| Variable | Default | Notes |
|---|---|---|
| `EVAL_SEED` | time-based | base seed; each game adds `stable_game_offset(game_id)` |
| `EVAL_MAX_ACTIONS` | unlimited | per-game action cap (0 = unlimited) |
| `EVAL_LOG_METRICS` / `EVAL_LOG_TRANSITIONS` / `EVAL_SAVE_VIS` | on / on / off | TensorBoard / `.npz` corpus / PNG heatmaps |
| `EVAL_RESET_ON_LEVEL` | on | reset model/optimizer/buffer at level boundary |
| `EVAL_RESULTS_DIR` | `results` | root for ALL run output |
| `EVAL_LABEL` | `change` | `change` (frame changed) or `novel` (new canonical state this level) |
| `EVAL_CANON_WARMUP` / `EVAL_CANON_REFRESH` | `200` / `250` | the online decoration mask |
| `EVAL_RETURN_MAP` | off | `custom_agents/return_map.py` |
| `EVAL_MAP_STALL` / `EVAL_MAP_CLICK_TRIES` / `EVAL_MAP_MAX_ROUTE` / `EVAL_MAP_RETURN` | `200` / `20` / `100` / `1` | map tuning (`docs/1-goose-upgrades/option-1-return-map.md`) |
| `EVAL_UPGRADES` | unset | comma list from `bars`, `attempt`, `map_gated`, `deadclick` |
| `EVAL_GB` / `EVAL_GB_BOOK` | off / unset | Glassbox's S5 guide in the agent (`gb/s5_act.py`); the book = an offline run under `results/glassbox/` (unset: GB-free); hooks tagged `# [gb]` |

Always run with `PYTHONHASHSEED=0`. Removed on 2026-10-06, each stops the run
with a pointer to its git tag: `EVAL_MASK_TRIED` and the upgrades `graded`,
`map_objects`, `map_diverse`.

## Where things stand (18 Oct 2026, evening)

**Part 1 is done:** `mb_gated_att` adopted 28 Sep (112 levels). **Part 2 is
closed:** Coach NO-GO; Rulebook v1 FAIL (2 of 8); Rulebook v2 closed after CP1
(FAIL 2 of 8, 0.0% wrong at T1). **Part 3, Glassbox, is ahead of its calendar:**
G0, G0b, G1 run (M1 done); M2's pieces built and smoke-tested; **waiting on Matt
to settle prereg §5's OPEN items and register 5a, then 5b**. Plan:
`docs/3-glassbox/glassbox-plan.md` (§7.4 is the running log). Gates:
`docs/3-glassbox/glassbox-prereg.md`: sections 3 (G0), 3b (G0b), 4 (G1)
registered and run; **5 (counterfactual + G2) DRAFTED 18 Oct, not registered**;
1, 2, 8–10 still drafts; 6–7 not written.

- **Timeline (Matt, 16 Oct):** every part tested at least once by about **Sun
  1 Nov** (soft); every gate run by **Fri 27 Nov** (hard). Course dates: Fri 30
  Oct full draft, Mon 2 Nov idea statement, Mon 7 Dec final. Plan §7.
- **Built** (`custom_agents/gb/`, one file per stage; `gb/README.md` has each
  one's status, `make gb-help` its command):
  - **S0 MASK** `s0_mask.py`: each bar in the decoration mask run to the
    screen's edges (`[mask] complete_bars`); off, it rebuilds CP1's evidence
    exactly. Also CP1's level rule for any game.
  - **S1 SEE** `s1_see.py` (VERSION 3): objects, what happened to them,
    "explained" (S1's measure; the plan's "closure" cannot fail), relations,
    feature bits, roles, exploration matrix, facts text (golden test).
  - **H1 VOCAB** `h1_vocab.py`, **registered `glassbox_vocab_v1`, frozen**: S1's
    relation code, `features()`, ten primitives, `CARD`. Rules get them through
    `gb/core/contract.py`.
  - **S2 WRITE v1** `s2_write.py`: CP1's procedure and budget, plus S1's facts,
    H1's card, one mechanism per rule, guards before near-miss calls. **S2 POOL**
    `s2_pool.py` (18 Oct): finished runs' candidates on the same evidence pooled
    into a new run, no LLM (one source = a clean copy, to re-run S3/S4).
  - **S3 CHECK**: grading `s3_check.py` (CP1's procedure), guard fitter
    `s3_guards.py`. Not built: priority composition, trust tiers.
  - **S4 CARRY v1** `s4_carry.py`: CP1's re-binding; plus (18 Oct) a role
    tie-break among the exact bindings applying to the most fit cases
    (`[carry] roles_in_bind`, **off by default**, so `make gb-g0b-control` still
    equals CP1) and the online wait `ready()`. Not built: repair calls.
  - **S5 ACT v1** `s5_act.py` (18 Oct): prune (carried colour no-ops, no LLM:
    >= 20 clicks on a level, none changing; the book's no-op rules, per group
    the known no-op with most moves, re-bound on EVERY level's own moves after
    the wait, retried at twice the moves up to 4,000), goal hints (never the
    background; boost only where a hint click is possible), `GbMap(UpgradedReturnMap)`
    (buttons a re-bound no-op rule covers count as tried). Weights reach Goose by
    rejection sampling with the guide's own generator (exactly p*w; Goose's
    stream untouched). Rules run in one spawned child (`RuleWorker`, timeout ->
    rules off). The book's CHANGING rules are not used online yet (no probes,
    no IMAGINE). **Hook:** `EVAL_GB=1` (+ `EVAL_GB_BOOK=<run>`; unset = GB-free),
    16 lines tagged `# [gb]` in `action.py`; **`cpu_check` with it unset:
    identical on every move** (goose/ft09 1,970, mb_gated_att/tu93 1,960; 18 Oct).
    Replay = the offline counterfactual (`--stage replay`, `make gb-replay`,
    `make gb-cf`), refused on dev games without `--cf`/`CF=1`.
  - **H3 TEMPLATES v0** `h3_templates.py`: the plan's smoke test only (seen
    games only): 0 of 21 templates adopted on lp85/ls20.
  - **S6 SHOW v0** `s6_show.py`: report page (funnel, case viewer) per run.
  - **Not built:** H2 SCHEDULER (online LLM), S5 probes, IMAGINE, contact goals.
  - Drivers: `tools/gb_offline.py` (`--stage mask|see|pool|write|check|carry|
    guards|replay|templates|show`); online runs `tools/gb_sweep.py` (GB and
    GB-free arms, 4 at a time, verdicts vs A0 with a 16-run A0 manifest;
    `make gb-online` seen games, `make gb-g2 BOOK=.. G2=1` dev games only after
    §5b is registered); verdicts `tools/gb_gate.py`.
- **Proven equal to CP1:** `make gb-import-cp1` (437 of 437); `make
  gb-g0b-control`. 226 live tests.
- **Results** (each written up in `docs/3-glassbox/`):
  - **G0** (16 Oct): the guard fitter rescued nothing; m0r0's near-misses were
    a decoration-mask gap (row 63's bar). **G0b** (16 Oct): with S0, all 11
    trusted; m0r0 0% → 53.5% of changed cells; T1 51.7% (memory 53.9%).
  - **G1** (17 Oct, `g1-offline-rules.md`): **FAIL**, 3 of 8 above G0b, safety
    met. tu93 0 → 59.0%, tr87 0 → 35.5% (first trusted rules ever), dc22 11.8 →
    29.3%; ft09, g50t, vc33, cd82 lost ground; the new books carry worse.
  - **Pooled book** (18 Oct, `pool-g0b-g1.md`, exploratory, `pool_g0b_g1`):
    never below the better of G0b/G1, above both on dc22 40.4%, g50t 18.1%,
    cd82 6.6%; T1 equals the better source's on every game (beats both
    baselines on dc22, ft09). Role tie-break: m0r0's G1 rules 6.8% → 26.5% at
    T1, pooled book unchanged (tuned on dev games, labelled).
  - **Smoke runs** (18 Oct, online, pooled book, 20k actions each): ft09 x2,
    g50t, cd82, lp85 runner; no errors, ~85-128 act/s, re-binding pauses 1-3 s.
    Not evidence of levels.
- **Next = Matt's checkpoint (prereg §5, OPEN items, recommendations first):**
  (a) the book: **`pool_g0b_g1`** (alt. G1's or G0b's alone); (b) LLM online in
  G2: **none** (H2 not built; GB runs 4 at a time, ~6 h) vs build H2 first;
  (c) roles in S5's re-binding: **leave as built**; (d) probes: **G2 without
  them**, add for dev sweep 2. Then: register 5a → `make gb-cf BOOK=pool_g0b_g1
  CF=1` (CPU, ~1 h) → write it up → register 5b → `make gb-g2 BOOK=pool_g0b_g1
  G2=1` (GPU, ~6 h; check AC power) → verdict. Target G2 Tue 27 – Thu 29 Oct.
- **Still OPEN, Matt's:** register sections 1, 2, 8–10; confirm "explained" as
  the perception number; plan §10 (a) (advisor and the 30 Oct draft) and (b)
  (teammates' modules in S5).
- **Known limits:** S1's `sits_on` is read from neighbouring colours (pinned in
  `tests/test_gb_vocab.py`); the binding search's time budget leaves bindings
  untried on tr87 when the machine is busy (run S4 on a quiet machine); the
  `ARC-AGI-3-Agents` submodule's local edits make manifests say `+dirty`
  (prereg §11). `--set` overrides are recorded in stage headers, not in the
  run's copied `config.toml`.
- **Frontier models:** everything is the local model. The only frontier use is
  an optional showcase in late November (plan §6.5, limits in prereg §10).

## Glassbox rules

- **Stage files are numbered in data order** (`s0_mask.py` … `s6_show.py`,
  helpers `h1_vocab.py` … `h3_templates.py`). Each reads and writes files through
  `gb/artifacts.py`, has its own section in `glassbox.toml`, its own
  `tests/test_gb_<stage>.py`, and one `make gb-<stage>` target added to `gb.mk`
  when it is built. Every module starts with the problem it fixes.
- **`gb/core/` is inherited, not new.** Change it only in ways that leave CP1's
  registered numbers unchanged: run `tests/test_gb_regression.py` (and
  `make gb-import-cp1` for all 8 games) before committing any change to it.
  Anything that should change a result goes in a stage file, behind a knob.
- **`h1_vocab.py` is registered** (`glassbox_vocab_v1`, artifacts/registry.json):
  any edit makes `tools/artifacts.py verify` fail, so a change is version 2
  with its own registration (and it holds S1's relation code too).
- **`legacy/rulebook/` is frozen.** Never edit its prompts (`wm/rules.py`,
  `wm/rules2.py`) or tools: v1's and v2's answer caches are keyed on the
  prompts. Glassbox copies helpers from it; it never imports it.
- **The agent hook** (`EVAL_GB=1`, plan §5.5) goes in `action.py` on lines
  tagged `# [gb]` only, removable with `sed -i '/\[gb\]/d'`, like
  `[return-map]` (25 lines) and `[upgrades]` (26). The map extension subclasses
  `ReturnMap` (`GbMap`, via `upgrades.UpgradedReturnMap`); never edit `return_map.py` or `canon.py` for it.
- **Pre-registration:** nothing in a registered section is edited after its run
  starts (corrections go in section 11). Tiers are in `tiers.py`; the
  untouched tier is only ever run inside the confirm; cross-game artifacts are
  built from dev and seen games only and registered (`tools/artifacts.py`)
  before any held-out use.
- **The LLM server** (`experiments/llm/serve.sh`, Qwen3.8-27B AWQ INT4, port
  8018, alias `rulebook`) takes ~90% of the GPU offline. Ask Matt before any
  GPU run or any run over 30 minutes; the PC must stay on AC power with
  GPU-heavy Windows apps closed.

## Key conventions

- ALL run output goes under `results/` (gitignored, `EVAL_RESULTS_DIR`):
  `results/runs/<ts>/<game>/`, `results/sweeps/`, `results/local_suite.csv`,
  `results/confirm_upgrade/` (A0's 75 runs, also Glassbox's recordings),
  `results/rulebook/` (v1/v2, registered), `results/glassbox/` (new).
  **Nothing in `results/` is backed up by git, and nothing there is moved:**
  manifests and the registry name those exact paths.
- Manifests (`run_dir<TAB>game<TAB>seed<TAB>arm`) are read through `manifest.py`
  only. Grid helpers come from `custom_agents/gridtools.py` only.
- `legacy/` is a leaf: it may import the baseline; nothing live imports it, and
  baseline tooling never invokes it or `.venv-llm`. `make check` enforces this.
- The `arc-agi` package (provides `arcengine`) is needed for the local engine
  but not declared in `requirements.txt`; install it separately.
- With `EVAL_LABEL` at its default the agent is the old baseline. Every new
  feature gets its own `EVAL_*` flag (default off) and its own line tag.
- Any change that should not alter behaviour is checked with
  `tools/cpu_check.py` before it is committed, for `goose` and `mb_gated_att`.
- Parallel runs share the GPU: 4 at once total ~162 act/s vs 140 for one.
- Do NOT change `EVAL_RESET_ON_LEVEL` semantics or any hyperparameters
  (learning rate, `train_frequency`, batch size, buffer capacity, confidence
  coefficients) without explicit approval; they confound ablations.
- Commit and push only when asked.

## Metric definitions

All metrics use the unified indicator-cell canonicalizer (`metrics_common.py`):
fixed tickers (cell changing in >=95% of transitions) plus rotating tickers
(compact cell set covering >=95% of tiny <=2-cell transitions, if those make up
>=30% of the run).

**Never report an exploration metric alone.** A high `meaningful_change_rate`
only means the frame keeps changing. Read change rate, redundancy and coverage
together.

- `n_actions` (RECORDED moves; per-action rates divide by it) vs `actions_taken`
  (the action counter; caps, level-up times, budgets, curves and AULC use it).
- `unique_states_per_action`: the coverage number. `discovery_auc` is
  DEPRECATED for cross-run comparison; never report it without the former.
- `novelty_late_per_1k`: new canonical states per 1k actions over the final
  10%; the stall detector. Baseline medians: tu93 0.0, dc22 2.4, ls20 7.4,
  g50t 9.9, ft09 955.
- Levels vs. budget, AULC, RHAE: `analyze_curves.py`. Always report AULC next to
  its `T_max`. At 100k actions RHAE is ~0 for every arm, so levels stay primary.
- Uplift: game-dependent metrics need the matched random floor (`make random`).
- Verdict rules (`paired_compare.py --rule`): `adopt` (levels >=, wins >
  losses), `dev` (levels >=, wins >= losses), `confirm` (levels strictly more,
  wins > losses). All add "no game worse on every seed".
- Glassbox's per-stage metrics (closure, compile rate, coverage, tiers, T0/T1/T2,
  avoidable waste, masking risk, rule survival): plan §6.1.

Censoring: most runs never reach level k. Always report "k/n seeds reached" next
to any actions-to-level median.

## Known measurement caveats

- **The rotating-ticker detector misses most tickers** (found 2026-09-04, NOT
  fixed in the scorer: changing it moves every published number, an advisor
  decision). ft09, ls20, ar25, lp85, dc22 and g50t are scored with an EMPTY
  decorative mask; on ft09 that inflates `unique_states_per_action` ~1.4x. The
  agent's `bars` upgrade and the rule evidence mask use the component-level fix
  (`gridtools.tick_cells`); the published scorer does not.
- **Runs are not reproducible past the first training step** on the GPU
  (identical for ~300 actions, then CUDA kernels diverge). **Never claim a
  byte-identical comparison between two arms**; compare distributions across
  seeds. CPU runs ARE reproducible (`cpu_check.py`).
- **Recordings changed on 2026-10-06.** Before, the move that ended an attempt
  was not recorded; after, it is (`game_over=1`). The 75 A0 reference runs are
  the old kind (median run recorded 98% of its actions, tu93 88%). Levels are
  unaffected; per-action rates are not strictly comparable across the change.
- **The adopted agent's walk-back bandit** credits a new level's first attempt
  to whichever arm was active before the level-up (noted in `upgrades.py`). Left
  as confirmed; changing it needs its own test.
- **Dropout stays on while Goose picks actions** (never put in eval mode).
  Inherited from upstream; part of the baseline. Do not "fix" it.
