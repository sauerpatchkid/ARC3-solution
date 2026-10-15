# CLAUDE.md — StochasticGoose (Team B, ARC-AGI-3)

Current state first, history by pointer. Results and reasoning live in
`docs/plans/`; this file says what is true now and what must not be broken.

## What this repo is

ARC-AGI-3 capstone agent. Three stages: the **agent** (`custom_agents/action.py`,
CNN change predictor), the **corpus logger** (`TransitionLogger` in
`eval_common.py`, writes `.npz` shards), and the **scorers** (`compute_metrics.py`
for the local pipeline, `legacy/summarize_runs.py` for the legacy API path — both
share the indicator-cell canonicalizer from `metrics_common.py`).

Two multi-game drivers sit on top of `run_local.py` and share its game loop
(`run_local.play`): `run_curriculum.py` (ONE persistent brain across a game list;
default agent only) and `sweep.sh` (games × seeds × arms, each a fresh process).

**It is a shared repo.** Teammates clone it and add their own agents, so three
things are load-bearing and must not be casually changed:

- `custom_agents/__init__.py` — the agent REGISTRY. Every agent lives in
  `custom_agents/` and is one line here; `TEMPLATE.py` is the starting point.
- `benchmark.py` — the FROZEN test set (`smoke`/`quick`/`standard`/`full`).
  Changing a suite invalidates every comparison already made with it; add a new
  suite instead. `make bench SUITE=x AGENT=y` runs one; `make compare` puts two
  side by side and refuses mismatched suites.
- `check_repo.py` (`make check`) — no file outside `legacy/` may import it and no
  baseline tooling may run it; every registered agent imports and has the
  runner's surface; every third-party import is declared. Run it before pushing.

Working branch: `refactor/shared-baselines`. `main` is deliberately left 40+
commits behind; do not merge or open a PR into it unless asked.

## The agents

| `--agent` | What it is |
|---|---|
| `goose` | Semester-1 baseline. Defaults unchanged, so teammates' comparisons hold |
| `mb_gated_att` | **The adopted Goose** (2026-09-28): `EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt`, fixed in `custom_agents/presets.py`. 112 vs 79 levels on 25 games × 3 seeds (`docs/plans/upgrade-confirm-results.md`) |
| `random` | Matched uniform-random floor, for uplift ratios |

A preset name means the same thing forever: for a new adopted configuration add
a new preset, never edit an old one.

## Running

```bash
make suites                                   # smoke / quick / standard / full
make bench SUITE=standard AGENT=mb_gated_att  # backgrounded, ~3.8 h
make compare M1=<manifest> M2=<manifest>

EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \
    uv run python run_local.py --game ls20 --agent mb_gated_att   # ~120 act/s
uv run python compute_metrics.py results/runs/<ts>/ls20/transitions \
    --game ls20 --agent mb_gated_att --seed 0 --suite results/local_suite.csv

make sweep DRY_RUN=1          # ad-hoc games/seeds, NOT comparable between people
make random                   # random floor;  make curves MANIFEST=<manifest>
make long                     # ft09 + tu93, 2M actions each, ~4 h/game
uv run python tools/paired_compare.py --base <manifest>:<arm> --new <manifest>:<arm> \
    [--rule adopt|dev|confirm] [--expect N]
```

Finished experiments keep their launchers: `make planb-dev|planb-confirm|map-dev|
map-confirm` (arm sweeps, shared `planb-status|pause|stop`), `make upgrade-screen`,
`upgrade-screen2`, `upgrade-confirm` (each with `-status` / `-pause`).
`make clean` deletes every recorded run and needs `CONFIRM=yes`. API path
(unchanged, slower): `make action`.

**Did a change alter the agent?** GPU runs cannot say (see caveats). Record
2,000 moves on the CPU before and after and compare move for move:

```bash
uv run python tools/cpu_check.py run --out /tmp/chk/before --game ft09   # ~20 min
uv run python tools/cpu_check.py compare /tmp/chk/before /tmp/chk/after
```

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
| `EVAL_LABEL` | `change` | training label: `change` (frame changed) or `novel` (new canonical state this level) |
| `EVAL_CANON_WARMUP` / `EVAL_CANON_REFRESH` | `200` / `250` | transitions before the online indicator-cell mask starts, and its recompute cadence |
| `EVAL_RETURN_MAP` | off | map of the level + walk back to untried screens (`custom_agents/return_map.py`) |
| `EVAL_MAP_STALL` / `EVAL_MAP_CLICK_TRIES` / `EVAL_MAP_MAX_ROUTE` / `EVAL_MAP_RETURN` | `200` / `20` / `100` / `1` | map tuning; see `docs/plans/option-1-return-map.md` |
| `EVAL_UPGRADES` | unset | comma list from `bars`, `attempt`, `map_gated`, `deadclick` (`custom_agents/upgrades.py`) |

Always run with `PYTHONHASHSEED=0`. Removed 2026-10-06, each stops the run with a
pointer to its git tag: `EVAL_MASK_TRIED` (Plan B's dropped mask) and the upgrade
names `graded`, `map_objects`, `map_diverse`.

## Focus games

After the 25-game sweep (`results/sweeps/sweep_20260727_225828_summary.md`):
**12 of 25 games complete at least one level** with the baseline at 100k actions.

- **ft09** — learnable, reliable level completions. 100% ACTION6 clicks and a
  99.1% raw change rate, so the change label is nearly constant here.
- **ls20** — null contrast, completes nothing. Exploration-only.
- **ar25 / cd82 / lp85** — the other level-2 games; no semester-1 medians.
- **tu93** — weak and seed-dependent; long-horizon runs only, paired with ft09.

## Semester 2: where things stand (2026-10-15)

Goose side, done: Plan B (novelty label adopted, tried mask dropped), Option 1
(return map), two upgrade screens and the 25-game confirm that adopted
`mb_gated_att`. Plans and results: `docs/plans/plan-B-*`, `option-1-*`,
`upgrade-*`.

LLM track, all local with Qwen3.8-27B (`cyankiwi/Qwen3.8-27B-AWQ-INT4`, served
from `.venv-llm`; the only cached model):

- **Coach** (LLM suggests what to try at a stall): offline probe NO-GO, at
  chance (`docs/plans/coach-probe-results.md`). Archived in `legacy/coach_track/`.
- **Rulebook v1** (LLM writes replay-checked rules): Stage A NO-GO, 1 of 3 games
  (`docs/plans/rulebook-stageA.md`); offline rule test Tier 0a G1 FAIL, 2 of 8
  (`docs/plans/rulebook-tier0a.md`). Verdicts stand; never re-scored. Code:
  `custom_agents/wm/`, `tools/wm_offline.py`, `experiments/rulebook/`. Git tag
  `rulebook-v1` is the code that produced them. Full v1 record:
  `results/rulebook-full-writeup.md` (not in the repo).
- **Rulebook v2** — the CURRENT plan: `docs/plans/rulebook-v2.md`, checkpoints
  CP0–CP5. Every gate is pre-registered in `docs/plans/rulebook-v2-prereg.md`
  before the run it judges; nothing in a registered section is edited after its
  run starts (corrections go in its section 10). Report dates, fixed: 30 Oct
  full draft, 2 Nov idea lock, 7 Dec final.
  - **CP0 DONE 2026-10-07** (`docs/plans/rulebook-v2-cp0-results.md`). Near-miss
    refinement: 2 of 7 groups reached an exact rule (expected 2–4), so CP1 goes
    ahead as written; both exact rules got there by covering less, and rules
    still did not carry to the next level. v1's books on v2's ruler: 1 of 8
    beats both baselines (dc22), ft09 wrong on 58.8%.
  - **CP1 DONE 2026-10-15: gate FAIL** (`docs/plans/rulebook-v2-cp1-results.md`).
    Primary 2 of 8 (dc22, ft09; needs 3), safety met on all 8 (T1 wrong 0.0%
    everywhere; v1 was wrong on 58.8% of ft09). Re-binding fixed transfer where
    rules existed (ft09 now beats both baselines), but tu93, tr87 and m0r0 got no
    trusted rule at all, and g50t/cd82's rules explain only part of each change.
    Results registered as `rulebook_v2_cp1_tier0a`.
  - **NEXT: CP2, from 16 Oct** (rule-guided Goose, online; plan section 5,
    gates in pre-registration section 5). Decided 15 Oct (Matt): CP2 uses v2's
    pipeline, not the v1-form rules plan section 2.3 wrote for a failed CP1.
  - **CP2 starting notes.**
    - *Order:* (1) the offline counterfactual (pre-registration 5.1: avoidable
      waste and masking risk on the confirm recordings; no GPU, no LLM); (2) the
      build; (3) `tools/cpu_check.py` on `goose` and `mb_gated_att` with the
      flag off (must match move for move); (4) the dev sweep, 8 games x seeds
      0-1 x 100k, arms A0 (the 75 confirm runs, reused), C2, C2-free; gate
      `paired_compare --rule dev --expect 16`.
    - *To build:* a guide in `action.py` behind `EVAL_WM=guide` (new
      `# [wm]` tag, off by default, removable like `[return-map]`); online
      triggers (warm-up at 2,000 moves, level-up re-binding after the fit
      window, 5 counterexamples, stall); the reweighting (no-op x0.1, probe
      boosts, 5% cap, 30 probe moves); a `ReturnMap` SUBCLASS that counts a
      trusted no-op's (screen, action) as tried (never edit `return_map.py`);
      goal hints; C2-free's strict carried no-ops (every click on the colour,
      >= 20, changed nothing). Budget per run: 16 requests, k = 2, 48
      completions, 400,000 tokens, 12,288-token answers with the code-only
      retry (measured fine, plan 10.5).
    - *Reuse:* `wm/` as built for CP1 (`bind.search`, `book.compose`,
      `book.trust`, `rules2`), the LLM server beside Goose (plan 5.4: one Goose
      run at a time, close GPU-heavy Windows apps).
  - **Pre-registration status.** Sections 1–5 are binding in full; 4.4 (the
    interval) was settled on 13 Oct. How CP1's text is read is in section 10
    (13 Oct entry): five readings Matt decided (a group's search stops only at a
    trusted rule the lint passes) and the as-built choices he kept (14 Oct).
    Still OPEN and Matt's: 6.4 (CP3's "no higher": per game or pooled), before
    28 Oct.
  - **Frontier models.** Everything is the local model, and most of the report
    is to be our own system on it (Matt, 13 Oct). The only frontier uses are
    optional: one helper-extraction pass in CP3 (plan 6.2), and a showcase in
    the last week of November (plan 9A, limits registered in pre-registration
    7.2): the same pipeline with a frontier rule writer, reported as a short
    comparison. Never a gate, nothing flows back, dev and seen games only, cut
    first if time is short.
  - **CP1 code.**
    - `wm/contract.py`: the v2 rule (`PARAMS` read statically, five slot types,
      optional `bind()`; a v1 rule is `PARAMS = {}` claiming every cell).
      `wm/lint.py`: the literal lint. `wm/bind.py`: re-binding by search (one
      child per rule and level, successive halving, time budget) and running a
      rule with a binding. `wm/book.py`: trust across levels and the composed
      book (cell by cell; a cell claimed two ways is UNKNOWN). `wm/rules2.py`:
      the v2 prompts (first round, refinement, residual, two-level); v1's text
      in `wm/rules.py` is untouched. `evidence.py` now keeps per-move order
      (`moves`: case, own outcome, action number, attempt, `t`); `metrics.py`
      has per-move rates and the gate's interval (`gate_interval`).
    - `tools/wm_tier0a_v2.py`: the run (`--evidence-only`, `--from-cache`,
      `--no-residual`). Jobs: `experiments/rulebook/llm_run.sh start
      tier0a2-smoke` (lp85, ~55 min) and `start tier0a2` (the gate run, done:
      9 h on 14–15 Oct; never start it again into the same folder - it is
      registered). Output: `results/rulebook/v2/`.
    - Smoke test, 13–14 Oct (`results/rulebook/v2/tier0a_v2_smoke/`): the whole
      pipeline ran with the real model, and a re-score from its answer cache
      matches it exactly. The model does use `PARAMS`. Half the answers ran out
      of room (v1 22%, CP0 34%); the code-only retry rescues most. 3 of 8
      first-round answers imported something (not allowed, as in v1).
    - 14 Oct: an outside plan update was reviewed (plan section 10.5). Nothing in
      CP1 changed; two CP2 details were added (the guide also acts inside the
      return map; C2-free carries only strict no-ops). Thinking-cap measurement
      (`tools/wm_thinkcap.py`, job `thinkcap`): cutting answers at CP2's 12,288
      tokens and finishing with the code-only retry lost no admitted rule on 208
      recorded answers; cutting at 6,144 lost 5 of 8.
    - `tools/wm_baseline.py` gives the gate's "v1, for comparison" row on the
      same moves with the same interval (`results/rulebook/v2/v1_under_v2_cp1/`;
      CP0's folder is registered and never rewritten): 1 of 8 (dc22).
    - Measured on the recordings: per move, memory is much stronger than per
      case (tu93 0.85, tr87 0.48, m0r0 0.45), because most moves repeat cases.
      vc33's fit split has 2 changing cases; ft09's fit mask is empty.
    - Not built (cut lines): the simplification pass, `api.t` move-by-move
      grading (`Api` has `t`/`attempt` slots, always None).
    - *Keep v1 reproducible:* do not edit `tools/wm_offline.py` or the prompt
      text in `wm/rules.py` (v1's answer cache is keyed on the prompts). v2 runs
      write under `results/rulebook/v2/`.
  - Offline LLM jobs run detached through
    `experiments/rulebook/llm_run.sh start|status|stop <job>` (~2 h for the
    near-miss job). While one runs, do not edit `custom_agents/wm/*.py` or
    `gridtools.py`: its check workers re-import them. The server takes ~88% of
    the GPU, so ask whether Matt is using it; the PC must stay on AC power.
  - v1's result folders are frozen: registered, read-only, copy in
    `results/rulebook_v1_frozen_2026-10-07.tar.gz`. `tier0a.sh` refuses to run.

`legacy/llm_track/` (semester-1 LLM work, all NO-GO or unfinished) is frozen.

## Key conventions

- ALL run output goes under `results/` (gitignored, override with
  `EVAL_RESULTS_DIR`): `results/runs/<ts>/<game>/` (corpus, `run_config.json`,
  `run_end.json`, tensorboard, `metrics.json`), `results/sweeps/` (manifests,
  summaries, logs), `results/local_suite.csv`, `results/confirm_upgrade/` (the
  75 `mb_gated_att` reference runs), `results/rulebook/`. Nothing in `results/`
  is backed up by git; v1's rule candidates exist only there.
- Manifests (`run_dir<TAB>game<TAB>seed<TAB>arm`) are read through
  `manifest.py` only. Grid helpers (bars, blobs, objects) come from
  `custom_agents/gridtools.py` only.
- ALL agents live in `custom_agents/` and are registered in its `__init__.py`.
- `legacy/` (`llm_track/`, `coach_track/`, API-path scripts) must stay a leaf: it
  may import the baseline, nothing outside it may import it, and baseline
  tooling must never invoke it or `.venv-llm`. `make check` enforces both.
- The `arc-agi` package (provides `arcengine`) is needed for the local engine
  but not declared in `requirements.txt` — install separately.
- With `EVAL_LABEL` at its default the agent is the old baseline. The return
  map and the upgrades are removable: every hook line in `action.py` ends in
  `# [return-map]` (25 lines) or `# [upgrades]` (26, plus 2 each in the Makefile
  and `check_repo.py`), and `sed -i '/\[return-map\]/d'` (or `[upgrades]`)
  restores the file without it byte for byte. Keep that property; new features
  get their own tag and an `EVAL_*` flag that defaults to off. Map variants
  SUBCLASS `ReturnMap`; never edit `return_map.py` or `canon.py` for a candidate.
- Any change that should not alter behaviour is checked with
  `tools/cpu_check.py` before it is committed, for `goose` and `mb_gated_att`.
- Parallel runs share the GPU: 4 at once total ~162 act/s vs 140 for one.
- Do NOT change `EVAL_RESET_ON_LEVEL` semantics or any hyperparameters
  (learning rate, `train_frequency`, batch size, buffer capacity, confidence
  coefficients) without explicit approval — they'd confound ablation results.
- Commit and push only when asked.

## Metric definitions

All metrics use the unified indicator-cell canonicalizer (`metrics_common.py`):
fixed tickers (cell changing in >=95% of transitions) plus rotating tickers
(compact cell set covering >=95% of tiny <=2-cell transitions, if those make up
>=30% of the run).

**Never report an exploration metric alone.** A high `meaningful_change_rate`
only means the frame keeps changing — an agent jiggling a decorative animation
scores 100%. Read change rate, redundancy and coverage together.

- `n_actions` vs `actions_taken` — two counts, do not mix them. `n_actions` is
  RECORDED moves (what every per-action rate divides by). `actions_taken` is the
  action counter (what `EVAL_MAX_ACTIONS` caps and level-up times are on).
  Budgets, curves and AULC use `actions_taken`.
- `unique_states_per_action` — the coverage number. Use this, not `discovery_auc`.
- `discovery_auc` — DEPRECATED for cross-run comparison (normalized by the final
  unique-state count, so it measures curve shape only). Never report it without
  `unique_states_per_action` beside it.
- `novelty_late_per_1k` — new canonical states per 1k actions over the final 10%.
  The stall detector, and a leading indicator. Baseline medians: tu93 0.0 (dead),
  dc22 2.4, ls20 7.4, g50t 9.9, ft09 955 (still discovering).
- `series` in `metrics.json` — per-1000-action novelty / meaningful / redundancy.
- Levels vs. budget, AULC, RHAE — `analyze_curves.py`. Always report AULC next
  to its `T_max`.
- Uplift — every game-dependent metric needs the matched random floor
  (`make random`) to be comparable across games.
- Verdict rules (`tools/paired_compare.py --rule`): `adopt` (levels >=, wins >
  losses; Plan B, Option 1, upgrades), `dev` (levels >=, wins >= losses; Rulebook
  dev gates), `confirm` (levels strictly more, wins > losses; Rulebook confirms).
  All add "no game worse on every seed".

Censoring discipline: most runs never reach level k. Always report
"k/n seeds reached" next to any actions-to-level median.

## Known measurement caveats

- **The rotating-ticker detector misses most tickers** (found 2026-09-04, NOT
  fixed in the scorer: changing the canonicalizer moves every published number,
  so it is an advisor decision). Its unit is the whole transition, so a ticker is
  only seen when it ticks alone. ft09, ls20, ar25, lp85, dc22 and g50t are scored
  with an EMPTY decorative mask; on ft09 that inflates `unique_states_per_action`
  about 1.4x. The agent's own `bars` upgrade and the Rulebook's evidence mask use
  the component-level fix (`gridtools.tick_cells`); the published scorer does not.
- **Runs are not reproducible past the first training step.** Two runs with the
  same `EVAL_SEED` are identical for ~300 actions, then nondeterministic CUDA
  kernels diverge the weights. `EVAL_SEED` fixes the initial conditions, not the
  trajectory. **Never claim a byte-identical comparison between two arms**;
  compare distributions across seeds. (CPU runs ARE reproducible: `cpu_check.py`.)
- **Recordings changed on 2026-10-06.** Before: the move that ended each attempt
  was not recorded (the action number steps by 3 there). After: it is recorded
  with `game_over=1` (steps by 2: the reset is an action, not a move). The 75
  `mb_gated_att` reference runs are the old kind: the median run recorded 98% of
  its actions, tu93 88%. Levels are unaffected. Per-action rates are not strictly
  comparable across the change on games with many game overs, and AULC recomputed
  now is 0.001–0.04 higher than in older reports (the budget window used the
  recorded count).
- **The adopted agent's walk-back bandit** credits a new level's first attempt to
  whichever arm was active before the level-up (`upgrades.py`, noted in the
  code). Left as confirmed; changing it needs its own test.
- **Dropout stays on while Goose picks actions** (the model is never put in eval
  mode). Inherited from upstream; part of the baseline. Do not "fix" it.
