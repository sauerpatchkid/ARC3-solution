# CLAUDE.md — StochasticGoose (Team B, ARC-AGI-3)

What is true now and what must not be broken. History lives in `docs/` (map:
`docs/README.md`); finished code lives in `legacy/` (index: `legacy/README.md`).

## The project in four parts

The README, the docs and the folders all follow these four, so keep new work in them:

1. **Goose upgrades (no LLM)**: the exploring agent, Goose (a CNN that learns
   online which actions change the screen), and the semester-2 improvements to
   it. 54 → 79 → 112 levels on 25 games × 3 seeds × 100k.
   Code: `custom_agents/action.py`, `canon.py`, `return_map.py`, `upgrades.py`,
   `carry.py`, `presets.py` (index: `custom_agents/README.md`). Docs:
   `docs/1-goose-upgrades/`.
   Launchers: `legacy/goose_experiments/`.
2. **Glassbox (current)**: Goose plus a local LLM that writes small rules,
   each replay-checked, in separate stages that each fix one problem.
   Code: `custom_agents/gb/` (one file per stage; `gb/README.md` has the
   problem each fixes). Docs: `docs/3-glassbox/`.
3. **Previously tried methods**: the judge, the Coach, Rulebook v1 and v2. All
   stopped at their gates. Docs: `docs/2-llm-history/`. Code: frozen in `legacy/`.
4. **Future methods**: ideas that are not built. `future/README.md` has this
   week's list, the nine ideas of the 10 Nov review
   (`future/improvement-lists-2026-11-10.md`) and the small fixes still open.
   An idea is built in its real home (`custom_agents/` or `custom_agents/gb/`),
   off by default behind its own flag; its write-up goes to `docs/` once it has
   a result.

The `docs/` folders are numbered in the order things happened (1 Goose
upgrades, 2 LLM history, 3 Glassbox), which is why Glassbox's folder is `3-`.

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
uv run python tools/levels_by_budget.py --base <manifest>:<arm> --new <manifest>:<arm> \
    [--games a,b] [--by-game]             # WHEN levels came: by budget, AULC, level races; no verdict
bash experiments/goose/carry_test.sh verdict  # the carry test (finished 13 Nov, FAIL): its tables again; also status
uv run python tools/gb_funnel.py --runs g1,pool_g0b_g1,g3_gb   # what each Glassbox stage did, per game (CPU)
uv run python tools/frozen_moves.py --runs <manifest>:<arm>    # moves on a screen nothing changed (CPU)
uv run python tools/probe_engine.py [--games a,b]              # frames per move, what ACTION7 does (CPU)
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
| `EVAL_UPGRADES` | unset | comma list from `bars`, `attempt`, `map_gated`, `deadclick`, `carry_noops`, `carry_hints` (the last two: `custom_agents/carry.py`, not adopted) |
| `EVAL_GB` / `EVAL_GB_BOOK` | off / unset | Glassbox's S5 guide in the agent (`gb/s5_act.py`); the book = an offline run under `results/glassbox/` (unset: GB-free); hooks tagged `# [gb]` |

Always run with `PYTHONHASHSEED=0`. Removed on 2026-10-06, each stops the run
with a pointer to its git tag: `EVAL_MASK_TRIED` and the upgrades `graded`,
`map_objects`, `map_diverse`.

## Where things stand (13 Nov 2026)

**Goose upgrades: done** (`mb_gated_att` adopted 28 Sep, 112 levels; the carry
upgrade was tested 12-13 Nov and **failed**, so it is not adopted).
**Previously tried methods: closed** (Coach NO-GO; Rulebook v1 FAIL, 2 of 8;
Rulebook v2 closed after CP1: FAIL 2 of 8, 0.0% wrong at T1). **Glassbox: every
gate is run**, and G4 adopted it on weak evidence. **Now: the buffer, to Tue
17 Nov**, then the report (final Mon 7 Dec). The history is in `docs/`; this
section holds only what a new session needs.

### Glassbox's results

Each row is written up in `docs/3-glassbox/` under the file named, and the
write-up names its run folder under `results/glassbox/`. Gates are in
`docs/3-glassbox/glassbox-prereg.md` (sections 1 to 9 registered, 10 not used);
the running log is plan §7.4 (`docs/3-glassbox/glassbox-plan.md`).

| Test (prereg section) | Run | Verdict | The numbers | Write-up |
|---|---|---|---|---|
| G0 (3) | 16 Oct | no rescue | the guard fitter rescued nothing; m0r0's near-misses were a mask gap (row 63's bar) | `g0-guard-rescue.md` |
| G0b (3b) | 16 Oct | fixed | with S0 all 11 trusted; m0r0 0% → 53.5% of changed cells; T1 51.7% (memory 53.9%) | `g0b-bar-mask.md` |
| G1 (4) | 17 Oct | **FAIL** | 3 of 8 above G0b, safety met; tu93 0 → 59.0%, tr87 0 → 35.5%, dc22 11.8 → 29.3%; ft09, g50t, vc33, cd82 lost ground | `g1-offline-rules.md` |
| Pooled book `pool_g0b_g1` | 18 Oct | exploratory | never below the better of G0b and G1; above both on dc22 40.4%, g50t 18.1%, cd82 6.6% | `pool-g0b-g1.md` |
| Offline counterfactual (5a) | 18 Oct | as registered | carried no-ops would have covered vc33 60.1% and ft09 22.9% of A0's moves in each new level's first 2,000 actions; masking <= 0.05%; GB-free nearly the same | `cf-counterfactual.md` |
| G2 (5b) | 18-19 Oct | **FAIL** | GB 34 levels vs A0's 37 on the 16 dev runs (1 / 12 / 3); GB-free 35 | `g2-online.md` |
| A0 run again | 19 Oct | exploratory | unchanged `mb_gated_att` 33 vs its own 37 of 27 Sep; noise ~±2.6 levels per 16 runs | `a0-fresh.md` |
| dc22's no-op rules | 2 Nov | exploratory | 4,160 moves covered on 9 recordings, 0 changed anything; the level-2 stall is dc22's | `dc22-noop-check.md` |
| The book on every later level | 2 Nov | exploratory | right wherever it applies only on ft09 L3 and tr87 L2; plan §7.3's IMAGINE precondition is met by no game | `carry-later-levels.md` |
| G3 (6) | 2-3 Nov | **FAIL** | 0 of 792 template tries adopted on the 8 A/B-offline games; GB there: mean cell coverage 13.8% (G1 on dev 32.0%), 46% of answers out of room | `g3-templates.md` |
| S5 v2 check | 3 Nov | exploratory | with the agent's mask, v2 carries the counterfactual's colours on all 10 recordings | `s5-v2.md` |
| Dev sweep 2 (7a) | 4-5 Nov | **PASS** | GB 72 vs a same-day A0's 67 on 8 dev games x seeds 3-6 (6 / 24 / 2); ~1.6 SD | `dev-sweep-2.md` |
| **G4 (7)** | 7-8 Nov | **ADOPT** | GB 110 vs a same-day A0's 101 on 25 games x seeds 0-2 (12 / 57 / 6), no game worse on all 3 seeds | `g4-confirm.md` |
| Time to level | 10 Nov | exploratory | finished runs re-scored: by budget GB leads the same-day A0 by +1 / +2 / +6 / +9 at 2k / 10k / 30k / 100k (two unguided arms differ by 11 at 100k); level by level, GB finished levels 2 and up in fewer actions on 36 vs 17 in G4 and 23 vs 16 in dev sweep 2 (unguided vs unguided 24 vs 27; level 1 23 vs 28); G2's LLM-free arm 14 vs 7, as the guided one. **Did not repeat** in the registered carry test (13 Nov: 23 vs 22) | `time-to-level.md` |
| The funnel, and G4's replay | 12 Nov | report table | mean over 8 games, share of a level's changing cases: S1 describes 83%, an admitted rule gets 57%, the book gets the whole screen on 28% (pooled book, dev); 78% / 40% / 5% on the held-out games; rule survival online 92%; G4's replay carries the 66 colours the runs carried, 65 at the same level | `funnel.md` |

**How to read G4.** +9 is ~1.1 standard deviations of chance (the rule passes
36% with no effect); the same-day A0 came in 11 below A0's 27 Sep runs (same
code), and GB is level with those (110 vs 112). By tier: dev 54 vs 50 (tuned),
seen 20 vs 19, A/B-offline 22 vs 19, untouched 14 vs 13. The gain sits where
the guide acted: +7 on the 11 games where it changed >= 5% of moves, +1 on the
9 where it changed nothing. Outside the dev tier GB used no LLM output; its
changes there were mostly goal hints (s5i5 +3, r11l +2). G4 had no LLM-free arm.

**The carry test (12-13 Nov): FAIL.** The guide's LLM-free part (colour no-ops,
goal hints), tested alone as a Goose upgrade with a rule registered first:
`A6` (the adopted Goose + `carry_noops,carry_hints`) against `A5` (the adopted
Goose) on vc33, ft09, cd82, dc22, m0r0, lp85 and tu93 (placebo) x seeds 7-10 x
100k = 56 runs. Primary: on levels 2 and up of the six click games A6 finished
23 levels in fewer actions and A5 22 (+1, 95% interval [-15, +18]). Safety met:
87 vs 81 levels (8 / 15 / 5), no game worse on every seed; +3 of the +6 is tu93,
where the carry changed no move. Level 1, where the arms are the same agent,
split 13 vs 7: chance moves these counts more than the carry did. The carry
changed 9.4% of A6's moves. **Nothing adopted, no preset.** So the guide's
effect on levels and on time to level is **not shown**; the 10 Nov lead was
exploratory and did not repeat. Write-up:
`docs/1-goose-upgrades/carry-results.md`; tables `results/sweeps/carry_test_*`.

### What is built

`custom_agents/gb/`, one file per stage. `gb/README.md` has each file's full
status, `make gb-help` each command. 284 live tests.

| Piece | File | State |
|---|---|---|
| S0 MASK | `s0_mask.py` | built; `[mask] complete_bars` off rebuilds CP1's evidence exactly |
| S1 SEE | `s1_see.py` (VERSION 3) | built; "explained" is S1's measure (the plan's "closure" cannot fail) |
| H1 VOCAB | `h1_vocab.py` | built; **registered `glassbox_vocab_v1`, frozen** |
| S2 WRITE v1, S2 POOL | `s2_write.py`, `s2_pool.py` | built |
| S3 CHECK | `s3_check.py`, `s3_guards.py` | grading and the guard fitter built; priorities and trust tiers not built |
| S4 CARRY v1 | `s4_carry.py` | built, with `--later`; `[carry] roles_in_bind` is off by default, so `make gb-g0b-control` still equals CP1; repair calls not built |
| S5 ACT v2 | `s5_act.py` (VERSION 2) | built: prune, goal hints, `GbMap`. **v2 is on in `glassbox.toml`, off in the code's defaults.** Probes not built; IMAGINE and contact goals dropped 2 Nov (plan §7.3) |
| H3 TEMPLATES v1 | `h3_templates.py` | built; the catalog `artifacts/glassbox_templates_v1.jsonl` is **registered `glassbox_templates_v1`, frozen** |
| S6 SHOW v0 | `s6_show.py` | built: `make gb-report RUN=<run>` |
| H2 SCHEDULER | (no file) | not built: no LLM runs online |

- **The agent hook:** `EVAL_GB=1` (+ `EVAL_GB_BOOK=<run>`; unset = GB-free, no
  LLM rules), 16 lines tagged `# [gb]` in `action.py`. With it unset,
  `cpu_check` is identical on every move (goose/ft09 1,970, mb_gated_att/tu93
  1,960; 18 Oct). `s5_act` is imported only under `EVAL_GB=1`, so a change to
  it alone needs no `cpu_check`.
- **Drivers:** `tools/gb_offline.py` (`--stage mask|see|pool|write|check|carry|
  guards|replay|templates|catalog|show`); `tools/gb_sweep.py` (online arms GB,
  GB-free, A0-fresh, 4 at a time; `make gb-status` / `gb-pause` / `gb-stop`,
  `RESUME=1`); verdicts `tools/gb_gate.py` and `tools/paired_compare.py`.
- **What the drivers refuse:** the replay on dev games without `--cf`; S5
  online on dev games without `--g2`, `--dev2` or `--g4`; A/B-offline games
  without `--g3`; untouched games outside G4 (`--g4` runs G4 whole or not at
  all). Each flag belongs to a registered section that has been run. A new run
  on those games needs its own entry in the pre-registration first.
- **Proven equal to CP1:** `make gb-import-cp1` (437 of 437); `make
  gb-g0b-control`.
- **Settings:** `experiments/glassbox/glassbox.toml`, hash `8c8e1cb023c827d2`
  (G4's). 35 of its settings are read by no code and are marked `[unread]`;
  they stay because the hash is registered.

### Dates

Every gate was run by 8 Nov. **Everything in the plans is finished by Tue 17
Nov** (Matt, 3 Nov), then 2-3 weeks of write-up. Course: final report Mon 7 Dec.
Cut: the online ablations (±2.6 levels of noise per 16 runs makes 8-run arms
uninformative) and the frontier showcase (only if the buffer allows).

### Now (13 Nov)

- **Committed and pushed** (Matt, 12 Nov): the tidy-up (`26ddcb6`), the small
  fixes (`e4f0f77`), the re-score and probes (`2443dd3`), the buffer's CPU
  jobs (`44dbcbe`), the carry upgrade (`c2f8d85`), the carry test's launcher
  (`a9fec48`) and its registration (`57e178c`); and (Matt, 13 Nov) the carry
  test's result.
- **The carry test is finished (13 Nov): FAIL, nothing adopted** (item 4).
  Its write-up (`docs/1-goose-upgrades/carry-results.md`) and the pages that
  point to it are committed and pushed; no code changed. Matt, 13 Nov:
  "commit and push, keep the carry switches".
- **This week's list** (Matt, 10 Nov; changed by him on 12 Nov; `future/README.md`
  has the detail). GPU runs are asked for one by one:
  1. ~~Re-score G4 and dev sweep 2 on time-to-level~~ **done 10 Nov**
     (`docs/3-glassbox/time-to-level.md`, `tools/levels_by_budget.py`): the
     guide's LLM-free carry seemed to make levels after the first come sooner
     (exploratory; it did not repeat, item 4); the level count stays inside
     the noise.
  2. ~~Frozen runs~~ and 3. ~~the engine probe~~ **done 10 Nov**
     (`docs/1-goose-upgrades/input-probes.md`): no run is frozen; ACTION7 is an
     undo on three of its six games; in-between frames matter on ft09, sc25 and
     cd82 only.
  4. **Carry inside Goose** (it replaced "a reset watchdog and ACTION7"; Matt,
     12 Nov). `custom_agents/carry.py`, switches `carry_noops` and
     `carry_hints` in `EVAL_UPGRADES`: colour no-ops and goal hints carried
     from one level to the next, with no LLM. It is a port of S5 v2 with no
     rule book (GB-free). **Built 12 Nov.** `cpu_check`
     (`results/cpu_check/2026-11-12_carry/`): with the switches off,
     `goose`/ft09 and `mb_gated_att`/tu93 are identical on every move (1,970
     and 1,960); with them on, Goose makes GB-free's moves on ft09 (2,934 of
     2,934 over 3,000 actions; the upgrade changed 1,114 of them).
     **Its test was registered on 12 Nov (code `a9fec48`) and run 12-13 Nov**
     (`docs/1-goose-upgrades/carry-inside-goose.md`), in two parts, 56 of 56
     runs on `57e178c`. **Verdict 13 Nov: FAIL** (the numbers are above, under
     "The carry test"; write-up `docs/1-goose-upgrades/carry-results.md`).
     **Not adopted: no preset `mb_carry`.** The two switches stay in the code,
     off by default, like `deadclick` (Matt, 13 Nov: keep them). `bash
     experiments/goose/carry_test.sh verdict` prints the tables again.
  5. "LLM plays first, Goose takes over": **on hold** (Matt, 12 Nov).
  6. ~~Owed for the report~~ **done 12 Nov**: G4's 24 dev-game GB recordings
     replayed with the agent's mask (66 colours, 65 at the same level; the
     66th is a threshold case, `g4-confirm.md` "S5 online"); the funnel table
     (`docs/3-glassbox/funnel.md`, `tools/gb_funnel.py`); S6 pages for
     `pool_g0b_g1` and `g3_gb`, with the three funnel figures copied to
     `docs/3-glassbox/figures/`.
- The `glassbox` preset is skipped for now (Matt, 8 Nov). No IMAGINE, no
  contact goals (§7.3). Last handoff: `results/handoff/2026-11-08_g4_done.md`.

### Open, known limits

- **Still OPEN, Matt's:** how the report words G4 now that the carry test
  failed; the `glassbox` preset (G4 adopted GB; its book lives under
  `results/`, outside git); the instructor's OK for item 5 above, and
  which model it uses (on hold); confirm "explained" as the perception number; plan §10
  (a) (advisor and the 30 Oct draft) and (b) (teammates' modules in S5).
- **Known limits:** S1's `sits_on` is read from neighbouring colours (pinned in
  `tests/test_gb_vocab.py`); the binding search's time budget leaves bindings
  untried on tr87 when the machine is busy (run S4 on a quiet machine; H3's
  try too: G3's budget cut 118 of 792 tries short); the
  `ARC-AGI-3-Agents` submodule's local edits make manifests say `+dirty`
  (prereg §11). `--set` overrides are recorded in stage headers, not in the
  run's copied `config.toml`. S5 v1's online colour carry lagged its replay
  (dc22's check; fixed in v2). The agent's own mask (`canon.py`, the bars
  upgrade) still lacks bars' last cells; only S5 completes them. S0's mask misses a bar that changes 3 cells a move (tu93 L4;
  `gridtools.tick_cells` too); a mask kept across levels covers it.
- **Frontier models:** everything so far is the local model (Qwen3.8-27B). The
  plan's only frontier use, an optional showcase (plan §6.5), was cut (prereg
  §10 is not used). A hosted model for item 5 above would be new: Matt's
  decision, with the instructor.

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
  Two deletions were made on 10 Nov 2026 with Matt's OK, listed in
  `results/DELETED_2026-11-10.md`: the `tensorboard/` folder of every run
  (levels and metrics come from the `.npz` recordings) and 73 July run folders
  that no manifest lists. Every manifest row still resolves.
- `future/` holds ideas that are not built, as pages, never as live code. Built
  code goes in its real home behind a flag (see "The project in four parts").
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
