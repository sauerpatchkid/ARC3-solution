# Convenience targets. Override vars on the command line, e.g.
#   make local GAME=ft09 CAP=2000
#   make bench SUITE=standard AGENT=goose
#   make metrics DIR=results/runs/<ts>/ft09/transitions GAME=ft09
#
# START HERE (new clone):
#   make install && make check && make bench SUITE=smoke
#
# All outputs land under $(RESULTS) (gitignored):
#   results/runs/ results/sweeps/ results/local_suite.csv results/curriculum_suite.csv
SEED    ?= 0
CAP     ?= 2000
GAME    ?= ft09
GAMES   ?= ft09,dc22,ls20
RESULTS ?= results
AGENT   ?= goose
SUITE   ?= standard

.PHONY: help install check suites bench bench-fg compare local curriculum \
        sweep long random curves metrics label-diag planb-dev planb-confirm map-dev map-confirm planb-pause planb-status planb-stop tensorboard clean action baseline

help:
	@echo "Baselines (what teammates use):"
	@echo "  make install                        set up the venv"
	@echo "  make check                          health check: isolation, agents, deps"
	@echo "  make suites                         list the frozen benchmark suites"
	@echo "  make bench SUITE=standard AGENT=x   run a frozen suite (backgrounded); AGENT=goose is the"
	@echo "                                      semester-1 baseline, AGENT=mb_gated_att the adopted Goose"
	@echo "  make bench-fg SUITE=smoke           same, in the foreground"
	@echo "  make compare M1=<manifest> M2=<..>  put two agents side by side"
	@echo "  make local GAME=ft09 CAP=2000       one ad-hoc run"
	@echo ""
	@echo "Analysis:"
	@echo "  make metrics DIR=<transitions> GAME=x   score one run"
	@echo "  make curves MANIFEST=<manifest>         levels-vs-budget, AULC"
	@echo "  make label-diag                         Plan B step 1: change vs novel label rate per game"
	@echo "  make planb-dev [DRY_RUN=1] [RESUME=m]   Plan B step 3: dev sweep, arms A0 A1, detached"
	@echo "  make planb-status                       progress of the detached Plan B sweep"
	@echo "  make planb-confirm [DRY_RUN=1] [RESUME=m] Plan B step 4: Confirm tier, A0 vs A1, all 25 games x 3 seeds, detached"
	@echo "  make map-dev [DRY_RUN=1] [RESUME=m]      Option 1 dev test: novelty label with vs without the map, detached"
	@echo "  make map-confirm [DRY_RUN=1] [RESUME=m]  Option 1 Confirm: the map on all 25 games, vs the Plan B Confirm runs"
	@echo "  make planb-pause                        pause after the current run finishes (loses nothing; resume with RESUME=)"
	@echo "  make planb-stop                         stop it now (loses only the run in progress; resume with RESUME=)"
	@echo "  make upgrade-screen [DRY_RUN=1] [RESUME=m]  screen every candidate upgrade, short runs, ranked"  # [upgrades]
	@echo "  make tensorboard"
	@echo ""
	@echo "  make clean CONFIRM=yes                  delete EVERY recorded run under results/runs"
	@echo ""
	@echo "Archive: legacy/ (API-path scripts; semester-1 LLM track in legacy/llm_track/ and the"
	@echo "         Coach in legacy/coach_track/, both frozen)"
	@echo ""
	@echo "Add your own agent: see custom_agents/__init__.py and TEMPLATE.py"

install:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt
	@echo ""
	@echo "Now run: make check"

# Health check: legacy/ isolation, agent registry, dependencies, suites.
check:
	uv run python check_repo.py

# ---------------------------------------------------------------------------
# THE FROZEN BENCHMARK — everyone runs the same games/seeds/budget so results
# can be compared. Definitions live in benchmark.py; see that file before
# changing anything.
# ---------------------------------------------------------------------------
suites:
	uv run python benchmark.py

# Backgrounded (suites other than 'smoke' take hours). Log + manifest paths
# are printed immediately; the manifest is what `make compare` consumes.
bench:
	mkdir -p $(RESULTS)/sweeps
	BENCH=$(SUITE) AGENT=$(AGENT) nohup bash sweep.sh \
	  > $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log 2>&1 &
	@echo "started: $(SUITE) suite, agent=$(AGENT)"
	@echo "log:     $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log"
	@echo "watch:   tail -f $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log"

bench-fg:
	BENCH=$(SUITE) AGENT=$(AGENT) bash sweep.sh

# Side-by-side table. M1 is the baseline, M2+ are compared against it.
compare:
	uv run python compare.py $(M1) $(M2) $(M3) \
	  --out $(RESULTS)/sweeps/comparison.md

# Local engine, one game (fast dev path, ~120 act/s).
local:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_local.py --game=$(GAME)

# Persistent-brain curriculum across several games (cross-game transfer).
curriculum:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_curriculum.py --games=$(GAMES)

# Ad-hoc exploratory sweep (your own GAMES/SEEDS/CAP; NOT comparable between
# people - use `make bench` for anything you intend to report).
sweep:
	mkdir -p $(RESULTS)/sweeps
	nohup bash sweep.sh > $(RESULTS)/sweeps/sweep.log 2>&1 &

# Long single-seed run on the two games that actually complete levels, to see
# how far past level 2 the agent can get. 2M actions is ~4h/game at ~142 act/s,
# so ~8h for the pair - just under action.py's 7h55m per-process wall clock,
# which each game gets fresh because sweep.sh runs one process per game.
long:
	mkdir -p $(RESULTS)/sweeps
	GAMES="ft09 tu93" SEEDS="0" CAP=2000000 \
	nohup bash sweep.sh > $(RESULTS)/sweeps/long.log 2>&1 &

# Matched random-policy floor: same games/seeds/contract, uniform over the same
# masked 5+64x64 action space. ~2700 act/s (no model), so a full sweep is cheap.
# Needed to turn change-rate / redundancy / coverage into uplift ratios.
random:
	mkdir -p $(RESULTS)/sweeps
	AGENT=random GAMES="ft09 tu93 g50t dc22 ls20" SEEDS="0 1 2 3 4" CAP=200000 \
	nohup bash sweep.sh > $(RESULTS)/sweeps/random.log 2>&1 &

# Levels-vs-budget curves, AULC and (with --human-baselines) RHAE, from the
# metrics.json files a sweep already wrote. Pure post-processing.
#   make curves MANIFEST=results/sweeps/sweep_<stamp>.manifest
curves:
	uv run python analyze_curves.py $(MANIFEST) \
	--out $(RESULTS)/sweeps/curves_$(notdir $(basename $(MANIFEST)))

# Score a finished run's corpus and append a row to results/local_suite.csv.
metrics:
	uv run python compute_metrics.py $(DIR) \
	--game=$(GAME) --seed=$(SEED) --suite $(RESULTS)/local_suite.csv

# Plan B step 1 (docs/plans/plan-B-goose-novelty.md): replay every recorded
# corpus and report the positive rate of the change label vs the novelty label.
# Pure post-processing over results/runs; touches neither agent nor scorer.
label-diag:
	mkdir -p $(RESULTS)/diagnostics
	PYTHONHASHSEED=0 uv run python tools/label_diagnostic.py --results $(RESULTS)

# ---------------------------------------------------------------------------
# Arm sweeps: games x seeds x arms (A0 change label, A1 novelty label, A4
# novelty label + return map), DETACHED with setsid so they survive this
# terminal or the Claude window closing (not `wsl --shutdown` or the machine
# sleeping). DRY_RUN=1 prints the plan and ETA without running anything.
# RESUME=<manifest> skips the (game, seed, arm) runs already in it. They share
# one log and the planb-status / planb-pause / planb-stop targets below.
#
# The four named sweeps are the finished Plan B and Option 1 experiments, kept
# so their runs can be resumed or repeated; all four go through `arm-sweep`.
#   planb-dev      6 dev games x 3 seeds x 100k x A0 A1   (plan-B §7.3; A2 A3,
#                  the tried-action mask, were removed on 2026-10-06)
#   planb-confirm  25 games x 3 seeds x 100k x A0 A1, ~30 h   (plan-B §7.4)
#   map-dev        8 games x 3 seeds x 100k x A1 A4, ~10 h    (option-1 dev test)
#   map-confirm    25 games x 3 seeds x 100k x A4, ~15 h; A1 comes from the
#                  Plan B confirm runs ($(PB_CONFIRM)), not rerun:
#     uv run python tools/paired_compare.py --base $(PB_CONFIRM):A1 --new <manifest>:A4
# ---------------------------------------------------------------------------
PB_GAMES ?= ls20 dc22 g50t tu93 ft09 lp85
PB_SEEDS ?= 0 1 2
PB_CAP   ?= 100000
PB_ARMS  ?= A0 A1
PB_LOG   := $(RESULTS)/sweeps/planb_dev.log
PC_GAMES ?= $(shell uv run python -c "import benchmark; print(' '.join(benchmark.ALL_GAMES))")
PC_SEEDS ?= 0 1 2
PC_CAP   ?= 100000
PC_ARMS  ?= A0 A1
MD_GAMES ?= ls20 dc22 g50t tu93 ft09 lp85 re86 wa30
MD_SEEDS ?= 0 1 2
MD_CAP   ?= 100000
MD_ARMS  ?= A1 A4
PB_CONFIRM := results/sweeps/sweep_20260917_224431.manifest

.PHONY: arm-sweep
arm-sweep:
	mkdir -p $(RESULTS)/sweeps
ifeq ($(DRY_RUN),1)
	DRY_RUN=1 ARMS="$(SW_ARMS)" GAMES="$(SW_GAMES)" SEEDS="$(SW_SEEDS)" CAP=$(SW_CAP) bash sweep.sh
else
	@pgrep -f "^bash sweep\.sh" >/dev/null && { echo "a sweep is already running (make planb-status)"; exit 1; } || true
	RESUME="$(RESUME)" ARMS="$(SW_ARMS)" GAMES="$(SW_GAMES)" SEEDS="$(SW_SEEDS)" CAP=$(SW_CAP) \
	  setsid nohup bash sweep.sh >> $(PB_LOG) 2>&1 < /dev/null &
	@echo "started detached; log: $(PB_LOG)   progress: make planb-status   pause: make planb-pause"
endif

planb-dev:
	@$(MAKE) --no-print-directory arm-sweep SW_ARMS="$(PB_ARMS)" SW_GAMES="$(PB_GAMES)" SW_SEEDS="$(PB_SEEDS)" SW_CAP=$(PB_CAP)
planb-confirm:
	@$(MAKE) --no-print-directory arm-sweep SW_ARMS="$(PC_ARMS)" SW_GAMES="$(PC_GAMES)" SW_SEEDS="$(PC_SEEDS)" SW_CAP=$(PC_CAP)
map-dev:
	@$(MAKE) --no-print-directory arm-sweep SW_ARMS="$(MD_ARMS)" SW_GAMES="$(MD_GAMES)" SW_SEEDS="$(MD_SEEDS)" SW_CAP=$(MD_CAP)
map-confirm:
	@$(MAKE) --no-print-directory arm-sweep SW_ARMS="A4" SW_GAMES="$(PC_GAMES)" SW_SEEDS="$(PC_SEEDS)" SW_CAP=$(PC_CAP)

# Graceful pause: the sweep checks for this file between runs, finishes the
# run in progress (<= ~12 min), records it, and exits. Nothing is lost.
planb-pause:
	@pgrep -f "^bash sweep\.sh" >/dev/null || { echo "not running"; exit 0; }
	touch $(RESULTS)/sweeps/STOP
	@echo "pause requested; the current run will finish first (watch: make planb-status)"

# Stop the detached sweep. Every finished run is already in the manifest and
# local_suite.csv, so only the run in progress (<= ~12 min) is lost; resume
# with `make planb-dev RESUME=<manifest>` and it is redone.
planb-stop:
	@pgrep -f "^bash sweep\.sh" >/dev/null || { echo "not running"; exit 0; }
	-pkill -f "^bash sweep\.sh"
	-pkill -f "run_local.py"
	-pkill -f "compute_metrics.py"
	@sleep 2; echo "stopped. resume with: make planb-dev RESUME=$$(ls -t $(RESULTS)/sweeps/sweep_*.manifest | head -1)"

planb-status:
	@pgrep -f "^bash sweep\.sh" >/dev/null && echo "RUNNING" || echo "not running"
	@[ -f $(PB_LOG) ] && grep -cE "^>>> game=" $(PB_LOG) | sed 's/^/runs started: /' || true
	@[ -f $(PB_LOG) ] && grep -E "^>>> game=|Score changed|run_local\] done|^!!|Sweep complete|^Done" $(PB_LOG) | tail -8 || true

tensorboard:
	.venv/bin/tensorboard --logdir=$(RESULTS)/runs --port=6006

# Deletes EVERY recorded run under $(RESULTS)/runs - including the Plan B
# confirm runs that later comparisons still read. Nothing can rebuild them
# except rerunning the sweeps, so it refuses without CONFIRM=yes.
clean:
ifeq ($(CONFIRM),yes)
	rm -rf ./$(RESULTS)/runs
else
	@echo "make clean deletes every recorded run in $(RESULTS)/runs:"
	@du -sh ./$(RESULTS)/runs 2>/dev/null || echo "  (nothing there)"
	@echo "Sweep manifests and paired comparisons point into that folder."
	@echo "If you are sure:  make clean CONFIRM=yes"
	@exit 1
endif

# --- API path (original, unchanged apart from where recordings land) ---
action:
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action

baseline:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action --game=$(GAME)
-include experiments/upgrade_screen/screen.mk  # [upgrades]
-include experiments/upgrade_screen2/screen2.mk  # [upgrades2]
-include experiments/upgrade_confirm/confirm.mk  # [upgrades-confirm]
