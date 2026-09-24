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
        sweep long random curves metrics label-diag planb-dev planb-confirm map-dev planb-pause planb-status planb-stop tensorboard clean action baseline

help:
	@echo "Baselines (what teammates use):"
	@echo "  make install                        set up the venv"
	@echo "  make check                          health check: isolation, agents, deps"
	@echo "  make suites                         list the frozen benchmark suites"
	@echo "  make bench SUITE=standard AGENT=x   run a frozen suite (backgrounded)"
	@echo "  make bench-fg SUITE=smoke           same, in the foreground"
	@echo "  make compare M1=<manifest> M2=<..>  put two agents side by side"
	@echo "  make local GAME=ft09 CAP=2000       one ad-hoc run"
	@echo ""
	@echo "Analysis:"
	@echo "  make metrics DIR=<transitions> GAME=x   score one run"
	@echo "  make curves MANIFEST=<manifest>         levels-vs-budget, AULC"
	@echo "  make label-diag                         Plan B step 1: change vs novel label rate per game"
	@echo "  make planb-dev [DRY_RUN=1] [RESUME=m]   Plan B step 3: dev sweep, arms A0-A3, detached"
	@echo "  make planb-status                       progress of the detached Plan B sweep"
	@echo "  make planb-confirm [DRY_RUN=1] [RESUME=m] Plan B step 4: Confirm tier, A0 vs A1, all 25 games x 3 seeds, detached"
	@echo "  make map-dev [DRY_RUN=1] [RESUME=m]      Option 1 dev test: novelty label with vs without the map, detached"
	@echo "  make planb-pause                        pause after the current run finishes (loses nothing; resume with RESUME=)"
	@echo "  make planb-stop                         stop it now (loses only the run in progress; resume with RESUME=)"
	@echo "  make tensorboard"
	@echo ""
	@echo "Archive: legacy/ (API-path scripts; semester-1 LLM track in legacy/llm_track/, frozen)"
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

# Plan B step 3 (docs/plans/plan-B-goose-novelty.md §7.3), the SMALL first
# pass: 6 dev games x 3 seeds x 100k actions x 4 arms = 72 runs, ~15 h.
# Override PB_GAMES / PB_SEEDS / PB_CAP / PB_ARMS on the command line.
# Started with setsid so it survives this terminal or the Claude window
# closing (not `wsl --shutdown` or the machine sleeping). DRY_RUN=1 prints
# the plan and ETA without running anything. After a sleep/reboot, resume
# with RESUME=<its manifest>: finished (game, seed, arm) runs are skipped.
PB_GAMES ?= ls20 dc22 g50t tu93 ft09 lp85
PB_SEEDS ?= 0 1 2
PB_CAP   ?= 100000
PB_ARMS  ?= A0 A1 A2 A3
PB_LOG   := $(RESULTS)/sweeps/planb_dev.log
planb-dev:
	mkdir -p $(RESULTS)/sweeps
ifeq ($(DRY_RUN),1)
	DRY_RUN=1 ARMS="$(PB_ARMS)" GAMES="$(PB_GAMES)" SEEDS="$(PB_SEEDS)" CAP=$(PB_CAP) bash sweep.sh
else
	@pgrep -f "^bash sweep\.sh" >/dev/null && { echo "a sweep is already running (make planb-status)"; exit 1; } || true
	RESUME="$(RESUME)" ARMS="$(PB_ARMS)" GAMES="$(PB_GAMES)" SEEDS="$(PB_SEEDS)" CAP=$(PB_CAP) \
	  setsid nohup bash sweep.sh >> $(PB_LOG) 2>&1 < /dev/null &
	@echo "started detached; log: $(PB_LOG)   progress: make planb-status"
endif

# Plan B step 4 (plan §7.4), the Confirm tier: the dev-sweep pick (A1, the
# novelty label) against the baseline on every public game x 3 seeds x 100k.
# 25 games x 3 seeds x 2 arms = 150 runs, ~30 h. Same log, status, pause,
# stop and RESUME as planb-dev.
PC_GAMES ?= $(shell uv run python -c "import benchmark; print(' '.join(benchmark.ALL_GAMES))")
PC_SEEDS ?= 0 1 2
PC_CAP   ?= 100000
PC_ARMS  ?= A0 A1
planb-confirm:
	mkdir -p $(RESULTS)/sweeps
ifeq ($(DRY_RUN),1)
	DRY_RUN=1 ARMS="$(PC_ARMS)" GAMES="$(PC_GAMES)" SEEDS="$(PC_SEEDS)" CAP=$(PC_CAP) bash sweep.sh
else
	@pgrep -f "^bash sweep\.sh" >/dev/null && { echo "a sweep is already running (make planb-status)"; exit 1; } || true
	RESUME="$(RESUME)" ARMS="$(PC_ARMS)" GAMES="$(PC_GAMES)" SEEDS="$(PC_SEEDS)" CAP=$(PC_CAP) \
	  setsid nohup bash sweep.sh >> $(PB_LOG) 2>&1 < /dev/null &
	@echo "started detached; log: $(PB_LOG)   progress: make planb-status   pause: make planb-pause"
endif

# Option 1 dev test (docs/plans/option-1-return-map.md): the adopted agent (A1,
# novelty label) with and without the return map (A4), on the 6 Plan B dev
# games plus two keyboard games nothing has ever solved. 8 games x 3 seeds x
# 100k x 2 arms = 48 runs, ~10 h. Same log, status, pause, stop and RESUME.
MD_GAMES ?= ls20 dc22 g50t tu93 ft09 lp85 re86 wa30
MD_SEEDS ?= 0 1 2
MD_CAP   ?= 100000
MD_ARMS  ?= A1 A4
map-dev:
	mkdir -p $(RESULTS)/sweeps
ifeq ($(DRY_RUN),1)
	DRY_RUN=1 ARMS="$(MD_ARMS)" GAMES="$(MD_GAMES)" SEEDS="$(MD_SEEDS)" CAP=$(MD_CAP) bash sweep.sh
else
	@pgrep -f "^bash sweep\.sh" >/dev/null && { echo "a sweep is already running (make planb-status)"; exit 1; } || true
	RESUME="$(RESUME)" ARMS="$(MD_ARMS)" GAMES="$(MD_GAMES)" SEEDS="$(MD_SEEDS)" CAP=$(MD_CAP) \
	  setsid nohup bash sweep.sh >> $(PB_LOG) 2>&1 < /dev/null &
	@echo "started detached; log: $(PB_LOG)   progress: make planb-status   pause: make planb-pause"
endif

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

clean:
	rm -rf ./$(RESULTS)/runs

# --- API path (original, unchanged apart from where recordings land) ---
action:
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action

baseline:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action --game=$(GAME)
