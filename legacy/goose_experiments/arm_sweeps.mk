# legacy/goose_experiments/arm_sweeps.mk - the finished Plan B and Option 1
# launchers, moved out of the root Makefile on 16 Oct 2026 (unchanged otherwise).
# Pulled in by legacy/goose_experiments/experiments.mk; run from the repo root:
#   make -f legacy/goose_experiments/experiments.mk planb-status

.PHONY: label-diag planb-dev planb-confirm map-dev map-confirm planb-pause planb-status planb-stop

# Plan B step 1 (docs/1-goose-upgrades/plan-B-goose-novelty.md): replay every recorded
# corpus and report the positive rate of the change label vs the novelty label.
# Pure post-processing over results/runs; touches neither agent nor scorer.
label-diag:
	mkdir -p $(RESULTS)/diagnostics
	PYTHONHASHSEED=0 uv run python legacy/goose_experiments/label_diagnostic.py --results $(RESULTS)

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
	@$(MAKE) --no-print-directory -f legacy/goose_experiments/experiments.mk arm-sweep SW_ARMS="$(PB_ARMS)" SW_GAMES="$(PB_GAMES)" SW_SEEDS="$(PB_SEEDS)" SW_CAP=$(PB_CAP)
planb-confirm:
	@$(MAKE) --no-print-directory -f legacy/goose_experiments/experiments.mk arm-sweep SW_ARMS="$(PC_ARMS)" SW_GAMES="$(PC_GAMES)" SW_SEEDS="$(PC_SEEDS)" SW_CAP=$(PC_CAP)
map-dev:
	@$(MAKE) --no-print-directory -f legacy/goose_experiments/experiments.mk arm-sweep SW_ARMS="$(MD_ARMS)" SW_GAMES="$(MD_GAMES)" SW_SEEDS="$(MD_SEEDS)" SW_CAP=$(MD_CAP)
map-confirm:
	@$(MAKE) --no-print-directory -f legacy/goose_experiments/experiments.mk arm-sweep SW_ARMS="A4" SW_GAMES="$(PC_GAMES)" SW_SEEDS="$(PC_SEEDS)" SW_CAP=$(PC_CAP)

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

