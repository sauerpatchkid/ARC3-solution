# experiments/upgrade_screen/screen.mk - the upgrade screen's make targets.
# Pulled into the root Makefile by ONE line tagged [upgrades]; delete that line
# and this folder to remove the screen. See experiments/upgrade_screen/README.md.
SCREEN_JOBS ?= 4
SCREEN_DIR  := $(RESULTS)/screen
SCREEN_PID  := $(SCREEN_DIR)/screen.pid

.PHONY: upgrade-screen upgrade-screen-status upgrade-screen-pause upgrade-screen-rank

# All 12 arms, 2 seeds x 50k actions x 8 games (192 runs, ~17 h), SCREEN_JOBS at a time,
# detached. DRY_RUN=1 prints the plan. RESUME=<manifest.tsv> continues one.
upgrade-screen:
ifeq ($(DRY_RUN),1)
	uv run python experiments/upgrade_screen/screen.py --dry-run --jobs $(SCREEN_JOBS) $(if $(RESUME),--resume $(RESUME))
else
	@if [ -f $(SCREEN_PID) ] && kill -0 $$(cat $(SCREEN_PID)) 2>/dev/null; then echo "the screen is already running (make upgrade-screen-status)"; exit 1; fi
	mkdir -p $(SCREEN_DIR)
	setsid nohup uv run python experiments/upgrade_screen/screen.py --jobs $(SCREEN_JOBS) $(if $(RESUME),--resume $(RESUME)) \
	  >> $(SCREEN_DIR)/screen.log 2>&1 < /dev/null &
	@echo "started detached; log: $(SCREEN_DIR)/screen.log   progress: make upgrade-screen-status   pause: make upgrade-screen-pause"
endif

upgrade-screen-status:
	@if [ -f $(SCREEN_PID) ] && kill -0 $$(cat $(SCREEN_PID)) 2>/dev/null; then echo RUNNING; else echo "not running"; fi
	@[ -f $(SCREEN_DIR)/screen.log ] && echo "runs finished: $$(grep -c '^done ' $(SCREEN_DIR)/screen.log)" || true
	@[ -f $(SCREEN_DIR)/screen.log ] && grep -E '^(done|start|!!|===)' $(SCREEN_DIR)/screen.log | tail -6 || true

# Stop starting new runs; the runs in progress finish and are kept.
upgrade-screen-pause:
	@if [ -f $(SCREEN_PID) ] && kill -0 $$(cat $(SCREEN_PID)) 2>/dev/null; then touch $(SCREEN_DIR)/STOP; echo "pause requested: runs in progress will finish first"; else echo "not running"; fi

# Leaderboard for any manifest (the screen writes one itself when it finishes).
upgrade-screen-rank:
	uv run python experiments/upgrade_screen/rank.py $(MANIFEST)
