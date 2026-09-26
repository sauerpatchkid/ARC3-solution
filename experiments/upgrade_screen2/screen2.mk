# experiments/upgrade_screen2/screen2.mk - round 2 of the upgrade screen.
# Pulled into the root Makefile by ONE line tagged [upgrades2]; delete that
# line and this folder to remove round 2. Round 1 is untouched by all of this.
SCREEN2_JOBS ?= 4
SCREEN2_DIR  := $(RESULTS)/screen2
SCREEN2_PID  := $(SCREEN2_DIR)/screen.pid

.PHONY: upgrade-screen2 upgrade-screen2-status upgrade-screen2-pause upgrade-screen2-rank

upgrade-screen2:
ifeq ($(DRY_RUN),1)
	uv run python experiments/upgrade_screen2/screen2.py --dry-run --jobs $(SCREEN2_JOBS) $(if $(RESUME),--resume $(RESUME))
else
	@if [ -f $(SCREEN2_PID) ] && kill -0 $$(cat $(SCREEN2_PID)) 2>/dev/null; then echo "round 2 is already running"; exit 1; fi
	mkdir -p $(SCREEN2_DIR)
	setsid nohup uv run python experiments/upgrade_screen2/screen2.py --jobs $(SCREEN2_JOBS) $(if $(RESUME),--resume $(RESUME)) \
	  >> $(SCREEN2_DIR)/screen.log 2>&1 < /dev/null &
	@echo "started detached; log: $(SCREEN2_DIR)/screen.log   progress: make upgrade-screen2-status   pause: make upgrade-screen2-pause"
endif

upgrade-screen2-status:
	@if [ -f $(SCREEN2_PID) ] && kill -0 $$(cat $(SCREEN2_PID)) 2>/dev/null; then echo RUNNING; else echo "not running"; fi
	@[ -f $(SCREEN2_DIR)/screen.log ] && echo "runs finished: $$(grep -c '^done ' $(SCREEN2_DIR)/screen.log)" || true
	@[ -f $(SCREEN2_DIR)/screen.log ] && grep -E '^(done|start|!!|===)' $(SCREEN2_DIR)/screen.log | tail -6 || true

upgrade-screen2-pause:
	@if [ -f $(SCREEN2_PID) ] && kill -0 $$(cat $(SCREEN2_PID)) 2>/dev/null; then touch $(SCREEN2_DIR)/STOP; echo "pause requested: runs in progress will finish first"; else echo "not running"; fi

upgrade-screen2-rank:
	uv run python experiments/upgrade_screen2/rank2.py $(MANIFEST)
