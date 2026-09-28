# experiments/upgrade_confirm/confirm.mk - the 25-game test of round 2's winner.
# Pulled into the root Makefile by ONE line tagged [upgrades-confirm]; delete that
# line and this folder to remove it. Plan: docs/plans/upgrade-confirm.md.
CONFIRM_JOBS ?= 4
CONFIRM_DIR  := $(RESULTS)/confirm_upgrade
CONFIRM_PID  := $(CONFIRM_DIR)/screen.pid

.PHONY: upgrade-confirm upgrade-confirm-status upgrade-confirm-pause

upgrade-confirm:
ifeq ($(DRY_RUN),1)
	uv run python experiments/upgrade_confirm/confirm.py --dry-run --jobs $(CONFIRM_JOBS) $(if $(RESUME),--resume $(RESUME))
else
	@if [ -f $(CONFIRM_PID) ] && kill -0 $$(cat $(CONFIRM_PID)) 2>/dev/null; then echo "the confirm test is already running"; exit 1; fi
	mkdir -p $(CONFIRM_DIR)
	setsid nohup uv run python experiments/upgrade_confirm/confirm.py --jobs $(CONFIRM_JOBS) $(if $(RESUME),--resume $(RESUME)) \
	  >> $(CONFIRM_DIR)/screen.log 2>&1 < /dev/null &
	@echo "started detached; log: $(CONFIRM_DIR)/screen.log   progress: make upgrade-confirm-status   pause: make upgrade-confirm-pause"
endif

upgrade-confirm-status:
	@if [ -f $(CONFIRM_PID) ] && kill -0 $$(cat $(CONFIRM_PID)) 2>/dev/null; then echo RUNNING; else echo "not running"; fi
	@[ -f $(CONFIRM_DIR)/screen.log ] && echo "runs finished: $$(grep -c '^done ' $(CONFIRM_DIR)/screen.log) of 75" || true
	@[ -f $(CONFIRM_DIR)/screen.log ] && grep -E '^(done|start|!!|===)' $(CONFIRM_DIR)/screen.log | tail -6 || true

upgrade-confirm-pause:
	@if [ -f $(CONFIRM_PID) ] && kill -0 $$(cat $(CONFIRM_PID)) 2>/dev/null; then touch $(CONFIRM_DIR)/STOP; echo "pause requested: runs in progress will finish first"; else echo "not running"; fi
