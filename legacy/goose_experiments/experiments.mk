# legacy/goose_experiments/experiments.mk - launchers for the finished Goose
# experiments (semester 2): Plan B (novelty label), Option 1 (return map), the
# two upgrade screens and the 25-game upgrade confirm that adopted mb_gated_att.
# They are kept so their runs can be resumed or repeated. Run from the repo root:
#
#   make -f legacy/goose_experiments/experiments.mk <target>
#
#   label-diag                        Plan B step 1: change vs novelty label rate per game
#   planb-dev | planb-confirm         Plan B dev and confirm sweeps (arms A0 A1)
#   map-dev | map-confirm             Option 1 dev and confirm sweeps (arms A1 A4)
#   planb-status | planb-pause | planb-stop   control any of those four sweeps
#   upgrade-screen | upgrade-screen2 | upgrade-confirm   (each with -status, -pause)
#
# All take DRY_RUN=1 (print the plan) and RESUME=<manifest> (skip finished runs).
# Results: docs/1-goose-upgrades/.
RESULTS ?= results

include legacy/goose_experiments/arm_sweeps.mk
include legacy/goose_experiments/upgrade_screen/screen.mk
include legacy/goose_experiments/upgrade_screen2/screen2.mk
include legacy/goose_experiments/upgrade_confirm/confirm.mk
