#!/usr/bin/env bash
# Start the Stage A rule-finding run DETACHED (its own session), so it survives
# this terminal, the Claude Code window closing and a dropped connection.
# It cannot survive `wsl --shutdown` or the machine sleeping: those stop the
# whole VM, which is what killed the 2026-09-11 attempt 90 seconds in.
#
#   bash legacy/llm_track/rules_overnight.sh          start it (~4 h; checker gate first)
#   bash legacy/llm_track/rules_overnight.sh status   how far along it is
#   bash legacy/llm_track/rules_overnight.sh stop     stop it and free the GPU
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/legacy_llm/rules/stageA
LOG=$OUT/run.log
PATTERN='^\.venv-llm/bin/python -m legacy.llm_track\.rule_writer'

running() { pgrep -f "$PATTERN" >/dev/null; }

case "${1:-start}" in
  status)
    if running; then echo "RUNNING (pid $(pgrep -f "$PATTERN" | head -1))"
    else echo "not running"; fi
    [ -f "$LOG" ] && grep -E "^\[rules\]|^    (ft09|lp85|ls20)|^exit=" "$LOG" | tail -10
    [ -f "$OUT/report.md" ] && echo "REPORT READY: $OUT/report.md"
    exit 0 ;;
  stop)
    running || { echo "not running"; exit 0; }
    pkill -f "$PATTERN"; sleep 10
    pgrep -f "VLLM::EngineCor[e]" >/dev/null && { pkill -f "VLLM::EngineCor[e]"; sleep 5; }
    echo "stopped; GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    exit 0 ;;
esac

running && { echo "already running (pid $(pgrep -f "$PATTERN" | head -1))"; exit 1; }

# vLLM asks for 90% of the card at startup, so a Windows app holding several GB
# makes it fail to allocate. Earlier successful runs started with 2-4 GB in use.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${USED:-0}" -gt 6000 ] && [ -z "${FORCE:-}" ]; then
  echo "GPU already has ${USED} MiB in use (needs to be a few GB at most)."
  echo "Close GPU-heavy Windows apps, then start again - or FORCE=1 to try anyway."
  exit 1
fi

# The checker must pass before a night is spent on the run.
.venv/bin/python -m legacy.llm_track.rule_referee selftest > results/legacy_llm/rules/selftest_prerun.log 2>&1 || {
  echo "checker selftest FAILED - not starting"; tail -6 results/legacy_llm/rules/selftest_prerun.log; exit 1; }
echo "checker selftest passed"

[ -s "$LOG" ] && { mv "$OUT" "${OUT}_$(date +%m%d_%H%M)"; echo "previous attempt archived"; }
mkdir -p "$OUT"
setsid nohup env HF_HUB_OFFLINE=1 timeout 43200 \
  .venv-llm/bin/python -m legacy.llm_track.rule_writer > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "started (pid $(pgrep -f "$PATTERN" | head -1)); about 4 hours -> $OUT/report.md"
echo "watch it:  bash legacy/llm_track/rules_overnight.sh status"
