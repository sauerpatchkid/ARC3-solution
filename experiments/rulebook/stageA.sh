#!/usr/bin/env bash
# Rulebook step 1: rerun the frozen Stage A rule-finding test with Qwen3.8-27B
# (docs/plans/rulebook-stageA.md). The code is legacy/llm_track/rule_writer.py,
# unchanged, run as its own process in .venv-llm; nothing imports it.
#
#   experiments/rulebook/stageA.sh          start it DETACHED (survives this window)
#   experiments/rulebook/stageA.sh status   how far along it is
#   experiments/rulebook/stageA.sh stop     stop it and free the GPU
#
# It cannot survive the PC sleeping or `wsl --shutdown` (that is what killed the
# 2026-09-11 attempt): keep the machine on AC power, where it never sleeps.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${STAGEA_MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4}"
OUT_ROOT=results/rulebook
OUT=$OUT_ROOT/stageA
LOG=$OUT/run.log
PATTERN='^\.venv-llm/bin/python -m legacy\.llm_track\.rule_writer'

running() { pgrep -f "$PATTERN" >/dev/null; }

case "${1:-start}" in
  status)
    if running; then echo "RUNNING (pid $(pgrep -f "$PATTERN" | head -1), started $(ps -o lstart= -p "$(pgrep -f "$PATTERN" | head -1)"))"
    else echo "not running"; fi
    [ -f "$LOG" ] && grep -E "^\[rules\]|^    (ft09|lp85|ls20)|^exit=|Traceback|Error" "$LOG" | tail -12
    [ -f "$OUT/raw_outputs.jsonl" ] && echo "answers so far: $(wc -l < "$OUT/raw_outputs.jsonl")"
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

# vLLM asks for 90% of the card at startup: a Windows app (a game) holding
# several GB makes it fail to allocate.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${USED:-0}" -gt 4000 ] && [ -z "${FORCE:-}" ]; then
  echo "GPU already has ${USED} MiB in use (should be under ~3 GB). Close GPU-heavy apps, or FORCE=1."
  exit 1
fi

# The checker must pass before a day is spent on the run.
mkdir -p "$OUT_ROOT"
.venv/bin/python -m legacy.llm_track.rule_referee selftest > "$OUT_ROOT/selftest_prerun.log" 2>&1 || {
  echo "checker selftest FAILED - not starting"; tail -6 "$OUT_ROOT/selftest_prerun.log"; exit 1; }
echo "checker selftest passed"

[ -s "$LOG" ] && { mv "$OUT" "${OUT}_$(date +%m%d_%H%M)"; echo "previous attempt archived"; }
mkdir -p "$OUT"
setsid nohup env HF_HUB_OFFLINE=1 timeout 108000 \
  .venv-llm/bin/python -m legacy.llm_track.rule_writer --model "$MODEL" --out "$OUT_ROOT" \
  > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "started (pid $(pgrep -f "$PATTERN" | head -1)) with $MODEL -> $OUT"
echo "watch it:  experiments/rulebook/stageA.sh status"
