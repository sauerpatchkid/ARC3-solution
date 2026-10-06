#!/usr/bin/env bash
# Rulebook step 2: run the offline rule test (docs/plans/rulebook-tier0a.md) on
# the 8 dev games, DETACHED so it survives this window. Starts the LLM server if
# it is not already up, runs tools/wm_offline.py, then stops the server it started.
#
#   experiments/rulebook/tier0a.sh          start
#   experiments/rulebook/tier0a.sh status   progress so far
#   experiments/rulebook/tier0a.sh stop     stop the run and the server
#
# Answers are cached in results/rulebook/tier0a/llm_cache.jsonl, so a stopped run
# resumes where it left off. Keep the PC on AC power (it sleeps on battery) and
# GPU-heavy Windows apps closed: the server takes 90% of the card.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rulebook/tier0a
LOG=$OUT/run.log
RUN='tools/wm_offline\.py'
SRV='[b]in/vllm serve'

case "${1:-start}" in
  status)
    pgrep -f "$RUN" >/dev/null && echo "RUNNING" || echo "not running"
    [ -f "$LOG" ] && grep -E "^\[|^model:|G1|Traceback|Error" "$LOG" | tail -14
    [ -f "$OUT/report.md" ] && echo "REPORT READY: $OUT/report.md"
    exit 0 ;;
  stop)
    pkill -f "$RUN"; sleep 2
    P=$(pgrep -f "$SRV"); [ -n "$P" ] && kill $P
    sleep 8; echo "stopped; GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    exit 0 ;;
esac

pgrep -f "$RUN" >/dev/null && { echo "already running"; exit 1; }
mkdir -p "$OUT"
# Size the server to what is free: vLLM refuses to start when its share of the card
# is not available, and on WSL an oversubscribed card spills into system RAM and
# crawls. 600 MiB is left as headroom. The share only changes how many answers
# run at once, never what they are.
read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
GPU_MEM=$(python3 -c "print(min(0.90, int((($TOTAL - $USED - 600) / $TOTAL) * 100) / 100))")
if python3 -c "import sys; sys.exit(0 if $GPU_MEM < 0.80 else 1)"; then
  echo "GPU has ${USED} MiB in use: too little left for the model (share would be $GPU_MEM). Close GPU-heavy apps."
  exit 1
fi
echo "GPU: ${USED} MiB in use by other apps -> server share $GPU_MEM"
export RULEBOOK_GPU_MEM=$GPU_MEM
setsid nohup bash -c '
  started=0
  if ! curl -s -m 2 http://127.0.0.1:8018/v1/models >/dev/null; then
    experiments/rulebook/serve.sh > results/rulebook/serve.log 2>&1 &
    started=1
    for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8018/v1/models >/dev/null && break; sleep 5; done
  fi
  uv run python tools/wm_offline.py --out '"$OUT"'
  echo "exit=$?"
  if [ "$started" = 1 ]; then P=$(pgrep -f "[b]in/vllm serve"); [ -n "$P" ] && kill $P; fi
' > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "started -> $LOG"
echo "watch it:  experiments/rulebook/tier0a.sh status"
