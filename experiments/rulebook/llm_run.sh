#!/usr/bin/env bash
# Rulebook v2: run an offline LLM job DETACHED, so it survives this window.
# Starts the LLM server if it is not already up (sized to the GPU memory that is
# free), runs the job, then stops the server it started. Answers are cached in
# the job's output folder, so a stopped job resumes where it left off.
#
#   experiments/rulebook/llm_run.sh start  <job>
#   experiments/rulebook/llm_run.sh status <job>
#   experiments/rulebook/llm_run.sh stop   <job>
#
# Jobs (each writes under results/rulebook/v2/ - never into v1's frozen folders):
#   nearmiss    CP0: near-miss refinement with a bandit (tools/wm_refine.py, ~1.5-2 h)
#
# Keep the PC on AC power (it sleeps on battery) and GPU-heavy Windows apps
# closed: the server takes up to 90% of the card. v1's launchers (tier0a.sh,
# stageA.sh) are kept for the record and refuse to overwrite their results.
set -uo pipefail
cd "$(dirname "$0")/../.."
case "${2:-}" in
  nearmiss) TOOL=tools/wm_refine.py; OUT=results/rulebook/v2/nearmiss ;;
  *) echo "usage: $0 start|status|stop <job>   (jobs: nearmiss)"; exit 1 ;;
esac
LOG=$OUT/run.log
RUN="python ${TOOL//./\\.} --out"     # the job's own command line, nothing that merely mentions the tool
SRV='[b]in/vllm serve'

case "${1:-}" in
  status)
    pgrep -f "$RUN" >/dev/null && echo "RUNNING" || echo "not running"
    [ -f "$LOG" ] && grep -E "^\[|^model:|exact|Traceback|Error|failed|exit=" "$LOG" | tail -16
    [ -f "$OUT/report.md" ] && echo "REPORT READY: $OUT/report.md"
    exit 0 ;;
  stop)
    pkill -f "$RUN"; sleep 2
    P=$(pgrep -f "$SRV"); [ -n "$P" ] && kill $P
    sleep 8; echo "stopped; GPU now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 start|status|stop <job>"; exit 1 ;;
esac

pgrep -f "$RUN" >/dev/null && { echo "already running"; exit 1; }
mkdir -p "$OUT"
# Size the server to what is free: vLLM refuses to start when its share of the card
# is not available, and on WSL an oversubscribed card spills into system RAM and
# crawls. 600 MiB is left as headroom. The share only changes how many answers
# run at once, never what they are.
if ! curl -s -m 2 http://127.0.0.1:8018/v1/models >/dev/null; then
  read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
  GPU_MEM=$(python3 -c "print(min(0.90, int((($TOTAL - $USED - 600) / $TOTAL) * 100) / 100))")
  if python3 -c "import sys; sys.exit(0 if $GPU_MEM < 0.80 else 1)"; then
    echo "GPU has ${USED} MiB in use: too little left for the model (share would be $GPU_MEM). Close GPU-heavy apps."
    exit 1
  fi
  echo "GPU: ${USED} MiB in use by other apps -> server share $GPU_MEM"
  export RULEBOOK_GPU_MEM=$GPU_MEM
fi
setsid nohup bash -c '
  started=0
  if ! curl -s -m 2 http://127.0.0.1:8018/v1/models >/dev/null; then
    experiments/rulebook/serve.sh > results/rulebook/serve.log 2>&1 &
    started=1
    for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8018/v1/models >/dev/null && break; sleep 5; done
  fi
  uv run python '"$TOOL"' --out '"$OUT"'
  echo "exit=$?"
  if [ "$started" = 1 ]; then P=$(pgrep -f "[b]in/vllm serve"); [ -n "$P" ] && kill $P; fi
' > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "started -> $LOG"
echo "watch it:  $0 status ${2}"
