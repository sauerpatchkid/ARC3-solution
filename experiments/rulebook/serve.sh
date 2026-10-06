#!/usr/bin/env bash
# Rulebook (docs/plans/llm-rulebook.md): serve the rule-writing LLM over HTTP,
# OpenAI-compatible, for the OFFLINE rule test (tools/wm_offline.py).
#
# Unlike the Coach server this one is multimodal (rule prompts carry pictures),
# has a 32k context (prompt + up to 20,480 answer tokens, thinking on) and takes
# 90% of the card, so no Goose run fits beside it. Same settings Stage A ran
# with in-process (legacy/llm_track/rule_writer.py); here it is a server so the
# agent's environment never imports vLLM. Nothing in the root Makefile or
# sweep.sh calls this script.
#
#   experiments/rulebook/serve.sh            # start (foreground); Ctrl-C to stop
#   RULEBOOK_MODEL=<hf id> experiments/rulebook/serve.sh
#   RULEBOOK_GPU_MEM=0.87 experiments/rulebook/serve.sh   # when other apps hold GPU memory
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.venv-llm"
MODEL="${RULEBOOK_MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4}"
PORT="${RULEBOOK_PORT:-8018}"
GPU_MEM="${RULEBOOK_GPU_MEM:-0.90}"     # share of the card; lower it when Windows apps hold GPU memory

[ -x "$VENV/bin/vllm" ] || { echo "no $VENV - build it: make -C legacy/llm_track env"; exit 1; }

# The same machine fixes as legacy/llm_track/judge.py:setup_vllm_env.
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CU="$(ls -d "$VENV"/lib/python3.*/site-packages/nvidia/cu13 2>/dev/null | head -1 || true)"
if [ -z "${CUDA_HOME:-}" ] && [ -n "$CU" ] && [ -x "$CU/bin/nvcc" ]; then
    export CUDA_HOME="$CU" PATH="$CU/bin:$PATH"
fi
export PATH="$VENV/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name rulebook \
    --host 127.0.0.1 --port "$PORT" \
    --seed 0 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image": 10, "video": 0}' \
    --reasoning-parser qwen3 \
    "$@"
