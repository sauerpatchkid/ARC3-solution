#!/usr/bin/env bash
# The local LLM server for Glassbox's S2 WRITE (custom_agents/gb/s2_write.py):
# Qwen3.8-27B (AWQ INT4) over HTTP, OpenAI-compatible, on 127.0.0.1:8018.
#
# Multimodal (prompts carry up to 10 pictures), 32k context (prompt plus up to
# 20,480 answer tokens, thinking on). Offline it takes ~90% of the card, so no
# Goose run fits beside it; online, start it with a smaller share
# (LLM_GPU_MEM) and run one Goose at a time (plan section 8). The served alias
# is "rulebook", kept from Rulebook v2 so gb/core/llm.py and its answer-cache
# keys work unchanged. Nothing in the root Makefile or sweep.sh calls this.
#
#   experiments/llm/serve.sh                     # start (foreground); Ctrl-C to stop
#   LLM_GPU_MEM=0.87 experiments/llm/serve.sh    # when other apps hold GPU memory
#
# First time on a machine: experiments/llm/make_env.sh (builds .venv-llm).
# Copied from legacy/rulebook/experiments/serve.sh on 16 Oct 2026; same settings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # the repo
VENV="$ROOT/.venv-llm"
MODEL="${LLM_MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4}"
PORT="${LLM_PORT:-8018}"
GPU_MEM="${LLM_GPU_MEM:-0.90}"     # share of the card; lower it when Windows apps hold GPU memory

[ -x "$VENV/bin/vllm" ] || { echo "no $VENV - build it: experiments/llm/make_env.sh"; exit 1; }

# Machine fixes found in semester 1 (legacy/llm_track/judge.py:setup_vllm_env).
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
