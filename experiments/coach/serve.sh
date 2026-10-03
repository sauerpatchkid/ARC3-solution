#!/usr/bin/env bash
# Coach (docs/plans/llm-coach.md): serve the in-loop LLM over HTTP, OpenAI-compatible.
#
# The agent never imports vLLM: it runs here, in .venv-llm (its own torch), and the
# coach talks to it over localhost. Nothing in the root Makefile or sweep.sh calls
# this script (make check enforces that for .venv-llm).
#
#   experiments/coach/serve.sh                     # Qwen3.8-27B, sized to sit beside one Goose run
#   COACH_MODEL=<other hf id> experiments/coach/serve.sh
#
# Sizing (step 2, 2026-10-03): Coach prompts are <= 2k tokens with <= 400 tokens out,
# so the KV cache is fixed at COACH_KV_BYTES (default 1 GiB: Qwen3.8-27B needs 0.81 GiB
# for one 8k-token request)
# instead of taking 90% of the card the way the semester-1 offline runs did.
# --language-model-only skips the vision encoder (prompts are text only).
# disable_any_whitespace forces compact JSON: pretty-printed answers ran into the
# 400-token cap and were cut off (11 of 40 in the first fit check).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.venv-llm"
MODEL="${COACH_MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4}"
PORT="${COACH_PORT:-8017}"
KV_BYTES="${COACH_KV_BYTES:-1073741824}"
MAX_LEN="${COACH_MAX_LEN:-8192}"

[ -x "$VENV/bin/vllm" ] || { echo "no $VENV - build it: make -C legacy/llm_track env"; exit 1; }

# The same machine fixes as legacy/llm_track/judge.py:setup_vllm_env: no FlashInfer
# sampler (needs nvcc), the pip nvcc for any other JIT, and the venv's bin/ (ninja).
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CU="$(ls -d "$VENV"/lib/python3.*/site-packages/nvidia/cu13 2>/dev/null | head -1 || true)"
if [ -z "${CUDA_HOME:-}" ] && [ -n "$CU" ] && [ -x "$CU/bin/nvcc" ]; then
    export CUDA_HOME="$CU" PATH="$CU/bin:$PATH"
fi
export PATH="$VENV/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name coach \
    --host 127.0.0.1 --port "$PORT" \
    --seed 0 \
    --language-model-only \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --kv-cache-memory-bytes "$KV_BYTES" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
    "$@"
