#!/usr/bin/env bash
# Build .venv-llm, the LLM server's own environment (vLLM pins its own torch, so it
# must never share the baseline's .venv). Needs a uv-managed Python: vLLM's Triton
# backend compiles a C helper at startup and needs Python.h.
# Copied from legacy/llm_track/Makefile's `env` target on 16 Oct 2026.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv python install 3.12
rm -rf .venv-llm
uv venv .venv-llm --python 3.12 --managed-python
VIRTUAL_ENV=.venv-llm uv pip install -r experiments/llm/requirements-llm.txt
