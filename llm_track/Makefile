# llm_track/Makefile - the LLM track's commands, kept OUT of the root Makefile
# so that baseline tooling never touches LLM code or its environment (the
# repo's `make check` enforces that).
#
# Run from the repo root:   make -C llm_track <target>
# Every recipe cd's to the repo root, so paths match the rest of the repo.
ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
RESULTS    ?= results
OUT        := $(RESULTS)/llm/probeA
GAMES      ?= ft09,ar25,cd82,lp85,ls20
JUDGES     ?= Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B
IMG_JUDGES ?= Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B

.PHONY: help env scan probe probe-img

help:
	@echo "LLM track (Matt's; isolated from the baselines - see llm_track/README.md)"
	@echo "  make -C llm_track env         (re)build .venv-llm, the vLLM serving env"
	@echo "  make -C llm_track scan        corpus stats: masks, dedupe, buckets, anchors"
	@echo "  make -C llm_track probe       Probe A, text: Qwen 4B + 9B -> $(OUT)/report.md"
	@echo "  make -C llm_track probe-img   Probe A, pictures: Qwen 2B + 4B -> same report"

# The serving venv. Separate from .venv on purpose (vLLM pins its own torch),
# and built on a uv-MANAGED Python because vLLM's Triton backend compiles a C
# helper at startup that needs Python.h, which the system Python here lacks.
env:
	cd $(ROOT) && uv python install 3.12
	cd $(ROOT) && rm -rf .venv-llm
	cd $(ROOT) && uv venv .venv-llm --python 3.12 --managed-python
	cd $(ROOT) && VIRTUAL_ENV=.venv-llm uv pip install -r llm_track/requirements-llm.txt

# Corpus statistics: decorative masks, signature dedupe, stratification
# buckets and hindsight-anchor census. Read-only over results/runs.
scan:
	cd $(ROOT) && uv run python -m llm_track.scan --games=$(GAMES) \
	  --limit=50 --sample=30000 --out $(RESULTS)/llm/scan_stage1.json

# Probe A (design section 7), text: build the frozen pair set, have each judge
# label every pair in both orders, score against the pre-registered criteria,
# then run the post-hoc hindsight check (NOT pre-registered).
probe:
	cd $(ROOT) && uv run python -m llm_track.probe_pairs --out $(OUT)
	cd $(ROOT) && for m in $(JUDGES); do \
	  HF_HUB_OFFLINE=1 .venv-llm/bin/python -m llm_track.judge --model $$m \
	    --pairs $(OUT)/pairs.jsonl || exit 1; \
	done
	cd $(ROOT) && uv run python -m llm_track.probe_report --dir $(OUT)
	cd $(ROOT) && uv run python -m llm_track.probe_hindsight --dir $(OUT)

# Probe A with pictures of the board: same pairs and criteria, each move also
# shown before/after. Needs `probe` to have built pairs.jsonl first.
probe-img:
	cd $(ROOT) && uv run python -m llm_track.probe_images --pairs $(OUT)/pairs.jsonl
	cd $(ROOT) && for m in $(IMG_JUDGES); do \
	  HF_HUB_OFFLINE=1 .venv-llm/bin/python -m llm_track.judge --model $$m \
	    --pairs $(OUT)/pairs.jsonl --images $(OUT)/images/images.json || exit 1; \
	done
	cd $(ROOT) && uv run python -m llm_track.probe_report --dir $(OUT)
