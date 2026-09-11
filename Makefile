# Convenience targets. Override vars on the command line, e.g.
#   make local GAME=ft09 CAP=2000
#   make bench SUITE=standard AGENT=goose
#   make metrics DIR=results/runs/<ts>/ft09/transitions GAME=ft09
#
# START HERE (new clone):
#   make install && make check && make bench SUITE=smoke
#
# All outputs land under $(RESULTS) (gitignored):
#   results/runs/ results/sweeps/ results/local_suite.csv results/curriculum_suite.csv
SEED    ?= 0
CAP     ?= 2000
GAME    ?= ft09
GAMES   ?= ft09,dc22,ls20
RESULTS ?= results
AGENT   ?= goose
SUITE   ?= standard

.PHONY: help install check suites bench bench-fg compare local curriculum \
        sweep long random curves metrics tensorboard clean action baseline

help:
	@echo "Baselines (what teammates use):"
	@echo "  make install                        set up the venv"
	@echo "  make check                          health check: isolation, agents, deps"
	@echo "  make suites                         list the frozen benchmark suites"
	@echo "  make bench SUITE=standard AGENT=x   run a frozen suite (backgrounded)"
	@echo "  make bench-fg SUITE=smoke           same, in the foreground"
	@echo "  make compare M1=<manifest> M2=<..>  put two agents side by side"
	@echo "  make local GAME=ft09 CAP=2000       one ad-hoc run"
	@echo ""
	@echo "Analysis:"
	@echo "  make metrics DIR=<transitions> GAME=x   score one run"
	@echo "  make curves MANIFEST=<manifest>         levels-vs-budget, AULC"
	@echo "  make tensorboard"
	@echo ""
	@echo "LLM track: Matt's, isolated in llm_track/ with its own Makefile - see llm_track/README.md"
	@echo ""
	@echo "Add your own agent: see custom_agents/__init__.py and TEMPLATE.py"

install:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt
	@echo ""
	@echo "Now run: make check"

# Health check: llm_track isolation, agent registry, dependencies, suites.
check:
	uv run python check_repo.py

# ---------------------------------------------------------------------------
# THE FROZEN BENCHMARK — everyone runs the same games/seeds/budget so results
# can be compared. Definitions live in benchmark.py; see that file before
# changing anything.
# ---------------------------------------------------------------------------
suites:
	uv run python benchmark.py

# Backgrounded (suites other than 'smoke' take hours). Log + manifest paths
# are printed immediately; the manifest is what `make compare` consumes.
bench:
	mkdir -p $(RESULTS)/sweeps
	BENCH=$(SUITE) AGENT=$(AGENT) nohup bash sweep.sh \
	  > $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log 2>&1 &
	@echo "started: $(SUITE) suite, agent=$(AGENT)"
	@echo "log:     $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log"
	@echo "watch:   tail -f $(RESULTS)/sweeps/bench_$(SUITE)_$(AGENT).log"

bench-fg:
	BENCH=$(SUITE) AGENT=$(AGENT) bash sweep.sh

# Side-by-side table. M1 is the baseline, M2+ are compared against it.
compare:
	uv run python compare.py $(M1) $(M2) $(M3) \
	  --out $(RESULTS)/sweeps/comparison.md

# Local engine, one game (fast dev path, ~120 act/s).
local:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_local.py --game=$(GAME)

# Persistent-brain curriculum across several games (cross-game transfer).
curriculum:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_curriculum.py --games=$(GAMES)

# Ad-hoc exploratory sweep (your own GAMES/SEEDS/CAP; NOT comparable between
# people - use `make bench` for anything you intend to report).
sweep:
	mkdir -p $(RESULTS)/sweeps
	nohup bash sweep.sh > $(RESULTS)/sweeps/sweep.log 2>&1 &

# Long single-seed run on the two games that actually complete levels, to see
# how far past level 2 the agent can get. 2M actions is ~4h/game at ~142 act/s,
# so ~8h for the pair - just under action.py's 7h55m per-process wall clock,
# which each game gets fresh because sweep.sh runs one process per game.
long:
	mkdir -p $(RESULTS)/sweeps
	GAMES="ft09 tu93" SEEDS="0" CAP=2000000 \
	nohup bash sweep.sh > $(RESULTS)/sweeps/long.log 2>&1 &

# Matched random-policy floor: same games/seeds/contract, uniform over the same
# masked 5+64x64 action space. ~2700 act/s (no model), so a full sweep is cheap.
# Needed to turn change-rate / redundancy / coverage into uplift ratios.
random:
	mkdir -p $(RESULTS)/sweeps
	AGENT=random GAMES="ft09 tu93 g50t dc22 ls20" SEEDS="0 1 2 3 4" CAP=200000 \
	nohup bash sweep.sh > $(RESULTS)/sweeps/random.log 2>&1 &

# Levels-vs-budget curves, AULC and (with --human-baselines) RHAE, from the
# metrics.json files a sweep already wrote. Pure post-processing.
#   make curves MANIFEST=results/sweeps/sweep_<stamp>.manifest
curves:
	uv run python analyze_curves.py $(MANIFEST) \
	--out $(RESULTS)/sweeps/curves_$(notdir $(basename $(MANIFEST)))

# Score a finished run's corpus and append a row to results/local_suite.csv.
metrics:
	uv run python compute_metrics.py $(DIR) \
	--game=$(GAME) --seed=$(SEED) --suite $(RESULTS)/local_suite.csv

tensorboard:
	.venv/bin/tensorboard --logdir=$(RESULTS)/runs --port=6006

clean:
	rm -rf ./$(RESULTS)/runs

# --- API path (original, unchanged apart from where recordings land) ---
action:
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action

baseline:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	RECORDINGS_DIR=$(RESULTS)/recordings \
	uv run ARC-AGI-3-Agents/main.py --agent=action --game=$(GAME)
