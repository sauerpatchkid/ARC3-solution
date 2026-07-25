# Convenience targets. Override vars on the command line, e.g.
#   make local GAME=ft09 CAP=2000
#   make curriculum GAMES=ft09,dc22,ls20 CAP=200000
#   make metrics DIR=results/runs/<ts>/ft09/transitions GAME=ft09
#
# All outputs land under $(RESULTS) (gitignored):
#   results/runs/ results/sweeps/ results/local_suite.csv results/curriculum_suite.csv
SEED    ?= 0
CAP     ?= 2000
GAME    ?= ft09
GAMES   ?= ft09,dc22,ls20
RESULTS ?= results

install:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt

# Local engine, one game (fast dev path, ~120 act/s).
local:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_local.py --game=$(GAME)

# Persistent-brain curriculum across several games (cross-game transfer).
curriculum:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run python run_curriculum.py --games=$(GAMES)

# Backgrounded overnight sweep (edit sweep.sh CONFIG or override GAMES/SEEDS/CAP).
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
