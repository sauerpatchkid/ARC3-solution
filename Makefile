# Convenience targets. Override vars on the command line, e.g.
#   make local GAME=ft09 CAP=2000
#   make curriculum GAMES=ft09,dc22,ls20 CAP=200000
#   make metrics DIR=runs/<ts>/ft09/transitions GAME=ft09
SEED  ?= 0
CAP   ?= 2000
GAME  ?= ft09
GAMES ?= ft09,dc22,ls20

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
	nohup bash sweep.sh > sweep.log 2>&1 &

# Score a finished run's corpus and append a row to local_suite.csv.
metrics:
	uv run python compute_metrics.py $(DIR) \
	--game=$(GAME) --seed=$(SEED) --suite local_suite.csv

tensorboard:
	.venv/bin/tensorboard --logdir=runs --port=6006

clean:
	rm -rf ./runs

# --- API path (original, unchanged) ---
action:
	uv run ARC-AGI-3-Agents/main.py --agent=action

baseline:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run ARC-AGI-3-Agents/main.py --agent=action --game=$(GAME)
