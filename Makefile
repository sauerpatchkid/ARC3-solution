action:
	uv run ARC-AGI-3-Agents/main.py --agent=action

install:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt

tensorboard:
	.venv/bin/tensorboard --logdir=runs --port=6006

clean:
	rm -r ./runs
baseline:
	PYTHONHASHSEED=0 EVAL_SEED=$(SEED) EVAL_MAX_ACTIONS=$(CAP) \
	uv run ARC-AGI-3-Agents/main.py --agent=action --game=$(GAME)
