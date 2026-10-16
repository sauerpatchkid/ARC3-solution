# experiments/glassbox/gb.mk - Glassbox commands, one per stage (plan section 4.3).
# Included by the root Makefile with ONE line tagged [gb]. Nothing here starts the
# LLM server or touches the LLM's environment (check_repo.py enforces that). Start
# the server by hand when a stage needs it: experiments/llm/serve.sh
#
# Stage commands are added here as each stage is built (plan section 7):
#   make gb-see     RUN=<name> GAMES=tu93,m0r0      S1 SEE, CPU, about a minute per game
#   make gb-write   RUN=<name> GAMES=lp85 [DRY=1]   S2 WRITE (DRY=1: render prompts, no LLM)
#   make gb-check   RUN=<name> GAMES=m0r0           S3 CHECK from cached candidates, CPU
#   make gb-carry   RUN=<name> GAMES=ft09           S4 CARRY, CPU
#   make gb-offline RUN=<name> TIER=dev             S1 -> S4 for a whole tier
#   make gb-replay  RUN=<an A0 run>                 S5 ACT against a recorded run (no game, no GPU)
#   make gb-report  RUN=<name> [COMPARE=<run>]      S6 SHOW
#   make gb-case    RUN=<name> CASE=<id>            one case through S1-S4, every step printed

.PHONY: gb-help gb-config gb-test gb-import-cp1

gb-help:
	@echo "Glassbox (custom_agents/gb/README.md, docs/3-glassbox/):"
	@echo "  make gb-config     the settings (experiments/glassbox/glassbox.toml) and their hashes"
	@echo "  make gb-test       Glassbox's tests"
	@echo "  make gb-import-cp1 re-run Rulebook v2's CP1 with gb/core into results/glassbox/cp1_import/;"
	@echo "                     every number is checked against CP1's registered one (~80 s, CPU)"
	@echo "  Stage commands are added as each stage is built; the planned list is at the top of"
	@echo "  experiments/glassbox/gb.mk."

gb-config:
	uv run python custom_agents/gb/artifacts.py

gb-test:
	uv run python -m pytest tests/test_gb_*.py -q

gb-import-cp1:
	uv run python tools/gb_offline.py --import-cp1
