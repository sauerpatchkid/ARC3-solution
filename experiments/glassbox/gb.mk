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
#   make gb-replay  RUN=<new> GAMES=lp85 [BOOK=<run>] [SEEDS=0,1,2]   S5 ACT against A0's recordings (no game, no GPU)
#   make gb-report  RUN=<name> [COMPARE=<run>]      S6 SHOW
#   make gb-case    RUN=<name> CASE=<id>            one case through S1-S4, every step printed

.PHONY: gb-help gb-config gb-test gb-import-cp1 gb-report gb-mask gb-see gb-write gb-check gb-carry gb-guards gb-g0b gb-g0b-control gb-g1 gb-pool gb-replay

gb-help:
	@echo "Glassbox (custom_agents/gb/README.md, docs/3-glassbox/):"
	@echo "  make gb-config     the settings (experiments/glassbox/glassbox.toml) and their hashes"
	@echo "  make gb-test       Glassbox's tests"
	@echo "  make gb-import-cp1 re-run Rulebook v2's CP1 with gb/core into results/glassbox/cp1_import/;"
	@echo "                     every number is checked against CP1's registered one (~80 s, CPU)"
	@echo "  make gb-mask RUN=<new> [FROM=cp1_import]   S0: evidence rebuilt from CP1's recordings with each"
	@echo "                     bar in the decoration mask run to the screen's edges; FROM's candidates copied"
	@echo "  make gb-see RUN=<name> [GAMES=a,b] [FORCE=1]   S1: objects, what happened to them, closure (CPU)"
	@echo "  make gb-write RUN=<name> [DRY=1] [FROM_CACHE=1]   S2: the LLM writes rules (needs the server:"
	@echo "                     experiments/llm/serve.sh; DRY=1 renders prompts only; FROM_CACHE=1 replays)"
	@echo "  make gb-check RUN=<name>    S3: every candidate graded on its level; the book (CPU)"
	@echo "  make gb-carry RUN=<name>    S4: the book's rules re-bound on the next level; T0, T1 (CPU)"
	@echo "  make gb-guards RUN=<name> [GAMES=a,b] [FORCE=1]   S3's guard fitter: near-miss rules split by a"
	@echo "                     guard from S1's facts, re-graded; book and T1 before and after (CPU). On"
	@echo "                     CP1's candidates (RUN=cp1_import) this IS G0: it refuses without G0=1, which"
	@echo "                     is for after pre-registration section 3 is registered"
	@echo "  make gb-g0b        G0b (prereg section 3b): S0 -> S1 -> S3 -> S4 -> guards into"
	@echo "                     results/glassbox/g0b_bars/, from CP1's candidates (CPU)"
	@echo "  make gb-g0b-control the same with S0's bars off (S0 -> S3 -> S4): must equal CP1 exactly"
	@echo "  make gb-g1         G1 (prereg section 4): S0 -> S1 -> S2 (the LLM: start experiments/llm/serve.sh"
	@echo "                     first; ~10-16 GPU hours) -> S3 -> S4 -> guards into results/glassbox/g1/, then"
	@echo "                     the verdict against G0b (tools/gb_gate.py)"
	@echo "  make gb-pool [POOL_RUN=pool_g0b_g1] [POOL_FROM=g0b_bars,g1]   S2 POOL: two runs' candidates on the"
	@echo "                     same evidence pooled (no LLM), then S1 -> S3 -> S4 -> S6 and pool_report.md"
	@echo "                     beside both sources' books (exploratory, not a gate; CPU)"
	@echo "  make gb-replay RUN=<new> GAMES=lp85 [BOOK=<run>] [SEEDS=0,1,2]   S5's guide fed A0's recordings:"
	@echo "                     avoidable waste and masking risk in each new level's first 2,000 actions (CPU);"
	@echo "                     BOOK's no-op rules carried (none: GB-free). On a dev game this IS the offline"
	@echo "                     counterfactual: it refuses dev games without CF=1 (after prereg section 5a)"
	@echo "  make gb-report RUN=<name>   S6: results/glassbox/<name>/s6_show/index.html (funnel, case viewer)"
	@echo "  Stage commands are added as each stage is built; the planned list is at the top of"
	@echo "  experiments/glassbox/gb.mk."

gb-config:
	uv run python custom_agents/gb/artifacts.py

gb-test:
	uv run python -m pytest tests/test_gb_*.py -q

gb-import-cp1:
	uv run python tools/gb_offline.py --import-cp1

RUN ?= cp1_import
# The root Makefile gives GAMES a default (ft09,dc22,ls20) for its own targets.
# Glassbox's commands take GAMES only when it is given (command line or
# environment); otherwise they run every game in the run.
GB_GAMES = $(if $(filter command line environment,$(origin GAMES)),--games $(GAMES))

gb-see:
	uv run python tools/gb_offline.py --stage see --run $(RUN) $(GB_GAMES) $(if $(FORCE),--force)

FROM ?= cp1_import
gb-mask:
	uv run python tools/gb_offline.py --stage mask --run $(RUN) --from $(FROM) $(GB_GAMES) $(if $(FORCE),--force)

gb-write:
	uv run python tools/gb_offline.py --stage write --run $(RUN) $(GB_GAMES) $(if $(DRY),--dry) $(if $(FROM_CACHE),--from-cache)

gb-check:
	uv run python tools/gb_offline.py --stage check --run $(RUN) $(GB_GAMES) $(if $(FORCE),--force)

gb-carry:
	uv run python tools/gb_offline.py --stage carry --run $(RUN) $(GB_GAMES) $(if $(FORCE),--force)

# G0b, as registered in docs/3-glassbox/glassbox-prereg.md section 3b: CP1's candidates on evidence whose
# decoration mask has its bars run to the screen's edges, then CP1's book and T1 procedure, then the guards.
gb-g0b:
	$(MAKE) gb-mask RUN=g0b_bars FROM=cp1_import
	$(MAKE) gb-see RUN=g0b_bars
	$(MAKE) gb-check RUN=g0b_bars
	$(MAKE) gb-carry RUN=g0b_bars
	$(MAKE) gb-guards RUN=g0b_bars G0=1

# G0b's control: S0 with the bars off must rebuild CP1's evidence, and S3 and S4 must then give CP1's book,
# T0, T1 and re-bindings on every game (cp1_import.matches).
G0B_OFF = --set mask.complete_bars=false
gb-g0b-control:
	for st in mask check carry; do uv run python tools/gb_offline.py --stage $$st --run g0b_control --from cp1_import $(G0B_OFF) || exit 1; done
	uv run python tools/gb_offline.py --compare-cp1 --run g0b_control

# G1, as registered in docs/3-glassbox/glassbox-prereg.md section 4: new rules written by the LLM (S2) on
# evidence with S0 on, then CP1's book and T1 procedure and the guards, judged against G0b's numbers.
gb-g1:
	uv run python tools/gb_offline.py --stage mask --run g1 --from cp1
	uv run python tools/gb_offline.py --stage see --run g1
	uv run python tools/gb_offline.py --stage write --run g1
	uv run python tools/gb_offline.py --stage check --run g1
	uv run python tools/gb_offline.py --stage carry --run g1
	uv run python tools/gb_offline.py --stage guards --run g1
	uv run python tools/gb_gate.py g1 --run g1 --base g0b_bars

gb-guards:
	uv run python tools/gb_offline.py --stage guards --run $(RUN) $(GB_GAMES) $(if $(FORCE),--force) $(if $(G0),--g0)

gb-report:
	uv run python tools/gb_offline.py --stage show --run $(RUN)

# The pooled book (exploratory, not a gate; custom_agents/gb/s2_pool.py): G0b's (CP1's) and G1's candidates
# on their shared evidence, the book chosen and carried by CP1's procedure, set beside both sources' books.
POOL_RUN ?= pool_g0b_g1
POOL_FROM ?= g0b_bars,g1
gb-pool:
	uv run python tools/gb_offline.py --stage pool --run $(POOL_RUN) --from $(POOL_FROM) $(GB_GAMES)
	uv run python tools/gb_offline.py --stage see --run $(POOL_RUN) $(GB_GAMES)
	uv run python tools/gb_offline.py --stage check --run $(POOL_RUN) $(GB_GAMES)
	uv run python tools/gb_offline.py --stage carry --run $(POOL_RUN) $(GB_GAMES)
	uv run python tools/gb_offline.py --stage pool --report --run $(POOL_RUN) --from $(POOL_FROM) $(GB_GAMES)
	uv run python tools/gb_offline.py --stage show --run $(POOL_RUN)

# S5's replay on A0's recordings (custom_agents/gb/s5_act.py). On dev games it is the offline counterfactual,
# pre-registration section 5a: CF=1 only after that section is registered.
SEEDS ?= 0
gb-replay:
	uv run python tools/gb_offline.py --stage replay --run $(RUN) $(GB_GAMES) --seeds $(SEEDS) $(if $(BOOK),--book $(BOOK)) $(if $(CF),--cf)
