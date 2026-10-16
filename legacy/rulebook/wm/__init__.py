"""wm - the Rulebook plan's world model (docs/plans/llm-rulebook.md).

Goose explores; a local LLM writes each game's mechanics as small Python rules;
every rule is checked against all the moves the run has recorded; only rules
that replay that evidence exactly may be used for planning.

Built so far (Tier 0a, the offline rule test - no agent changes):
  sandbox.py   static checks + child process for LLM-written code
  evidence.py  one level's recorded moves as a checked index: distinct
               (screen, action) keys, their outcomes, the decoration mask
  check.py     run a rule over the evidence; admitted / plan-eligible / no-op;
               the assembled rule book with UNKNOWN; coverage and transfer
  rules.py     what the LLM is shown and asked (text + pictures), its answers
  llm.py       HTTP client for experiments/rulebook/serve.sh, with a cache

Rulebook v2 (docs/plans/rulebook-v2.md), CP0 and CP1:
  metrics.py   claimed-cell grades, T0, per-move rates and the gate's interval
  tiers.py, registry.py   held-out tiers; frozen artifacts and their hashes
  contract.py  the v2 rule: PARAMS (read statically), slot types, bind()
  lint.py      level facts written into a rule's code
  bind.py      re-binding PARAMS to a level by search; running a bound rule
  book.py      trust across levels; the composed book, cell by cell
  rules2.py    the v2 prompts (v1's in rules.py are left as they were)

Adapted from legacy/llm_track (rule_referee.py, rule_writer.py, heur_sandbox.py,
tickers.py, probe_images.py). Copied, not imported: legacy/ is a leaf.

HOW TO REMOVE IT
Delete this folder, tools/wm_*.py, tools/artifacts.py, experiments/rulebook/
and tests/test_wm_*.py. Nothing in the baseline imports it.
"""
