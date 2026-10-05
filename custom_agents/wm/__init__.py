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

Adapted from legacy/llm_track (rule_referee.py, rule_writer.py, heur_sandbox.py,
tickers.py, probe_images.py). Copied, not imported: legacy/ is a leaf.

HOW TO REMOVE IT
Delete this folder, tools/wm_offline.py, experiments/rulebook/ and
tests/test_wm_*.py. Nothing in the baseline imports it.
"""
