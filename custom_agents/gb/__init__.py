"""gb - Glassbox: Goose plus a local LLM that writes small, replay-checked rules.

Plan: docs/3-glassbox/glassbox-plan.md. Gates: docs/3-glassbox/glassbox-prereg.md.
One stage per file, numbered in the order data flows. Each stage reads files
and writes files (gb/artifacts.py), so one stage can be changed and re-run
without touching the others. The table in gb/README.md says what problem each
stage fixes.

  core/            checking machinery copied from Rulebook v2 (see core/__init__.py)
  artifacts.py     the artifact contract: run folders, stage headers, cache keys
  cp1_import.py    Rulebook v2's CP1 re-run with core/ and written as artifacts: the
                   check that core/ still reproduces CP1, and S6's first data

  s0_mask.py       S0 MASK   the decoration mask's bars run to the screen's edges (no LLM)
  s1_see.py        S1 SEE    objects, relations and object-level diffs (no LLM)
  s2_write.py      S2 WRITE  the LLM writes one-mechanism rules
  s3_check.py      S3 CHECK  replay and the book; s3_guards.py: the guard fitter (no LLM)
  s4_carry.py      S4 CARRY  re-bind rules on each new level (no LLM)
  s5_act.py        S5 ACT    prune, probe, imagine, goals: steers Goose online
  s6_show.py       S6 SHOW   one report page and a case viewer per run
  h1_vocab.py      H1 VOCAB      tested grid primitives the rules may call
  h2_schedule.py   H2 SCHEDULER  when the LLM runs online, inside a fixed budget
  h3_templates.py  H3 TEMPLATES  rules learned on dev games, tried on new games

Off by default: the agent only uses Glassbox with EVAL_GB=1, through lines in
action.py tagged `# [gb]` (removable with sed, like `[return-map]`).
"""
