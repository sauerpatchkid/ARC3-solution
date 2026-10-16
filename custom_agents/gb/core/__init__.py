"""gb.core - the checking machinery Glassbox inherits from Rulebook v1/v2.

COPIED, not imported, from legacy/rulebook/wm/ on 16 Oct 2026 (legacy/ is a
leaf). On that date the files were identical to v2's except for the paths to
the docs. Glassbox's own stages (gb/s1_see.py ... gb/h3_templates.py) build on
these; they do not change what these modules mean.

  evidence.py  one level's recorded moves as a checked index: distinct
               (screen, action) cases, outcomes, the decoration mask, splits
  sandbox.py   static checks and a child process for LLM-written code
  check.py     run a rule over the evidence; grades; the "memory" baseline
  contract.py  the v2 rule: PARAMS (read statically), slot types, optional bind();
               since 16 Oct it also hands rules H1's names (gb/h1_vocab.py)
  lint.py      level facts written literally into a rule's code
  bind.py      re-binding PARAMS to a new level by search, no LLM
  book.py      trust across levels; the composed book, cell by cell
  metrics.py   claimed-cell grades, cell/case coverage, T0, the gate's interval
  llm.py       HTTP client for the local server, with an answer cache

The rule for changing anything here: CP1's registered numbers must still come
out the same when CP1 is re-scored with this code. tests/test_gb_regression.py
checks 4 games in ~15 s; `make gb-import-cp1` checks all 8 (437 numbers, ~80 s).
A change that should alter results belongs in a stage file, behind its own
config knob.
"""
