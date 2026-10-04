# Coach offline probe (step 4): results, prompt v1 — NO-GO (3 Oct 2026)

The question: shown the text summary of a stuck or about-to-win screen, does the LLM pick the next experiment
better than random and no worse than a hand rule? Tool and pre-registered rule: `tools/advice_probe.py`
(written before any LLM answer was scored). Full report: `results/coach/probe_20261003_121021/report.md`.

- **Data:** the 24 mb_gated_att confirm runs of the 8 dev games, 110 points, 100 scored (51 just before a winning
  move, 49 at stuck points).
- **Settings:** summary `coach-summary-v1`; prompt v1 (`custom_agents/coach/prompt_v1.txt`, hash `f787df88a737`);
  Qwen3.8-27B (`cyankiwi/Qwen3.8-27B-AWQ-INT4`), thinking off, temperature 0.3. No frontier model (Matt's call).

## Verdict

| Pre-registered condition | Result | |
|---|---|---|
| 1. at least 15 scored points | 100 | pass |
| 2. LLM beats random (one-sided Poisson-binomial p < 0.05) | 21 hits vs 21.24 expected, p = 0.56 | **fail** |
| 3. not clearly below the hand rule | LLM 21 vs rule 14 (only-LLM 12, only-rule 5, p = 0.14) | pass |
| **G1** | | **NO-GO** |

| Points | n | random (expected) | hand rule | LLM |
|---|---:|---:|---:|---:|
| all | 100 | 21.2 | 14 | 21 |
| object targets (right object in top 3) | 25 | 3.5 | 0 | 5 |
| button targets (right button ranked first) | 75 | 17.8 | 14 | 16 |
| just before a winning move | 51 | 10.5 | 7 | 13 |
| stuck points | 49 | 10.7 | 7 | 8 |

Format and cost: 100 of 100 answers parsed (schema-forced JSON), median 149 tokens out, 5.0 s per answer with 4
running at once.

## Why (diagnosis, after the verdict)

1. **Three quarters of the points ask "which arrow key next?" on movement games** (tu93 34, dc22 19, g50t 8).
   - The summary says what each button does ("colour 4 moved by (+0, −5)"). It does not say where the moving
     object is relative to walls or a goal, so the right direction can't be read from it.
   - The LLM tied all four buttons in 25 of 75 answers and often fell back on a 5-4-3-2 countdown.
   - Its button score (16) is chance (17.8).
2. **"Prefer untried" is the wrong rule for these targets.** The winning or next-new-screen object is usually one
   Goose had already clicked many times. The rule's untried-first order hit 0 of 25 object points; a
   "most-clicked first" order would hit 11 of 25.
   - Part of this is how the test labels stuck points: the "right answer" is whatever Goose did next, so it leans
     toward Goose's own favourite objects. Pre-win targets are real winning moves, but they too were objects
     Goose had clicked a lot.
3. **The LLM did a little better than chance on objects** (5 vs 3.5 of 25; too few points to mean anything). Its
   hypotheses read sensibly but generically ("move colour 4 to a target location"). On vc33 it twice called the
   bottom-row buttons, which are the winning moves, unimportant.

## What this means for the plan

By the pre-registered rule, prompt v1 is a NO-GO. The plan allows prompt and summary tuning on the dev games until
the G1 freeze (16 Oct), with the 17 held-out games as the clean test. A v2 would need:
- the moving object's position and what lies next to it, for movement games;
- distinct weights for buttons (no ties);
- no "prefer untried" instruction.

Even so, the button question may not be answerable from a text summary. The decision between one tuning round and
switching to Rulebook is Matt's.
