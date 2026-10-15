# Rulebook v2, CP1: results (15 Oct 2026)

CP1 tests whether rules written with their level facts as parameters, re-fitted
to the next level without the LLM, predict that level better than two simple
baselines. The gate was pre-registered in `docs/plans/rulebook-v2-prereg.md`
section 4, with every reading of it settled before the run (section 10, 13–14
Oct). Everything here was tuned on dev games.

The run: `tools/wm_tier0a_v2.py` (job `tier0a2`), commit `03e9fd1`, started
14 Oct 21:16, finished 15 Oct 06:20. Files: `results/rulebook/v2/tier0a_v2/`
(report, per-game JSON with every candidate rule, evidence, answer cache).

## 1. Verdict: FAIL

| | Criterion | Result |
|---|---|---|
| Primary | T1 beats both baselines, both 95% intervals above zero, on ≥ 3 of 8 games | **2 of 8** (dc22, ft09) |
| Safety | T1 wrong on ≤ 5% of the transfer level's changing cases, on every game | **met on all 8** (0.0% on every game) |

Pass needs both, so **CP1 fails its gate**. v1, measured the same way on the
same moves, beat both baselines on 1 of 8 (dc22) and was wrong on 58.8% of
ft09's changing cases. The plan put the odds of passing at about 60%.

## 2. The gate's numbers (T1, per recorded move of the score split)

| game | rules carried | book | nothing | memory | book − nothing [95%] | book − memory [95%] | beats both | wrong |
|---|---:|---:|---:|---:|---|---|---|---:|
| tu93 | 0 | 0.036 | 0.036 | 0.851 | +0.000 [+0.000, +0.000] | −0.815 [−0.834, −0.795] | no | 0.0% |
| tr87 | 0 | 0.000 | 0.000 | 0.476 | +0.000 [+0.000, +0.000] | −0.476 [−0.495, −0.462] | no | 0.0% |
| dc22 | 3 | 0.255 | 0.154 | 0.234 | +0.101 [+0.097, +0.105] | +0.021 [+0.004, +0.037] | **yes** | 0.0% |
| g50t | 2 | 0.091 | 0.091 | 0.150 | +0.000 [+0.000, +0.000] | −0.059 [−0.062, −0.056] | no | 0.0% |
| vc33 | 0 | 0.751 | 0.751 | 0.751 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | no | 0.0% |
| ft09 | 3 | 0.181 | 0.030 | 0.148 | +0.151 [+0.146, +0.155] | +0.033 [+0.027, +0.039] | **yes** | 0.0% |
| m0r0 | 0 | 0.062 | 0.062 | 0.454 | +0.000 [+0.000, +0.000] | −0.392 [−0.402, −0.383] | no | 0.0% |
| cd82 | 3 | 0.149 | 0.131 | 0.173 | +0.018 [+0.017, +0.018] | −0.024 [−0.027, −0.022] | no | 0.0% |

"Book" reads unclaimed cells as unchanged; a move is right only if the whole
next screen is. Intervals resample attempts (blocks of 50 moves on vc33), 2,000
times, seed 0.

## 3. What worked

1. **Re-binding removed every wrong prediction.** Untouched (T0), the books were
   wrong on 8.7% of g50t's changing cases, 100% of vc33's and 0.6% of dc22's.
   After re-fitting their parameters on the first 300 moves, and dropping the
   rules that no binding made exact there, T1 was wrong on 0.0% everywhere.
2. **ft09 is the case v1 failed on, and it now carries.** v1's rules had the
   colours 8 and 9 written in and were wrong on 59% of the next level. v2's red
   rule was re-bound from target colour 9 to 12, and its bar rule from tile
   colour 9 to 12. T0 (untouched) did not beat memory (0.100 vs 0.148); T1 does
   (0.181). The plan expected "≥ 90% right on the blue rule's applicable
   cases": the tile rule's claimed cells are right on 99.9% of the 1,346 moves
   it applies to.
3. **The model used the contract.** Of 389 candidates that ran, 344 declared
   PARAMS. 79 were marked level-bound by the lint, and none of those reached T1.

## 4. Where it failed

| Failure | Games | What the numbers say |
|---|---|---|
| No trusted rule at all | tu93, tr87, m0r0 | tu93: no rule was admitted (best 77% claimed-exact). tr87: no rule got more changed cells right than wrong. m0r0: 11 rules admitted, none exact (best 99.8%); its training level has cases seen to end in more than one way, which trust excludes |
| Rules carried, but explain only part of each change | g50t, cd82 | Every carried rule is 100% right on the cells it claims, but a move only counts if the whole screen is right. g50t's two rules never make a whole move right; cd82's three get the whole screen right on 6.3% of changing cases, enough to beat "nothing" but not memory |
| Too little to re-fit on | vc33 | Its only changing group has 2 changing cases in the first 300 moves; no binding of its trusted rule was exact on them, so it was dropped, and the one repair call did not produce a trusted rule |
| Narrower than v1 | dc22 | It still beats both, by less than v1's book (book − memory +0.021 vs v1's +0.107). v2 found no trusted ACTION2 rule, which v1 had |

Residual prompts (57 calls) added two rules to books: g50t's "ACTION5 moves the
agent block to its home cell" and ft09's bar rule. Neither made more whole moves
right on the next level: g50t's pair still leaves part of each change
unclaimed, and ft09's bar rule adds 0.1 point of cases. The two-level repair
calls (T2) changed no game's numbers.

So the plan's diagnosis was half right. The *form* of the rules was v1's
transfer problem (ft09 shows it, and safety is fixed), but on most games the
model does not write exact rules at all, or only for part of a change, and that
now limits the result more than transfer does.

## 5. Cost

551 requests, 662 answers, 7.66 million generated tokens, 9 h 4 min on the GPU.
193 of 439 answers ran out of room before writing code (44%; v1 22%, CP0 34%).
At most 4 groups were worked on at once, shared across the 8 games.

## 6. Caveats

- One recording per game, dev games only, tuned on them.
- tr87's training level has 40,000 cases. The binding search ran out of time on
  1,379 bindings of 4 tr87 rules, which were therefore never called exact (as
  registered). They were the bindings that applied least in the first 50 cases,
  and all 4 rules had no binding that got a changed cell right, so a different
  tr87 result is unlikely, but it was not checked.
- Per move, memory is strong wherever moves repeat the same screens (tu93 0.85,
  tr87 0.48, m0r0 0.45), which makes the primary hard on exactly the games with
  no trusted rule.
- On ft09, 4.3% of the moves the red rule covers are screens seen to end two
  ways; no rule that reads only the screen can get those, and the per-move rate
  counts them as misses.

## 7. What the plan says next

Plan section 2.3, "CP1 fails": CP2 still runs, with v1-form rules plus the
LLM-free parts (the dead-click filter carried by colour, goal hints); CP4
narrows to games whose rules are exact; CP1 is reported as a diagnosis with the
numbers above. CP2 starts 16 Oct (slip rule, section 10.2).

**Open, Matt's:** the plan wrote "v1-form rules" for CP2 because it assumed v2's
pipeline might not exist by then. It does now, and its rules are never wrong at
T1 where v1's were wrong on 59% of ft09. Using v2's pipeline (re-binding and the
safety filter) in CP2 would follow the plan's intent rather than its letter;
it changes no registered CP2 criterion (prereg section 5 names the arms and
budgets, not the rule form).
