# G3: templates on held-out games — result (3 Nov 2026)

*Judged by pre-registration section 6 (`glassbox-prereg.md`, registered and
pushed in `b9f1988` before the run): catalog `glassbox_templates_v1` (40
templates), config `6c5d95983cf6fa97`, code unchanged since `734600b` apart from
the launcher fix of section 11 entry 6 (`ebba7e1`), one answer cache
(`G3_SHARE=1`).*

## In one line

**FAIL, as expected: no template was adopted on any of the 8 held-out games (0
of 792 tries), so GB+T equals GB on every game** (more rules and coverage on 0
of 8; T1 wrong 2.75% and overrun 46.3% in both arms). Not one try got a single
changed cell right. GB alone, the first held-out test of S1-S4, reaches a mean
cell coverage of 13.8% (G1 on the dev games: 32.0%) with 16 trusted rules;
s5i5's book beats both "nothing" and "memory" at T1, and tn36's is wrong on
8.1% of its transfer level's changing cases.

## The run

- `make gb-g3 G3=1 G3_SHARE=1`, Mon 2 Nov 21:46:57 to Tue 3 Nov 10:14:51
  (12 h 28 min), commit `ebba7e1`, model `cyankiwi/Qwen3.8-27B-AWQ-INT4`,
  vocabulary `glassbox_vocab_v1`. Log `results/glassbox/g3/g3_log.txt`; runs
  `results/glassbox/g3_gb/` (GB) and `g3_gbt/` (GB+T); verdict
  `g3_gbt/g3_verdict.{json,md}`.
- GB: S0 and S1 21:47-21:58; **S2 21:58 to 07:35 (9 h 37 min)**; S3, S4 and the
  guard fitter to 07:44. GB+T: S0 and S1 07:44-07:55; H3's try 07:55-08:59
  (1 h 4 min); S2 08:59-10:05 (1 h 5 min: cached answers, grading only); S3, S4,
  the guard fitter and the verdict 10:05-10:15.
- **Valid.** The code is unchanged since `734600b` except section 11 entry 6's
  two `gb.mk` lines (`git diff 734600b -- custom_agents/gb tools/gb_offline.py
  tools/gb_gate.py experiments/glassbox artifacts/glassbox_templates_v1.jsonl`);
  the catalog's sha256 is `c263262ecffd8d75` and it matched its registry entry
  when loaded (the try refuses an unregistered catalog on a held-out game); no
  group failed in either arm; the server never needed a retry. Both runs record
  `ebba7e1+dirty`, as before only because of the `ARC-AGI-3-Agents`
  submodule's local edits.
- **One cache, as registered.** GB+T's S2 sent no request to the server: its
  copy of GB's cache held 586 answers before and after. Its 285 calls repeat
  GB's line for line (the same group, call, claimed-exact share, cells right and
  tokens). The one difference is bookkeeping: on sc25 the binding search's time
  budget left a few more bindings untried in GB+T (495 of 512 instead of 494 on
  one rule), so one first-round rule that GB graded as applying to no move
  stopped with a runtime error under the binding GB+T reached (64 graded
  candidates instead of 65; neither admitted). No gated or reported number
  differs.

## The verdict (`tools/gb_gate.py g3`)

Primary (more trusted rules and higher cell coverage than GB on >= 5 of 8):
**0 of 8, not met.** T1 wrong, pooled: 2.75% vs 2.75% (4,278 of 155,749
changing cases in each arm): met. Overrun, pooled: 46.3% vs 46.3% (204 of 441
answers): met. **FAIL.**

| Game | Cell coverage (GB -> GB+T) | Trusted | Templates tried / adopted / in book / carried | T1 per move (GB -> GB+T) | Requests (GB -> GB+T) |
|---|---|---|---|---|---|
| ar25 | 0.0% -> 0.0% | 0 -> 0 | 64 / 0 / 0 / 0 | 27.0% -> 27.0% | 36 -> 36 |
| lf52 | 0.9% -> 0.9% | 1 -> 1 | 144 / 0 / 0 / 0 | 12.5% -> 12.5% | 46 -> 46 |
| re86 | 21.3% -> 21.3% | 4 -> 4 | 80 / 0 / 0 / 0 | 0.3% -> 0.3% | 45 -> 45 |
| s5i5 | 22.4% -> 22.4% | 2 -> 2 | 96 / 0 / 0 / 0 | 21.7% -> 21.7% | 30 -> 30 |
| sc25 | 0.1% -> 0.1% | 1 -> 1 | 136 / 0 / 0 / 0 | no transfer level | 63 -> 63 |
| sp80 | 3.0% -> 3.0% | 2 -> 2 | 104 / 0 / 0 / 0 | 5.2% -> 5.2% | 54 -> 54 |
| su15 | 42.6% -> 42.6% | 2 -> 2 | 96 / 0 / 0 / 0 | 10.2% -> 10.2% | 36 -> 36 |
| tn36 | 20.2% -> 20.2% | 4 -> 4 | 72 / 0 / 0 / 0 | 0.7% -> 0.7% | 14 -> 14 |

Reported: distinct templates reused 0; LLM requests saved 0 on every game;
negative transfer none; template coincidence none (no template adopted).

## The templates: where the 792 tries went

For each group S2 would call, every template of the group's family (16 button
or 24 click) was re-bound on the training level by `bind.search` (no binding
preferred, 240 s per group shared among them) and graded with v2's trust
(`g3_gbt/h3_templates/<game>.json`).

| Game | Groups | Tries | Binding failed | Applies to no move | Applies (ambiguous / one binding) | Cells right, best try | Cut short by the budget |
|---|---|---|---|---|---|---|---|
| ar25 | 4 | 64 | 24 | 28 | 4 / 8 | 0 | 0 |
| lf52 | 6 | 144 | 42 | 60 | 36 / 6 | 0 | 12 |
| re86 | 5 | 80 | 20 | 35 | 20 / 5 | 0 | 25 |
| s5i5 | 4 | 96 | 32 | 36 | 24 / 4 | 0 | 4 |
| sc25 | 7 | 136 | 49 | 62 | 16 / 9 | 0 | 39 |
| sp80 | 6 | 104 | 24 | 55 | 20 / 5 | 0 | 30 |
| su15 | 4 | 96 | 24 | 44 | 16 / 12 | 0 | 8 |
| tn36 | 3 | 72 | 21 | 39 | 12 / 0 | 0 | 0 |
| **All** | **39** | **792** | **236** | **359** | **148 / 49** | **0** | **118** |

"Binding failed": no binding was exact on any move; "applies to no move": the
best binding's rule applies to none of the level's moves (flagged *binding
untested*). The 197 tries that applied to at least one move got no changed cell
right (0 cells right and 0 predicted changes that did not happen; S3's reason
for a gain of 0 reads "it claims more changes that did not happen than changes
that did"). The widest, sc25's ACTION3 under template 2, applies to 23,624
moves, its claims are all right on 75.9% of them (trust needs 95%), and it gets
no changed cell right. The 240 s budget left bindings untried in 118 tries,
mostly on the three largest training levels (sc25 25,943 cases, sp80 23,795,
re86 11,727); the 674 full tries got no cell right either.

## GB on games no artifact was built from

Section 6's "G1's table for these 8 games" (GB's numbers; GB+T's are the same).
S1's "explained" is on the training level. T1: per move, with "nothing" and
"memory" in brackets and the differences' 95% intervals; "wrong" is the share
of the transfer level's changing cases.

| Game | S1 explained | Cell coverage | Case coverage | Trusted | T1 (nothing, memory) | T1 − nothing | T1 − memory | Wrong | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | 88.8% | 0.0% | 0.0% | 0 | 27.0% (27.0%, 27.5%) | 0 | −0.006 [−0.018, 0.000] | 0% | 32 | 907,434 |
| lf52 | **0.0%** | 0.9% | 0.0% | 1 | 12.5% (12.5%, 23.4%) | 0 | −0.109 [−0.126, −0.093] | 0.03% | 40 | 1,283,662 |
| re86 | 62.3% | 21.3% | 0.0% | 4 (1 rescued) | 0.3% (0.3%, 1.0%) | 0 | −0.007 [−0.008, −0.006] | 0.24% | 40 | 1,170,883 |
| **s5i5** | 100% | 22.4% | 22.1% | 2 | 21.7% (20.0%, 20.0%) | **+0.017 [0.016, 0.019]** | **+0.018 [0.015, 0.020]** | 2.37% | 26 | 844,076 |
| sc25 | 71.0% | 0.1% | 0.0% | 1 | no transfer level | | | | 56 | 1,296,682 |
| sp80 | 99.5% | 3.0% | 0.3% | 2 | 5.2% (2.7%, 27.4%) | +0.025 [0.024, 0.026] | −0.222 [−0.228, −0.217] | 0.02% | 48 | 1,187,552 |
| su15 | 99.1% | **42.6%** | 0.0% | 2 | 10.2% (10.3%, 14.0%) | −0.001 [−0.001, −0.001] | −0.038 [−0.040, −0.035] | 0.90% | 32 | 796,073 |
| tn36 | 100% | 20.2% | 20.2% | 4 | 0.7% (0.7%, 15.6%) | 0 | −0.149 [−0.150, −0.148] | **8.06%** | 11 | 253,915 |

**The LLM's work:** 285 calls after the first round (324 requests), 442
candidates, 7.74M tokens (G1 on the dev games: 245 calls, 6.27M). 204 of 441
answers ran out of room (46.3%; G1: 35%, CP1: 44%). 304 of the 330 graded
candidates and 11 of the 15 trusted rules S2 wrote call an H1 primitive (G1: 304
of 309, 29 of 30). Mean cell coverage 13.8% (G1: 32.0%); 16 trusted rules
(G1: 32).

**Flags (section 9).** *Perception gap*: lf52 (S1 explains 0.0% of its 113
changing cases). *Template coincidence*: none (no template adopted). *Negative
transfer* between the arms: none (equal).

## Against the expectations (section 6)

| Expectation | Result |
|---|---|
| few templates adopted | none, on all 8 games (0 of 792 tries) |
| GB+T equals or nearly equals GB on most games | equal on all 8 |
| the primary fails (pass about 10%) | 0 of 8: FAIL |
| the wrong and overrun conditions hold | equal in both arms: held |
| GB's cell coverage below G1's on the dev games | mean 13.8% vs 32.0%: met |

## What the run shows

**1. The catalog's rules do not carry to new games at all.** Re-binding can
change which colour, object, size or step a rule uses, not what the rule says
happens. In 792 tries on 8 games none got a single changed cell right: none of
the catalog's 40 rules, whatever it is re-bound to, describes anything that
happens on these games. The seen-game check (ls20: 0 of 64) said the same
before the run.

**2. The rule writer does about half as well on games it was not built on.**
Mean cell coverage 13.8% against G1's 32.0% on the dev games, with more answers
running out of room (46% vs 35%). The spread is wide: su15 42.6% (an agent
square that jumps to a background click, and steps toward a clicked edge), s5i5
22.4%, re86 21.3%, tn36 20.2%; but ar25 0.0% though S1 explains 88.8% of its
changes, and lf52, which S1 cannot see (0.0% explained), 0.9%.

**3. Carrying is still the weak step.** Of the 7 games with a transfer level,
the book beats "nothing" on s5i5 and sp80 only, and also "memory" on s5i5, so
s5i5 meets CP1's own criterion (both intervals above zero), which before this
only dc22 and ft09 had (G0b, the pooled book; G1's books on no game). tn36's
three click rules all carry, but on level 2 their claims are right on only 7.6%
of the moves they apply to and they get no changed cell right, like m0r0's book
on its level 3 (`carry-later-levels.md`). S5 v1 uses only a book's no-op rules
online, so wrong changing rules like these do not reach the agent.

## What it means

The registered reading (plan §6.3): *"Fail: templates are not adopted, the
confirm runs GB, and G3 is reported as a result in its own right (reuse is a
finding either way)."* The confirm itself runs only if dev sweep 2 (11-13 Nov:
a same-day A0 arm, 4 seeds per arm) shows GB >= A0 (plan §6.3). What did cross
games is H1's vocabulary, not whole rules: 304 of GB's 330 graded candidates
call an H1 primitive. GB in the confirm has no book for the 17 games outside the
dev tier (H3 adopts nothing and H2 is not built), so there it plays as GB-free.
