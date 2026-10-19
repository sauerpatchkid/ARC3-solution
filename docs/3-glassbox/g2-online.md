# G2: the online dev gate — result (19 Oct 2026)

*Judged by pre-registration section 5b (`glassbox-prereg.md`, registered and
pushed in `4174874` before the run): book `pool_g0b_g1`, no LLM online, S5 v1
as built (prune, hints, map; no probes), config `6c5d95983cf6fa97`, code
unchanged since `d111172`.*

## In one line

**FAIL: Goose guided by S5 completed 34 levels on the 16 dev runs against A0's
37 (paired 1 better, 12 the same, 3 worse); no game was worse on both seeds.**
The LLM-free arm did the same (35 levels, 1 / 12 / 3), so the book's no-op rules
added nothing measurable, as the counterfactual predicted. Every difference is
within the seed-to-seed noise these 16 runs show where the guide did nothing.

## The run

- `make gb-g2 BOOK=pool_g0b_g1 G2=1`, Sun 18 Oct 21:46 to Mon 19 Oct 03:45
  (6 h 0 min), 32 runs (GB and GB-free × 8 dev games × seeds 0-1 × 100,000
  actions), 4 at a time on the GPU, ~37 act/s each. Log
  `results/glassbox/g2/g2_log.txt`; runs, manifests and the three verdicts in
  `results/glassbox/g2/online/`.
- **Valid.** 32 of 32 runs finished at the cap; the run-set check (`--expect 16`)
  passed for every comparison; no rules process was switched off. 28 runs
  record commit `a2cc3cf`, a CLAUDE.md-only commit made after G2 started (no
  code differs from `4174874`; prereg section 11).

## Per game

Levels completed per seed (s0, s1). "Guide changed" is the share of GB's moves
the guide replaced (covered no-ops resampled, or hint boosts); "covered" is the
share of Goose's first draws a carried no-op covered.

| Game | A0 | GB | GB-free | GB: guide changed | GB: covered | Note |
|---|---|---|---|---|---|---|
| tu93 | 4, 4 | 4, 4 | 4, 4 | 0.0%, 0.0% | 0.0% | no no-op rule, no clicks: unguided |
| tr87 | 1, 1 | 1, **0** | **0**, 1 | 0.0%, 0.0% | 0.0% | unguided: the losses are noise |
| vc33 | 4, 4 | 4, **5** | 4, 4 | 9.2%, 26.6% | 9.7%, 9.4% | the one win |
| ft09 | 2, 2 | 2, 2 | 2, 2 | 4.0%, 3.4% | 4.3%, 3.7% | |
| dc22 | 3, 3 | **1**, 3 | **4**, 3 | 2.7%, 21.6% | 2.3%, 2.8% | see 3 below |
| m0r0 | 2, 2 | 2, **1** | 2, **1** | 0.6%, 1.2% | 0.7%, 1.3% | GB-free changed 0.0% on s1 |
| cd82 | 2, 1 | 2, 1 | **1**, 1 | 34.2%, 1.6% | 3.4%, 1.0% | |
| g50t | 1, 1 | 1, 1 | 1, 1 | 1.0%, 0.7% | 1.1%, 0.9% | |
| **Total** | **37** | **34** | **35** | | | |

**Verdicts** (`tools/paired_compare.py`, section 5b's rule): GB vs A0, 34 vs 37,
1 / 12 / 3, rule 1 FAIL, rule 2 (no game worse on both seeds) PASS: **FAIL**.
Reported: GB-free vs A0, 35 vs 37, 1 / 12 / 3, FAIL; GB vs GB-free, 34 vs 35,
3 / 11 / 2.

**Online safety.** Covered moves taken that changed something (the online
masking count): vc33 3 and 4 in ~99,000 moves per run, m0r0 1 per run (both
arms), 0 elsewhere. Each dropped the colour or demoted the rule at once (vc33:
2-4 colours dropped, 2 rules demoted per GB run).

## Against the expectations (section 5b)

| Expectation | Result |
|---|---|
| tu93 and tr87 play as A0 (the guide does nothing there) | the guide changed 0.0% of moves; tu93 4, 4 in every arm; tr87 lost a level on one seed in each arm |
| The effect shows on vc33 and ft09, little elsewhere | vc33 +1 level (GB s1), ft09 equal; dc22 s0 -2 (GB) |
| GB and GB-free within 2 levels | 34 vs 35: met |
| Pass uncertain, about 40% | FAIL |

## What the run shows

**1. The noise floor is a level per game.** Where the guide changed nothing
(tu93, tr87, and m0r0 s1 for GB-free), runs still differ from A0 by a level:
tr87 lost one in each arm, m0r0 s1 in both. A0's own seeds vary as much: its
seed 2 (not in G2) reached tu93's level 4 at 14,905 actions and ft09's level 2
at 3,554, where seeds 0-1 took 31,934-92,872 and 21,002-26,514. A -3 total on 16
runs sits inside that spread.

**2. Faster levels on the click games are not attributable.** On vc33 and ft09
the guided runs mostly reached levels sooner than A0's seeds 0-1 (ft09 level 2:
GB 7,933 and 8,750 actions, GB-free 3,664 and 4,073, A0 21,002 and 26,514). But
the unguided tu93 runs were faster too (level 3 at 2,452-7,850 against A0's
19,316 and 66,390), and A0's seed 2 is as fast as either. Two seeds cannot
separate the guide from luck here.

**3. dc22 s0 is the one loss the guide may have caused.** GB stopped at level 2
(reached at 7,068 actions) where GB-free went on to 4 levels and A0 to 3. On
level 2 the only difference between the arms was GB's two no-op rules (dc22's
red- and blue-click rules, re-bound at move 1,000); no move they covered was
ever seen to change anything (no contradiction). A rule that wrongly covers a
click that matters only in some states would look exactly like this, and so
would luck. One run cannot tell them apart.

**4. The LLM's no-op rules add nothing measurable.** GB and GB-free tie (34 vs
35, 3 / 11 / 2), as the counterfactual predicted (the rules added at most 3.6
points of coverage to the colour rule). In v1, that is all the LLM contributes
online; its changing rules (tu93 59%, tr87 35.5%) are not used.

## What it means

The registered reading (plan §6.3): *"G2 fails: M3 still builds; the 2 Nov
statement stands; the confirm runs only if dev sweep 2 shows GB at or above A0
on the dev games."* Beyond that, the run points at three things, each Matt's
call:

- **Give the LLM's rules something to do online.** In v1 only no-op rules act,
  and the colour rule already does their job. Probes (aimed at what the book
  can't explain) and IMAGINE (tu93's and m0r0's rules predict the whole screen
  on 40-52% of the transfer level's moves) are where the changing rules would
  first matter.
- **Measure the noise before the next sweep.** With a level per game of
  seed-to-seed spread, 2 seeds cannot show a 1-2 level effect. A fresh A0 run
  of the same 16 (about 3 h on the GPU) would say how far today's runs differ
  from 27 Sep's, and more seeds would narrow it.
- **Check dc22's no-op rules** on dc22 s0's level 2 recording (offline, no GPU)
  before they are carried again.
