# Dev sweep 2: GB against the same-day A0 — result (5 Nov 2026)

*Judged by pre-registration section 7a (`glassbox-prereg.md`, registered and
pushed in `3386f77` before the run): arms A0-fresh and GB, S5 VERSION 2
(`colour_judge = "level_end"`, `mask_bars = true`), book `pool_g0b_g1`, config
`8c8e1cb023c827d2`, code unchanged since `ecae24b`, 8 dev games × seeds 3-6 ×
100,000 actions. A dev-game result: S5 v2 and the book were built and tuned on
these games, so it is labelled as tuned.*

## In one line

**PASS: Goose guided by S5 v2 completed 72 levels on the 32 dev runs, against
67 for the adopted Goose run beside it (paired 6 better, 24 the same, 2
worse); no game was worse on all four seeds.** So G4 runs, with GB (prereg
§7a, "What it decides"). The gain is small against the noise: +5 levels is
about 1.6 standard deviations of the gap two unguided arms show by chance
(below). Every gain is on the three click games where the guide changed the
most moves (vc33, dc22, cd82); on the five where it changed few or none, all 20
pairs tied.

## The run

- `make gb-dev2 BOOK=pool_g0b_g1 DEV2=1`, Wed 4 Nov 22:43 to Thu 5 Nov 14:52:
  64 runs (A0-fresh and GB × 8 dev games × seeds 3-6 × 100,000 actions), 4 at
  a time on the GPU, in the order seed, game, arm. **Paused 08:03-12:41** at
  Matt's request (he needed the GPU), with the sweep's STOP file: the 4 runs in
  progress finished first, so no run was cut short; resumed with `--resume`
  after the code, config hash and AC power were checked again (prereg §11
  entry 8). Running time 11 h 29 min (52 runs in 9 h 19 min, the last 12 in
  2 h 10 min). Log `results/glassbox/dev2/dev2_log.txt`; runs, manifest and
  verdict in `results/glassbox/dev2/online/`.
- **Valid.** 64 of 64 runs ended at the 100,000-action cap (`run_end.json`);
  the run-set check (`--expect 32`) passed; no rules process was switched off.
  Every run records commit `3386f77` (the §7a registration), `+dirty` only
  because of the `ARC-AGI-3-Agents` submodule's local edits.
- Throughput: A0-fresh 38.8 act/s, GB 38.6 act/s (4 runs at once): the guide
  costs no measurable speed.

## Per game

Levels completed per seed (s3, s4, s5, s6). "Guide changed" is the share of
GB's moves the guide replaced (a covered no-op drawn again, or a hint boost);
"covered" is the share of Goose's first draws a carried no-op covered.

| Game | A0-fresh | GB | W / T / L | GB: guide changed | GB: covered | GB: colours carried / dropped |
|---|---|---|---|---|---|---|
| tu93 | 4, 4, 4, 4 | 4, 4, 4, 4 | 0 / 4 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| tr87 | 1, 1, 1, 1 | 1, 1, 1, 1 | 0 / 4 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| vc33 | 4, 4, **2**, 4 | 4, 4, 4, 4 | 1 / 3 / 0 | 21.5, 28.3, 12.1, 49.4% | 12.7-31.3% | 28 / 0 |
| ft09 | 2, 2, 2, 2 | 2, 2, 2, 2 | 0 / 4 / 0 | 2.4, 4.4, 4.4, 3.2% | 2.6-4.7% | 10 / 0 |
| dc22 | 1, 1, **4**, 1 | **3, 3**, 1, **2** | 3 / 0 / 1 | 17.3, 16.6, 2.5, 3.8% | 2.1-3.4% | 15 / 0 |
| m0r0 | 2, 2, 2, 2 | 2, 2, 2, 2 | 0 / 4 / 0 | 2.0, 2.2, 1.5, 1.8% | 1.6-2.4% | 27 / 12 |
| cd82 | 2, 1, **2**, 1 | 2, **2**, 1, **2** | 2 / 1 / 1 | 29.3, 10.1, 2.5, 32.4% | 2.1-3.3% | 7 / 0 |
| g50t | 1, 1, 1, 1 | 1, 1, 1, 1 | 0 / 4 / 0 | 1.2-1.8% | 1.4-2.1% | 0 / 0 |
| **Total** | **67** | **72** | **6 / 24 / 2** | | | 87 / 12 |

**Verdict** (`tools/paired_compare.py --rule dev --expect 32`, section 7a's
rule; `verdict_gb_vs_a0_fresh.md`): levels 72 >= 67 and wins 6 >= losses 2,
PASS; no game worse on every seed, PASS: **PASS** (the tool prints "ADOPT", its
word for a passed rule). Bootstrap interval for the level gap, resampling
games: +1 to +10 (descriptive, not part of the rule). Levels of 2 and above:
A0-fresh 35, GB 40 (every run of both arms completed level 1).

Read with the levels: GB's runs also found more distinct states on the click
games (mean unique states per run: vc33 36,561 → 58,120, dc22 24,566 → 35,649,
cd82 22,168 → 31,679, ft09 70,349 → 83,640) with more late novelty (vc33 335 →
531, dc22 30 → 320 per 1k actions), partly because states of later levels
count too.

## How big +5 is

- **The noise.** The 8-game seed totals of unguided runs are now nine: A0's
  27 Sep runs and A0 run again (19, 18, 16, 17, 16; `a0-fresh.md`) and
  A0-fresh here (17, 16, 18, 16). Their standard deviation is about 1.1 levels,
  so with 4 seeds per arm the gap between two arms' 32-run totals varies by
  about ±3.2 levels from the agent alone (section 7a's estimate was ±3.7, from
  the first five). +5 is about 1.6 of those: a pass under the registered rule,
  short of two standard deviations. (GPU runs do not repeat past the first few
  hundred actions, so pairing by seed narrows nothing.)
- **GB's seed totals** (19, 19, 16, 18; mean 18.0) lie in the same 16-19 band
  as every 8-game seed total measured so far (17 of them, guided or not); the
  unguided mean is 17.0.
- **Where the gains are.** All 6 wins and both losses are on vc33, dc22 and
  cd82, the games where the guide changed the most moves (10-49% on most of
  their runs). On tu93 and tr87 (guide changed 0.0%) and ft09, g50t and m0r0
  (1.2-4.4%), all 20 pairs tied, so this time those games show no noise at
  all; in G2, tr87 lost a level in each arm. A small real effect would look like
  this; so could chance on three high-variance games (dc22 alone ranges 1 to 4
  levels without a guide).
- **One confound to avoid.** Within a game, the guide changes more moves in
  runs that reach more levels (colours are carried at a level-up and act on the
  levels after it), so the per-run link between "guide changed" and levels
  says nothing about cause (dc22: 17.3% and 16.6% on the seeds GB reached 3,
  2.5% on the seed it reached 1).

## Against the expectations (section 7a)

| Expectation, written before running | Result |
|---|---|
| GB within noise of A0-fresh | +5, about 1.6 standard deviations: within two |
| The effect shows on vc33 and ft09, a little on m0r0 and cd82 | vc33 +2, cd82 +1; ft09 none (2 levels in all 8 runs); m0r0 none; **dc22 +2**, not expected |
| Nothing on tu93 and tr87 | none: all 8 pairs tied, guide changed 0.0% |
| Pass about 50% | passed |

## S5 online

- **Colours carried as no-ops:** 87 over the 32 GB runs (vc33 28, m0r0 27,
  dc22 15, ft09 10, cd82 7; none on tu93, tr87, g50t).
- **Online masking** (a covered move that changed a live cell drops the colour
  or demotes the rule at once): 12 colours dropped, all on m0r0, and 6 rules
  demoted (vc33 4, m0r0 2), in 3.16 million GB moves.
- **The book's no-op rules:** 82 re-bindings, 229 "not re-bound yet" waits
  (S4's online wait, retried at twice the moves), 6 demotions, 0 rule processes
  switched off.
- **Goal hints:** 54 set, 25 dropped after 2,000 moves without effect.
- **Not yet done:** section 7a also reports the colours carried online against
  S5 v2's agent's-mask replay of the same recordings (as `s5-v2.md` did for the
  smoke). It is a CPU replay of the 32 GB recordings, left for the next session
  (no new run was started after the sweep, at Matt's request).

## What it means

- **G4 runs** (plan §6.3, prereg §7a): GB, that is S5 v2 with the book
  `pool_g0b_g1`, against **a same-day A0 arm on all 75 runs** (Matt, 5 Nov),
  with A0's 27 Sep runs (112 levels) reported beside it. It is registered in
  prereg section 7 before it starts.
- **What G4 tests.** The book has rules for the 8 dev games only. On the other
  17 games GB is colour no-ops and goal hints with no LLM output, so G4 mostly
  measures S5's LLM-free parts on games they were not tuned on; the dev tier is
  where the LLM's no-op rules act, and it is labelled as tuned.
- **What to expect.** If the effect here is real, it comes from pruning dead
  clicks on click games, so G4's gain, if any, should come from the click games
  among the 17. G4's rule (strictly more levels, paired wins > losses, no game
  worse on all 3 seeds) is stricter than this one; at 3 seeds per arm the noise
  on a 75-run total is larger than here.
- **Against G2.** G2 (S5 v1, 34 vs A0's 37 from 27 Sep) and this sweep (S5 v2,
  72 vs 67 from the same days) differ in the guide, the comparator and the
  seeds at once, so the change from FAIL to PASS cannot be credited to v2 alone.
  Both are within about two standard deviations of no effect.
