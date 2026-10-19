# The offline counterfactual — result (18 Oct 2026)

*Judged by pre-registration section 5a (`glassbox-prereg.md`, registered and
pushed in `d9a6c80` before the run): book `pool_g0b_g1`, config
`6c5d95983cf6fa97`, code unchanged since `d111172`. Not a gate: nothing passes
or fails, and no S5 setting is changed after it.*

## In one line

**On A0's own recordings, S5's carried no-ops would have covered 60.1% of
vc33's moves and 22.9% of ft09's in the first 2,000 actions of each new level,
every one of them a move that did nothing; masking risk is at most 0.05%
(m0r0, 6 moves).** The registered expectation is met on ft09 and vc33 and
missed on m0r0 (5.4%), whose play there is only 9.7% clicks. Almost all of it
comes without the LLM: GB-free covers the same on 5 of the 8 games.

## The run

- `make gb-cf BOOK=pool_g0b_g1 CF=1`, Sun 18 Oct 18:55 to 19:02 (7 min, CPU),
  commit `d9a6c80`; log `results/glassbox/cf_gb/cf_log.txt`; summaries
  `cf_gb/s5_act/replay_summary.md` (GB) and `cf_free/s5_act/replay_summary.md`
  (GB-free); one `<game>_s<seed>.replay.json` per recording, with every event.
- 24 recordings (8 dev games × A0's seeds 0-2), all replayed, no errors.
  **Valid** as registered.

## Per game

Levels 2 and up, the first 2,000 actions of each level, pooled over the three
seeds. Waste and masking are shares of the window's recorded moves; "clicks" is
the share of those moves that are clicks (no-ops carried by colour can only
cover clicks).

| Game | Levels | Moves | Clicks | GB: avoidable waste | GB: masking | GB-free: waste | Covered by (GB) |
|---|---|---|---|---|---|---|---|
| vc33 | 12 | 19,953 | 100% | **60.1%** (11,986) | 0% | 60.1% | colour 11,986; rule 1,315 (overlap) |
| ft09 | 6 | 11,658 | 100% | **22.9%** (2,664) | 0% | 19.3% | colour 2,247; rule 2,398 |
| dc22 | 7 | 13,855 | 13.3% | 8.4% (1,164) | 0% | 8.4% | colour 1,164; rule 132 |
| cd82 | 4 | 7,848 | 8.4% | 6.3% (494) | 0% | 5.0% | colour 391; rule 131 |
| m0r0 | 6 | 11,844 | 9.7% | 5.4% (640) | **0.05%** (6) | 5.4% | colour 646; rule 127 |
| g50t | 3 | 5,910 | 0% | 1.3% (76) | 0% | 0.0% | rule 76 (ACTION4) |
| tu93 | 12 | 16,663 | 0% | 0.0% | 0% | 0.0% | — (no no-op rule) |
| tr87 | 3 | 5,910 | 0% | 0.0% | 0% | 0.0% | — (no no-op rule) |

## Against the expectation (section 5a)

| Expectation | Result |
|---|---|
| Avoidable waste >= 10% on ft09, vc33, m0r0 (GB) | ft09 22.9%, vc33 60.1%: met; **m0r0 5.4%: not met** |
| Masking risk <= 1% on every game | at most 0.05% (m0r0): met |
| GB-free covers most of what GB covers | equal on dc22, m0r0, vc33, tu93, tr87; GB adds 3.6 points on ft09, 1.3 on cd82 and g50t: met |

## What it shows

**1. Carried no-ops are reliable.** 17,030 covered moves over 53 levels and 24
recordings; 6 of them changed something, all on m0r0's level 3 (2 per seed),
where colours carried from earlier levels (each with 20+ clicks that did
nothing) started doing something. Each such colour is dropped at its first
contradiction, as designed (m0r0 seed 0: colours 5, 15 and 8 dropped on
level 3).

**2. Where Goose clicks, it wastes a lot on colours that never do anything.**
On vc33, 60% of A0's moves in a new level's first 2,000 actions are clicks on
colours that did nothing over 20+ clicks on an earlier level (86% on level 3
of seed 0). On ft09 it is 23%. This is the room S5's prune has online.

**3. m0r0 misses for a plain reason.** Only 9.7% of A0's moves there are clicks
(it is mostly a button game), so carried click no-ops can cover at most about
10% of them; they covered 5.4%, over half of all its clicks. v2's expectation
counted m0r0 as a click game because its orange-click no-op rule was its most
reliable rule (921 of 932), not because Goose clicks much there.

**4. The LLM's no-op rules add little on top of the colour rule.** The book's
rules cover moves the colour rule already covers (vc33, dc22, m0r0) or a few
more (ft09 +3.6 points, cd82 +1.3); its one button rule (g50t ACTION4) covers
1.3% of g50t's moves. So in G2, GB and GB-free are expected to differ little;
section 5b says a tie is reported plainly.

## What it means for G2

- Prune has real room on vc33 and ft09, little on the movement games, none on
  tu93 and tr87 (no no-op rules, no clicks). Whether covering those moves turns
  into levels is what G2 measures; a covered move is only made 10 times less
  likely (alpha = 0.1), not removed.
- Masking is low enough that prune is unlikely to hide a level's mechanism.
- Next: register section 5b (its OPEN items are settled), then `make gb-g2
  BOOK=pool_g0b_g1 G2=1` (GPU, about 6 h).
