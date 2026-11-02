# dc22's no-op rules over whole levels — exploratory (2 Nov 2026)

*Not pre-registered; it changes no verdict. The follow-up to G2's point 3
(`g2-online.md`), option (c) Matt chose on 19 Oct: check dc22's no-op rules
offline before they are carried again.*

## In one line

**The book's two dc22 no-op rules never covered a move that changed anything:
over whole levels, on all 9 dc22 recordings we have (A0's seeds 0-2, A0 run
again on seeds 0-1, G2's four runs), they covered 4,160 moves and 0 changed
something.** The stall G2 attributed to them (GB s0: 1 level, GB-free s0: 4)
also happens with no guide at all: A0's seed 2 and both fresh A0 runs stopped
on level 2 the same way (`a0-fresh.md`). The rules can be carried again.

## Why it was checked

In G2, dc22 s0 finished 1 level with the guide and 4 without. On level 2 the
only difference between the arms was the book's two no-op rules (#67, a red
click on the right; #122, a blue click), re-bound at move 1,000. The
counterfactual (prereg 5a) looked at each level's first 2,000 actions only, and
dc22's level 2 lasts 12,000-93,000 moves.

## What was run

S5's replay (`s5_act.replay`, the counterfactual's code) with its window set to
the whole level, `--book pool_g0b_g1`, one run per set of recordings
(`results/glassbox/dc22_whole/run.sh`; 2 Nov, commit `ab81d38`, CPU, 2.5 min):

    uv run python tools/gb_offline.py --stage replay --run dc22_whole_a0 --book pool_g0b_g1 \
        --games dc22 --seeds 0,1,2 --cf --set act.waste_window=1000000000
    ... --run dc22_whole_fresh  --set 'run.manifest="results/glassbox/a0_fresh/online/manifest.tsv"' --arm A0-fresh
    ... --run dc22_whole_gb     --set 'run.manifest="results/glassbox/g2/online/manifest.tsv"' --arm GB
    ... --run dc22_whole_gbfree --set 'run.manifest="results/glassbox/g2/online/manifest.tsv"' --arm GB-free

The replay re-binds each rule on every level from that level's own moves, as S5
does online, and asks of every recorded move whether a rule or a carried colour
covers it and whether it changed anything (under S0's mask for the level). The
replay now counts the two kinds separately (`masking_by`) and takes `--arm` for
a manifest with several arms.

## Per recording

| Recording | Levels done | Moves, levels 2 and up | Rule covers (all levels) | ... that changed something | Colour covers (levels 2 and up) | ... that changed something |
|---|---|---|---|---|---|---|
| A0 27 Sep s0 | 3 | 92,738 | 529 | 0 | 800 | 0 |
| A0 27 Sep s1 | 3 | 87,940 | 864 | 0 | 1,012 | 0 |
| A0 27 Sep s2 | 1 | 93,283 | 351 | 0 | 416 | 0 |
| A0 fresh s0 | 1 | 92,252 | 440 | 0 | 362 | 0 |
| A0 fresh s1 | 1 | 94,032 | 414 | 0 | 338 | 0 |
| GB s0 | 1 | 92,404 | 361 | 0 | 883 | 0 |
| GB s1 | 3 | 89,929 | 224 | 0 | 443 | 0 |
| GB-free s0 | 4 | 93,413 | 421 | 0 | 493 | 2 |
| GB-free s1 | 3 | 94,264 | 556 | 0 | 815 | 0 |
| **Total** | | **830,255** | **4,160** | **0** | **5,562** | **2** |

On every recording both rules were re-bound on level 2 to the same binding the
online GB runs used (#67 `button 5, swap 0`; #122 `agent 3, other 3`), after
402-1,000 moves. The 2 colour-covered moves that changed something are on
GB-free s0's level 5 (colours carried from earlier levels), where the colour
was dropped at once, as designed.

## What it shows

**1. The rules are clean.** Five of these recordings won level 2. Had the rules
covered a click that mattered there, it would show as a covered move that
changed something; none did, on any level. The online runs agree: no rule was
ever contradicted in G2.

**2. The stall is dc22's, not the book's.** Level 2 stopped 4 of the 9 runs
(A0 27 Sep s2, both fresh seeds, GB s0) for the rest of their 100,000 actions,
3 of them with no guide.

**3. Online, colours are carried later, or not at all, compared with the
replay.** Each G2 GB-free run, replayed with no book (`carry_lag_gbfree`, same
command with `--arm GB-free` and the five click games), against what that same
run carried online (`glassbox/events.jsonl`): colours carried over the run,
online vs replay.

| Game | Seed 0 | Seed 1 | Online's lag |
|---|---|---|---|
| ft09 | 3 vs 3 | 2 vs 2 | none |
| vc33 | 7 vs 7 | 7 vs 7 | the same colours, one level later (after level 2, not 1) |
| dc22 | 4 vs 4 | 3 vs 4 | one or two levels later; one colour never |
| m0r0 | 1 vs 7 | 0 vs 3 | most never carried |
| cd82 | 0 vs 0 | 0 vs 3 | never carried |

On GB s0's dc22 level 1, about half the clicks on colours 3, 4 and 5 changed raw
cells (74 of 168, 92 of 200, 33 of 57), none of them under S0's mask: they moved
decorations only. S5 judges a click online under the agent's own mask *at that
moment*, and the earliest of these clicks (colour 5 at moves 6 and 10, colour 3
at 22) came before the agent's mask exists (`EVAL_CANON_WARMUP` = 200 moves). One
such click keeps a colour from being carried for the whole level. On m0r0 the
likely cause is the one G0 found in S0's mask: the last cells of the move-budget
bar, which the agent's mask misses too. So the counterfactual (prereg 5a), which
judges clicks under S0's mask, overstates what online prune covers on vc33, dc22,
m0r0 and cd82 (ft09 is exact).

## What it means

- The book's dc22 no-op rules can be carried again as they are.
- **A small S5 change would close most of the gap in point 3** (Matt's call, a
  new S5 version, tuned on dev games): judge a colour at the level's end, not
  click by click: keep per colour the cells its clicks changed, and carry it if
  none of those cells is live under the level's final mask (or S0's, which also
  covers m0r0's bar end). Whether more prune turns into levels is a separate
  question: G2's GB and GB-free are within noise of A0 (`a0-fresh.md`).
