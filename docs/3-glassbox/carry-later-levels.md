# The book on every later level: what IMAGINE would have to work with — exploratory (2 Nov 2026)

*Not pre-registered; it changes no verdict. It measures IMAGINE's precondition
(plan 7.3) where it matters, on the levels where Goose stops, before IMAGINE is
built (Matt, 19 Oct: "IMAGINE first: an offline replay on A0's recordings
first").*

## In one line

**Re-bound on a later level, the pooled book predicts the moves that level has
not seen yet only on some games: on the level where Goose stops, it gets the
whole next screen right wherever it applies on ft09 (level 3) and tr87 (level 2),
partly on dc22 and cd82, and never on tu93, g50t, m0r0 or vc33.** And the plan's
condition for IMAGINE, rules that together claim every live cell, holds on no
game at any level: rules claim only the cells they touch.

## Why it was measured

IMAGINE (plan 5.5) would let the map use predicted moves: for a screen and an
action never tried there, the book predicts the next screen, and a new predicted
screen becomes a place worth walking to when Goose stalls. Its precondition was
read so far one level out (T1: on tu93 and m0r0 the carried book gets the whole
screen right on 40.7% and 51.7% of the transfer level's moves, `pool-g0b-g1.md`).
But Goose stops later: tu93 on level 5 in all 9 runs we have, m0r0 on level 3 in
7. IMAGINE would have to work there, on moves not yet tried.

## What was run

S4's new `later_levels` (`custom_agents/gb/s4_carry.py`, module doc "LATER
LEVELS"; 2 Nov, commit `ab81d38`, CPU, about 9 minutes):

    uv run python tools/gb_offline.py --stage carry --later --run later_pool --book pool_g0b_g1 \
        --games tu93,m0r0,dc22,vc33,ft09,g50t,cd82,tr87 --seeds 0,1,2

On A0's 24 dev recordings, for every level after the book's training level:
each book rule is re-bound on the level's first 1,000 moves (S5's online wait at
its longest; CP1's rebind: exact there, trusted over both levels), and the rules
kept are composed (CP1's book) on the rest of the level. Every (screen, action)
pair there that the first 1,000 moves never saw is scored once, at its first
occurrence: those are the pairs an imagined edge would stand in for. Scored:

- **applies**: the book claims at least one cell;
- **right**: the whole next screen is right, cells it leaves unclaimed taken as unchanged;
- **wrong**: a cell it claims is wrong;
- **claims every cell**: the plan's IMAGINE condition.

The mask is S0's, kept from level to level: the map's key, which IMAGINE would
predict, uses a mask that is sticky across levels. Skipped: levels under 1,200
moves (tu93 level 2, vc33 level 2). Levels with more than 40,000 pairs are
scored on the evidence index's uniform sample (ft09 level 3, vc33 level 5).
Output: `results/glassbox/later_pool/s4_carry/` (one `.later.json` per recording,
`later_summary.md`).

## Per game and level

Changing pairs only (pairs the level had not seen, whose move changed a live
cell), pooled over the seeds that reached the level. "Stop here" counts the 9
runs we have of each game (A0 27 Sep ×3, A0 fresh ×2, G2's GB ×2 and GB-free ×2)
that ended their 100,000 actions on that level. **Bold**: where most runs stop.

| Game | Level | Stop here (of 9) | Rules re-bound, per seed | New changing pairs | Book applies | Right where it applies | Wrong where it applies |
|---|---|---|---|---|---|---|---|
| tu93 | 3 | 0 | 2, 2, 2 | 335 | 37.9% | 100.0% | 0.0% |
| tu93 | 4 | 0 | 2, 2, 2 | 106 | 31.1% | 0.0% | 0.0% |
| tu93 | **5** | **9** | 1, 1, 1 | 1,435 | 11.6% | **0.0%** | 0.0% |
| m0r0 | 2 | 2 | 2, 2, 2 | 4,457 | 33.6% | 92.6% | 7.4% |
| m0r0 | **3** | **7** | 2, 2, 2 | 35,056 | 0.3% | **0.0%** | 100.0% |
| ft09 | 2 | 0 | 3, 3, 3 | 38,595 | 12.2% | 100.0% | 0.0% |
| ft09 | **3** | **9** | 2, 3, 3 | 114,699 | 20.4% | **100.0%** | 0.0% |
| tr87 | **2** | **6** | 1, 1, 1 | 20,207 | 25.6% | **100.0%** | 0.0% |
| g50t | **2** | **9** | 3, 3, 3 | 17,453 | 9.0% | **0.0%** | 2.7% |
| dc22 | **2** | **4** | 5, 5, 5 | 7,616 | 38.4% | 22.9% | 0.0% |
| dc22 | 3 | 0 | 5, 5 | 8,175 | 41.9% | 23.5% | 1.4% |
| dc22 | **4** | **4** | 3, 4 | 1,896 | 40.4% | 39.2% | 2.0% |
| cd82 | **3** | **4** | 6 | 17,281 | 22.0% | 28.8% | 0.7% |
| vc33 | 3-5 | **8** on level **5** | 0, on every level | 107,292 | 0% | — | — |

The book never claims every live cell on any pair, on any game and level
(the summary's last column: 0.0% everywhere). cd82's training level is 2, where
its other 5 runs stop.

## What it shows

**1. One level out, the carry is good wherever the rules re-bind.** tu93 level 3
and ft09 level 2: right on every new changing pair the book applies to; m0r0
level 2: 92.6%, with 7.4% wrong (the same CP1 rules were wrong on 4.3% of T1's
changing cases in G0b). This agrees with T1.

**2. Where Goose stops, it splits.**
- *ft09 level 3, tr87 level 2: the book carries perfectly.* It applies to a
  fifth to a quarter of the changing moves the level has not seen and gets the
  whole screen right on all of them.
- *tu93 level 5: new objects.* The agent's rule still claims its own move
  correctly (0 wrong), but other blocks move at the same time and no rule claims
  them: on seed 0, every pair the book applies to leaves changed cells unclaimed,
  a median of 18 cells a move on level 4 (rows 22-24, 28-30, 34-35) and 54 on
  level 5. Levels 4 and 5 add mechanics level 1 never showed.
- *m0r0 level 3: the book is wrong wherever it applies* (on 0.3% of pairs). Its
  rules re-bind on the first 1,000 moves but do not hold for the rest of the level.
- *g50t level 2:* applies to 9% and is never right (2.7% wrong).
- *vc33:* its one rule (a blue click) finds no exact binding on any later level.

**3. The plan's condition for IMAGINE holds nowhere.** "Axiom rules jointly
claim the whole live board" assumed rules that predict the whole screen. The
rules written here claim the cells they change and leave the rest to "nothing
changes". Plan 7.3's precondition (on some dev game, "the axiom rules of the
book claim the whole live board on at least 20% of a level's cases") is met by
no game; the plan's rule for that case is that IMAGINE is not built and these
numbers are reported as the reason. The 18 Oct reading that called it met
(`pool-g0b-g1.md`: the whole screen right per move, one level out) is the weaker
one used here: the composed prediction, unclaimed cells unchanged, with every
imagined step checked against the real screen when walked (plan 5.5 already
requires the check).

**4. tu93's level 4 needs the sticky mask.** On level 4 its move-budget bar
(row 63) changes 3 cells a move, which neither S0's ticker scan (`tick_frac`
0.0, nothing detected) nor `gridtools.tick_cells` (components of at most 2 cells)
catches, so level 4's own mask is empty and every move changes a live cell.
Kept from level 3, where the bar is masked, it is covered: on seed 0 the new
changing pairs go from 279 (level 4's own mask, a first pass on 19 Oct) to 33,
because most of them differed only in the bar.

## What it means

- **IMAGINE built on carried rules could act on 2 of the 8 dev games where Goose
  stops (ft09, tr87), partly on 2 (dc22, cd82), and not on 4 (tu93, g50t, m0r0,
  vc33).** On ft09 and tr87, a fifth to a quarter of the untried changing moves
  would get a right prediction. Whether a short search through such predictions
  finds a level-up is untested; ft09's levels are colour puzzles where more new
  screens are not the problem, and tr87 mostly reaches its level 2 late (level 1
  at 9,891-97,093 actions in the 6 runs that reached it, 5 of them after 45,000).
- **On tu93 and m0r0 the stop level brings what the book never saw.** Rules for
  it would have to be written on that level: by the LLM online (H2's stall
  trigger, plan 5.8) or not at all.
- These are Matt's calls (M3's focus, 19 Oct: IMAGINE first). Nothing here is
  tuned; the numbers are dev-game numbers.
