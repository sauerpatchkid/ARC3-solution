# G1: the offline rule test — result (17 Oct 2026)

*Judged by pre-registration section 4 (`glassbox-prereg.md`, registered in
`af208c7`, pushed before the run): the mask on, judged against G0b.*

## In one line

**FAIL: the new rule writing beats G0b on 3 of 8 games (it needed 5).** Safety
holds: T1 is wrong on at most 0.15% of changing cases (cd82). The 3 wins are
large and on games no earlier attempt could do: **tu93 goes from 0% to 59.0%**
of the training level's changed cells and **tr87 from 0% to 35.5%**, both their
first trusted rules ever; dc22 goes from 11.8% to 29.3%. Where CP1 already had
good rules (ft09 97.7%, g50t 17.7%), starting from scratch found less, and the
new books carry to the next level worse than CP1's did.

## The run

- `make gb-g1`, Sat 17 Oct 13:33 to 21:15 (7 h 42 min), commit `af208c7` (code
  unchanged since `bd0552b`), config `5b7a7b2684e37ce5`, model
  `cyankiwi/Qwen3.8-27B-AWQ-INT4`, vocabulary `glassbox_vocab_v1`; all as
  registered. Log: `results/glassbox/g1/g1_log.txt`; verdict:
  `g1/g1_verdict.{json,md}`; report page: `g1/s6_show/index.html`.
- **Valid.** No group failed; on every game the guard stage's "before" equals
  S3's book and S4's T1 (`before_matches_import`); S4's searches all finished.
  S2's binding search ran out of its time budget with bindings untried on tr87
  (1,548) and cd82 (199), as CP1's did on tr87; section 4 reports this, it does
  not void.

## Per game (G0b → G1)

Training level: the composed book's cell and case coverage, trusted rules.
Transfer level: T1 per move (G0b's in brackets), the differences from "nothing"
and "memory" with their 95% intervals, and the wrong share of changing cases.

| Game | Cell coverage | Case coverage | Trusted | T1 (G0b) | T1 − nothing | T1 − memory | Wrong | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|
| **tu93** | **0.0% → 59.0%** | 0.0% → 59.0% | 0 → 3 | 40.7% (3.8%) | +0.369 [0.365, 0.373] | −0.464 [−0.482, −0.444] | 0% | 25 | 784,449 |
| **tr87** | **0.0% → 35.5%** | 0.0% → 25.1% | 0 → 1 | 19.7% (0.0%) | +0.197 [0.189, 0.206] | −0.279 [−0.300, −0.262] | 0% | 26 | 651,245 |
| **dc22** | **11.8% → 29.3%** | 14.7% → 6.0% | 5 → 2 | 15.4% (25.5%) | 0 | −0.080 [−0.097, −0.064] | 0% | 47 | 1,000,245 |
| g50t | 17.7% → 9.0% | 0.0% → 0.0% | 3 → 3 | 9.1% (9.1%) | 0 | −0.059 [−0.062, −0.056] | 0% | 40 | 1,115,903 |
| vc33 | 10.6% → 6.1% | 0.0% → 0.0% | 2 → 1 | 75.1% (75.1%) | 0 | 0 | 0% | 8 | 214,797 |
| ft09 | 97.7% → 59.2% | 56.0% → 56.0% | 4 → 7 | 3.0% (18.6%) | 0 | −0.122 [−0.126, −0.117] | 0% | 11 | 217,439 |
| m0r0 | 53.5% → 53.5% | 53.0% → 53.0% | 11 → 4 | 6.8% (51.7%) | 0 | −0.471 [−0.482, −0.461] | 0% | 23 | 584,336 |
| cd82 | 5.2% → 4.2% | 8.3% → 4.0% | 6 → 11 | 13.7% (14.9%) | +0.006 [0.006, 0.007] | −0.036 [−0.038, −0.033] | 0.15% | 65 | 1,696,741 |

**Primary:** higher than G0b on tu93, tr87 and dc22: 3 of 8, not met.
**Safety:** met on all 8. **CP1's own criterion** (T1 beats "nothing" and
"memory", both intervals above zero): no game (G0b: dc22 and ft09).

**The LLM's work:** 245 calls after the first round, 385 candidates, 6.27M
tokens (CP1: 9.66M). 134 of 385 answers ran out of room (35%; CP1: 44%), and
every one of them then wrote code in its code-only retry. 304 of the 309 graded
candidates call an H1 primitive, and so do 29 of the 30 trusted rules S2 wrote.
**Guards:** inside the loop the fitter never found a guard; after it, it rescued
2 rules (cd82, ft09), neither changing a book's coverage.

## Against the expectations (section 4)

| Expectation | Result |
|---|---|
| tu93 gains a trusted movement rule and coverage above 0% (likely) | 3 trusted rules (they swap the agent's block), 59.0%: met |
| g50t and cd82 gain a little (likely) | both lower (9.0%, 4.2%): not met |
| dc22, ft09, m0r0 stay about where they are | dc22 up to 29.3%; m0r0 equal; ft09 down to 59.2% |
| tr87 may stay at 0% | 35.5%, with its first trusted rule |
| the primary is uncertain (~45%) | 3 of 8 |
| safety holds; m0r0 the risk | met everywhere; m0r0 0% wrong |

## What the run shows

**1. The new rule writing reaches games CP1 could not.** tu93 and tr87 had no
trusted rule in CP0, CP1 or G0b. Now tu93's three rules (the agent's block swaps
with a slot above, beside or across a bar) explain 59.0% of its changed cells;
two of them carry to level 3, where the book beats "nothing" by 37 points. tr87's one rule (two aligned
tiles shift along a bar, wrapping at its end) explains 35.5% across 13,903 moves.
dc22's book explains 2.5 times as many cells as CP1's.

**2. It starts from scratch, so it loses what CP1 had.** G1's books hold only
G1's rules. On ft09 the red-click group got no rule into the book, and the
blue-click rules cover 3,564 changed cells where CP1's covered 6,192. g50t's and
vc33's new books are smaller than CP1's. The gate compares a fresh book with
CP1's best, game by game, so each of these counts as a loss.

**3. The new books carry worse.** T1 equals "nothing changes" on ft09 (all four
rules apply to no move of the 300-move fit split, flagged *binding untested*),
on m0r0 (rules as good as CP1's on level 1, 53.5% each way, but the re-binding is
*ambiguous* and the one chosen applies to nothing on level 2), and on dc22.
CP1's rules for the same games, on the same evidence (G0b), carried: ft09 18.6%,
m0r0 51.7%, dc22 25.5%. The fit split, not
the rules' accuracy, is what fails here: re-binding on 300 moves cannot tell
apart parameter values the rule's own level never distinguished.

**4. Guards were not the bottleneck.** Inside the loop the fitter never found a
guard; after it, it rescued 2 rules, without changing coverage. As in G0, the near-misses
that limit the books are not ones a screen fact separates.

## What it means

The registered reading (plan §6.3): *"G1 fails: M2 goes ahead with whatever
trusted rules exist (S5 only uses trusted and axiom rules, so it stays safe),
and G1 is reported as a diagnosis with its funnel."* Beyond that, the data point
at three things, each Matt's call:

- **Keep rules across runs.** A book drawn from CP1's and G1's trusted rules
  together would hold both tu93's new rules and ft09's old ones. It costs no LLM
  call; it is a new comparison, so it would be reported as exploratory, not as G1.
- **Re-binding needs more than a 300-move window**, or hints: S4's planned role
  hints in `bind()` and its online wait for evidence (plan §5.4) are aimed at
  exactly the *untested* and *ambiguous* flags seen here.
- **The rule writer can be pointed at what CP1 lacked** (residual groups, games
  with no rule) rather than rewriting groups CP1 already explained.
