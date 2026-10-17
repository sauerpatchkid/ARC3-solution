# G0b: the bar mask — result (16 Oct 2026)

*Judged by pre-registration section 3b (`glassbox-prereg.md`, registered in
`763d253`). A diagnostic, not a gate. It follows from G0
(`g0-guard-rescue.md`).*

## In one line

**Masking the whole of m0r0's move-budget bar turns all 11 of its near-miss
rules into trusted rules, and its book goes from explaining 0% to 53.5% of the
training level's changed cells.** On the next level the book is right on 51.7%
of moves, against 6.8% for "nothing", but below "memory" (53.9%), and wrong on
4.3% of the changing cases: the one expectation not met.

## The run

- `make gb-g0b`, 16 Oct, commit `763d253` (code unchanged since `78c461c`),
  CPU, about 6.5 minutes; config hash `2f789d8dca149dfe`; all as registered. Logs
  in `results/glassbox/g0b_bars/` (`g0b_log.txt`, `control_log.txt`).
- **Valid.** The control (`make gb-g0b-control`, bars off) was run at the same
  commit just before and printed "ALL EQUAL TO CP1": CP1's book, T0, T1 and
  re-bindings on all 8 games. So every difference below comes from the mask.
  The manifest says `763d253+dirty` for the same reason as G0's (the harness
  submodule's local edits only; prereg section 11).

## Expectations, written before running

| Expectation | Result | |
|---|---|---|
| m0r0: all 11 near-miss rules become trusted | 11 of 11 trusted (each 100% claimed-exact, 606 or 575 moves) | met |
| m0r0: the book's cell coverage rises above 0% | 0% → 53.5% (case coverage 0% → 53.0%), 2 rules | met |
| tu93: no newly trusted rule | none | met |
| the other six: CP1's books, T1 within 1 point | same books; T1 moves by +0.5 points (ft09) at most | met |
| no game's T1 wrong share above 0% | m0r0: 4.3% | **not met** |

## Per game (CP1 → G0b)

Training level: admitted / trusted rules, book rules, cell and case coverage.
Transfer level: T1 per move for the book, "nothing" and "memory" (S0 masks more
cells for every arm), the differences with their 95% intervals, and the wrong
share of changing cases.

| Game | Admitted / trusted | Book | Cell cov. | Case cov. | T1 book | nothing | memory | T1 − nothing | T1 − memory | Wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| cd82 | 7 / 6 → 7 / 6 | 4 → 4 | 5.2% | 8.3% | 14.9% | 13.1% | 17.3% | +0.018 [0.017, 0.018] | −0.024 [−0.027, −0.022] | 0% |
| dc22 | 5 / 5 → 5 / 5 | 3 → 3 | 11.8% | 14.7% | 25.5% | 15.4% | 23.4% | +0.101 [0.097, 0.105] | +0.021 [0.004, 0.037] | 0% |
| ft09 | 4 / 4 → 4 / 4 | 3 → 3 | 97.7% | 56.0% | 18.1% → 18.6% | 3.0% | 14.8% → 15.2% | +0.156 [0.151, 0.160] | +0.034 [0.028, 0.040] | 0% |
| g50t | 3 / 3 → 3 / 3 | 2 → 2 | 17.7% | 0.0% | 9.1% | 9.1% | 15.0% | 0 | −0.059 [−0.062, −0.056] | 0% |
| **m0r0** | **11 / 0 → 11 / 11** | **0 → 2** | **0% → 53.5%** | **0% → 53.0%** | **6.2% → 51.7%** | 6.2% → 6.8% | 45.4% → 53.9% | **+0.449 [0.444, 0.454]** | **−0.022 [−0.033, −0.010]** | **0% → 4.3%** |
| tr87 | 0 / 0 | 0 | 0% | 0% | 0% | 0% | 47.6% | 0 | −0.476 [−0.495, −0.462] | 0% |
| tu93 | 0 / 0 | 0 | 0% | 0% | 3.6% → 3.8% | 3.6% → 3.8% | 85.0% → 87.1% | 0 | −0.833 [−0.852, −0.813] | 0% |
| vc33 | 1 / 1 → 2 / 2 | 1 → 1 | 10.6% | 0.0% | 75.1% | 75.1% | 75.1% | 0 | 0 | 0% |

(One number where CP1 and G0b agree.) As in CP1, dc22 and ft09 beat both
baselines; m0r0 now beats "nothing" by 45 points but not "memory".

**S0's report.** Conflicting cases on the training level: m0r0 13 → 0, tu93
11 → 0, g50t 93 → 93, the others 0; on the score splits m0r0 758 → 0, tu93
96 → 0, ft09 33 → 0. Mask cells: m0r0 122 → 128, tu93 60 → 64, vc33, cd82, tr87
63 → 64 (training level), as registered.

**The guard fitter** found nothing new: the only admitted-but-not-trusted rule
left is cd82's, with 1 wrong case (it needs 3).

## m0r0 on the next level

The book is rules 2 (ACTION1) and 31 (ACTION2), each moving m0r0's two blue
objects one step. S4 re-bound both on level 2's fit split with the step changed
from 5 to 4, flagged *binding ambiguous* (several bindings were exact on those
300 moves), and both are trusted over the two levels.

The 91 wrong cases (73 on ACTION2, 18 on ACTION1, of 2,138 changing cases; none
seen to end two ways) are a different mechanism, not a wrong step. Where T1 is
right, the two objects move 4 rows together; where it is wrong, they jump 24 to
40 rows, sideways by different amounts. The fit split showed that twice in 300
moves, so re-binding could not have known it. That is the case the plan's
safety bar for G1 is about (T1 wrong ≤ 5% on every game): m0r0 would pass it at
4.3%, narrowly.

## What it means

- **The cause G0 found was the right one.** One gap in the decoration mask kept
  11 correct rules out of the book; closing it costs nothing elsewhere: the
  other seven games' books are unchanged.
- **S0 should be on for G1** (section 4 decides, naming the mask as a
  difference from CP1). Its evidence is still CP1's on every game where no bar
  was cut short.
- **Transfer is now m0r0's limit**, not trust: a trusted rule met a mechanism on
  level 2 that level 1 never showed. That points at S4's online wait (more
  evidence before carrying) and at guards on the next level, where a rule that
  carried can be a near-miss with real wrong cases.
