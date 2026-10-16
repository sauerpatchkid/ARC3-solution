# Plan B dev sweep — results (2026-09-17)

Sweep: 6 dev games × 3 seeds × 100k actions × 4 arms = 72 runs, local engine,
`make planb-dev`. Manifest `results/sweeps/sweep_20260915_200509.manifest`;
summary and curves beside it. Arms: A0 baseline (change label, no mask),
A1 novelty label, A2 tried-action mask, A3 both. Same seeds across arms; runs
are not byte-reproducible past ~300 actions (CUDA), so read distributions.

## Levels reached (seeds reaching / 3; median actions to that level)

| game | A0 baseline | A1 novelty label | A2 tried mask | A3 both |
|---|---|---|---|---|
| ls20 | 0/3 | **L1 3/3** (38k) | 0/3 | L1 2/3 (61k) |
| tu93 | 0/3 | **L2 2/3** (22k) | L2 1/3 (17k) | 0/3 |
| g50t | 0/3 | **L1 2/3** (16k) | 0/3 | L1 2/3 (21k) |
| ft09 | L2 3/3 (5.3k) | L2 3/3 (8.3k) | L2 3/3 (11.3k) | L2 3/3 (9.1k) |
| lp85 | L5 3/3 (18k), L7 2/3 | L5 3/3 (27k), L7 1/3 | L5 3/3 (43k), L7 2/3 | L5 3/3 (42k), L7 1/3 |
| dc22 | 0/3 | 0/3 | 0/3 | 0/3 |

ls20, tu93 and g50t had never produced a level in any earlier run (16, 8 and 9
baseline runs respectively).

## Mechanism metrics (mean over seeds)

| game | arm | unique states | redundancy | late novelty /1k | act/s |
|---|---|---:|---:|---:|---:|
| g50t | A0 → A1 | 5,370 → 28,981 | 0.86 → 0.49 | 18 → 245 | 138 → 136 |
| tu93 | A0 → A1 | 186 → 1,155 | 0.99 → 0.97 | 0.0 → 0.2 | 144 → 143 |
| ls20 | A0 → A1 | 6,566 → 9,609 | 0.81 → 0.75 | 11 → 6 | 138 → 139 |
| dc22 | A0 → A2 | 3,782 → 288 | 0.76 → 0.38 | 7 → 0 | 140 → 136 |
| lp85 | A0 → A1 | 52,770 → 78,921 | 0.017 → 0.024 | 914 → 919 | 143 → 142 |

Throughput is within 3% on every arm. The sampler's all-zero guard
(`Agent/degenerate_samples`) fired in 0 of 72 runs; the one crash it fixes
happened once, before the fix, on tu93 A3.

## Verdict

**Adopt A1 (novelty label) for the Confirm tier. Drop the tried-action mask.**

- The pre-registered success bar (§6 of the plan) is met by A1 on criterion
  (a): a level reached on a dev game that A0 never reached — three games.
- Where A1 does not add levels (ft09, lp85) it reaches the same levels, with
  the mid-level medians somewhat slower (ft09 L2 5.3k → 8.3k; lp85 L3 12.6k →
  23.5k). Seed ranges overlap and the final level counts match within one
  seed, so this is "watch on Confirm", not a regression call.
- The mask (A2) adds nothing on its own, and on dc22 it collapses exploration
  (unique states 3,782 → 288, late novelty 0). Combined with the label (A3) it
  erases the label's tu93 win (2/3 → 0/3): tu93 needs repeated moves, which is
  exactly the risk the plan listed. Not worth a Confirm slot.
- dc22 moved for nobody; it stays a null contrast.

## Next

Confirm tier (plan §7.4): A0 vs A1 on all runnable public games × 3 seeds ×
100k. Adoption rule: A1 wins on the primary metrics and is not worse than A0
on any game beyond seed noise.
