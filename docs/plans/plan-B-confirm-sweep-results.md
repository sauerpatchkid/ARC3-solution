# Plan B Confirm tier — results (2026-09-22)

Sweep: all 25 public games × 3 seeds × 100k actions × 2 arms = 150 runs,
local engine, `make planb-confirm`. Manifest
`results/sweeps/sweep_20260917_224431.manifest`; summary and curves beside it.

- **A0** — baseline: change label, no mask (`EVAL_LABEL=change`).
- **A1** — the dev-tier pick: novelty label (`EVAL_LABEL=novel`), no mask.

The tried-action mask (dev arms A2/A3) was dropped after the dev tier; see
`plan-B-dev-sweep-results.md`. Runs are not byte-reproducible past ~300
actions (CUDA), so comparisons are paired across seeds, never per-trajectory.
No crashes or restarts in the Confirm portion; the sampler's all-zero guard
never fired.

## Headline

| measure | A0 | A1 |
|---|---:|---:|
| total levels completed (75 runs each) | 54 | **79** |
| games reaching ≥1 level | 11/25 | **18/25** |
| paired (game, seed) wins / ties / losses | — | **23 / 51 / 1** |
| throughput (act/s, mean) | 140.2 | 139.3 |

## Per game (levels per seed)

| game | A0 | A1 | W/T/L | note |
|---|---|---|---|---|
| ar25 | 2,2,2 | 2,2,2 | 0/3/0 | |
| bp35 | 0,0,0 | 0,0,0 | 0/3/0 | |
| cd82 | 2,2,2 | 2,2,2 | 0/3/0 | coverage 13× |
| cn04 | 1,1,1 | 1,1,1 | 0/3/0 | |
| dc22 | 0,0,0 | 0,0,0 | 0/3/0 | null contrast |
| ft09 | 2,2,2 | **3**,2,2 | 1/2/0 | L3 first time in any run |
| g50t | 0,0,0 | 0,**1**,**1** | 2/1/0 | new |
| ka59 | 0,0,0 | 0,0,0 | 0/3/0 | |
| lf52 | 0,0,0 | 0,0,0 | 0/3/0 | frozen screen, both arms |
| lp85 | 7,5,5 | 5,5,7 | 1/1/1 | tie on medians (see below) |
| ls20 | 0,0,0 | **1,1,1** | 3/0/0 | new, every seed |
| m0r0 | 1,1,1 | 1,1,1 | 0/3/0 | |
| r11l | 1,1,1 | 1,1,1 | 0/3/0 | |
| re86 | 0,0,0 | 0,0,0 | 0/3/0 | |
| s5i5 | 0,0,0 | **1,1,1** | 3/0/0 | new, every seed |
| sb26 | 0,0,0 | **1**,0,**1** | 2/1/0 | new |
| sc25 | 0,0,0 | 0,0,0 | 0/3/0 | |
| sk48 | 1,1,1 | 1,1,1 | 0/3/0 | |
| sp80 | 1,1,1 | **2,2,2** | 3/0/0 | L2 every seed |
| su15 | 0,0,0 | 0,**1**,0 | 1/2/0 | new |
| tn36 | 0,0,0 | 0,0,**1** | 1/2/0 | new |
| tr87 | 1,1,0 | 1,1,**1** | 1/2/0 | |
| tu93 | 0,0,0 | 0,**2,2** | 2/1/0 | new, straight to L2 |
| vc33 | 1,0,1 | **2,2,2** | 3/0/0 | L2 every seed |
| wa30 | 0,0,0 | 0,0,0 | 0/3/0 | |

**lp85 is a tie, not a loss.** Both arms reach L5 on 3/3 seeds and L7 on 1/3;
A1's medians are slightly faster (L3 17.6k → 13.6k, L7 80.5k → 76.3k). The
per-seed vectors differ only in which seed got the long run. The dev tier
showed the same pattern and the same conclusion.

## Mechanism: coverage

Median unique canonical states per action, on games where A0 was effectively
stuck:

| game | A0 | A1 | ratio |
|---|---:|---:|---:|
| sb26 | 0.0001 | 0.0940 | 759× |
| s5i5 | 0.0007 | 0.0648 | 97× |
| vc33 | 0.0006 | 0.0096 | 15× |
| cd82 | 0.0279 | 0.3694 | 13× |
| g50t | 0.0540 | 0.3762 | 7× |
| tu93 | 0.0018 | 0.0054 | 3× |
| bp35 | 0.0686 | 0.2027 | 3× |
| ls20 | 0.0662 | 0.0901 | 1.4× |

Read with redundancy and levels (per CLAUDE.md): on sb26, s5i5, vc33, g50t
and tu93 the coverage gain comes with new level completions, so it is not
decoration-jiggling.

## Where A1 is not better

- **Speed to the first level is unchanged at best.** Paired actions-to-L1 on
  the games both arms clear is mixed in both directions with no pattern.
- **AULC (levels weighted early in log budget) drops on 6 games** — ar25
  0.95 → 0.63, cn04 0.45 → 0.34, m0r0 0.49 → 0.45, cd82, tr87 — while rising
  on 9 (vc33 0.21 → 0.87, tu93/ls20/s5i5/g50t 0.00 → 0.16–0.20, sk48
  0.18 → 0.40). Sum over games 8.07 → 9.32. The novelty label spends early
  actions exploring rather than repeating what already worked, so on games A0
  solves quickly the same level arrives later. Final level counts are equal on
  all six.
- **7 games remain at zero for both arms** (bp35, dc22, ka59, lf52, re86,
  sc25, wa30). lf52 sits on a frozen screen under both arms (21 vs 18 unique
  states in 100k actions) — worth a separate look, unrelated to the label.

## Adoption decision

**Adopt A1 as the baseline agent.** The plan's adoption rule (§6) required a
Confirm-tier win on the primary metrics with no game worse than seed noise:

- primary metric (levels): 79 vs 54, 23 paired wins against 1 loss, and that
  loss is a median-level tie;
- no game is worse on level counts in every seed;
- cost: 0.6% throughput.

Pre-registered success bar (a) — "a new level reached on a dev game that A0
never reached" — is met on 7 games, 6 of them outside the dev set.

`EVAL_LABEL=novel` still defaults to off, so teammates' baselines are
unchanged unless they opt in. Flipping the default is a separate decision.

## Next

Plan A (`plan-A-llm-advisor.md`) builds on this agent: its B0 arm is A1.
Step 1 there is the offline advice-quality probe over existing corpora, which
needs no GPU and no agent changes.
