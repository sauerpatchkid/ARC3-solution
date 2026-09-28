# Upgrade confirm — results (2026-09-28): ADOPT

75 new runs of **mb_gated_att** (novelty label + return map with progress bars
masked, walk back only while it pays, half credit within an attempt) on all 25
games × seeds 0–2 × 100k actions, 2026-09-27 22:19 to 2026-09-28 ~13:45 (paused
09:41–10:12 at Matt's request), no failed runs. Compared with the novelty
agent's 75 runs from the Plan B Confirm sweep, same games, seeds and budget.
Verdict computed by `tools/paired_compare.py`:
`results/confirm_upgrade/20260927_221915/verdict.md`. Plan and rule:
`upgrade-confirm.md` (written before launch).

```
EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt
```

## Pre-registered rule: PASS

| rule | result |
|---|---|
| 1. Levels ≥ novelty and paired wins > losses | **112 vs 79**; 31 better, 36 same, 8 worse: pass |
| 2. No game worse on every seed | none: pass |

## Headline

| | novelty agent (adopted until now) | mb_gated_att |
|---|---:|---:|
| levels finished, 75 runs each | 79 | **112** |
| games with at least one level | 18 of 25 | **22 of 25** |
| same game and seed: better / same / worse | | 31 / 36 / 8 |
| single-run speed, act/s (ft09, tu93; 10k actions) | 118, 126 | 130, 125 |

For the record: the original baseline (change label) finished 54 levels on 11
games in the same test.

## Per game (levels per seed)

| game | novelty | new | note |
|---|---|---|---|
| dc22 | 0,0,0 | **3,3,1** | first levels on this game in any 25-game test |
| tu93 | 0,2,2 | **4,4,4** | |
| vc33 | 2,2,2 | **4,4,4** | |
| ls20 | 1,1,1 | **2,2,2** | first level 2 ever on ls20 |
| m0r0 | 1,1,1 | **2,2,2** | first level 2 ever on m0r0 |
| s5i5 | 1,1,1 | 2,1,3 | |
| bp35 | 0,0,0 | 1,1,1 | new |
| ka59 | 0,0,0 | 0,1,1 | first level ever on ka59 |
| lf52 | 0,0,0 | 1,1,0 | new |
| re86 | 0,0,0 | 1,1,0 | new |
| su15 | 0,1,0 | 1,1,1 | |
| g50t | 0,1,1 | 1,1,1 | |
| tn36 | 0,0,1 | 1,0,0 | mixed |
| lp85 | 5,5,7 | 5,7,2 | mixed; one seed stopped at level 2 |
| ar25, cn04, r11l, sk48, tr87 | | | tied on every seed |
| cd82 | 2,2,2 | 2,1,1 | worse on 2 seeds |
| ft09 | 3,2,2 | 2,2,2 | worse on 1 seed |
| sp80 | 2,2,2 | 2,1,2 | worse on 1 seed |
| sb26 | 1,0,1 | 0,0,0 | worse on 2 seeds; the only game that lost its level |
| sc25, wa30 | 0 | 0 | still nothing |

## Reading it

- **The map's failure is fixed.** The original map passed its 8-game test and
  then failed this rule because it cost the games already solved (ar25 and tr87
  worse on both seeds). With the gate and the bar fix, ar25 and tr87 tie on
  every seed; the gate stopped walking back on ar25 (map chose 0% of its moves)
  while keeping it where it pays (tu93, vc33, su15, bp35).
- **The gains are broad, not one game:** 12 games improved on their best level,
  5 of them from nothing, and 3 reached a level never seen before in any run
  (dc22 level 3, ls20 level 2, m0r0 level 2, plus ka59's first level).
- **The losses are real and small:** 8 of 75 game-seeds, 5 games, none worse on
  every seed. cd82 and sb26 lost on 2 seeds each and are worth a look; sb26
  spent its runs nearly frozen (late novelty 9.5 vs 557), a pattern seen on
  sb26 before in other arms. lp85 is high-variance for every version.
- **No speed cost.** The report's throughput column (139 vs 36 act/s) compares
  runs made one at a time with runs made four at a time on a shared GPU, so it
  is not a like-for-like number; measured one at a time, both versions run at
  118–130 act/s.
- Caveat carried from the pipeline: scores use the shared decoration detector,
  which misses some progress bars (CLAUDE.md, "Known measurement caveats");
  levels, the headline here, are unaffected.

## What changes

mb_gated_att becomes the adopted Goose for semester 2 comparisons, replacing
the novelty-label-only agent. It stays behind switches: nothing changes for a
teammate who does not set them. Making it the default is a team decision.
