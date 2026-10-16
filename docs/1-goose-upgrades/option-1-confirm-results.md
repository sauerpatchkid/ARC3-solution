# Option 1 Confirm tier — results after two seeds (2026-09-25)

Sweep: `make map-confirm`, the map arm (A4 = novelty label + return map) on all
25 public games, 100k actions. Stopped by choice after seeds 0 and 1: 50 of the
planned 75 runs, no crashes. Seed 2 was not run; it can be added later with
`make map-confirm RESUME=results/sweeps/sweep_20260924_233007.manifest`.

Compared with the novelty-only arm (A1) from the Plan B Confirm sweep, same
games and seeds (`results/sweeps/sweep_20260917_224431.manifest`; the A1 code
path is unchanged, verified action-for-action). Table:
`results/sweeps/map_confirm_2seeds.md`, produced by

```bash
uv run python tools/paired_compare.py \
    --base results/sweeps/sweep_20260917_224431.manifest:A1 \
    --new  results/sweeps/sweep_20260924_233007.manifest:A4
```

## Headline

| measure, first two seeds | novelty only | novelty + map |
|---|---:|---:|
| levels finished, 50 matched runs | 50 | 58 |
| games with at least one level | 17 of 25 | 18 of 25 |
| same game and seed: better / same / worse | | 11 / 30 / 9 |
| throughput, act/s | 139.4 | 138.2 |

## Pre-registered rule: NOT MET

| rule | result |
|---|---|
| 1. Levels: map ≥ novelty-only, and paired wins > losses | 58 vs 50, 11 wins vs 9 losses: pass |
| 2. No game worse on every seed | ar25 and tr87 are worse on both seeds: **fail** |

The rule was written for three seeds. With two, "worse on every seed" is an
easier bar to fail, so this is a stopped-early reading, not a final one. It is
still the result as run: the map is not adopted as the default.

## Where the map helps and where it hurts

**Helps, on games Goose was stuck on:**

- tu93: levels 5 and 3 vs 0 and 2.
- vc33: level 4 on both seeds vs level 2.
- First levels ever recorded on bp35 (both seeds) and lf52 (one seed); tn36,
  su15, g50t and m0r0 also gained.

**Hurts, on games Goose already solved:** ar25 (1,1 vs 2,2), tr87 (0,0 vs
1,1), and one seed each on cd82, cn04, ft09 and sp80. On these the map chooses
about 50% of all actions, almost all walking back after game overs, and unique
states per run roughly halve. The walk back is crowding out exploration that
was already working.

sb26 lost its one level, but the map-arm runs there sat on a frozen screen
(11 unique states, map share 0%), the same frozen-screen behaviour seen before
in other arms. That loss is probably not the map.

## What next

The pattern points at the game-over walk back, not the map as such. The
follow-up is a "stuck-only" variant that keeps the stall route and turns off
the walk back (`EVAL_MAP_RETURN=0`, already a switch). It needs its own dev
test with a pre-registered rule; nothing was retuned during this Confirm.

The map stays in the code behind its off switch, with this document, as a
documented experiment (see `option-1-return-map.md`, "How to remove it").
