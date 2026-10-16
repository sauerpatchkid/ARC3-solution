# Upgrade confirm — round 2's winner on all 25 games

Status: DONE 2026-09-28, verdict ADOPT (112 vs 79 levels, 31/36/8, no game worse on
every seed). Results: `upgrade-confirm-results.md`.

## What is tested

**mb_gated_att**, round 2's winner (`upgrade-screen2-results.md`): the novelty
label plus the return map with progress bars masked, walk back only while it
pays, and half credit for screens new within an attempt.

```
EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt
```

Against the adopted agent (novelty label only, arm A1), on all 25 public games
× seeds 0, 1, 2 × 100k actions: the same games, seeds and budget as the Plan B
Confirm sweep. The novelty agent's 75 runs are reused from that sweep
(`results/sweeps/sweep_20260917_224431.manifest`, arm A1); its code path has not
changed since (with the new switches unset the agent takes the same actions,
checked action-for-action). Only the 75 new runs are made, four at a time,
about 14 hours.

## Pre-registered rule (the Confirm rule used for the novelty label and the map)

Computed by `tools/paired_compare.py`, paired by (game, seed):

1. Levels summed over the 75 pairs: new ≥ novelty, **and** paired wins exceed
   losses.
2. No game where the new version completes fewer levels than novelty on
   **every** seed.

Pass both: it replaces the novelty agent as the adopted Goose (still behind
switches; teammates' defaults unchanged). Fail either: the novelty agent stays
adopted and the result is reported as it is.

Read beside the levels: which games moved, the map's share of actions, route
outcomes and the walk-back chooser's decisions per game.

## Why this is worth 14 hours

Two short screens (2 seeds, 50k) put the fixed map family well ahead of the
novelty agent (round 1: 30 vs 14 levels on 8 games; round 2: 41 vs 23 on 11),
and the winner was worse than novelty on only 1 of 22 game-seeds. The original
map passed its 8-game test and then failed this 25-game rule because it hurt
games already solved (ar25, tr87); this test checks the fix holds across all
25 games at full length.

## Running it

```bash
make upgrade-confirm DRY_RUN=1
make upgrade-confirm
make upgrade-confirm-status
make upgrade-confirm-pause
make upgrade-confirm RESUME=results/confirm_upgrade/<stamp>/manifest.tsv
```

When the runs finish, the verdict is written to
`results/confirm_upgrade/<stamp>/verdict.md`.

## Separation

`experiments/upgrade_confirm/` reuses the screen runner as a library and adds
one Makefile line tagged `# [upgrades-confirm]`. No agent code.
