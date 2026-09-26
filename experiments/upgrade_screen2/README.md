# experiments/upgrade_screen2 — upgrade screen, round 2

Combinations of round 1's two best ideas (map_bars, map_gated) with the two
cheap ideas that helped without losses (attempt, deadclick), on round 1's 8
games plus 3 new ones. Plan and pre-registered rule:
`docs/plans/upgrade-screen2.md`. Round 1's best is saved at git tag
`upgrade-screen-round1`.

```bash
make upgrade-screen2 DRY_RUN=1     # 8 arms x 11 games x 2 seeds x 50k = 176 runs, ~17 h
make upgrade-screen2
make upgrade-screen2-status
make upgrade-screen2-pause
make upgrade-screen2 RESUME=results/screen2/<stamp>/manifest.tsv
```

`screen2.py` reuses round 1's runner as a library (it swaps in its own arms,
games and results folder and never edits a round-1 file); `rank2.py` compares
every arm with round 1's best (map_bars) and with the adopted agent.

**Removing it:** delete this folder, then
`sed -i '/\[upgrades2\]/d' Makefile`. No agent code belongs to round 2.
