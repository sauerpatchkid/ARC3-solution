# Goose upgrade experiments (semester 2, archived 16 Oct 2026)

These launchers ran the experiments that turned the semester-1 Goose into
`mb_gated_att`. **The upgrades themselves are live code** in `custom_agents/`;
only the sweeps that tested them live here.

| Step | Upgrade (live file, switch) | Test | Result |
|---|---|---|---|
| Plan B | novelty label: learn from "reached a new screen this level", not "the screen changed" (`canon.py`, `EVAL_LABEL=novel`) | `planb-dev`, `planb-confirm` (`label-diag` was step 1) | adopted: 79 vs 54 levels on 25 games × 3 seeds |
| Option 1 | return map: remember the level's screens and walk back to untried ones (`return_map.py`, `EVAL_RETURN_MAP=1`) | `map-dev`, `map-confirm` | dev 45 vs 32; not adopted alone (ar25, tr87 worse) |
| Upgrade screens | nine candidate upgrades in two rounds (`upgrades.py`, `EVAL_UPGRADES=...`) | `upgrade-screen`, `upgrade-screen2` | winner: map + bars + map_gated + attempt |
| Upgrade confirm | the winner as one agent, `--agent mb_gated_att` (`presets.py`) | `upgrade-confirm` | **adopted: 112 vs 79 levels**, 31 / 36 / 8 better / same / worse |

Run from the repo root:

```bash
make -f legacy/goose_experiments/experiments.mk upgrade-confirm DRY_RUN=1
make -f legacy/goose_experiments/experiments.mk planb-status
```

Write-ups: `docs/1-goose-upgrades/`. The 75 `mb_gated_att` runs from the
upgrade confirm (`results/confirm_upgrade/20260927_221915/`) are Glassbox's A0
reference and its recordings.

Moved here from `experiments/upgrade_screen*/`, `experiments/upgrade_confirm/`,
`tools/label_diagnostic.py` and the root Makefile on 16 Oct 2026. The move
changed only the paths these files use to find the repo.
