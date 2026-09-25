# experiments/upgrade_screen — the upgrade screen

A short screening sweep over nine candidate improvements to the novelty label
and the return map, ranked against the adopted agent. Built and smoke-tested,
not yet run. Reasoning and evidence: `docs/plans/upgrade-screen.md`
(technical) and `docs/reports/Goose_Upgrade_Options.docx` (plain language).

```bash
make upgrade-screen DRY_RUN=1      # 12 arms x 8 games x 2 seeds x 50k = 192 runs, ~17 h
make upgrade-screen                # detached; keep the machine awake
make upgrade-screen-status
make upgrade-screen-pause          # runs in progress finish and are kept
make upgrade-screen RESUME=results/screen/<stamp>/manifest.tsv
```

Files: `screen.py` (runner: parallel jobs, per-run results directories,
resume, pause), `rank.py` (leaderboard and the pre-registered "promising"
rule), `screen.mk` (make targets, included from the root Makefile by one
tagged line). The candidates themselves are in `custom_agents/upgrades.py`.

**Removing it.** Delete this folder, `custom_agents/upgrades.py` and
`tests/test_upgrades.py`, then:

```bash
sed -i '/\[upgrades\]/d' custom_agents/action.py Makefile check_repo.py
```

That restores each of those three files byte for byte. The baseline, the
novelty label and the return map do not depend on anything here.
