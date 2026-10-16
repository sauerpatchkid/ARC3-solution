# legacy/: finished work, kept for the record

Everything here is **frozen**. Nothing in the live repo imports it, and the root
Makefile and `sweep.sh` never run it (`make check` enforces both). Code in here
may import the live baseline, never the other way round.

The folders follow the project's story. The write-ups for each step are in
`docs/` under the same numbers.

| Folder | What it is | Outcome | Write-ups |
|---|---|---|---|
| `semester1/` | the original API-path scripts (suite runner, summarizer, game probe, 25-game sweep) and the cleanup handoff | superseded by `run_local.py`, `sweep.sh` and the frozen benchmark | — |
| `llm_track/` | semester-1 LLM work: pairwise judge, heuristic writer, first rule finder | all NO-GO or unfinished | its own `README.md` |
| `goose_experiments/` | launchers for the semester-2 **Goose upgrades**: Plan B, Option 1, two upgrade screens and the upgrade confirm | adopted `mb_gated_att`, 54 → 79 → 112 levels (the agent code itself is live, in `custom_agents/`) | `docs/1-goose-upgrades/` |
| `coach_track/` | the **Coach**: an LLM suggests what to try when Goose stalls | NO-GO, at chance (21 hits vs 21.2 expected) | `docs/2-llm-history/` |
| `rulebook/` | **Rulebook v1 and v2**: an LLM writes replay-checked rules | v1 FAIL (2 of 8); v2 CP1 FAIL (2 of 8, 0.0% wrong at T1) | `docs/2-llm-history/` |

What replaced them:
- **Goose:** `mb_gated_att` is the live reference agent (A0).
- **LLM track:** Glassbox (`docs/3-glassbox/`, code in `custom_agents/gb/`). It
  starts from a **copy** of Rulebook v2's checking machinery
  (`custom_agents/gb/core/`), not an import.
