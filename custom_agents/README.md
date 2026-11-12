# custom_agents/: every agent

Two things live here: **Goose and its improvements** (the files in this
folder) and **Glassbox** (the `gb/` folder, which has its own
[README](gb/README.md)).

## Goose and its improvements

Goose is a small CNN that learns, while it plays, which moves do something,
and samples its moves from that prediction. Each improvement is its own file
behind its own switch, off by default, so `--agent goose` with no flags is
still the semester-1 baseline.

| File | What it is | Switch | Result |
|---|---|---|---|
| `action.py` | Goose: the network, the sampler, the training step, the recording of every move | `--agent goose` | 54 levels |
| `canon.py` | Novelty label: learn "did this move reach a screen not seen before on this level?", with blinking cells masked out | `EVAL_LABEL=novel` | 79 levels, adopted 22 Sep |
| `return_map.py` | Return map: remember which move leads where, and walk back to untried places when stuck or after a game over | `EVAL_RETURN_MAP=1` | 45 vs 32 on the dev games; not adopted alone |
| `upgrades.py` | `bars` (progress bars masked), `attempt` (half credit for a screen new this attempt), `map_gated` (walk back only while it pays), `deadclick` (not adopted) | `EVAL_UPGRADES=bars,map_gated,attempt` | screened in two rounds |
| `presets.py` | **`mb_gated_att`, the adopted Goose (A0)**: the three switches above, fixed under one name | `--agent mb_gated_att` | **112 levels**, adopted 28 Sep |
| `carry.py` | Carry: colours that do nothing when clicked (`carry_noops`) and what shrank before a win (`carry_hints`), carried to the next level. No LLM; the same moves as Glassbox's LLM-free arm | `EVAL_UPGRADES=...,carry_noops,carry_hints` | built 12 Nov; its test is running; not adopted |

Levels are on all 25 public games × 3 seeds × 100,000 actions. Write-ups:
[`docs/1-goose-upgrades/`](../docs/1-goose-upgrades/).

`action.py` is the only file that calls the others, and it does so on tagged
lines: `# [return-map]`, `# [upgrades]`, `# [gb]`. Deleting a tag's lines
(`sed -i '/\[upgrades\]/d' custom_agents/action.py`) removes that feature.

## Shared pieces

| File | What it is |
|---|---|
| `__init__.py` | **The agent registry.** One line per agent; `--agent <name>` looks it up here |
| `TEMPLATE.py` | A working random agent with the whole contract wired up. Copy it to start your own |
| `random_agent.py` | The matched random floor: uniform over the same action space Goose samples from |
| `gridtools.py` | Helpers for 64×64 screens (bars, blobs, objects), written once and shared |
| `view_utils.py` | Heatmap pictures (`EVAL_SAVE_VIS=1`) |

## Adding an agent

Copy `TEMPLATE.py`, add one line to `REGISTRY` in `__init__.py`, run
`make check`. The root [README](../README.md#2-adding-your-own-agent) has the
steps.
