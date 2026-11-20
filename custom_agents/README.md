# custom_agents/: every agent

Three things live here: **Goose and its improvements** (the files in this
folder), **Glassbox** (the `gb/` folder, which has its own
[README](gb/README.md)) and the **Knowledge Planner** (`kp_bridge.py` and the
`kp_ref/` folder).

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
| `carry.py` | Carry: colours that do nothing when clicked (`carry_noops`) and what shrank before a win (`carry_hints`), carried to the next level. No LLM; the same moves as Glassbox's LLM-free arm | `EVAL_UPGRADES=...,carry_noops,carry_hints` | **not adopted** (13 Nov): its test failed. Later levels came no sooner (23 level races against 22); 87 vs 81 levels on 7 games, inside the noise |
| `upgrades.py` | `map_sched`: a bandit over five map schedules (how soon the map steps in, how far it walks back, untried buttons first), in `map_gated`'s place. `objclick`: where a click lands is chosen by shape, not by cell | `EVAL_UPGRADES=bars,map_sched,attempt,objclick` | **not adopted**: [upgrade screen 3](../docs/1-goose-upgrades/upgrade-screen3-results.md) (16-17 Nov, exploratory, one seed, 5,000 actions). Levels on 25 games: the adopted Goose 15; `map_sched` 18, `objclick` 19, both 19; all three upgrades together 22, which clears the noise |
| `train_more.py` | Train more often, and on fresher data: a gradient step every N actions, with the newest stored moves in every batch | `EVAL_TRAIN_EVERY=1 EVAL_TRAIN_RECENT=16` (defaults `5` and `0`) | **not adopted**: upgrade screen 3. Alone 20 levels against 15 (inside the noise); with `map_sched` 21, with all three 22 (both clear it). About 3 times the time per action |

Levels are on all 25 public games × 3 seeds × 100,000 actions. Write-ups:
[`docs/1-goose-upgrades/`](../docs/1-goose-upgrades/).

`action.py` is the only file that calls the others, and it does so on tagged
lines: `# [return-map]`, `# [upgrades]`, `# [train]`, `# [gb]`, `# [kp]`. Deleting a tag's lines
(`sed -i '/\[upgrades\]/d' custom_agents/action.py`) removes that feature.

## The Knowledge Planner

A groupmate's planner for the levels after the first, run unedited behind
Goose. Off by default. Write-ups:
[`docs/4-knowledge-planner/`](../docs/4-knowledge-planner/).

| File | What it is | Switch | Result |
|---|---|---|---|
| `kp_bridge.py` | The bridge: it loads the planner, shows it every screen Goose sees, books Goose's own clicks with it, and sends the planner's click when it has one. An error inside the planner switches it off for the rest of the run | `EVAL_KP=1` (`EVAL_KP_PLAN=0`: watch and learn only; settings in `experiments/kp/kp.toml`) | **not adopted**: its [registered test](../docs/4-knowledge-planner/kp-results.md) passed on 19 Nov. Goose with the screen 3 upgrades, 25 games x 2 seeds x 5,000 actions: 62 levels with the planner against 44 and 47 without. All of the gain is on ft09 and lp85, the games it was built on |
| `kp_ref/` | The groupmate's four files (their agents E18, E19, E20 and the knowledge bank), byte for byte. **Never edited**; a test checks their hashes | loaded by the bridge only | |

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
