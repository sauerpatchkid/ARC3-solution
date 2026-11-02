# A0 run again: the noise floor — exploratory (run 19 Oct, written 2 Nov 2026)

*Not pre-registered; it changes no verdict. G2 was registered against A0's 27 Sep
runs (prereg 5b) and stays a FAIL. This is the follow-up Matt chose on 19 Oct
("measure the noise", option (b) of `g2-online.md`).*

## In one line

**The adopted Goose, run again on G2's 16 dev runs with nothing changed,
completed 33 levels where its 27 Sep runs completed 37 (paired 1 better, 12 the
same, 3 worse; dc22 worse on both seeds).** Both of G2's arms are above it: GB
34 (2 / 12 / 2), GB-free 35 (3 / 9 / 4). G2's −3 against the old runs is the
size of A0's own run-to-run spread, so G2 shows no cost from S5, and no gain.

## The run

- `make gb-a0-fresh` (`tools/gb_sweep.py --run a0_fresh --arms A0-fresh --games
  dev --seeds 0,1`): `mb_gated_att` with `EVAL_GB` unset, 8 dev games × seeds
  0-1 × 100,000 actions, 4 at a time on the GPU. Mon 19 Oct 16:47 to 19:48 (3 h
  1 min), ~40 act/s per run (A0's 27 Sep runs: ~35). Output:
  `results/glassbox/a0_fresh/online/` (manifest, logs, the comparisons below).
- A PC restart stopped the first attempt (started 15:42) at ~27,000 actions
  into its first four runs; all 16 were rerun from the start with `--resume`.
  The four partial run folders stay where they are and are in no manifest.
- **The agent is the same.** Every change to the agent's code since 27 Sep was
  checked move for move on the CPU (`tools/cpu_check.py`: the 6 Oct recording
  fix, the 18 Oct `# [gb]` hook with `EVAL_GB` unset). The one scoring change
  (3 Oct: the engine's own level count) can add a level finished on a run's
  final move, never remove one. All 16 runs record commit `2eabedd`; 12 of them
  also record uncommitted edits to `gb/` stage files, their tests and
  `tools/gb_offline.py` (made during the sweep), none of which the agent imports
  with `EVAL_GB` unset (`action.py` was unmodified in every run).

## Per game

Levels completed per seed. A0 27 Sep has seed 2 as well (not in G2).

| Game | A0, 27 Sep (s0, s1, s2) | A0 fresh (s0, s1) | GB (G2) | GB-free (G2) |
|---|---|---|---|---|
| tu93 | 4, 4, 4 | 4, 4 | 4, 4 | 4, 4 |
| tr87 | 1, 1, 1 | 1, **0** | 1, 0 | 0, 1 |
| dc22 | 3, 3, **1** | **1, 1** | 1, 3 | 4, 3 |
| g50t | 1, 1, 1 | 1, 1 | 1, 1 | 1, 1 |
| vc33 | 4, 4, 4 | 4, 4 | 4, 5 | 4, 4 |
| ft09 | 2, 2, 2 | 2, 2 | 2, 2 | 2, 2 |
| m0r0 | 2, 2, 2 | 2, 2 | 2, 1 | 2, 1 |
| cd82 | 2, 1, 1 | 2, 2 | 2, 1 | 1, 1 |
| **Total, s0-1** | **37** | **33** | **34** | **35** |
| Per seed (8 games) | 19, 18, 16 | 17, 16 | 17, 17 | 18, 17 |

`tools/paired_compare.py --expect 16` (files in `a0_fresh/online/`):

| New | Base | Levels | Better / same / worse | Worse on both seeds | Dev rule (G2's) |
|---|---|---|---|---|---|
| A0 fresh | A0 27 Sep | 33 vs 37 | 1 / 12 / 3 | dc22 | FAIL (`a0_fresh_vs_a0.md`) |
| GB | A0 fresh | 34 vs 33 | 2 / 12 / 2 | none | PASS (`gb_vs_a0_fresh.md`) |
| GB-free | A0 fresh | 35 vs 33 | 3 / 9 / 4 | cd82 | FAIL (`gb_free_vs_a0_fresh.md`) |

## What it shows

**1. A0 against itself moves by as much as G2 did.** The same agent lost 4
levels on the same 16 runs (bootstrap interval over games −13 to +2). Under the
rule G2 was judged by, A0 itself would have failed against A0. The nine
8-game seed totals in the table, guided or not, all lie between 16 and 19.

**2. How big the noise is.** The five unguided seed totals (19, 18, 16, 17, 16)
have a standard deviation of about 1.3 levels (five values, so rough). With 2
seeds per arm, the gap between two arms' 16-run totals then varies by about
±2.6 levels (one standard deviation) from the agent alone: G2's −3 and this run's
−4 are 1.2 and 1.5 of those. Telling a real effect from noise at two
standard deviations would take about 4 seeds per arm for +2 levels per seed (+4
per 16 runs), and about 14 for +1 level per seed. (GPU runs do not repeat past the
first few hundred actions, so pairing by seed narrows nothing.)

**3. Most of the spread is two games.** dc22 (1 to 4 levels) and cd82 (1 to 2)
account for most of the differences. dc22's level 2 stopped 4 of the 9 runs,
3 of the 5 with no guide (A0 27 Sep s2, both fresh seeds): the stall GB s0 hit
in G2 is common without any guide (`dc22-noop-check.md`).

**4. Drift since 27 Sep cannot be told from noise.** The fresh seeds (17, 16)
are below 27 Sep's seeds 0-1 (19, 18) but level with its seed 2 (16).

## What it means

- **G2's reading changes, its verdict does not.** G2 fails as registered; read
  against a same-day A0 it would pass its rule (GB 34 vs 33, 2 / 12 / 2). Neither
  shows that S5 v1 changes levels: GB, GB-free and A0 are all within each
  other's noise on these 16 runs.
- **Dev sweep 2 (11-13 Nov) needs a design decision before it is registered**
  (Matt's): a same-day A0 arm beside GB, and more seeds than 2 if it is to show
  anything short of a large effect. Each extra seed is 8 runs per arm, about
  1.5 h on the GPU for an arm with no online LLM.
