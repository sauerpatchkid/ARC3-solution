# G4, the confirm: GB against the same-day A0 on all 25 games — result (8 Nov 2026)

*Judged by pre-registration section 7 (`glassbox-prereg.md`, registered and
pushed in `11d76cb` before the run). Arms: A0-fresh, the adopted Goose
(`mb_gated_att`) with no guide, and GB, the same agent guided by S5 VERSION 2
(`colour_judge = "level_end"`, `mask_bars = true`) with the book `pool_g0b_g1`.
Config `8c8e1cb023c827d2`, code unchanged since `9b1ac79`, 25 games × seeds 0-2
× 100,000 actions, 150 runs. S5 v2 and the book were built and tuned on the 8
dev games, so the dev tier is labelled as tuned. The held-out tiers' per-game
rows were sealed until the verdict and opened right after it (8 Nov, logged in
`artifacts/registry.json`), as Matt asked on 7 Nov (plan §7.4 item 19).*

## In one line

**ADOPT: Goose guided by S5 v2 completed 110 levels on the 75 runs, against
101 for the adopted Goose run beside it (paired 12 better, 57 the same, 6
worse); no game was worse on all three seeds.** As the pre-registration said
it would be, this is weak evidence. +9 levels is about one standard deviation
of the gap two unguided arms show by chance, and the rule passes 36% of the
time with no effect at all. The same-day A0 came in 11 levels below A0's own
27 Sep runs (112); GB is level with those (110 vs 112). The gain sits on the
games where the guide changed the most moves. Outside the 8 dev games GB used
no LLM output (the book has no rules there), so the +5 of the +9 found there
cannot come from the LLM. It sits on the games where S5's goal hints, an
LLM-free part of S5, changed the most moves.

## The run

- **Command and times.** `make gb-g4 BOOK=pool_g0b_g1 G4=1` ran from Sat 7 Nov
  09:19 to Sun 8 Nov 14:59: 150 runs (A0-fresh and GB × 25 games × seeds 0-2 ×
  100,000 actions), 4 at a time on the GPU, in the order seed, game, arm.
  Log: `results/glassbox/g4/g4_log.txt`. Runs, manifest, verdicts and the
  opened sealed files: `results/glassbox/g4/online/`.
- **Pause.** The sweep was paused 08:29-11:25 on 8 Nov at Matt's request with
  `make gb-pause`, asked at 08:09. The 4 runs in progress finished first, so
  none was cut short; 130 of 150 were done. It resumed with `RESUME=1` after
  these checks: the code unchanged since `9b1ac79`, HEAD still `11d76cb`, the
  config hash `8c8e1cb023c827d2`, the PC on AC power, the GPU idle (prereg §11
  entry 9).
- **Running time:** 26 h 44 min (registered estimate 27-30 h). A run took
  26-57 min (median 43).
- **Valid.**
  - 150 of 150 runs ended at the 100,000-action cap (`run_end.json`), and the
    run-set check (`--expect 75`) passed.
  - No run log has an error, and no rules process was switched off.
  - The PC neither slept nor rebooted (up since 7 Nov 09:04).
  - Every run records commit `11d76cb` (the §7 registration), `+dirty` only
    because of the `ARC-AGI-3-Agents` submodule's local edits.
  - Every GB run records S5 VERSION 2, `level_end`, `mask_bars` and the book
    `pool_g0b_g1`. No A0-fresh run records a guide.
  - The numbers below come from `results/glassbox/g4/analysis/g4_numbers.py`
    (its output, `g4_numbers.txt`, is beside it).
- **Throughput:** A0-fresh 40.0 act/s, GB 39.8 act/s (4 runs at once). The
  guide costs no measurable speed.

## By tier

| Tier | Games | A0, 27 Sep | A0-fresh | GB | GB vs A0-fresh: W / T / L |
|---|---|---|---|---|---|
| Dev (tuned) | 8 | 53 | 50 | **54** | 5 / 18 / 1 |
| Seen | 2 | 20 | 19 | **20** | 1 / 4 / 1 |
| A/B-offline | 8 | 25 | 19 | **22** | 4 / 17 / 3 |
| Untouched (the clean held-out tier) | 7 | 14 | 13 | **14** | 2 / 18 / 1 |
| **All** | **25** | **112** | **101** | **110** | **12 / 57 / 6** |

**Verdict** (`tools/paired_compare.py --rule confirm --expect 75`, section 7's
rule; `verdict_gb_vs_a0_fresh.md`):

- levels 110 > 101 and wins 12 > losses 6: PASS;
- no game worse on every seed: PASS;
- so **ADOPT**.

The bootstrap interval for the level gap, resampling games, is −1 to +19
(descriptive, not part of the rule).

Reported beside the verdict, not gating:

- **GB against A0's 27 Sep runs** (`gb_vs_a0.md`): 110 vs 112, 10 / 53 / 12;
  bootstrap −11 to +7. Under the confirm rule this comparison would say DO NOT
  ADOPT.
- **A0-fresh against A0's 27 Sep runs** (`a0_fresh_vs_a0.md`, the drift since
  27 Sep): 101 vs 112, 4 / 57 / 14; bootstrap −21 to −2. A0-fresh is below on
  every tier, most on A/B-offline (19 vs 25).
- **Games with at least one level** (of 25): A0 27 Sep 22, A0-fresh 21, GB 20.
  Runs with at least one level (of 75): 61, 56, 57. The guide changed 0.0% of
  moves on every game GB scored nothing on (sc25, tn36, ka59, sb26, wa30), so
  these differences are chance.
- **Levels of 2 and above** (each run's levels past its first): A0 27 Sep 51,
  A0-fresh 45, GB 53.

## Per game

Each cell lists the levels completed on seeds 0, 1 and 2. In a pair that
differs, the higher of A0-fresh and GB is in bold. The last three columns
describe GB's runs:

- **Guide changed:** the share of GB's moves the guide replaced (a covered
  no-op drawn again, or a hint boost).
- **Covered:** the share of Goose's first draws that a carried no-op covered.
- **Colours:** colours carried as no-ops / colours dropped (a covered click
  changed a live cell).

The book's no-op rules exist for 6 of the dev games (none for tu93 and tr87)
and for no other game.

| Game | A0, 27 Sep | A0-fresh | GB | W / T / L | Guide changed | Covered | Colours |
|---|---|---|---|---|---|---|---|
| *Dev (tuned)* | | | | | | | |
| tu93 | 4, 4, 4 | 4, 4, 4 | 4, 4, 4 | 0 / 3 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| tr87 | 1, 1, 1 | 0, 1, 1 | **1**, 1, 1 | 1 / 2 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| dc22 | 3, 3, 1 | **3**, 3, 1 | 1, 3, 1 | 0 / 2 / 1 | 2.9, 23.0, 2.7% | 2.3-3.7% | 12 / 0 |
| g50t | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 0 / 3 / 0 | 1.3, 1.1, 1.6% | 1.3-2.0% | 0 / 0 |
| vc33 | 4, 4, 4 | 2, 4, 4 | **4**, 4, 4 | 1 / 2 / 0 | 20.8, 24.3, 19.8% | 21.9-26.9% | 21 / 0 |
| ft09 | 2, 2, 2 | 2, 2, 2 | 2, **3**, 2 | 1 / 2 / 0 | 5.4, 16.3, 5.0% | 5.3-5.9% | 10 / 0 |
| m0r0 | 2, 2, 2 | 2, 1, 2 | 2, **2**, 2 | 1 / 2 / 0 | 1.8, 2.1, 1.4% | 1.6-2.3% | 16 / 7 |
| cd82 | 2, 1, 1 | 2, 2, 1 | 2, 2, **2** | 1 / 2 / 0 | 34.2, 19.3, 11.4% | 3.2-4.1% | 7 / 1 |
| **Dev** | **53** | **50** | **54** | **5 / 18 / 1** | | | 66 / 8 |
| *Seen* | | | | | | | |
| ls20 | 2, 2, 2 | 1, **2**, 1 | **3**, 1, 1 | 1 / 1 / 1 | 0.0% on every seed | 0.0% | 0 / 0 |
| lp85 | 5, 7, 2 | 5, 5, 5 | 5, 5, 5 | 0 / 3 / 0 | 2.7, 2.2, 2.9% | 1.2-1.3% | 10 / 0 |
| **Seen** | **20** | **19** | **20** | **1 / 4 / 1** | | | 10 / 0 |
| *A/B-offline* | | | | | | | |
| ar25 | 2, 2, 2 | 2, 2, 2 | 2, 2, 2 | 0 / 3 / 0 | 35.0, 35.4, 36.1% | 0.2-0.3% | 3 / 3 |
| lf52 | 1, 1, 0 | 1, 0, 0 | 1, 0, **1** | 1 / 2 / 0 | 22.1, 0.0, 12.7% | 0.0% | 1 / 0 |
| re86 | 1, 1, 0 | **1**, 0, 0 | 0, 0, **1** | 1 / 1 / 1 | 0.0% on every seed | 0.0% | 0 / 0 |
| s5i5 | 2, 1, 3 | 1, 1, 1 | 1, 1, **4** | 1 / 2 / 0 | 50.5, 52.9, 16.8% | 4.9-9.3% | 8 / 0 |
| sc25 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0 / 3 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| sp80 | 2, 1, 2 | **2**, **2**, 1 | 1, 1, **2** | 1 / 0 / 2 | 34.4, 34.0, 20.8% | 0.0-0.6% | 3 / 0 |
| su15 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 0 / 3 / 0 | 1.2, 1.3, 1.2% | 0.2-0.3% | 6 / 3 |
| tn36 | 1, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0 / 3 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| **A/B-offline** | **25** | **19** | **22** | **4 / 17 / 3** | | | 21 / 6 |
| *Untouched* | | | | | | | |
| bp35 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 0 / 3 / 0 | 23.9, 21.3, 22.4% | 0.2-0.5% | 6 / 0 |
| cn04 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 0 / 3 / 0 | 0.8% on every seed | 0.0% | 3 / 0 |
| ka59 | 0, 1, 1 | 0, 0, 0 | 0, 0, 0 | 0 / 3 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| r11l | 1, 1, 1 | 1, 1, 1 | **2**, **2**, 1 | 2 / 1 / 0 | 48.5, 48.7, 49.4% | 0.0-0.2% | 3 / 1 |
| sb26 | 0, 0, 0 | **1**, 0, 0 | 0, 0, 0 | 0 / 2 / 1 | 0.0% on every seed | 0.0% | 0 / 0 |
| sk48 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 0 / 3 / 0 | 26.9, 35.8, 33.5% | 0.0-0.2% | 5 / 0 |
| wa30 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0 / 3 / 0 | 0.0% on every seed | 0.0% | 0 / 0 |
| **Untouched** | **14** | **13** | **14** | **2 / 18 / 1** | | | 17 / 1 |
| **All 25** | **112** | **101** | **110** | **12 / 57 / 6** | 11.6% of all moves | | 114 / 15 |

## How big +9 is

- **The noise.** Prereg §7 measured it before the run by resampling each
  game's unguided runs (`results/glassbox/g4_noise/`). The gap between two
  75-run totals has a standard deviation of about 8 levels and stays within ±11
  levels 90% of the time. The confirm rule passes 36% of the time with no
  effect anywhere. +9 is about 1.1 standard deviations: it passes the
  registered rule, but it is inside the band chance alone produces.
- **The comparator came in low.**
  - A0-fresh is the same agent as A0's 27 Sep runs, yet it completed 11 fewer
    levels (4 / 57 / 14). Every change to the agent's code between 27 Sep and
    19 Oct was checked move for move on the CPU (`a0-fresh.md`). No commit has
    touched the agent's code since (the guide is imported only under
    `EVAL_GB=1`).
  - −11 is at the edge of the same ±11 band, so chance can explain it: a gap
    that size happens about 1 time in 10. So could something outside the code,
    such as the GPU's kernels (runs diverge after ~300 actions) or the engine.
  - The unguided runs of 19 Oct were also below 27 Sep's on the same 16 dev
    runs (33 vs 37); G4's A0-fresh made 34 on those 16.
  - These runs cannot tell whether 27 Sep was a good day or the unguided agent
    now does worse. Either way, the registered comparison is the same-day one,
    and GB ties the older one (110 vs 112).
- **Where the gap is** (descriptive; this split was not registered). The
  games, grouped by how much the guide changed their moves (mean over the 3
  seeds):

  | The guide changed | Games | Gap | W / T / L |
  |---|---|---|---|
  | nothing (< 0.5% of moves) | 9: tu93, tr87, ls20, re86, sc25, tn36, ka59, sb26, wa30 | +1 | 3 / 21 / 3 |
  | a little (0.5-5%) | 5: g50t, m0r0, lp85, su15, cn04 | +1 | 1 / 14 / 0 |
  | a lot (≥ 5%) | 11: dc22, vc33, ft09, cd82, ar25, lf52, s5i5, sp80, bp35, r11l, sk48 | +7 | 8 / 22 / 3 |

  The first row is a placebo: the guide could not have changed the outcome on
  those 9 games. Even so, 6 of their 27 pairs differ, 3 each way; that is the
  noise, seen directly. The gain sits where the guide acted, as a real effect
  would. Still, 8 wins against 3 losses in the last row is within chance: a
  fair coin gives 8 or more of 11 about 11% of the time.
- **Dev sweep 2's confound still applies within a game.** Colours and hints are
  set at a level-up and act on the levels after it, so runs that reach more
  levels give the guide more to change:
  - dc22: 23.0% on the seed where GB reached 3 levels, 2.7-2.9% on the seeds
    where it reached 1;
  - ft09: 16.3% on the seed where it reached 3.

  Grouping by game rather than by run limits this, because whether the guide
  acts at all depends mostly on whether the game has clicks.
- **dc22 s0 lost again:** GB 1 level, A0-fresh 3, as on seed 0 in G2 (GB 1,
  GB-free 4, A0 3). On the GPU a seed is not a replay. Over dev sweep 2 and G4,
  dc22 totals 14 levels in each arm (7 runs each).

## What the guide did outside the dev tier

- **No LLM output there.** The book has no rules for the 17 games outside the
  dev tier, so on them GB is colour no-ops and goal hints only: the same as
  GB-free.
- **Mostly hints.** On 7 of the 15 held-out games (ar25, lf52, s5i5, sp80,
  bp35, r11l, sk48) the guide changed 12-53% of moves on most seeds. Yet
  carried no-ops covered at most 0.6% of Goose's first draws there (s5i5
  4.9-9.3%). So those changes are goal hints: once a level is won, clicks on
  the two colours that vanished or shrank most during it are drawn twice as
  often on the levels after.
- **Results on those 7 games:** +5 (5 / 14 / 2).
  - s5i5 +3 (one seed: 4 levels vs 1);
  - r11l +2;
  - lf52 +1;
  - sp80 −1;
  - ar25, bp35 and sk48 0.
- **Results on the other 8 held-out games**, where the guide changed at most
  1.3% of moves: −1 (1 / 21 / 2).
- **Against dev sweep 2's guess.** It guessed that a held-out gain would come
  from pruning dead clicks on click games. The gain did come from click games,
  but through hints rather than pruning.

## Against the expectations (section 7)

| Expectation, written before running | Result |
|---|---|
| GB within noise of A0-fresh | +9, about 1.1 standard deviations: within noise |
| The rule passes 36% by chance, 54% if dev sweep 2's gain is real; adoption about 45% | passed |
| Read with the tier totals, and with whether the dev tier repeats dev sweep 2 | dev +4 (5 / 18 / 1) after dev sweep 2's +5 (6 / 24 / 2) on seeds 3-6: the same direction. vc33 and cd82 gained again; dc22, dev sweep 2's biggest gain, lost 2. Every tier ≥ 0 (seen +1, A/B-offline +3, untouched +1) |
| lp85 is noisy (±8 levels by chance) | 5, 5, 5 in both arms (27 Sep: 5, 7, 2) |

## S5 online

- **Moves:** the guide saw 7.39 million moves in GB's 75 runs and changed
  856,510 of them (11.6%).
- **Colours carried as no-ops:** 114 (dev 66, seen 10, A/B-offline 21,
  untouched 17). 15 were dropped when a covered click changed a live cell
  (m0r0 7, ar25 3, su15 3, cd82 1, r11l 1).
- **The book's no-op rules** (on dc22, g50t, vc33, ft09, m0r0 and cd82): 60
  re-bindings, 184 "not re-bound yet" waits, 5 demotions (vc33 3, m0r0 2),
  0 rule processes switched off.
- **Goal hints:** 86 set; 33 dropped after 2,000 moves without effect.

## What it means

- **G4 passes, so GB is adopted.** The rule is the one the project set before
  running it (plan §6.3, prereg §7). What is adopted: S5 v2 with the book
  `pool_g0b_g1`, on `mb_gated_att`. By the preset rule (CLAUDE.md), an adopted
  configuration gets a new preset, `glassbox`. Matt skipped it for now (8
  Nov); the book lives under `results/`, outside git.
- **How to state it.** "On all 25 public games, the guided Goose completed 110
  levels against 101 for the adopted Goose run beside it, passing the
  pre-registered confirm (12 better, 57 the same, 6 worse). The gain is about
  one standard deviation of chance, and GB's total equals A0's earlier one
  (110 vs 112)." Not "Glassbox beats Goose by 9 levels" without the noise.
- **What it says about the LLM.**
  - The LLM's rules act only on the 6 dev games whose book has them, alongside
    colour no-ops and hints, and those games were tuned on. GB gained +3 there
    (4 / 13 / 1).
  - G4 has no GB-free arm to split the rules from the rest. In G2, GB and
    GB-free were level (34 vs 35); in the counterfactual, the LLM's no-op rules
    added at most 3.6 points of covered moves.
  - On the 17 other games GB used no LLM output, and gained +5.
  - So G4 is evidence for S5's checked, LLM-free guidance (colour no-ops, goal
    hints) more than for LLM-written rules.
- **Next** (plan §7.4, buffer to Tue 17 Nov):
  - replay G4's GB recordings on the dev games with the agent's mask, as for
    dev sweep 2;
  - the funnel table and S6 figures for the report.

  The report (final Mon 7 Dec) gives this result with its noise.
