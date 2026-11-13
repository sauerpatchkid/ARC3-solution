# Time to level: what the guide changes (10 Nov 2026, exploratory)

**In one line.** Counted at 100,000 actions, the guided agent's gain is inside
the noise (G4). Looked at level by level, it finishes levels after the first in
fewer actions about twice as often as the unguided agent: **36 levels against
17 in G4**, where two unguided arms split 24 against 27. Level 1, where nothing
can be carried, splits evenly. The LLM-free arm does as well as the guided one.

**Status: exploratory.** No rule for these numbers was registered. The measure
was chosen on 10 Nov, after every run, and changed once that day (see "Limits").
Nothing here changes a verdict.

**Added 13 Nov: it did not repeat.** The LLM-free carry was then tested by
itself, registered, on new seeds
([`carry-results.md`](../1-goose-upgrades/carry-results.md)). On six click
games the carry arm finished 23 later levels in fewer actions and the plain
Goose 22 (interval [-15, +18]). Level 1 alone, with two identical agents, split
13 against 7. Read this page's level races as a lead that a registered test did
not confirm.

## Why look

Every verdict counts levels at the end of a run. The guide acts at the start of
each new level: it carries colours that did nothing on the level before, and
goal hints. So its effect should be on *when* a level is finished. That had
never been scored (the 10 Nov review, List 2 item 3a).

## What was measured

Finished runs only, read from each run's `metrics.json` (the action at which
each level was completed). Tool: `tools/levels_by_budget.py`. Every table and
the commands: `results/glassbox/time_to_level/` (`run.sh`).

1. **Levels by budget**: levels completed by 2,000, 10,000, 30,000 and 100,000
   actions.
2. **AULC**: the average completed level, weighted evenly in log action budget.
3. **Level races.** For each level that both runs of a pair started, which run
   finished it in fewer actions, counted from the level-up before it. If only
   one run finished the level, it wins when it took fewer actions than the
   other spent on the level without finishing; otherwise the race is undecided.

Three checks are built in:

- **A noise reference:** the same table for two arms of the same agent with no
  guide (A0's 27 Sep runs and the A0 run beside GB).
- **Level 1:** nothing is carried into a game's first level, so the guide
  cannot help there.
- **Games where the guide changed nothing** (tu93 above all: 0.0% of moves).

## Result 1: by budget, nothing shows early

G4, 25 games × seeds 0-2. Levels completed, summed over 75 runs:

| By | A0 27 Sep | Same-day A0 | GB | GB − same-day A0 (better / same / worse) | Same-day A0 − A0 27 Sep |
|---|---:|---:|---:|---|---|
| 2,000 actions | 25 | 27 | 28 | +1 (6 / 63 / 6) | +2 |
| 10,000 actions | 62 | 61 | 63 | +2 (11 / 51 / 13) | −1 |
| 30,000 actions | 78 | 79 | 85 | +6 (7 / 65 / 3) | +1 |
| 100,000 actions | 112 | 101 | 110 | +9 (12 / 57 / 6) | −11 |

The three arms are level at 2,000 and 10,000 actions. GB's lead opens later.
But two unguided arms differ by 11 at 100,000, more than GB's lead, so this
table cannot show a guide effect. AULC says the same: GB +2.79 over the
same-day A0 (95% interval −0.88 to +7.22; 29 better, 17 same, 29 worse).

## Result 2: level by level, levels after the first come sooner

Levels 2 and up: which arm finished a level in fewer actions.

| Comparison | Levels (games) | New faster | Base faster | New − base (95% interval, games resampled) |
|---|---|---:|---:|---|
| **G4: GB against the same-day A0** | 53 (12) | **36** | **17** | +19 (+7 to +36) |
| **Dev sweep 2: GB against the same-day A0** | 39 (6) | **23** | **16** | +7 (+2 to +13) |
| *Noise, G4: same-day A0 against A0 27 Sep* | 51 (11) | 24 | 27 | −3 (−10 to +5) |
| *Noise, dev games: A0 of 19 Oct against A0 27 Sep* | 20 (6) | 11 | 8 | +3 (−4 to +9) |

Guided against unguided on the same day, pooled: **59 against 33**. Unguided
against unguided, pooled: 35 against 35.

The two placebos behave as placebos:

| Where the guide cannot act | New faster | Base faster |
|---|---:|---:|
| Level 1, G4 (GB against the same-day A0) | 23 | 28 |
| Level 1, dev sweep 2 | 14 | 16 |
| tu93, levels 2 and up, G4 (guide changed 0.0% of moves) | 5 | 4 |
| tu93, levels 2 and up, dev sweep 2 | 6 | 6 |

Where both runs finished a level after the first, GB's median was 0.76 of the
same-day A0's actions in G4 (40 levels) and 0.78 in dev sweep 2 (31 levels).

## Result 3: where it sits, and whose it is

G4, GB against the same-day A0, levels 2 and up:

| Group | Games with a race | GB faster | A0 faster |
|---|---:|---:|---:|
| Guide changed ≥ 5% of moves (11 games) | 8 | 18 | 9 |
| Guide changed 0.5 to 5% (5 games) | 2 | 12 | 3 |
| Guide changed < 0.5% (9 games) | 2 | 6 | 5 |
| Dev tier (tuned) | 6 | 19 | 11 |
| Seen tier (ls20, lp85) | 2 | 11 | 3 |
| Held-out tiers (15 games) | 4 | 6 | 3 |

- **lp85 carries a large share:** 10 against 2 on its 12 levels (5 against 6
  in the noise reference). Without lp85 G4 is 26 against 15.
- **Few held-out games get past level 1**, so the held-out tiers have only 9
  races (6 against 3).
- **It is the LLM-free part.** In G2 the arm with no LLM rules anywhere
  (GB-free) won 14 races against A0's 7, the same as the guided arm (14
  against 7). Guided against LLM-free was 7 against 13. G2's A0 runs are from
  27 Sep, not the same day, so these three are weaker than the rows above.

## Limits

- **Exploratory, and the measure was changed once.** The first version counted
  only levels both runs finished. That favours whichever arm finished fewer
  levels, because its slow levels drop out, so it was replaced by the race
  rule above. Both versions point the same way.
- **Few games decide it.** Twelve games in G4 have a level after the first
  that both runs started. The intervals resample those games and are rough.
- **Dev sweep 2 alone is not clear of chance:** 23 of 39 (59%) against the dev
  games' noise reference of 11 of 19 (58%), which is small. G4 carries the
  result (68% against 47%).
- **It is about speed, not levels.** The level count at 100,000 actions is
  still inside the noise, as G4's write-up says.

## What it means

1. **For the report.** State G4 as registered (ADOPT, within noise). Add this
   as an exploratory finding: the guide's LLM-free carry makes levels after the
   first come sooner.
2. **For what to build.** It supports moving the LLM-free carry (colour no-ops,
   goal hints) into Goose itself: the review's List 1, item 2, which is not on
   this week's list.
3. **For later tests.** Level races see an effect that the level count misses,
   at no extra cost. A test of a change that acts within levels should register
   them beside the level count, before the run.
