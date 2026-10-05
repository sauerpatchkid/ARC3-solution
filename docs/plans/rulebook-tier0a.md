# Rulebook step 2: the offline rule test (Tier 0a)

Rulebook's first gate (`llm-rulebook.md` §6.4, G1). It asks one question before the remaining ~2,000 lines are
built: **can the local LLM write exact rules for most of what happens in a level, from one run's recordings?**
No agent changes: recorded mb_gated_att runs only.

## What it does

For each of the 8 dev games, from one recorded run:
1. **Levels.**
   - The *training level* is the run's first level with at least 500 recorded moves.
   - The *transfer level* is the next level after it with at least 50 distinct cases.
2. **Evidence** (`custom_agents/wm/evidence.py`). The training level's moves are indexed.
   - A case is a distinct (screen with decorations blanked, action).
   - Each case keeps what happened and how often.
   - A case that led to two different results is flagged as a conflict, never silently dropped.
3. **Rule writing** (`wm/rules.py`, `wm/llm.py`).
   - Actions are grouped: each button, and clicks by the colour clicked.
   - For each group the LLM sees the level's first board and up to 6 recorded moves, as before/after pictures
     plus the exact changed cells as text. This is Stage A's format.
   - It writes candidate rules as code.
4. **Checking** (`wm/check.py`, `wm/sandbox.py`). Every candidate runs in a sandboxed child process on every
   case of the level.
   - *Admitted:* applies to at least 20 moves, at least 95% right, and predicts real changes.
   - *Plan-eligible:* admitted, right on every case it applies to, and none of those cases is a conflict.
   - *Known no-op:* correctly says "nothing changes" on at least 20 moves.
5. **The rule book.** Per group, the best plan-eligible rule plus the best known no-op.
   - A case no rule covers, or where two rules disagree, is UNKNOWN.
   - UNKNOWN is never treated as "nothing changes".
6. **Scores.**
   - *Coverage:* the share of the training level's distinct screen-changing cases the book predicts exactly.
     UNKNOWN and wrong both count as misses.
   - *Transfer:* the same book, unrepaired, on the transfer level, beside "nothing changes" and a memory baseline.
   - *Conflicts:* how many cases have more than one recorded result.

## Pre-registered, 2026-10-05, before any Tier 0a LLM answer

| Item | Value |
|---|---|
| Games | tu93, tr87, dc22, g50t, vc33, ft09, m0r0, cd82 |
| Runs | seed 0 of `results/confirm_upgrade/20260927_221915` (mb_gated_att, 100k moves) |
| Model | `cyankiwi/Qwen3.8-27B-AWQ-INT4`, thinking on, 20,480-token answers, one code-only retry, one repair attempt |
| Budget | k = 4 candidates per group; round 2 (with feedback) only for groups with nothing usable; at most 12 groups per game |
| Check set | every case of the level, capped at 40,000 by a uniform sample |
| **G1** | **PASS when coverage is ≥ 80% on at least 3 of the 8 games** |
| Reported beside it | coverage of the looser 95% book, conflicts, transfer |
| If G1 fails | one day to try other settings on these dev games, then narrow the build to the games and groups that pass |

**Two clarifications made before any LLM answer.** Both were made after seeing only the recordings' sizes and what
the decoration mask flagged.

1. **Which level.** The plan's G1 says "level 1".
   - Moves spent on level 1 in these runs: cd82 9, ft09 665, tu93 761, m0r0 2,672, vc33 3,572, g50t 5,498,
     dc22 6,065, tr87 95,107.
   - Nine moves can't support any rule, and the plan itself writes rules only once enough evidence has accumulated.
   - So the training level is the first with at least 500 moves: level 2 for cd82, level 1 for the other seven.
2. **Mask guard.** Ticker cells are accepted only within 4 cells of a screen edge, which is mb_gated_att's
   bar-detector rule.
   - Without it, Stage A's detector masked a small sliding object as decoration in a unit test.
   - The existing notes record the same false positive on tr87's playfield.

## Evidence at a glance (no LLM)

| game | training level | moves | distinct cases | changing cases | conflicts | action groups |
|---|---:|---:|---:|---:|---:|---:|
| tu93 | 1 | 760 | 124 | 78 | 11 | 4 |
| tr87 | 1 | 95,106 | 68,097 (40,000 checked) | all | 0 | 4 |
| dc22 | 1 | 6,064 | 1,069 | 347 | 0 | 9 |
| g50t | 1 | 5,497 | 571 | 371 | 93 | 5 |
| vc33 | 1 | 3,571 | 2,890 | 67 | 0 | 5 |
| ft09 | 1 | 664 | 581 | 91 | 0 | 4 |
| m0r0 | 1 | 2,671 | 1,385 | 991 | 13 | 8 |
| cd82 | 2 | *(filled in by the run)* | | | | |

- **tr87:** every button moves a selection marker, wrapping around the screen, so every move changes the board.
- **g50t:** 16% of its cases are conflicts, which points to state the screen doesn't show. Rules touching those
  cases can be admitted but not plan-eligible.

## Smoke test on ls20 (not a dev game) and one fix, before the dev run

`results/rulebook/tier0a_smoke/report.md`, 2026-10-05 16:05–16:40. Its purpose was to prove the pipeline with the
real model: pictures, 4 answers per request, retries, checks, cache and report. It ran clean. It also showed:

- **A port bug, fixed before the dev run.**
  - Of 16 round-1 answers, 13 ran out of their 20,480 tokens while thinking.
  - The code-only retry then showed the model an *empty* earlier answer, because the server returns thinking
    separately from the answer. Stage A's in-process run showed the retry its own reasoning.
  - `rules.tail()` now keeps the reasoning of a truncated answer, with a test. Nothing else changed.
- **No trusted rule on ls20.**
  - The best rule, for ACTION1, was right on 97% of its moves: admitted but not plan-eligible. ls20 also has 71
    conflicting cases.
  - Strict coverage was 0%; the 95% book covered 21%.
  - Stage A's ls20 rules were also 95–98%, never 100%, so this looks like the game (timer effects), not the
    pipeline. It is the risk G1 exists to measure.
- **Speed.** 35 minutes for 4 groups over two rounds; about 11k tokens per answer; 110–290 tokens/s, limited by
  the 97,621-token KV cache. The 8 dev games (about 47 groups) should take roughly 6–7 hours.

## Running it

The dev run is prepared but **not started**. It was launched once on 2026-10-05 and stopped about two minutes in
at Matt's request, before any answer was saved, so it will start clean.

```bash
experiments/rulebook/tier0a.sh start     # detached; starts the LLM server, runs the 8 games, stops the server
experiments/rulebook/tier0a.sh status    # progress
experiments/rulebook/tier0a.sh stop      # stop it and free the GPU
```

- **Time:** about 6-7 hours.
- **GPU:** the server takes 90% of the card (~30 GB), so GPU-heavy Windows apps must be closed.
- **Power:** the PC must stay on AC power; it sleeps on battery.
- **Resuming:** answers are cached in `results/rulebook/tier0a/llm_cache.jsonl`, so a stopped run resumes where it
  left off.

## Results

*(filled in when the run finishes)*
