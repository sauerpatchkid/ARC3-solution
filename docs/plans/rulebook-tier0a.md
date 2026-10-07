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
| cd82 | 2 | 29,520 | 4,281 | 2,324 | 0 | 11 |

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

## Results (2026-10-05 22:27 to 2026-10-06 03:30, 5 h) — G1 FAIL, 2 of 8

Full report: `results/rulebook/tier0a/report.md`; per game: `results/rulebook/tier0a/<game>.json`.

**Run.** The server ran at 87% of the card, because Windows apps held 3.6 GB. The launcher now sizes it to what is
free; this changes how many answers run at once, not the answers.
- LLM: 200 requests, 449 answers, 4.66 million tokens.
- Candidates: 446. Of these, 98 (22%) never reached code and 307 were checked.
- **17 were admitted (≥ 95% right) and 10 were plan-eligible (right on every case).**

**G1: coverage ≥ 80% on at least 3 of the 8 games → 2 games (vc33, ft09) → FAIL.**

| game | training level | **coverage** (trusted rules) | coverage of the looser 95% book | best rules | transfer level: book vs nothing / memory | wrong on transfer |
|---|---:|---:|---:|---|---|---:|
| tu93 | 1 | **0%** | 0% | 91–92% right (2 of 4 buttons) | 0.179 vs 0.179 / 0.179 | 0% |
| tr87 | 1 | **25%** | 25% | 1 of 4 buttons exact | 0.000 vs 0.000 / 0.000 | 0% |
| dc22 | 1 | **25%** | 42% | 3 of 4 buttons exact (narrow), 1 at 92% | **0.429 vs 0.277 / 0.277** | 0% |
| g50t | 1 | **0%** | 0% | 60–76% right | 0.201 vs 0.201 / 0.201 | 0% |
| vc33 | 1 | **100%** | 100% | the one changing group, exact | 0.835 vs 0.835 / 0.835 | 0% |
| ft09 | 1 | **100%** | 100% | both changing groups, exact | 0.046 vs 0.046 / 0.046 | **59%** |
| m0r0 | 1 | **0%** | 48% | 90–96% right (4 buttons) | 0.378 vs 0.378 / 0.378 | 0% |
| cd82 | 2 | **0%** | 0% | 0–45% right | 0.146 vs 0.146 / 0.146 | 0% |

**What the test found:**

1. **Exact rules come only for simple, one-step click mechanics.**
   - vc33 (blue buttons shift two boundaries) and ft09 (clicking a tile recolours it) reached 100%.
   - The 95% book would not change the verdict either: it also clears 80% on only those two games.
2. **On movement games the model is close but not exact.**
   - Its rules describe the mechanic correctly in words and are 90–97% right: tu93 92%, m0r0 96%, ls20 97% in the
     smoke test, dc22's fourth button 92%.
   - The misses are edge cases: walls, timers, and cases where the same screen gave two results (g50t has 93 such
     cases, tu93 has 11).
   - Under "100% or not trusted" that is 0% coverage.
3. **Mechanics with several moving parts were out of reach.**
   - cd82 (rotating and painting shapes): no rule above 45%.
   - tr87 (cycling glyphs): 0% on three of four buttons.
4. **Transfer is the weakest link.** Only dc22's rules carried over.
   - dc22: its three exact movement rules were right on all 517 level-2 cases they applied to, and the book beat
     both baselines.
   - ft09: the rule hard-coded level 1's colour ("turns it red") and was **wrong on 59%** of level 2's changing
     cases. Stage A's version, learned from many runs, said "the level's other main colour" and transferred at
     98.9%.
   - vc33 and tr87: the rules were tied to level 1's layout and applied to nothing on level 2.
   - Nowhere did memory beat "nothing changes", so the rule book is the only thing that transfers at all.
5. **Round 2 did most of the work.** Round 2 shows the model its mistakes.
   - It produced tr87's, vc33's and dc22's exact rules, and one of ft09's.
   - The plan's own setting stopped refining a group as soon as one rule was *admitted*. So m0r0's two 96% rules
     and dc22's 92% rule never got a second round.
   - 22% of answers ran out of their 20,480 tokens before writing code.

**What it means.** By the pre-registered gate, the full Rulebook build is not justified as designed.
- **What is established.**
  - A local 27B can write replay-exact rules for simple click mechanics from one run's recordings.
  - It gets movement mechanics roughly right.
  - Its exact rules, where it has them, make no wrong predictions on the level they were written for.
- **What is not established.**
  - Coverage of most of a level on most games.
  - Reliable transfer to the next level without repair.

The pre-registration allows one day of trying other settings on these dev games, then narrowing the build to the
games and groups that pass. The obvious setting to try is the one finding 5 points at: keep giving feedback rounds
until a rule is exact, not merely admitted. Any result from that is dev-tuned and must be labelled so.

## Changes to the code after this run (2026-10-06)

The results above were produced by the code at git tag `rulebook-v1` and stand as
run. Three things changed afterwards, none of which moves the G1 verdict:

- **A "known no-op" now needs conflict-free evidence**, as a plan-eligible rule
  always did. One rule in the books above is affected: g50t's ACTION5 no-op
  (184 moves, 2 keys with conflicting outcomes) would no longer be trusted.
  g50t's coverage was 0% either way.
- **The sandbox denies its forbidden names as attributes too**
  (`np.ma.core.builtins.open` used to pass the static check). None of the 344
  candidates that passed the check uses such an attribute.
- **The ticker scan uses the shared vectorised helper** (`gridtools.tick_cells`).
  All 16 evidence files of this run rebuild bit for bit.

The LLM client also retries a dropped connection, and one failed game no longer
ends the others (no report is written for an incomplete set).

