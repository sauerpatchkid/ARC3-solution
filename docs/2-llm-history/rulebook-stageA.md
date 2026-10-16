# Rulebook step 1: the Stage A rerun

Rulebook (`llm-rulebook.md`) starts by finishing last semester's rule-finding test, Stage A. Its pass/fail
criteria were fixed on 2026-09-11, before any rule-writing run, and are not changed here
(`legacy/llm_track/rule_writer.py` docstring).
- **R1, better predictions.** On at least 2 of ft09, lp85 and ls20, the rule book's exact-prediction rate on
  the test moves beats both baselines ("nothing changes" and "memory"), with the 95% bootstrap CI of each
  difference above 0.
- **R2, rules exist.** Every game's rule book has at least one rule that predicts a change.
- **GO** if R1 and R2.

## Why it never finished

The 2026-09-11 overnight attempt loaded the model, started generating, and the log stops 90 seconds later with
no error. `legacy/llm_track/rules_overnight.sh` records the cause: the WSL VM was stopped (Windows sleep or
`wsl --shutdown`). It was not a code fault, so no code changes. This machine's Windows power plan
(checked 2026-10-04) never sleeps on AC power and sleeps after 10 minutes on battery, so the run must stay plugged
in.

## Setup, declared 2026-10-04 before the run starts

| Item | Value | vs. the 2026-09-11 attempt |
|---|---|---|
| Code | `legacy/llm_track/rule_writer.py` unchanged, run as `python -m legacy.llm_track.rule_writer` | same |
| Criteria | R1 and R2 above | same (frozen) |
| Data | `results/legacy_llm/rules/data` (ft09 and lp85: train level 1, test level 2; ls20: 8 runs train, 8 test) | same |
| Budget | 5 rounds × 8 candidates per action group, thinking on, 20,480-token answers, code-only retry | same |
| **Model** | **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** | **changed** |
| Output | `results/rulebook/stageA/` (the failed attempt stays in `results/legacy_llm/rules/stageA/`) | new folder |
| Launcher | `experiments/rulebook/stageA.sh` (start / status / stop; same pre-run checker self-test and GPU check as `rules_overnight.sh`, 30 h timeout) | new |

**About the model change:**
- Stage A's docstring says the model "is chosen before the run". It was the Qwen3.5-35B-A3B, which was deleted on
  2026-10-03 when Matt chose Qwen3.8-27B as the project's only local model.
- The test set has never been scored by any model, so choosing the model now, before the run, keeps the test
  clean.
- Expect it to be slower than the 35B (about 3× on the Coach calls), so the run should take most of a day rather
  than about 4 hours.
- A dense Qwen3.6-27B found the same ft09 rules as the 35B in the 2026-09-11 smoke tests, at 3.6× the time.

## Results (2026-10-04, 09:49–16:28, 6 h 38 min) — NO-GO

Full report: `results/rulebook/stageA/report.md`.

**Run:** Qwen3.8-27B, thinking on, 5 rounds × 8 candidates over 23 action groups. It wrote 1,093 candidates; 728
were checked, 503 accepted, and 105 of the accepted ones predict a change. Round times rose from 62 to 89 minutes
as the feedback grew.

**The test**, scored once (exact-prediction rate on unseen moves, 95% CIs bootstrapped over runs):

| game | test | rule book | nothing | memory | beats both? |
|---|---|---|---|---|---|
| ft09 | level 2 (a new level) | **0.657** (0.635–0.683) | 0.117 | 0.117 | **yes** (+0.51 to +0.57) |
| lp85 | level 2 (a new level) | 0.365 | 0.365 | 0.365 | no (tie) |
| ls20 | 8 new runs of the same level | **0.908** (0.898–0.918) | 0.254 | 0.919 (0.912–0.926) | no (−0.021 to −0.002 vs memory) |

- R1 better predictions: **FAIL** (1 of 3 games; 2 needed).
- R2 rules exist: **PASS** (every game has a rule that predicts a change).
- **NO-GO** by the criteria fixed on 2026-09-11.

**What the rules were, game by game:**
- **ft09: the mechanic transferred to a new level.**
  - "Clicking a blue block turns the whole block into the level's other main colour" is exact on all 596 training
    moves. On level 2, which the model never saw, it applied to 2,221 moves and was right on **98.9%**.
  - The second rule (red → blue) never applies on level 2.
- **lp85: one rule, and it didn't transfer.**
  - It found one exact rule after 4 rounds: the right green plus rotates the ring of tiles clockwise (100% on 386
    training moves).
  - The rule is tied to level 1's layout, so it applies to **0** moves on level 2.
  - The other changing groups, such as the red arrow and the left/right edges, never produced an accepted rule.
    The model described the rotation correctly in words but never coded it exactly enough. The other 7 groups got
    only "does nothing" rules.
- **ls20: the rules are good and general; the test favours memory.**
  - All four movement rules ("ACTION4 slides the 5×5 token 5 columns right if the cells are all grey track") are
    95–98% right on **unseen** runs.
  - The book reaches 0.908 on the test, against 0.254 for "nothing changes".
  - It loses to "memory" by about one point, because ls20's test replays the same level. Memory has already seen
    almost every board there; Stage A's own notes called ls20 "the hard game" for this reason.

**What it means for Rulebook:**
- **The working part.** Qwen3.8-27B can write exact mechanics as code, checked against thousands of moves, and on
  simple mechanics they carry over: ft09 to a new level, ls20 to new runs.
- **The failing part.** A multi-step mechanic (lp85's ring rotation) wasn't coded exactly. The one exact rule it
  wrote was tied to positions and didn't transfer.
- **Not Rulebook's gate.** Rulebook's plan reports Stage A as it comes out (evidence for link L2, transfer). Its own
  first gate is Tier 0a, G1: rule coverage ≥ 80% on level 1 for ≥ 3 of the 8 dev games.
- **Time cost.** Stage A's full budget took 6.6 hours for 23 groups. Rulebook plans a much smaller per-run budget (k
  = 4, 2 rounds), but the 27B's speed is the main cost driver.
