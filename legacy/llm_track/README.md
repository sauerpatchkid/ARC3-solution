# ARCHIVED 2026-09-15 — semester-1 LLM track, kept for the record

This folder is the LLM work from the end of semester 1 (Probe A, Probe C and
the rule-finding Stage A), moved from `llm_track/` to `legacy/llm_track/` at
the start of semester 2. It is frozen: nothing new goes here, nothing in the
baseline imports it, and the new plans (`docs/plans/plan-A-llm-advisor.md`,
`docs/plans/plan-B-goose-novelty.md`) start from the baseline, not from this
code. Its outputs were renamed from `results/llm/` to `results/legacy_llm/`.

Outcome, in one line each:
- **Probe A** (pairwise LLM judge, text and pictures, Qwen 2B-9B): NO-GO, chance on anchors.
- **Probe C** (Qwen-9B writes scoring heuristics for ft09): NO-GO, never beat a one-line rule.
- **Rule-finding Stage A** (Qwen-35B writes exact mechanics rules): the smoke
  test found ft09's tile-flip rule; the full overnight run died at model load
  (`results/legacy_llm/rules/stageA/run.log`) and was not restarted. No result.

Still reusable from here: `tickers.py` (component-level ticker detector, see
CLAUDE.md caveats), `corpus.py` (shard iteration), the vLLM serving notes
below, and the `.venv-llm` environment and cached Qwen models on this machine.
The paths in this README and Makefile were rewritten to the new location
(`make -C legacy/llm_track help`, `python -m legacy.llm_track.<module>`), so
the commands should still run, but they are not maintained.

---

# LLM track (`legacy/llm_track/`): Matt's LLM work, isolated from the baselines

This folder is self-contained: code, commands, serving environment, design docs
and results. Teammates working on the baselines can ignore it entirely, because
nothing outside it depends on it (see *Isolation* below).

- Design: `docs/295B-llm-track-plan.md` (option survey) and
  `docs/295B-llm-design-detailed.md` (component design)
- The transition-record contract: `SCHEMA.md`

## Status (2026-09-11)

Week 1 (serializer, corpus scan) is done. **Probe A, the go/no-go gate for the
design's backbone (P1: "the LLM judges which move looks like progress"), came
back NO-GO with both text and pictures.** The code is kept as a record of what
was tried.

**Now: LLM-written heuristics (design P2), Phase 1.** Qwen-9B writes small
scoring functions; the recorded games decide which ones survive. Step 1 is done:
`heur_api.py` (what a heuristic may use), `heur_sandbox.py` (static checks, then
a child process with a timeout) and `heur_referee.py` (70 recorded completions,
27,466 moves labelled toward/away from the solved board). `make -C legacy/llm_track
referee-selftest` checks it: the oracle control scores 1.000 on ft09, the sandbox
rejects all 8 kinds of bad code, and the best one-line rule on ft09 level 2 (the
held-out test level) scores 0.536.

**Probe C (2026-09-11): NO-GO.** `heur_writer.py` (`make -C legacy/llm_track probe-c`):
Qwen3.5-9B saw level 1 of ft09 and wrote 4 rounds x 8 candidate heuristics, with
feedback and one repair attempt each; the winner was picked on level 1 and graded
once on level 2. C1 FAIL (level-2 AUC 0.500, CI 0.500-0.500), C2 FAIL (below the
best one-line rule, 0.536), C3 PASS (valid code in all 4 rounds, 18 of 32
graded). Qwen never beat a one-line rule even on level 1 (best 0.533 vs 0.534),
feedback rounds did not improve it, and the winner hard-coded a board position
from level 1. Its plans misread the mechanic ("swap with the workshop colour",
"turn blue to red"); none encoded ft09's actual rule, making the framed block
match the example patterns. Pipeline lessons, all found on level 1 only: with
thinking on, Qwen spent its whole budget deliberating and wrote no code, so
thinking is off; three API misreadings (a button slot for ACTION6, region cells
as a mask) led to `r.mask`, an explicit ACTION6 note and one style example; a
sandbox bug masked numpy errors as `KeyError: '__import__'` and is fixed.

**Now: rule-finding (design P3), Stage A** (`rule_referee.py`, `rule_writer.py`;
see *Rule-finding, Stage A* below). Instead of guessing the goal, Qwen writes
rules for what each action does to the board, and the recordings check them
exactly. The checker, data and baselines are built and self-tested. In the
first smoke run (ft09 training moves only), Qwen3.5-35B worked out ft09's level-1
click mechanic by itself. The dense Qwen3.6-27B found the same rules but took
3.6x as long, so the 35B runs the full Stage A (started 2026-09-11).

## Commands

Run from the repo root (`-C llm_track` points make at this folder's Makefile):

```bash
make -C legacy/llm_track help
make -C legacy/llm_track env         # (re)build .venv-llm; see "Serving environment"
make -C legacy/llm_track scan        # corpus stats: masks, dedupe, buckets, anchors
make -C legacy/llm_track probe       # Probe A, text: Qwen 4B + 9B -> report.md
make -C legacy/llm_track probe-img   # Probe A, pictures: Qwen 2B + 4B -> report.md
make -C legacy/llm_track rules-data      # rule-finding: build the rules dataset (CPU)
make -C legacy/llm_track rules-selftest  # rule-finding: check the checker and baselines
make -C legacy/llm_track rules-smoke     # rule-finding: ft09, training moves only (GPU)
make -C legacy/llm_track rules           # rule-finding Stage A: full run, then the one test
make -C legacy/llm_track rules-overnight # the same run detached (~4 h), survives this window
make -C legacy/llm_track rules-status    # how far the detached run has got
```

A long run must be started detached (`rules-overnight`): anything started from a
terminal or the Claude Code window dies with it. Nothing survives
`wsl --shutdown` or the machine sleeping - both stop the whole VM, which killed
the first overnight attempt 90 seconds in (2026-09-11).

Everything writes under `results/legacy_llm/` (gitignored).

## Data harvest

Only ft09 has enough recorded level completions (31 level-1, 25 level-2) to
grade a heuristic; every other game has one or two. `make -C legacy/llm_track harvest`
runs plain, unmodified Goose overnight with new seeds (100+) on the games that
complete levels, with per-game budgets sized from when their levels were
reached (`harvest.sh` lists them). `harvest-plan` previews it without running.

## Isolation: why the baselines cannot be affected

1. **No baseline file imports it.** The repo's `make check` fails if one ever
   does. The arrow points one way: this package reads the corpus and the
   canonicalizer, and nothing reads this package.
2. **No baseline tooling runs it.** Its commands live in this folder's Makefile,
   and `make check` also fails if the root Makefile or `sweep.sh` ever invokes
   LLM code or `.venv-llm`.
3. **Separate virtualenv.** Serving runs in `.venv-llm` (vLLM 0.28 with its own
   torch 2.13/cu130), so it can never bump the baseline's pinned torch 2.8.0.
4. **Offline only.** Nothing here is on any agent's per-action path.

## Serving environment

`.venv-llm` must be built on a uv-MANAGED Python, not the system one: vLLM's
Triton backend compiles a C helper at startup and needs `Python.h`, and the
system Python 3.12 has no headers (no python3.12-dev, no passwordless sudo).
`make -C legacy/llm_track env` does this; pins are in `requirements-llm.txt`.

`judge.py` also switches off FlashInfer's sampler (its warm-up compiles CUDA
code and needs `nvcc`; greedy decoding never uses it), and defaults to
`--gpu-mem 0.90` rather than a value derived from `nvidia-smi`, which on WSL
over-reports the memory in use. `setup_vllm_env()` also puts the venv's `bin/`
on PATH, because FlashInfer's JIT runs `ninja` from there.

Big models on the 5090 (32 GB; checked 2026-09-11): Qwen3.5-35B-A3B GPTQ-Int4
loads in 21.1 GiB and leaves 2.9 GiB of KV cache at `--gpu-mem 0.90` (about 105k
tokens), and runs at about 250 tokens/s over 6 streams.
nvidia/Qwen3.6-35B-A3B-NVFP4 does not start: FlashInfer JIT-compiles an sm_120
FP8 GEMM, and its bundled CUDA headers do not match the pip nvcc 13.x. The dense
Qwen3.6-27B (cyankiwi AWQ 4-bit) runs too, in 19.2 GiB, but is about 3.6x slower
here.

## Probe A: can a small LLM judge ARC-AGI-3 moves?

The test set (`probe_pairs.py`) has 794 frozen pairs:
- 596 **anchor** pairs: a real move from just before a level was completed vs a
  real move from 200+ moves earlier in the same level
- 198 **sanity** pairs: a real move vs a ticker-only or no-change move

Each pair is asked in both orders; only the letter is used. The go/no-go
criteria are fixed in `probe_report.py`'s docstring and were written before any
model ran. Do not edit them after seeing results.

**Text (Qwen3.5-4B, -9B): NO-GO.** Both score 1.00 on the sanity tiers (the
format is understood) but chance on anchors (4B 0.497, 9B 0.502, CI
0.485-0.518; the "bigger change wins" rule scores 0.509). On ~91% of anchor
pairs they answer by position in both orders, i.e. they have no preference.

**Hindsight check (post-hoc, `probe_hindsight.py`, not pre-registered).** The
label is not the problem: moves 2-5 before a level-up bring the board toward its
pre-solve state 75% of the time vs 48% for controls, and a rule that sees the
board scores 0.64 on the same pairs. The progress signal is in the board state,
which a one-line description of a move omits.

**Pictures (Qwen3.5-2B, -4B): also NO-GO.** Before/after pictures of the whole
board (`probe_images.py`), same pairs and criteria: sanity 0.91-0.99, anchors
2B 0.508 and 4B 0.515 (CI 0.484-0.546) vs the size rule's 0.509. Pictures made
the models more decisive (ties 91% -> 65%) but not more correct (decisive
accuracy ~0.54). ft09 stays at 0.50: its goal is shown only as example blocks,
which is ARC-style rule induction.

**Conclusion:** the pairwise LLM judge (design P1) does not work at 2B-9B, with
text or images. The hindsight signal (0.64) is the best available referee for
any replacement mechanism.

## Rule-finding, Stage A (in progress)

The judge and the heuristics both needed Qwen to guess the GOAL, and it could
not. Rule-finding asks for something that can be checked exactly: the
MECHANICS, i.e. what one action does to the board. A rule is two small functions,
`applies()` and `predict()`. It is kept only if it predicts the next board
exactly (every non-ticker cell) on at least 95% of the 20 or more recorded moves
it claims (`rule_referee.py`). Nobody describes the game. Moves are grouped
automatically (each button; clicks by the colour clicked), and for each group
Qwen sees six recorded moves as pictures plus the exact changed cells.

- **Data** (`rules-data`): ft09 and lp85 learn from level 1 and are tested on
  level 2. ls20 never reaches level 2, so it learns from 8 runs and is tested on
  8 others. About 4,000 training and 4,000 test moves per game, spread evenly
  over runs.
- **Frozen screens are dropped.** 6 of 37 ft09 and 3 of 12 lp85 Goose runs sit on
  a screen that no action changes (not even the ticker) for most of their
  budget, up to 199k of 200k moves. Unfiltered, they were 58% of ft09's test
  sample. This may be worth a look on the baseline side; the LLM track only
  filters it out.
- **Ticker gaps are masked.** A few progress-bar cells change too rarely for the
  ticker detector to catch (ft09 row 63: 2 cells; ls20 row 61: 3), so the gaps
  inside each ticker row are filled. On ft09 these cells alone held a correct
  tile-flip rule at 94%.
- **Baselines (no LLM), exact-prediction rate on the test moves:**
  - "nothing changes": ft09 0.117, lp85 0.365, ls20 0.254
  - "memory" (repeat what the same board and action did in training, or else
    the same 9x9 click neighbourhood): ft09 0.117, lp85 0.365, ls20 0.919

  Memory cannot transfer to a new level. But when the test replays the same
  level (ls20) it has seen almost every board, so ls20 is the hard game. R1
  needs 2 of the 3.
- **Checker self-test** (`rules-selftest`):
  - On a synthetic game with known mechanics: correct rules score 100%, an
    off-by-one rule 0%, and a swapped-click rule applies to 3 of 104 moves.
  - ft09 clicks decode as (row, col): 840 tile flips are predicted only when the
    click is read that way, and 0 only when it is read swapped.
  - A "nothing changes" rule book equals the nothing baseline, and bad code is
    rejected at the right stage.
- **Pass/fail criteria** (R1, R2) are in `rule_writer.py`'s docstring, fixed
  before any Qwen rule-writing run.
- **Search budget:** 5 rounds x 8 candidates per action group (40 per group, 23
  groups over the three games), about 4 hours on the 5090.
- **Lesson from the model smoke test:** on a picture-only toy (a block that moves
  2 cells right), Qwen3.5-35B wrote 0 of 6 correct rules with and without
  thinking. It misread colours and distances. So the rule prompts give the exact
  changed cells as text as well as the pictures.
- **First smoke run** (Qwen3.5-35B, thinking on, 2 rounds, ft09 training moves
  only, 11 min):
  - 36 of 40 round-1 rules said "clicking colour X does nothing". Such a rule
    never changes a prediction, so rule books and feedback now rank rules by
    their gain over "nothing changes", and R2 now needs an accepted rule that
    predicts changes.
  - On the red and blue tiles, Qwen found the real mechanic by itself: "clicking
    a red block inside the dark-gray container turns it blue". With the ticker
    gaps masked, that rule is exactly right on all 562 of its training moves,
    and the blue rule on 99.5% of 800.
  - 10 of 48 answers ran out of thinking room (16k tokens) before writing code.
- **Model comparison** (same pipeline and data, ft09 training moves, 2 rounds):
  both models found the tile-flip mechanic. Qwen3.5-35B-A3B wrote two
  single-colour rules; the dense Qwen3.6-27B wrote one rule for both colours.
  With the ticker gaps masked, each rule book predicts 99.9% of the training
  moves exactly ("nothing changes": 74.6%). The 35B took 12 min and the 27B
  43 min, so the 35B, the largest model that runs here, is Stage A's model.
- **Prompt tweak:** each move now lists up to 24 changed areas (was 8), because
  one lp85 button press changes about 20 (its whole ring of squares rotates).
- **First Stage A attempt was stopped and restarted** (2026-09-11, before the
  test set was touched; kept in `results/legacy_llm/rules/stageA_aborted_16k`). With a
  16k thinking budget, 41 of 138 round-1 answers never reached code: 23 of 24 on
  ls20 and most of lp85's changing groups, while their thinking held sensible
  mechanics ("ACTION4 moves the 2x2 pattern 5 columns right if the target area
  is empty"). Only ft09, whose rules are short, came through. The budget is now
  20,480 tokens, the prompt asks for brief reasoning, and any answer that runs
  out of room is asked for its code alone with thinking off.

## Week-1 measurements

On the Stage-1 games the serializer costs 0.11-0.29 ms per transition (2-6% of
the agent's 5.2 ms model budget). Signature dedupe ranges from 38x (ls20) to
1.1x (ar25), so any pair sampler must cap per bucket per game rather than sample
proportionally.
