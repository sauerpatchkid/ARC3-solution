# Rulebook v1 and v2 (archived 16 Oct 2026)

**The idea.** Goose explores. A local LLM (Qwen3.8-27B) writes the game's
mechanics as small Python rules, and every rule is replayed against the moves
Goose recorded. Only rules that replay exactly are trusted.

**How it ended.**

| Version | Test | Result | Write-up |
|---|---|---|---|
| v1 | Stage A: can the model write any exact rule? | NO-GO, 1 of 3 games | `docs/2-llm-history/rulebook-stageA.md` |
| v1 | Tier 0a: offline rule test on 8 dev games | G1 FAIL, 2 of 8. Rules did not carry to the next level (ft09 wrong on 58.8%) | `docs/2-llm-history/rulebook-tier0a.md` |
| v2 | CP0: near-miss refinement | 2 of 7 groups made exact, both by covering less | `docs/2-llm-history/rulebook-v2-cp0-results.md` |
| v2 | CP1: rules that re-bind to each new level | Gate FAIL, 2 of 8. Safety met on all 8 (0.0% wrong at T1) | `docs/2-llm-history/rulebook-v2-cp1-results.md` |

What v2 fixed: rules now carry between levels. What it did not fix: the model
writes too few exact rules (tu93, tr87 and m0r0 got none). That diagnosis is
why Glassbox (`docs/3-glassbox/glassbox-plan.md`) replaced it.

**What Glassbox kept.** The checking machinery (`evidence`, `sandbox`,
`contract`, `lint`, `bind`, `book`, `check`, `metrics`, `llm`) was **copied** to
`custom_agents/gb/core/`. `tiers.py` and `registry.py` moved to the repo root.
Nothing outside `legacy/` imports this folder, and `make check` enforces that.
The copies here stay frozen, so v1 and v2 can always be re-scored exactly as
they ran.

## Layout

| Path | What |
|---|---|
| `wm/` | the world-model package. v1: `sandbox`, `evidence`, `check`, `rules` (prompts), `llm`. v2 added `metrics`, `tiers`, `registry`, `contract`, `lint`, `bind`, `book`, `rules2` (prompts) |
| `tools/wm_offline.py` | v1's offline rule test (Tier 0a) |
| `tools/wm_refine.py` | v2 CP0: near-miss refinement with a bandit |
| `tools/wm_tier0a_v2.py` | v2 CP1: the gate run |
| `tools/wm_thinkcap.py` | what is lost if thinking is cut at 12,288 or 6,144 tokens |
| `tools/wm_baseline.py` | v1's rule books scored on v2's ruler |
| `experiments/` | `serve.sh` (the LLM server), `llm_run.sh` (v2 jobs), `tier0a.sh` and `stageA.sh` (v1; they refuse to overwrite their results) |
| `tests/` | 66 unit and end-to-end tests (mock LLM, no GPU) |

## Reproduce

```bash
uv run python -m pytest legacy/rulebook/tests -q
```

To re-score a finished v2 run from its answer cache (no server, CPU only), copy
its `llm_cache.jsonl` into an empty folder and pass that folder as `--out`:

```bash
uv run python legacy/rulebook/tools/wm_tier0a_v2.py --from-cache --games lp85 --out <empty folder with the cache>
```

Never point `--out` at a registered folder; the tool refuses. The registered
results are `rulebook_v1_results`, `rulebook_v2_cp0_nearmiss` and
`rulebook_v2_cp1_tier0a` in `artifacts/registry.json`. They live under
`results/rulebook/`, which is not in git, and v1's are also packed in
`results/rulebook_v1_frozen_2026-10-07.tar.gz`.

**Do not edit** the prompt text in `wm/rules.py` or `wm/rules2.py`, or the
tools. The answer caches are keyed on the prompts.

## History

| Git tag | What it points at |
|---|---|
| `rulebook-v1` | the code that produced v1's results |
| `rulebook-v2` | the code that produced CP0 and CP1 (`8c42e75`) |

This folder was moved from `custom_agents/wm/`, `tools/wm_*.py`,
`tests/test_wm_*.py` and `experiments/rulebook/` on 16 Oct 2026. The move
changed only the lines that locate the repo root, the `wm` package and these
tools.
