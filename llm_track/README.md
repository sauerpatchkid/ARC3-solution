# LLM track (`llm_track/`): Matt's LLM work, isolated from the baselines

This folder is self-contained: code, commands, serving environment, design docs
and results. Teammates working on the baselines can ignore it entirely, because
nothing outside it depends on it (see *Isolation* below).

- Design: `docs/295B-llm-track-plan.md` (option survey) and
  `docs/295B-llm-design-detailed.md` (component design)
- The transition-record contract: `SCHEMA.md`

## Status (2026-09-10)

Week 1 (serializer, corpus scan) is done. **Probe A, the go/no-go gate for the
design's backbone (P1: "the LLM judges which move looks like progress"), came
back NO-GO with both text and pictures.** The code is kept as a record of what
was tried. The next direction (LLM-written heuristics, P2, or rule-finding, P3)
has not been started.

## Commands

Run from the repo root (`-C llm_track` points make at this folder's Makefile):

```bash
make -C llm_track help
make -C llm_track env         # (re)build .venv-llm; see "Serving environment"
make -C llm_track scan        # corpus stats: masks, dedupe, buckets, anchors
make -C llm_track probe       # Probe A, text: Qwen 4B + 9B -> report.md
make -C llm_track probe-img   # Probe A, pictures: Qwen 2B + 4B -> report.md
```

Everything writes under `results/llm/` (gitignored).

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
`make -C llm_track env` does this; pins are in `requirements-llm.txt`.

`judge.py` also switches off FlashInfer's sampler (its warm-up compiles CUDA
code and needs `nvcc`; greedy decoding never uses it), and defaults to
`--gpu-mem 0.90` rather than a value derived from `nvidia-smi`, which on WSL
over-reports the memory in use.

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

## Week-1 measurements

On the Stage-1 games the serializer costs 0.11-0.29 ms per transition (2-6% of
the agent's 5.2 ms model budget). Signature dedupe ranges from 38x (ls20) to
1.1x (ar25), so any pair sampler must cap per bucket per game rather than sample
proportionally.
