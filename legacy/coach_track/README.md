# Coach track (archived 2026-10-06)

The **Coach** was the first semester-2 LLM option: when Goose stalls, a local LLM
reads a text summary of the screen and suggests what to try
(`docs/plans/llm-coach.md`). It was built through its offline probe, which came
back **NO-GO**: prompt v1 scored at chance (21 hits against 21.2 expected from
random picks, `docs/plans/coach-probe-results.md`). The LLM track then moved to
Rulebook. Nothing here was ever wired into the agent.

It is kept because the result is part of the record, not because anything uses
it. Like the rest of `legacy/`, it is a leaf: it imports the baseline
(`custom_agents/upgrades.py`, `custom_agents/gridtools.py`), and nothing in the
baseline imports it (`make check` enforces that).

| Path | What |
|---|---|
| `coach/history.py` | the screen summary the LLM reads (objects, per-button effects, previous wins) |
| `coach/advice.py`, `coach/prompt_v1.txt` | the advice format, its validator, the hand rule and the LLM client |
| `tools/advice_probe.py` | the offline probe and its pre-registered gate |
| `tools/coach_summary.py` | print the summary at chosen points of a recorded run |
| `experiments/serve.sh`, `experiments/fit_check.py` | serve the LLM beside Goose and measure memory and latency |
| `tests/` | its unit tests |

```bash
uv run python -m pytest legacy/coach_track/tests -q
uv run python legacy/coach_track/tools/advice_probe.py --no-llm
```

Moved here from `custom_agents/coach/`, `tools/`, `experiments/coach/` and
`tests/`; the pre-move layout is at the git tag `pre-cleanup-2026-10-06`.
