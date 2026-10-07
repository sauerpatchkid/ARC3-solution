"""coach - the Coach plan's LLM advisor (docs/plans/llm-coach.md).

When Goose stalls, a local LLM reads a short text summary of the screen and of
what has been tried this level, and suggests what to try next. This package
holds everything Coach adds; the agent only calls it from lines tagged
`# [advisor]` in custom_agents/action.py (none yet).

  history.py   History: per-level record fed one transition at a time, and
               render(): the text the LLM reads (step 3)

HOW TO REMOVE IT
Delete this folder, legacy/coach_track/tests/test_coach_history.py and legacy/coach_track/tools/coach_summary.py.
Nothing else in the baseline imports it.
"""
