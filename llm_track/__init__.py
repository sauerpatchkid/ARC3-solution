"""llm_track — the LLM track ("Judge -> Head -> Heuristics").

See 295B-llm-track-plan.md (option survey) and 295B-llm-design-detailed.md
(component design). This package holds everything that runs OFFLINE or
PERIODICALLY; nothing here is imported by the agent's per-action path yet.

Modules
-------
corpus      shard iteration helpers shared by every offline tool
tickers     component-level decorative-cell detection (C1's ticker feature)
serializer  C1: transition -> feature record -> text for the Judge
scan        CLI: corpus statistics (signature dedupe, tickers, anchors)
"""
