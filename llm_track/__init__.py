"""llm_track — the LLM track, isolated from the baselines.

Status, commands and results: README.md. Design: docs/295B-llm-track-plan.md
(option survey) and docs/295B-llm-design-detailed.md (component design).
Everything here runs OFFLINE; nothing is imported or run by any baseline file
or tool (the repo's `make check` enforces that).

Modules
-------
corpus           shard iteration, random access, run selection
tickers          component-level decorative-cell detection
serializer       C1: transition -> feature record -> text
scan             CLI: corpus statistics (dedupe, tickers, anchors)
probe_pairs      Probe A: the frozen pair set
judge            Probe A: vLLM judge, text or --images (runs in .venv-llm)
probe_images     Probe A: before/after pictures for the image judge
probe_report     Probe A: scores and the pre-registered go/no-go
probe_hindsight  Probe A: post-hoc check - is the signal in the board?
"""
