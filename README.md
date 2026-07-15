# ARC-AGI-3 Capstone — Team B: Instrumented StochasticGoose Baseline

Fork of [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution) (StochasticGoose, winner of the ARC-AGI-3 developer preview, by Dries Smit / Tufa Labs). This fork adds the instrumentation and evaluation machinery for our SJSU CMPE 295 capstone: reproducible seeding, action caps, a transition corpus, and timing metrics — the foundation for our three-baseline comparison (**StochasticGoose / Blind Squirrel / random**) and the upgrade ablations that follow.

**If you're implementing another baseline agent (random, Blind Squirrel): read §5, "The comparability contract." Your agent must honor the same environment variables and emit the same corpus schema, or our numbers won't be comparable.**

---

## 1. What StochasticGoose is (30 seconds)

A CNN takes the current 64×64 game frame and predicts, for each of ~4101 possible moves (5 buttons + 4096 click cells), the probability that the move will *change the frame at all*. The agent samples its next move in proportion to those probabilities, observes whether the frame changed, and uses that free 0/1 label to train online during play. Buffer and model are fully reset at each new level. It learns *which* actions do something — never *what* they do. Details: upstream README + our code walkthrough doc.

## 2. Setup (WSL2 / Linux)

Tested on Windows 11 + WSL2 (Ubuntu 24.04) with an RTX 5090; any CUDA Linux box should work.

```bash
# prerequisites: git, make, build-essential, uv (https://docs.astral.sh/uv/)
git clone --recurse-submodules https://github.com/sauerpatchkid/ARC3-solution.git
cd ARC3-solution

# API key (free, from https://three.arcprize.org/user)
cp ARC-AGI-3-Agents/.env.example ARC-AGI-3-Agents/.env
#   edit .env: ARC_API_KEY=<your key>
#   NOTE: the file is .env.example (dot), upstream README says .env-example — ours is correct

# apply the two required harness patches (submodule edits, tracked as a patch file here)
git -C ARC-AGI-3-Agents apply ../harness_patches.patch

make install
```

Gotchas we hit so you don't have to:
- **WSL users:** clone into the Linux filesystem (`~/...`), never `/mnt/c/...` — file I/O there is several times slower and the corpus logger will feel it. Torch + CUDA ≈ 8.4 GB installed; make sure the drive hosting your WSL vhdx has 20+ GB free (a full disk mid-install corrupted our first ext4 filesystem).
- **RTX 5090 / Blackwell:** if torch complains about `sm_120` kernels, reinstall with `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- GPU passthrough in WSL uses the *Windows* NVIDIA driver only; verify with `nvidia-smi` inside Ubuntu.

Verify: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 3. Running the instrumented agent

```bash
# smoke test: one game, fixed seed, 2000-action cap (~2.5 min at typical server fps)
make baseline GAME=vc33 SEED=0 CAP=2000

# original behavior (all games, no cap, stop on WIN or 8h):
make action
```

`make baseline` expands to environment variables that control everything:

| Env var | Default | Meaning |
|---|---|---|
| `GOOSE_SEED` | unset (time-based) | Base seed. Each game adds a deterministic md5-derived offset, so multi-game runs get distinct but reproducible streams. Set `PYTHONHASHSEED=0` too (the Makefile target does). |
| `GOOSE_MAX_ACTIONS` | 0 (unlimited) | Hard per-game action cap. |
| `GOOSE_LOG_METRICS` | true | TensorBoard scalars (loss, score, buffer size, timing). |
| `GOOSE_LOG_TRANSITIONS` | true | Write the transition corpus (see §4). |
| `GOOSE_SAVE_VIS` | false | Expensive PNG heatmaps; smoke-test debugging only. |

Every run writes `runs/<timestamp>/<game_id>/run_config.json` recording the seed, flags, and cap, plus the git commit + uncommitted diff (via `utils.py`) — no run is ever mystery data.

## 4. The transition corpus

With `GOOSE_LOG_TRANSITIONS=true`, every transition (before any dedup/filtering) is logged to compressed shards:

```
runs/<timestamp>/<game_id>/transitions/shard_00000.npz, shard_00001.npz, ...
```

Each shard (≤1000 transitions) contains parallel arrays:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `frames` | (N, 64, 64) uint8 | Frame before the action (color indices 0–15, **not** one-hot; last animation frame) |
| `actions` | (N,) int32 | Unified action index: 0–4 = ACTION1–5; 5 + (64·y + x) = click at (x, y) |
| `next_frames` | (N, 64, 64) uint8 | Frame after the action (this is what the upstream buffer *doesn't* store — required for forward-dynamics training) |
| `changed` | (N,) uint8 | 1 if any cell differs between frame and next_frame |
| `levels` | (N,) int32 | Game score (= level) at the time of the action |
| `action_nums` | (N,) int64 | Global action counter |
| `wall_ms` | (N,) float32 | Wall-clock since previous decision (includes server round-trip) |
| `model_ms` | (N,) float32 | Model-only compute (inference + any training step) — the number that transfers to the offline Kaggle sandbox |

Inspect any corpus:

```bash
uv run inspect_corpus.py runs/<ts>/<game_id>/transitions
# reproducibility check between two runs:
uv run inspect_corpus.py <run1>/transitions --verify-seed <run2>/transitions
```

## 5. The comparability contract (baseline implementers: this is your spec)

For the three-baseline comparison to be valid, every agent must:

1. **Honor the same knobs.** Read `GOOSE_SEED` (with the same md5 per-game offset — copy `_stable_game_offset()` from `custom_agents/action.py`) and `GOOSE_MAX_ACTIONS`. Seed *all* RNGs you use (`random`, `numpy`, `torch` if applicable). Run under `PYTHONHASHSEED=0`.
2. **Emit the same corpus.** Copy the `TransitionLogger` class verbatim from `custom_agents/action.py` and log every transition with the schema above — including agents with no model (random agent: `model_ms=0` for pure sampling time, or time your selection code). Log *before* any agent-internal filtering.
3. **Write `run_config.json`** with at minimum: game_id, seed, seed_source, max_actions, agent name.
4. **Count RESET actions against the cap** (the harness's `action_counter` does this naturally — don't circumvent it).
5. **Random agent definition** (so it's not a strawman): uniform over the frame's `available_actions`; if ACTION6 is drawn, uniform over all 4096 coordinates. Same masking information Goose gets.
6. **Pass the reproducibility test** before your numbers count: same game + same seed twice → `inspect_corpus.py --verify-seed` reports identical action sequences.

Suggested: keep the `GOOSE_*` names even in non-Goose agents (ugly but unambiguous), or we rename all to `EVAL_*` together — decide at the protocol meeting, then it's frozen.

## 6. Evaluation protocol (agreed 3-tier design)

- **Tier 0 — smoke (~5 min):** 1 game × 1 seed × 2K cap. Regression check after changes. Suggested game: `vc33` (click-only, easiest for AI).
- **Tier 1 — dev eval (overnight):** frozen dev set (~8 of the 25 public games, chosen after a 1-seed × 2K pilot over all 25) × 3 fixed seeds (0, 1, 2) × 10K action cap. Metrics: levels solved within cap, actions at each level-up (progress curve), wall/model ms (median + p95), plus per-agent internals kept out of cross-agent tables. **No RHAE at this tier** — capped runs aren't comparable to official scorecards.
- **Tier 2 — full benchmark (milestones only):** all 25 public games, official protocol, real RHAE via server scorecards.

Dev set, cap, and seeds are decided once at the team meeting and never changed.

## 7. Findings so far (from smoke tests, Jul 15)

- **Model compute is ~4% of wall-clock** (median 2.1 ms model vs 54 ms wall at ~14 fps): the bottleneck is server round-trip, which doesn't exist in the offline Kaggle sandbox. Goose's true offline speed is likely hundreds of actions/sec.
- **vc33 is click-only** (`available_actions = [6]`) and contains a per-action progress indicator: exactly 1 cell changes per action (median), never the clicked cell, so the global "did anything change?" signal is constant 1.0 and carries **zero information**. On such games baseline Goose is structurally equivalent to the random agent — a testable Tier 1 prediction and the cleanest motivation for the per-cell change-mask upgrade (Phase 2).
- Known upstream quirks documented in our code-walkthrough doc: hash-dedup set never prunes on buffer eviction; time-based seeding (fixed here); `torch.cuda.empty_cache()` every train step; only the last animation frame is used.
- Scorecard API returned zeros on a capped run — under investigation before Tier 2; does not affect Tier 1.

## 8. Repo layout & docs

```
custom_agents/action.py    # the agent + our instrumentation (all changes commented)
inspect_corpus.py          # corpus stats + seed-verification tool
harness_patches.patch      # the two required submodule edits (apply per §2)
Makefile                   # install / action / baseline / tensorboard / clean
ARC-AGI-3-Agents/          # official harness (submodule, pinned)
```

Companion docs (shared drive / on request): Team B project plan v2 (phase ladder, deadlines) and the StochasticGoose code walkthrough (line-level reference, quirks list, metrics plan).

## 9. Credits & license

StochasticGoose by Dries Smit (Tufa Labs), adviser Jack Cole — [upstream repo](https://github.com/DriesSmit/ARC3-solution). Official harness by the ARC Prize Foundation. Instrumentation and evaluation machinery: Team B, SJSU CMPE 295 (2026). License: upstream terms apply to inherited code; our additions intended for CC0/MIT-0 release per ARC Prize 2026 eligibility rules (final licensing to be confirmed at team meeting — flag before adding substantial new code).
