# C1 transition record — schema v1.0

The serializer (`llm_track/serializer.py`) turns one transition into a record.
Everything downstream — the Judge's prompt, the scorer `g`, the heuristic
sandbox, the run digest — reads this and nothing else, so this file is the
interface contract for the whole LLM track.

Computed only from `(frame, action_idx, next_frame, level)`, all of which the
`.npz` corpus already contains. No object layer, no new play. When the object
layer lands it adds fields; it does not change the ones below.

## Provenance

| Field | Meaning |
|---|---|
| `schema_version` | `"1.0"`. A trained `g` artifact is only valid for its schema version. |
| `game`, `action_num`, `level` | Run coordinates. `level` is the score when the action was **taken**. |
| `level_delta`, `is_level_completing` | Hindsight only (offline). `is_level_completing` marks the action that ended the level — see *Anchors* below. |

## Action

| Field | Meaning |
|---|---|
| `action_type` | 1–6. 6 is a click. |
| `click` | `[row, col]` or `null`. Decoded from the unified index `5 + 64*y + x`. |
| `click_color_before` / `_after` | Colour under the click point, or `-1` for non-clicks. |

## Global change

| Field | Meaning |
|---|---|
| `changed`, `n_changed`, `frac_changed` | Did anything change, how much. |
| `n_changed_ticker`, `n_changed_nonticker` | Split by the decorative mask. |
| `ticker_only` | **The decisive flag.** The frame moved, but only decoration moved. |
| `color_before[16]`, `color_after[16]`, `color_delta[16]` | Whole-frame cell counts per colour. "4 cells of colour 9 vanished" falls out of `color_delta`. |
| `n_distinct_colors_before` / `_after`, `background_color` | Cheap scene context. |

## Components (`components`, up to 8, largest first)

Connected components (8-connectivity) of the **non-ticker** changed cells.

| Field | Meaning |
|---|---|
| `size`, `bbox` `[y0,x0,y1,x1]`, `centroid` | Extent. |
| `color_before` / `_after` | Dominant colour each side. |
| `n_colors_before` / `_after` | How mixed the component is — 1 means a clean recolour. |
| `color_delta` | Per-colour cell flow **within** the component. Without it a component whose colours merely rearrange reads as "colour 3 → colour 3", which tells the Judge nothing. |

## Moves (`moves`)

Rigid translations, detected **per colour**, not per component: the cells that
changed away from colour *c* are where a *c*-coloured object was; the cells that
changed into *c* are where it is now. Same shape ⇒ a translation.

| Field | Meaning |
|---|---|
| `color`, `size`, `dy`, `dx` | The shape and its displacement. |

Per-colour is essential: an object moving less than its own width merges its
vacated and arrived cells into **one** component, which a component-pair matcher
cannot see. That is the normal case on ls20, where this detector correctly
reports "a 25-cell colour-3 shape moved by (0, −5)".

Blind to: rotation, scaling, occlusion, and two same-coloured objects moving
differently (reports nothing rather than guessing).

## Text rendering

Deterministic template, fixed on purpose — a prompt-format change is a code
change with a version bump. Target ≤200 tokens. Example (ft09):

```
ACTION6 at (row 54, col 53), colour 9->8 | changed 38 cells (0.93%) |
2 of them in the ticker region | comp1: 36 cells (6x6) at row 52, col 52
[colour 8 +36, colour 9 -36] | counts colour 8: 448->484; colour 9: 720->684
```

## Numeric vector

`to_vector(rec)` → 91 float32 values in the fixed order `VECTOR_FIELDS`. This is
what `g` and the heuristic sandbox consume. Layout is frozen per schema version.

## Signature (dedupe unit)

`signature(rec)` hashes the structural content and deliberately **excludes**
`action_num` and `level`, so the same event at action 900 and action 90,000 is
one signature with multiplicity 2. The Judge only ever labels unique signatures.

Measured dedupe (30k sampled transitions per game, newest run):

| Game | Unique signatures | Dedupe | Dominant buckets |
|---|---|---|---|
| ls20 | 785 | **38.2×** | move 73%, ticker_only 24% |
| cd82 | 3,614 | 8.3× | large 47%, ticker_only 32%, unchanged 19% |
| ft09 | 5,980 | 5.0× | large 96%, unchanged 4% |
| lp85 | 6,657 | 4.5× | unchanged 80%, move 16% |
| ar25 | 26,257 | **1.1×** | large 85%, move 13% |

**Budget consequence.** The design assumed a 10–100× reduction. Only ls20
delivers it. ar25 is effectively undedupable, so a flat "label every unique
signature" policy would spend the entire budget on one game. The pair sampler
must cap per bucket per game rather than sampling proportionally.

## Anchors (hindsight ground truth)

`is_level_completing` marks the action that ended a level — the strongest label
in the study. Two caveats:

1. **Historical corpora do not have it.** The agent cleared `prev_frame` before
   logging, so the completing transition was never written; the only trace is a
   gap of 2 in `action_nums`. `corpus.is_post_fix_boundary()` detects this. All
   66 level-ups in the current 14.1M-transition corpus are pre-fix. Runs
   recorded after the fix have it.
2. **`action_nums` is not contiguous** regardless: a game over also drops two
   actions (~1.5% on ft09). Always align on `action_nums`, never on position.

## Known limitation: mask stability

The decorative mask depends on how much of a run is scanned, because a ticker
may only tick during some phases. Measured on lp85: scanning 25 shards gives 60
masked cells (a vertical bar in column 0) and `tick_frac` 0.39; scanning 50
shards gives 0 masked cells and `tick_frac` 0.20 — the run's later phase has
far fewer lone ticks, dropping it under the 0.30 guard. ft09 is stable in the
mask but its `tick_frac` still moves (0.94 → 0.89).

Consequence: **the mask must be computed once per corpus over a fixed, recorded
window and stored with the labels**, or `g` will be trained against one mask and
applied under another. Do not recompute it ad hoc.
