"""probe_hindsight.py — POST-HOC diagnostic for Probe A. Not pre-registered.

Probe A came back NO-GO: on the anchor tier both Qwen models scored at chance.
Two explanations predict opposite next steps:

  (a) the signal is there, but the TEXT description of a single move does not
      carry it — it describes the change, not the board — so show the model the
      board (e.g. as an image);
  (b) "a move just before a level-up" is not a usable progress label at all,
      so no representation will help.

This separates them using hindsight that no judge ever sees: S* is the board
one move before the level was solved. A move is "toward" S* if it leaves fewer
non-ticker cells different from S* than there were before it. If anchors move
toward S* much more often than controls, the signal exists and lives in the
board state, which the text omits: explanation (a).

CAVEAT: anchors 1 move before the level-up are circular (S* IS their result),
so they are reported separately and excluded from the headline. The oracle rule
uses hindsight, so it shows the signal EXISTS in board terms; it does not show
that a judge could infer the goal from a picture of the board. Part of the
anchors' toward-rate is also structural — any trajectory that reaches S* soon
after must approach it at the end, even a random one — so this shows anchors
ARE toward-goal moves (what a judge should reward), not that Goose chose them
deliberately.

Kept separate from probe_report.py on purpose: that report contains only what
was fixed before any model ran. This was written after seeing the result.

    uv run python -m llm_track.probe_hindsight   # -> results/llm/probeA/hindsight.md
"""
import argparse
import collections
import json
import os

import numpy as np

from .corpus import GOOSE, CorpusReader, find_corpora, game_of
from .probe_pairs import build_masks


def oracle_score(rows):
    """Prefer the move that brings the board closer to S*; ties = 0.5."""
    return float(np.mean([1.0 if r["rx"] > r["ry"] else 0.0 if r["rx"] < r["ry"] else 0.5
                          for r in rows])) if rows else float("nan")


def direction(values):
    v = np.array(values)
    return (f"{np.mean(v > 0):.2f} | {np.mean(v < 0):.2f} | {np.mean(v == 0):.2f}"
            if len(v) else "- | - | -")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/llm/probeA")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    with open(os.path.join(a.dir, "pairs.jsonl")) as f:
        anchors = [p for p in map(json.loads, f) if p["tier"] == "anchor"]
    games = {p["game"] for p in anchors}
    # Same fixed-window masks the pair builder used.
    masks, _, _ = build_masks([c for c in find_corpora(a.results, agent=GOOSE)
                               if game_of(c[0]) in games])

    by_run = collections.defaultdict(list)
    for p in anchors:
        by_run[p["event"].rsplit("/", 1)[0]].append(p)

    rows = []
    for key, ps in by_run.items():
        run_id, name = key.split("/")
        r = CorpusReader(os.path.join(a.results, "runs", run_id, name, "transitions"))
        idx = {int(n): k for k, n in enumerate(r.scalars["action_nums"])}
        for p in ps:
            an = int(p["event"].rsplit("/", 1)[1])
            live = ~masks[p["game"]]
            s_star = r.get(idx[an])["next_frame"]

            def toward(side):
                t = r.get(idx[side["action_num"]])
                return (int(((t["frame"] != s_star) & live).sum())
                        - int(((t["next_frame"] != s_star) & live).sum()))

            rows.append({"game": p["game"], "steps": p["x"]["steps_before_levelup"],
                         "rx": toward(p["x"]), "ry": toward(p["y"])})

    nc = [q for q in rows if q["steps"] >= 2]
    L = ["# Probe A — post-hoc hindsight check (NOT pre-registered)\n",
         "Does a move bring the board closer to S*, the board one move before the "
         "level was solved? Hindsight no judge sees; see the module docstring.\n",
         "| moves | n | toward | away | neutral |", "|---|---|---|---|---|"]
    for lab, sel in (("anchors, 1 before (circular)", lambda q: q["steps"] == 1),
                     ("anchors, 2-5 before", lambda q: 2 <= q["steps"] <= 5),
                     ("anchors, 6+ before", lambda q: q["steps"] >= 6)):
        rs = [q for q in rows if sel(q)]
        L.append(f"| {lab} | {len(rs)} | " + direction([q["rx"] for q in rs]).replace(" | ", " | ") + " |")
    L.append(f"| controls (200+ earlier) | {len(rows)} | "
             + direction([q["ry"] for q in rows]) + " |")
    L += ["\n'Prefer the move that brings the board closer to S*', scored like the "
          "judge (0.50 = chance). Compare with the judges' anchor score in report.md.\n",
          "| pairs | n | score |", "|---|---|---|",
          f"| all anchor pairs | {len(rows)} | {oracle_score(rows):.3f} |",
          f"| **excluding circular 1-move anchors** | {len(nc)} | **{oracle_score(nc):.3f}** |",
          f"| ft09 only (excl. circular) | {sum(q['game'] == 'ft09' for q in nc)} | "
          f"{oracle_score([q for q in nc if q['game'] == 'ft09']):.3f} |",
          f"| other games (excl. circular) | {sum(q['game'] != 'ft09' for q in nc)} | "
          f"{oracle_score([q for q in nc if q['game'] != 'ft09']):.3f} |"]

    text = "\n".join(L) + "\n"
    with open(os.path.join(a.dir, "hindsight.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
