#!/usr/bin/env python3
"""wm_baseline.py - Rulebook v1's rule books under v2's measuring tools (docs/plans/
rulebook-v2.md sections 3.1 and 3.3). No LLM, no GPU.

v1's verdicts are final and are not re-scored here. This puts v1's books on the
ruler CP1 will be judged with, so that "v1, for comparison" in the CP1 gate is
measured the same way as v2 will be:

  training level   the three numbers (claimed-cell exactness, cell coverage,
                   case coverage). A v1 rule claims every cell, so its
                   claimed-cell exactness is its whole-board exactness.
  transfer level   T0 on the SCORE split: the book untouched, on the moves after
                   the first 300, beside "nothing changes" and memory (which also
                   gets the first 300). The share of changing cases it gets
                   WRONG is the CP1 safety number.

T1 (after re-binding) and T2 (after one repair) do not exist for v1: nothing in
a v1 rule can be re-bound.

Since 13 Oct 2026 the transfer level is also scored the way CP1's gate is
(rulebook-v2-prereg.md section 4.4): per recorded move, with the 95% interval of
book - "nothing" and book - "memory" from resampling attempts (wm.metrics). That
is the gate's "v1, for comparison" row. The rates over distinct cases are kept
beside it.

  uv run python tools/wm_baseline.py            # 8 dev games, a few minutes
Writes results/rulebook/v2/v1_under_v2_cp1/{report.md, <game>.json, <game>_fit.npz, <game>_score.npz}.
CP0's run of this tool (results/rulebook/v2/v1_under_v2/, without the intervals) is
a registered result and is never written to again.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from wm import registry, tiers  # noqa: E402
from wm.check import book_status, check_rule, coverage  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.metrics import (gate_interval, memory_hash, moves_right, nothing_hash, three_numbers,  # noqa: E402
                        transfer_t0, v1_book_hash)

V1 = os.path.join(ROOT, "results", "rulebook", "tier0a")
FIT_MOVES = 300


def run_game(game, out):
    v1 = json.load(open(os.path.join(V1, f"{game}.json")))
    book = [c["code"] for c in v1["book"]]
    train_path = os.path.join(V1, f"{game}_train.npz")
    train = LevelEvidence.load(train_path)
    res1 = [r for r in (check_rule(code, train_path) for code in book) if r["stage"] == "checked"]
    row = {"game": game, "book_rules": len(book), "train_level": train.level + 1,
           "train": three_numbers(res1, book_status(res1, len(train.actions)), train),
           "v1_coverage": v1["coverage"]["coverage"]}
    if v1.get("level2"):
        level = v1["level2"]["level"] - 1
        corpus = v1["corpus"] if os.path.isabs(v1["corpus"]) else os.path.join(ROOT, v1["corpus"])
        fit, score = LevelEvidence.split(game, corpus, level, FIT_MOVES, known_mask=train.mask)
        row.update(transfer_level=level + 1, fit_moves=int(fit.count.sum()),
                   score_moves=int(score.count.sum()) if score is not None else 0)
        if score is not None:
            p_fit, p_score = (os.path.join(out, f"{game}_{k}.npz") for k in ("fit", "score"))
            fit.save(p_fit)
            score.save(p_score)
            res2 = [r for r in (check_rule(code, p_score) for code in book) if r["stage"] == "checked"]
            st2 = book_status(res2, len(score.actions))
            t0 = transfer_t0(st2, train, fit, score)
            t0.pop("per_case")
            row["t0"] = dict(t0, changing_cases=coverage(st2, score)["changing_keys"],
                             v1_whole_level=v1["transfer"])
            row["t0_moves"] = gate_interval(score, {
                "book": moves_right(score, v1_book_hash(res2, score)),
                "nothing": moves_right(score, nothing_hash(score)),
                "memory": moves_right(score, memory_hash([train, fit], score))})
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(row, f, indent=1)
    t, m = row.get("t0"), row.get("t0_moves")
    print(f"[{game}] train: {row['train']}" + (f" | T0 on score split: book {t['book']:.3f} nothing {t['nothing']:.3f} "
                                               f"memory {t['memory']:.3f} wrong {t['wrong_on_changing']:.1%}; per move: "
                                               f"{m['rate']}, beats both: {m['beats_both']}" if t else " | no score split"),
          flush=True)
    return row


def _interval(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d["lo"] is not None else f"{d['point']:+.3f} (no interval)"


def report(out, rows):
    beat = [r["game"] for r in rows if r.get("t0", {}).get("beats_both")]
    safe = [r["game"] for r in rows if "t0" in r and r["t0"]["wrong_on_changing"] <= 0.05]
    gate = [r["game"] for r in rows if r.get("t0_moves", {}).get("beats_both")]
    L = ["# Rulebook v1's rule books under v2's measuring tools", "",
         "No LLM. v1's verdicts are final; this only re-measures its books the way CP1 will be measured "
         "(tools/wm_baseline.py).", "",
         "## Training level", "",
         "| game | level | book rules | claimed-cell exactness | cell coverage | case coverage (v1's strict number) |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["train"]
        L.append(f"| {r['game']} | {r['train_level']} | {r['book_rules']} | {t['claimed_exactness']:.1%} | "
                 f"{t['cell_coverage']:.1%} | {t['case_coverage']:.1%} |")
    L += ["", f"## Transfer level, T0 on the score split (moves after the first {FIT_MOVES})", "",
          "One-step exact rate over the score split's distinct cases. 'Wrong' is the share of changing cases "
          "the untouched book predicts wrongly (CP1's safety bar is 5%, for T1).", "",
          "| game | level | score moves | cases | book | nothing | memory (+ fit split) | beats both | wrong on changing cases |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        t = r.get("t0")
        L.append(f"| {r['game']} | " + (
            f"{r['transfer_level']} | {r['score_moves']:,} | {t['cases']:,} | {t['book']:.3f} | {t['nothing']:.3f} | "
            f"{t['memory']:.3f} | {'yes' if t['beats_both'] else 'no'} | {t['wrong_on_changing']:.1%} |"
            if t else "- | - | - | - | - | - | - | - |"))
    L += ["", f"Beats both baselines (point estimate): {len(beat)} of {len(rows)} ({', '.join(beat) or 'none'}). "
              f"Wrong on at most 5% of changing cases: {len(safe)} of {len(rows)} ({', '.join(safe) or 'none'})."]
    L += ["", "## Transfer level, T0 per move, with the gate's interval (pre-registration 4.4)", "",
          "One-step exact rate over the score split's recorded MOVES: a move is right if the prediction matches "
          "that move's own next screen. The differences are book minus baseline with their 95% interval, from "
          "2,000 resamples (seed 0) of attempts, or of blocks of 50 moves where a level has fewer than 20 "
          "attempts. 'Beats both' is the CP1 gate's primary criterion: both intervals above zero.", "",
          "| game | level | moves | resampled | book | nothing | memory (+ fit split) | book - nothing | book - memory | beats both |",
          "|---|---:|---:|---|---:|---:|---:|---|---|---|"]
    for r in rows:
        m = r.get("t0_moves")
        L.append(f"| {r['game']} | " + (
            f"{r['transfer_level']} | {m['moves']:,} | {m['units']} {m['unit']} | {m['rate']['book']:.3f} | "
            f"{m['rate']['nothing']:.3f} | {m['rate']['memory']:.3f} | {_interval(m['diff']['nothing'])} | "
            f"{_interval(m['diff']['memory'])} | {'yes' if m['beats_both'] else 'no'} |"
            if m else "- | - | - | - | - | - | - | - | - |"))
    L += ["", f"Beats both baselines with the interval above zero: {len(gate)} of {len(rows)} "
              f"({', '.join(gate) or 'none'}). This is the gate's 'v1, for comparison' entry."]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L[4:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", default=",".join(tiers.DEV))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "v2", "v1_under_v2_cp1"))
    a = ap.parse_args()
    inside = os.path.relpath(os.path.abspath(a.out), ROOT) + os.sep
    frozen = [e["name"] for e in registry.load()["artifacts"] if any(f.startswith(inside) for f in e["files"])]
    if frozen:
        sys.exit(f"{a.out} holds the registered result {frozen[0]!r}; write somewhere else")
    os.makedirs(a.out, exist_ok=True)
    report(a.out, [run_game(g, a.out) for g in a.games.split(",")])


if __name__ == "__main__":
    main()
