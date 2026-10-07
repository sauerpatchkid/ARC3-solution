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
a v1 rule can be re-bound. Intervals are not computed here; how to resample with
one recording per game is still open (rulebook-v2-prereg.md section 4.4).

  uv run python tools/wm_baseline.py            # 8 dev games, a few minutes
Writes results/rulebook/v2/v1_under_v2/{report.md, <game>.json, <game>_fit.npz, <game>_score.npz}.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from wm import tiers  # noqa: E402
from wm.check import book_status, check_rule, coverage  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.metrics import three_numbers, transfer_t0  # noqa: E402

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
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(row, f, indent=1)
    t = row.get("t0")
    print(f"[{game}] train: {row['train']}" + (f" | T0 on score split: book {t['book']:.3f} nothing {t['nothing']:.3f} "
                                               f"memory {t['memory']:.3f} wrong {t['wrong_on_changing']:.1%}" if t else " | no score split"),
          flush=True)
    return row


def report(out, rows):
    beat = [r["game"] for r in rows if r.get("t0", {}).get("beats_both")]
    safe = [r["game"] for r in rows if "t0" in r and r["t0"]["wrong_on_changing"] <= 0.05]
    L = ["# Rulebook v1's rule books under v2's measuring tools", "",
         "No LLM. v1's verdicts are final; this only re-measures its books the way CP1 will be measured "
         "(tools/wm_baseline.py). Point estimates: the interval method for CP1's gate is still open "
         "(rulebook-v2-prereg.md section 4.4).", "",
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
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L[4:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", default=",".join(tiers.DEV))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "v2", "v1_under_v2"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report(a.out, [run_game(g, a.out) for g in a.games.split(",")])


if __name__ == "__main__":
    main()
