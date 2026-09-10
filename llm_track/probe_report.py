"""probe_report.py — score Probe A: can a small LLM judge ARC-AGI-3 moves?

    uv run python -m llm_track.probe_report     # reads results/llm/probeA/

Reads pairs.jsonl and every labels_*.jsonl beside it, and writes report.md.

THE HEADLINE NUMBER: score = (wins + 0.5 * ties) / pairs.
  A tie counts as a coin flip, so 0.50 is chance whatever the tie rate, and a
  model cannot look good by refusing to decide. Decisive accuracy (wins over
  wins + losses) is reported beside it.

CONFIDENCE INTERVALS are bootstrapped over LEVEL-UP EVENTS on the anchor tier —
the pairs from one event share an anchor window and are not independent. On the
sanity tiers pairs are resampled directly.

BASELINES ON THE SAME PAIRS. "size" prefers whichever move changed more real
(non-ticker) cells; "moves" prefers whichever has more detected object moves.
A judge that only matches "size" has learned nothing a one-line rule does not
know. Beating chance is necessary; beating size is the point.

GO / NO-GO — criteria fixed 2026-09-10, BEFORE any model was run:
  G1 signal       9B anchor score: 95% CI lower bound above 0.50
  G2 beyond size  9B anchor score above the size baseline's
  G3 consistency  9B gives the same answer in both orders on >= 70% of
                  anchor pairs
  G4 format       9B scores >= 0.80 on each sanity tier
  GO if all four pass. Separately, if the 4B is within 0.05 of the 9B on the
  anchor tier, the 4B becomes the volume labeler (about 2x the throughput).
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

TIERS = ("sanity_ticker", "sanity_nochange", "anchor")
N_BOOT = 2000


def score_of(verdicts):
    v = list(verdicts)
    if not v:
        return float("nan")
    return (sum(x == "x" for x in v) + 0.5 * sum(x == "tie" for x in v)) / len(v)


def boot_ci(items, clusters, rng):
    """95% CI of score_of, resampling whole clusters."""
    by = collections.defaultdict(list)
    for it, c in zip(items, clusters):
        by[c].append(it)
    groups = list(by.values())
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(score_of([x for k in pick for x in groups[k]]))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def heuristic(p, key):
    """Baseline verdict: prefer the side with the larger feature value."""
    a, b = p["x"][key], p["y"][key]
    return "x" if a > b else "y" if b > a else "tie"


def summarize(rows, pairs, tier, rng):
    rs = [r for r in rows if r["tier"] == tier]
    if not rs:
        return None
    verdicts = [r["verdict"] for r in rs]
    clusters = [r["event"] if tier == "anchor" else r["pair_id"] for r in rs]
    wins = sum(v == "x" for v in verdicts)
    losses = sum(v == "y" for v in verdicts)
    raw = [a for r in rs for a in r["answers"] if a in ("A", "B")]
    return {
        "n": len(rs),
        "n_clusters": len(set(clusters)),
        "score": score_of(verdicts),
        "ci": boot_ci(verdicts, clusters, rng),
        "decisive_acc": wins / (wins + losses) if wins + losses else float("nan"),
        "tie_rate": sum(v == "tie" for v in verdicts) / len(rs),
        "consistency": sum(r["consistent"] for r in rs) / len(rs),
        "first_position_rate": (sum(a == "A" for a in raw) / len(raw)) if raw else float("nan"),
        "parse_fail": sum(a == "FAIL" for r in rs for a in r["answers"]) / (2 * len(rs)),
        "size": score_of(heuristic(pairs[r["pair_id"]], "n_changed_nonticker") for r in rs),
        "moves": score_of(heuristic(pairs[r["pair_id"]], "n_moves") for r in rs),
    }


def fmt(x, d=3):
    return "-" if x != x else f"{x:.{d}f}"


def model_rank(tag):
    """Sort models by size so the table reads small -> large."""
    for i, s in enumerate(("0.8B", "2B", "4B", "9B", "27B", "35B")):
        if s in tag:
            return i
    return 99


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/llm/probeA")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    with open(os.path.join(a.dir, "pairs.jsonl")) as f:
        pairs = {p["pair_id"]: p for p in map(json.loads, f)}
    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    label_files = sorted((p for p in glob.glob(os.path.join(a.dir, "labels_*.jsonl"))
                          if "_limit" not in p),
                         key=lambda p: model_rank(os.path.basename(p)))
    if not label_files:
        raise SystemExit(f"no labels_*.jsonl in {a.dir} - run llm_track.judge first")

    models = {}
    for lf in label_files:
        tag = os.path.basename(lf)[len("labels_"):-len(".jsonl")]
        with open(lf) as f:
            rows = [json.loads(ln) for ln in f]
        jm = json.load(open(lf.replace(".jsonl", ".meta.json")))
        models[tag] = {"rows": rows, "meta": jm,
                       "tiers": {t: summarize(rows, pairs, t, rng) for t in TIERS}}

    L = ["# Probe A — can a small LLM judge ARC-AGI-3 moves?\n",
         f"Pair set: {meta['n_pairs']} pairs ({meta['tiers']}), built "
         f"{meta['created']} at commit `{meta['git_commit']}`, serializer schema "
         f"v{meta['schema_version']}. Anchor tier: {meta['events_used']} level-up "
         f"events; {meta['identical_text_dropped']} pairs dropped because both "
         f"moves rendered to identical text.\n",
         "**score** = (wins + 0.5 x ties) / pairs, so **0.50 is chance**. "
         "**size** / **moves** are one-line rules scored on the same pairs. "
         "CI = 95% bootstrap over level-up events (anchor) or pairs (sanity).\n"]

    for t in TIERS:
        L.append(f"\n## {t}\n")
        L.append("| model | n | score | 95% CI | decisive acc | ties | "
                 "both-orders agree | picks 1st shown | unparsed | size rule | moves rule |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for tag, m in models.items():
            s = m["tiers"][t]
            if not s:
                continue
            L.append(f"| {tag} | {s['n']} | **{fmt(s['score'])}** | "
                     f"{fmt(s['ci'][0])}–{fmt(s['ci'][1])} | {fmt(s['decisive_acc'])} | "
                     f"{fmt(s['tie_rate'], 2)} | {fmt(s['consistency'], 2)} | "
                     f"{fmt(s['first_position_rate'], 2)} | {fmt(s['parse_fail'], 3)} | "
                     f"{fmt(s['size'])} | {fmt(s['moves'])} |")

    # --- per-game and by-distance breakdown of the anchor tier
    L.append("\n## anchor tier by game\n")
    L.append("| game | events | pairs | " + " | ".join(models) + " | size rule |")
    L.append("|---|---|---|" + "---|" * len(models) + "---|")
    games = sorted({p["game"] for p in pairs.values() if p["tier"] == "anchor"},
                   key=lambda g: -sum(1 for p in pairs.values()
                                      if p["tier"] == "anchor" and p["game"] == g))
    for g in games:
        ids = [k for k, p in pairs.items() if p["tier"] == "anchor" and p["game"] == g]
        ev = len({pairs[k]["event"] for k in ids})
        cells = []
        for tag, m in models.items():
            vs = [r["verdict"] for r in m["rows"] if r["pair_id"] in set(ids)]
            cells.append(fmt(score_of(vs)))
        size = score_of(heuristic(pairs[k], "n_changed_nonticker") for k in ids)
        L.append(f"| {g} | {ev} | {len(ids)} | " + " | ".join(cells) + f" | {fmt(size)} |")

    L.append("\n## anchor tier by distance to the level-up\n")
    L.append("Does the judge do better on moves closer to the completion?\n")
    L.append("| moves before level-up | pairs | " + " | ".join(models) + " |")
    L.append("|---|---|" + "---|" * len(models))
    for lo, hi in ((1, 5), (6, 15), (16, 60)):
        ids = {k for k, p in pairs.items() if p["tier"] == "anchor"
               and lo <= p["x"].get("steps_before_levelup", 0) <= hi}
        cells = [fmt(score_of(r["verdict"] for r in m["rows"] if r["pair_id"] in ids))
                 for m in models.values()]
        L.append(f"| {lo}-{hi} | {len(ids)} | " + " | ".join(cells) + " |")

    # --- agreement between models, and throughput
    tags = list(models)
    if len(tags) >= 2:
        L.append("\n## agreement between models (anchor tier)\n")
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                va = {r["pair_id"]: r["verdict"] for r in models[tags[i]]["rows"]
                      if r["tier"] == "anchor"}
                vb = {r["pair_id"]: r["verdict"] for r in models[tags[j]]["rows"]
                      if r["tier"] == "anchor"}
                common = va.keys() & vb.keys()
                same = sum(va[k] == vb[k] for k in common) / max(len(common), 1)
                both = [k for k in common if va[k] != "tie" and vb[k] != "tie"]
                dec = sum(va[k] == vb[k] for k in both) / max(len(both), 1)
                L.append(f"- {tags[i]} vs {tags[j]}: same verdict on {same:.2f} of "
                         f"pairs; {dec:.2f} where both decided ({len(both)} pairs)")
    L.append("\n## throughput\n")
    for tag, m in models.items():
        jm = m["meta"]
        L.append(f"- {tag}: {jm['pairs_per_sec']} pairs/s ({jm['n_calls']} calls in "
                 f"{jm['gen_sec']}s, model load {jm['load_sec']}s)")

    # --- the pre-registered decision
    big = next((t for t in reversed(tags) if "9B" in t), None)
    L.append("\n## Decision (criteria fixed before running — see module docstring)\n")
    if big is None:
        L.append("No 9B labels yet: the decision needs the 9B.")
    else:
        an = models[big]["tiers"]["anchor"]
        checks = [
            ("G1 signal: CI lower bound > 0.50", an["ci"][0] > 0.50,
             f"{fmt(an['ci'][0])}"),
            ("G2 beyond size: score > size rule", an["score"] > an["size"],
             f"{fmt(an['score'])} vs {fmt(an['size'])}"),
            ("G3 consistency >= 0.70", an["consistency"] >= 0.70,
             f"{fmt(an['consistency'], 2)}"),
        ]
        for t in ("sanity_ticker", "sanity_nochange"):
            s = models[big]["tiers"][t]
            if s:
                checks.append((f"G4 format: {t} >= 0.80", s["score"] >= 0.80,
                               f"{fmt(s['score'])}"))
        for name, ok, val in checks:
            L.append(f"- [{'PASS' if ok else 'FAIL'}] {name} ({val})")
        go = all(ok for _, ok, _ in checks)
        L.append(f"\n**{'GO' if go else 'NO-GO'}** for building on the {big} judge.")
        small = next((t for t in tags if "4B" in t), None)
        if small:
            d = models[small]["tiers"]["anchor"]["score"] - an["score"]
            L.append(f"\n4B vs 9B on anchors: {d:+.3f}. " +
                     ("Within 0.05 — use the 4B for volume labeling."
                      if abs(d) <= 0.05 else
                      "More than 0.05 apart — the 9B stays the labeler."))

    # --- a few rationales, for the report (never consumed by anything)
    if big:
        L.append(f"\n## Example anchor judgments ({big})\n")
        shown = collections.Counter()
        for r in models[big]["rows"]:
            if r["tier"] != "anchor" or shown[r["verdict"]] >= 2:
                continue
            p = pairs[r["pair_id"]]
            shown[r["verdict"]] += 1
            L.append(f"- **{r['verdict'].upper()}** ({p['game']}, anchor "
                     f"{p['x'].get('steps_before_levelup')} moves before level-up)  \n"
                     f"  anchor: `{p['x']['text'][:160]}`  \n"
                     f"  control: `{p['y']['text'][:160]}`  \n"
                     f"  reason (x shown first): _{r['reasons'][0]}_")

    text = "\n".join(L) + "\n"
    with open(os.path.join(a.dir, "report.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {os.path.join(a.dir, 'report.md')}")


if __name__ == "__main__":
    main()
