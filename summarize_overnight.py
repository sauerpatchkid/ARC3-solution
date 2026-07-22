#!/usr/bin/env python3
"""summarize_overnight.py - aggregate an overnight sweep into one report.

Reads the manifest written by run_overnight.sh (tab-separated:
run_dir <TAB> game <TAB> seed <TAB> arm), loads each run's metrics.json,
computes actions-to-each-level, aggregates across seeds by (game, arm), prints
a readable summary, and writes <out>.md and <out>.csv.

Usage:
  python summarize_overnight.py overnight_<stamp>.manifest --out overnight_<stamp>_summary
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict


def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            rows.append({"run_dir": parts[0], "game": parts[1],
                         "seed": int(parts[2]), "arm": parts[3]})
    return rows


def load_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return json.load(f)


def actions_to_levels(levelup_events):
    """levelup_events: list of [action_num, new_level] -> {level: action_num}."""
    out = {}
    for action_num, lvl in levelup_events or []:
        out[int(lvl)] = int(action_num)
    return out


def _clean(xs):
    return [x for x in xs if x is not None]


def med(xs):
    xs = _clean(xs)
    return round(st.median(xs), 2) if xs else None


def mean(xs):
    xs = _clean(xs)
    return round(st.mean(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="overnight_summary")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    for r in rows:
        try:
            r["m"] = load_metrics(r["run_dir"])
        except Exception as e:
            r["m"] = None
            print(f"[warn] no metrics for {r['run_dir']}: {e}")

    groups = defaultdict(list)
    for r in rows:
        if r["m"] is not None:
            groups[(r["game"], r["arm"])].append(r)

    max_level = max((r["m"].get("max_level", 0) for r in rows if r["m"]), default=0)

    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("# Overnight sweep summary")
    emit(f"(runs: {sum(len(v) for v in groups.values())}, "
         f"max level reached anywhere: {max_level})")
    emit("")

    csv_rows = []
    for (game, arm) in sorted(groups):
        grp = groups[(game, arm)]
        seeds = sorted(r["seed"] for r in grp)
        levels = [r["m"]["levels_completed"] for r in grp]
        a2l_all = [actions_to_levels(r["m"].get("levelup_events", [])) for r in grp]

        emit(f"## {game}  |  arm=reset_{arm}  |  seeds={seeds}")
        emit(f"- levels completed per seed: {levels}  "
             f"(median {med(levels)}, max {max(levels) if levels else 0})")
        for L in range(1, max_level + 1):
            reached = _clean([a.get(L) for a in a2l_all])
            if reached:
                emit(f"- actions to level {L}: median {med(reached)}  "
                     f"({len(reached)}/{len(grp)} seeds reached)  raw={sorted(reached)}")

        uniq = mean([r["m"]["unique_states"] for r in grp])
        auc = mean([r["m"]["discovery_auc"] for r in grp])
        chg = mean([r["m"]["meaningful_change_rate"] for r in grp])
        red = mean([r["m"]["redundancy"] for r in grp])
        ent = mean([r["m"]["entropy_delta_bits"] for r in grp])
        aps = mean([r["m"]["actions_per_sec"] for r in grp])
        emit(f"- unique_states~{uniq}  discovery_auc~{auc}  meaningful_change~{chg}")
        emit(f"- redundancy~{red}  entropy_delta~{ent}  act/s~{aps}")
        emit("")

        row = {"game": game, "arm": f"reset_{arm}", "n_seeds": len(grp),
               "levels_median": med(levels),
               "levels_max": max(levels) if levels else 0,
               "unique_states_mean": uniq, "discovery_auc_mean": auc,
               "meaningful_change_mean": chg, "redundancy_mean": red,
               "entropy_delta_mean": ent, "actions_per_sec_mean": aps}
        for L in range(1, max_level + 1):
            reached = _clean([a.get(L) for a in a2l_all])
            row[f"a2l{L}_median"] = med(reached) if reached else None
            row[f"a2l{L}_reached"] = f"{len(reached)}/{len(grp)}"
        csv_rows.append(row)

    # Persistence ablation verdict for ft09 (headline: actions-to-level-2)
    on_grp, off_grp = groups.get(("ft09", "on")), groups.get(("ft09", "off"))
    if on_grp and off_grp and max_level >= 2:
        def a2l2(grp):
            reached = _clean([actions_to_levels(r["m"].get("levelup_events", [])).get(2)
                              for r in grp])
            return med(reached), len(reached), len(grp)
        on_med, on_n, on_tot = a2l2(on_grp)
        off_med, off_n, off_tot = a2l2(off_grp)
        emit("## Persistence ablation verdict - ft09, actions-to-level-2")
        emit(f"- reset ON  : median {on_med}   ({on_n}/{on_tot} seeds reached L2)")
        emit(f"- reset OFF : median {off_med}   ({off_n}/{off_tot} seeds reached L2)")
        if on_med is not None and off_med is not None:
            if off_med < on_med:
                emit(f"- persistence looks FASTER to L2 by ~{round(on_med - off_med)} "
                     f"actions (median); also compare seeds-reached above.")
            elif off_med > on_med:
                emit(f"- persistence looks SLOWER to L2 by ~{round(off_med - on_med)} "
                     f"actions (median); possible negative transfer.")
            else:
                emit("- persistence and reset ~tied on median actions-to-L2.")
        elif on_n != off_n:
            emit("- arms differ mainly in HOW MANY seeds reached L2 (see counts above).")
        emit("- NOTE: small seed counts + GPU nondeterminism -> directional, not conclusive.")
        emit("")

    with open(args.out + ".md", "w") as f:
        f.write("\n".join(out_lines) + "\n")

    if csv_rows:
        allcols = []
        for r in csv_rows:
            for k in r:
                if k not in allcols:
                    allcols.append(k)
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=allcols)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

    print(f"\nwrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
