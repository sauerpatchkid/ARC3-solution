"""s2_pool.py - S2 POOL: the candidates of finished runs pooled into one run, with no LLM.

The problem it fixes: each S2 run writes its rules from scratch, so a new run
loses what an earlier one found. G1 (docs/3-glassbox/g1-offline-rules.md) gave
tu93 and tr87 their first trusted rules, but its ft09 book lost CP1's red-click
rule (97.7% -> 59.2% of changed cells), and judged game by game each such loss
counted against it. A book drawn from both runs' candidates can keep both.

What it does, per game. Every source run must hold the same evidence for the
game (its training, fit and score files, byte for byte): rules graded on
different evidence cannot be pooled. The evidence and S0's report are copied
from the first source. Then S2's files are written for the new run:

  candidates.jsonl  every source's candidates, source by source, each source's
                    followed by the narrowed rules its guard fitter trusted
                    (s3_check/<game>/guarded.jsonl), so the pool holds every
                    rule its sources' books could hold. Each gets a new id and
                    order (integers, in that order) and `pooled_from` (run, id);
                    a candidate with the same code and binding as an earlier one
                    is marked a duplicate.
  groups.json       the groups called, in first-seen order across the sources
  llm.json          no LLM calls; the sources, and how many candidates each gave

S3's grading and S4's carry then run unchanged, so the book is chosen from the
pool by CP1's procedure (per group, in the order the groups were called:
lint-clean first, most gain, at most 4, a later rule must add cells) and
re-bound on the next level's fit split. Nothing is asked again and nothing is
re-fitted, so a pooled book is a new comparison: it is reported as exploratory,
never as a gate's result. A pool of ONE run is a copy of its candidates (and its
guard fitter's rescues): a way to re-run S3 and S4 with other settings (S4's roles,
say) without touching the run itself.

`report()` sets the pool's book beside each source's (the guard fitter's
"after" book and T1, as G1's verdict reads them; else S3's book and S4's T1).

Reads  <source>/evidence/<game>_L*_{train,fit,score}.npz, <source>/s0_mask/<game>.json
       <source>/s2_write/<game>/candidates.jsonl, groups.json
       <source>/s3_check/<game>/guarded.jsonl        (if the source ran the guard fitter)
Writes <run>/evidence/<game>_L*_*.npz, <run>/s0_mask/<game>.json   (copies)
       <run>/s2_write/<game>/candidates.jsonl, groups.json, llm.json
       <run>/pool_report.{json,md}                   report(): the pool beside its sources
"""
import glob
import json
import os
import shutil

from gb.artifacts import file_sha256, plain

VERSION = 1                     # bump when what the output means changes (artifacts.py)


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evidence_files(run, game):
    """{file name: path} of a run's evidence splits for one game."""
    return {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(run.dir, "evidence", f"{game}_L*_*.npz")))}


def pooled(sources, game):
    """(candidates, groups, per-source counts) for one game: module doc, candidates.jsonl."""
    cands, groups, seen, counts = [], [], set(), {}
    for src in sources:
        s2 = os.path.join(src.dir, "s2_write", game)
        own = _jsonl(os.path.join(s2, "candidates.jsonl"))
        p_guarded = os.path.join(src.dir, "s3_check", game, "guarded.jsonl")
        rescued = [c for c in (_jsonl(p_guarded) if os.path.exists(p_guarded) else [])
                   if c.get("kind") == "narrowed" and c.get("trusted")]
        for c in own + rescued:
            key = (c.get("code"), json.dumps(c.get("binding"), sort_keys=True))
            cands.append(dict(c, id=len(cands), order=len(cands), pooled_from={"run": src.name, "id": c["id"]},
                              duplicate=bool(c.get("duplicate")) or key in seen))
            seen.add(key)
        with open(os.path.join(s2, "groups.json")) as f:
            groups += [g for g in json.load(f) if g not in groups]
        counts[src.name] = {"candidates": len(own), "rescued": len(rescued)}
    return cands, groups, counts


def run_game(run, game, sources, log=print, force=False):
    """S2 POOL for one game: `sources` (artifacts.Run, in order) pooled into `run`."""
    files = [evidence_files(s, game) for s in sources]
    if not files[0]:
        raise FileNotFoundError(f"{sources[0].name}: no evidence for {game}")
    for s, f in zip(sources[1:], files[1:]):
        if sorted(f) != sorted(files[0]) or any(file_sha256(f[n]) != file_sha256(files[0][n]) for n in f):
            raise ValueError(f"{game}: {s.name}'s evidence differs from {sources[0].name}'s; rules graded on "
                             "different evidence cannot be pooled")
    inputs = {f"{s.name}:{label}": p for s in sources for label, p in
              [("candidates", os.path.join(s.dir, "s2_write", game, "candidates.jsonl")),
               ("groups", os.path.join(s.dir, "s2_write", game, "groups.json")),
               ("guarded", os.path.join(s.dir, "s3_check", game, "guarded.jsonl"))] if os.path.exists(p)}
    inputs.update({f"evidence:{n}": p for n, p in files[0].items()})
    settings = {"sources": [s.name for s in sources]}
    if not force and run.is_current("s2_write", game, VERSION, settings, inputs):
        log(f"[{game}] the pool is current; skipped")
        return None
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    for name, p in files[0].items():
        shutil.copyfile(p, os.path.join(ev_dir, name))
    s0 = os.path.join(sources[0].dir, "s0_mask", f"{game}.json")
    if os.path.exists(s0):
        shutil.copyfile(s0, os.path.join(run.stage_dir("s0_mask"), f"{game}.json"))
    cands, groups, counts = pooled(sources, game)
    out = os.path.join(run.stage_dir("s2_write"), game)
    os.makedirs(out, exist_ok=True)
    p_cands, p_groups, p_llm = (os.path.join(out, f) for f in ("candidates.jsonl", "groups.json", "llm.json"))
    with open(p_cands, "w") as f:
        for c in cands:
            f.write(json.dumps(plain(c)) + "\n")
    with open(p_groups, "w") as f:
        json.dump(groups, f, indent=1)
    with open(p_llm, "w") as f:
        json.dump({"pooled_from": counts, "llm_calls": 0}, f, indent=1)
    run.write_header("s2_write", game, VERSION, settings, inputs,
                     {"candidates": p_cands, "groups": p_groups, "llm": p_llm}, code=__file__)
    log(f"[{game}] pool: " + ", ".join(f"{n} {c['candidates']} candidates + {c['rescued']} rescued"
                                       for n, c in counts.items())
        + f"; {sum(c['duplicate'] for c in cands)} duplicates; {len(groups)} groups")
    return counts


# ---- the pool beside its sources ------------------------------------------------------------------
def book_numbers(run_dir, game):
    """A run's book on the training level and its T1, as G1's verdict reads them: the guard
    fitter's "after" if it ran, else S3's book and S4's T1. Members are the run's own ids."""
    guards = glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.guards.json"))
    if guards:
        with open(guards[0]) as f:
            s = json.load(f)
        book, t1 = dict(s["book"]["after"]), (s.get("t1") or {}).get("after")
    else:
        with open(glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.book.json"))[0]) as f:
            b = json.load(f)
        book = dict(b["numbers"], members=b["members"])
        carry = glob.glob(os.path.join(run_dir, "s4_carry", f"{game}_L*.carry.json"))
        t1 = None
        if carry:
            with open(carry[0]) as f:
                c = json.load(f)
            t1 = dict(c["t1"], members=[r["id"] for r in c["rules"] if r["in_t1"]])
    out = {"cell_coverage": book["cell_coverage"], "case_coverage": book["case_coverage"],
           "wrong": book["wrong"], "members": book["members"]}
    if t1:
        pm = t1["per_move"]
        out.update(t1=pm["rate"]["book"], nothing=pm["rate"]["nothing"], memory=pm["rate"]["memory"],
                   t1_wrong=t1["wrong"], beats_both=pm["beats_both"], t1_members=t1["members"],
                   vs_nothing=pm["diff"]["nothing"], vs_memory=pm["diff"]["memory"])
    return out


def report(run, sources, games, log=print):
    """The pool's book beside each source's, per game: <run>/pool_report.{json,md}."""
    rows = {}
    for game in games:
        r = {s.name: book_numbers(s.dir, game) for s in sources}
        r["pool"] = p = book_numbers(run.dir, game)
        origin = {c["id"]: c["pooled_from"] for c in _jsonl(os.path.join(run.dir, "s2_write", game,
                                                                         "candidates.jsonl"))}
        p["members_from"] = [origin[i] for i in p["members"]]
        p["t1_members_from"] = [origin[i] for i in p.get("t1_members", [])]
        rows[game] = r
    names = [s.name for s in sources]
    best = lambda r: max(r[n]["cell_coverage"] for n in names)
    lines = [f"# Pooled book: {run.name} (from {', '.join(names)}), exploratory", "",
             "Training level: the book's cell coverage (case coverage). Transfer level: T1 per move "
             "(\"nothing\", \"memory\"), wrong share of changing cases, whether T1 beats both with both 95% "
             "intervals above zero. Book members by source run.", "",
             "| Game | " + " | ".join(f"Cells: {n}" for n in names) + " | Cells: pool | T1 pool (nothing, memory) | "
             + " | ".join(f"T1 {n}" for n in names) + " | Wrong | Beats both | Pool's members |",
             "|---|" + "---|" * (2 * len(names) + 5)]
    for game, r in rows.items():
        p = r["pool"]
        t1 = lambda x: f"{x['t1']:.1%}" if "t1" in x else "-"
        mark = "**" if p["cell_coverage"] > best(r) else ""
        members = ", ".join(f"{m['run']}:{m['id']}" for m in p["members_from"]) or "none"
        lines.append(f"| {game} | " + " | ".join(f"{r[n]['cell_coverage']:.1%} ({r[n]['case_coverage']:.1%})"
                                                for n in names)
                     + f" | {mark}{p['cell_coverage']:.1%}{mark} ({p['case_coverage']:.1%}) | "
                     + (f"{p['t1']:.1%} ({p['nothing']:.1%}, {p['memory']:.1%})" if "t1" in p else "-") + " | "
                     + " | ".join(t1(r[n]) for n in names)
                     + f" | {p.get('t1_wrong', 0):.2%} | {'yes' if p.get('beats_both') else 'no'} | {members} |")
    higher = [g for g, r in rows.items() if r["pool"]["cell_coverage"] > best(r)]
    lower = [g for g, r in rows.items() if r["pool"]["cell_coverage"] < best(r)]
    lines += ["", f"Pool above the better source on {len(higher)} of {len(rows)} games "
              f"({', '.join(higher) or 'none'}); below it on {len(lower)} ({', '.join(lower) or 'none'})."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run.dir, "pool_report.json"), "w") as f:
        json.dump(plain({"run": run.name, "sources": names, "games": rows, "higher": higher, "lower": lower}),
                  f, indent=1)
    with open(os.path.join(run.dir, "pool_report.md"), "w") as f:
        f.write(text)
    log(text)
    return rows
