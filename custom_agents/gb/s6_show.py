"""s6_show.py - S6 SHOW: one offline page per run, with a funnel per game and a case viewer.

The problem it fixes: our numbers are spread over many files, and nothing shows
one move's prediction next to what really happened, so it is hard to see where
and why a game breaks. (Plan section 5.6; this is v0: the funnel and the case
viewer. Stage panels, the online views and compare mode come in v1.)

Reads a run folder (the artifact contract: S3 and S4 write these files whether
they come from the CP1 import or from Glassbox's own stages):
  manifest.json                                  sessions, commits, config hash
  evidence/<game>_L<k>_{train,fit,score}.npz     the levels' cases
  s2_write/<game>/candidates.jsonl, llm.json     rule code; LLM use
  s3_check/<game>/rules.jsonl, rule_cases.npz    every rule's grades, per case
  s3_check/<game>_L<k>.book.json, .status.npz    the book and its status per case
  s4_carry/<game>_L<k+1>.carry.json              re-binding and T0/T1/T2
  s4_carry/<game>_L<k+1>.t1_status.npz           T1's status per case
  s1_see/<game>_L<k>_<split>.facts.{npz,jsonl}   S1's description of each case (optional)
Writes s6_show/:
  index.html          the page: self-contained, opens offline, no external files
  cases/<game>.json   the flagged cases the viewer pages through
  figures/funnel.png  the funnel, for the written report

It never changes a result. To draw a predicted board it re-runs the book's rules
on the cases it shows (in the sandbox, in a child process) and checks that each
re-run case gets exactly the status and cell counts the stage stored. Any
difference stops the build.
"""
import base64
import gzip
import json
import os
from datetime import datetime

import numpy as np

from gb import artifacts as A
from gb.core.book import PARTIAL
from gb.core.check import EXACT, UNKNOWN, WRONG
from gb.core.contract import Rule
from gb.core.evidence import LevelEvidence
from gb.core.sandbox import run_in_child

VERSION = 1
STATUS = {UNKNOWN: "UNKNOWN", EXACT: "EXACT", WRONG: "WRONG", PARTIAL: "PARTIAL"}
NEAR_CELLS = 3            # a WRONG case with at most this many wrong claimed cells is also NEAR
KNOWN_FLAGS = ("WRONG", "NEAR", "UNCLAIMED", "CONFLICT", "EXACT")   # the flags v0 can compute
# The games' own colours (copied from legacy/rulebook/wm/rules.py), so boards look like the game.
PALETTE_HEX = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
               "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
               "#FF851B", "#921231", "#4FCC30", "#A356D6"]
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s6_show_page.html")


# ---- reading one game's files -------------------------------------------------------------
def games_in(run_dir):
    s3 = os.path.join(run_dir, "s3_check")
    return sorted(f.split("_L")[0] for f in os.listdir(s3) if f.endswith(".book.json")) if os.path.isdir(s3) else []


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_game(run_dir, game):
    """Everything S6 needs for one game, from the run folder only."""
    s3, s4 = os.path.join(run_dir, "s3_check"), os.path.join(run_dir, "s4_carry")
    book_file = next(f for f in os.listdir(s3) if f.startswith(f"{game}_L") and f.endswith(".book.json"))
    k = int(book_file[len(game) + 2:].split(".")[0])
    g = {"game": game, "k": k, "book": json.load(open(os.path.join(s3, book_file))),
         "status": dict(np.load(os.path.join(s3, f"{game}_L{k}.status.npz"))),
         "rules": {r["id"]: r for r in _jsonl(os.path.join(s3, game, "rules.jsonl"))},
         "code": {c["id"]: c for c in _jsonl(os.path.join(run_dir, "s2_write", game, "candidates.jsonl"))},
         "train_path": os.path.join(run_dir, "evidence", f"{game}_L{k}_train.npz")}
    rc = os.path.join(s3, game, "rule_cases.npz")
    g["rule_cases"] = dict(np.load(rc)) if os.path.exists(rc) else None
    llm = os.path.join(run_dir, "s2_write", game, "llm.json")
    g["llm"] = json.load(open(llm)) if os.path.exists(llm) else None
    carry = [f for f in os.listdir(s4) if f.startswith(f"{game}_L") and f.endswith(".carry.json")] \
        if os.path.isdir(s4) else []
    g["carry"] = g["t1_status"] = g["s1_score"] = None
    g["s1_train"] = _s1(run_dir, f"{game}_L{k}_train")
    if carry:
        k2 = int(carry[0][len(game) + 2:].split(".")[0])
        g.update(k_next=k2, carry=json.load(open(os.path.join(s4, carry[0]))),
                 t1_status=dict(np.load(os.path.join(s4, f"{game}_L{k2}.t1_status.npz"))),
                 fit_path=os.path.join(run_dir, "evidence", f"{game}_L{k2}_fit.npz"),
                 score_path=os.path.join(run_dir, "evidence", f"{game}_L{k2}_score.npz"),
                 s1_score=_s1(run_dir, f"{game}_L{k2}_score"))
    return g


def _s1(run_dir, name):
    """S1's facts for one split, if S1 has run on it."""
    p = os.path.join(run_dir, "s1_see", name + ".facts.npz")
    if not os.path.exists(p):
        return None
    return {"arr": dict(np.load(p)),
            "recs": {r["k"]: r for r in _jsonl(os.path.join(run_dir, "s1_see", name + ".facts.jsonl"))}}


# ---- the funnel ---------------------------------------------------------------------------
def _explains(status, cells_right):
    """Per case: the book claims at least one changed cell right and no claimed cell wrong."""
    return np.isin(status, (EXACT, PARTIAL)) & (cells_right > 0)


def funnel(g, train):
    """One game's funnel. Every share is of the level's distinct changing cases."""
    ch = train.changed
    n = int(ch.sum())
    called = set(g["book"].get("groups_called") or [])
    in_called = ch & np.array([grp in called for grp in train.group])
    admitted = np.zeros(len(ch), bool)
    rc = g["rule_cases"]
    if rc is not None and len(rc["ids"]):
        ok = {i for i, r in g["rules"].items() if r.get("v2_admitted")}
        rows = np.array([int(i) in ok for i in rc["ids"]])
        if rows.any():
            admitted = (rc["applies"][rows] & rc["claimed_ok"][rows] & (rc["cells_right"][rows] > 0)).any(axis=0)
    st = g["status"]
    book_part = ch & _explains(st["status"], st["cells_right"])
    book_whole = ch & (st["status"] == EXACT)
    nums = g["book"]["numbers"]
    if n and round(int(book_whole.sum()) / n, 4) != nums["case_coverage"]:
        raise RuntimeError(f"{g['game']}: the funnel's whole-screen count disagrees with S3's case coverage")
    count = lambda m: int(m.sum())
    s1 = g["s1_train"]
    out = {"game": g["game"], "level": g["k"], "changing": n, "cases": len(ch),
           "s1": count(ch & s1["arr"]["explained"]) if s1 else None,
           "in_called": count(in_called), "admitted": count(admitted & ch), "book_part": count(book_part),
           "book_whole": count(book_whole), "book_rules": len(g["book"]["members"]),
           "cell_coverage": nums["cell_coverage"], "book_wrong": nums["wrong"], "transfer": None}
    if g["carry"] is not None:
        score = LevelEvidence.load(g["score_path"])
        s1, t1 = g["t1_status"], g["carry"]["t1"]
        sch = score.changed
        out["transfer"] = {
            "level": g["k_next"], "changing": int(sch.sum()), "rules": t1["rules_in"],
            "t1_part": count(sch & _explains(s1["status"], s1["cells_right"])),
            "t1_whole": count(sch & (s1["status"] == EXACT)), "wrong": t1["wrong"],
            "s1": count(sch & g["s1_score"]["arr"]["explained"]) if g["s1_score"] else None,
            "rate": t1["per_move"]["rate"], "diff": t1["per_move"]["diff"],
            "beats_both": t1["per_move"]["beats_both"], "moves": t1["per_move"]["moves"],
            "unit": t1["per_move"]["unit"], "units": t1["per_move"]["units"]}
    return out


def cp1_gate(funnels):
    """CP1's pre-registered gate (Rulebook v2 pre-registration section 4.3),
    recomputed from the run's files: T1 beats both baselines on >= 3 games, and
    is wrong on <= 5% of changing cases on every game."""
    primary = [f["game"] for f in funnels if f["transfer"] and f["transfer"]["beats_both"]]
    unsafe = [f["game"] for f in funnels if not f["transfer"] or f["transfer"]["wrong"] > 0.05]
    return {"name": "CP1's gate (Rulebook v2 pre-registration, section 4.3), recomputed from this run's files",
            "criteria": [
                {"label": "Primary: T1 beats both baselines, both 95% intervals above zero",
                 "value": f"{len(primary)} of {len(funnels)} games", "needs": "3 or more",
                 "pass": len(primary) >= 3, "games": primary},
                {"label": "Safety: T1 wrong on at most 5% of the next level's changing cases",
                 "value": f"{len(funnels) - len(unsafe)} of {len(funnels)} games", "needs": "every game",
                 "pass": not unsafe, "games": unsafe}],
            "pass": len(primary) >= 3 and not unsafe}


# ---- choosing the cases to show -----------------------------------------------------------------
def flags_of(ev, st, wanted):
    """{flag: case indices}, each list most-recorded case first."""
    s, ch = st["status"], ev.changed
    masks = {"WRONG": s == WRONG,
             "NEAR": (s == WRONG) & (st["cells_wrong"] <= NEAR_CELLS),
             "UNCLAIMED": ch & ~ev.frozen & np.isin(s, (UNKNOWN, PARTIAL)),
             "CONFLICT": ev.n_variants > 1,
             "EXACT": ch & (s == EXACT)}
    order = np.lexsort((np.arange(len(s)), -ev.count))
    return {f: [int(k) for k in order if masks[f][k]] for f in wanted if f in masks}


def pick(flag_cases, per_flag):
    """{case: [flags]} for the first `per_flag` cases of each flag."""
    out = {}
    for f, ks in flag_cases.items():
        for k in ks[:per_flag]:
            out.setdefault(k, []).append(f)
    return out


# ---- predictions, re-run in the sandbox ----------------------------------------------------------
def _predict_worker(members, ev_path, cases, q):
    """Child process: the composed prediction of `members` on each case, exactly
    as gb/core/book.py composes it (a cell claimed two ways is dropped)."""
    try:
        ev = LevelEvidence.load(ev_path)
        live = ev.live
        flat_live = live.ravel()
        layouts, rules = {None: ev.api.layout}, []
        for m in members:
            lp = m.get("layout_path")
            if lp not in layouts:
                layouts[lp] = LevelEvidence.load(lp).api.layout
            r = Rule(m["src"])
            rules.append((m["id"], r, r.values(m.get("binding") or {}, layouts[lp])))
        out = {}
        for k in cases:
            board, act, api = ev.inputs(k)
            flat, after = ev.before[k].ravel(), ev.after[k].ravel()
            said = []
            for rid, rule, p in rules:
                try:
                    if rule.applies(board, act, api, p):
                        pred, claim = rule.predict(board, act, api, p, live)
                        pf, cf = np.clip(pred, 0, 255).astype(np.uint8).ravel(), claim.ravel()
                        diff = np.flatnonzero(cf & (pf != flat))
                        said.append((rid, cf.copy(), diff, pf[diff]))
                except Exception:
                    pass
            pred = flat.copy()
            claimed = np.zeros(flat.size, bool)
            clash = np.zeros(flat.size, bool)
            if len(said) == 1:
                _, cf, diff, vals = said[0]
                claimed = cf
                pred[diff] = vals
            elif said:
                value = {}
                for rid, cf, diff, vals in said:
                    v = flat.copy()
                    v[diff] = vals
                    value[rid] = v
                    claimed |= cf
                touched = np.unique(np.concatenate([s[2] for s in said]))
                for c in touched:
                    vals = {int(value[rid][c]) for rid, cf, _, _ in said if cf[c]}
                    if len(vals) > 1:
                        clash[c] = True
                    elif vals:
                        pred[c] = vals.pop()
                claimed &= ~clash
            really = (flat != after) & flat_live
            wrong = claimed & (pred != after)
            status = (UNKNOWN if not claimed.any() else WRONG if wrong.any() else
                      EXACT if np.array_equal(pred[flat_live], after[flat_live]) else PARTIAL)
            out[k] = {"pred": pred.astype(np.uint8), "claimed": claimed, "clash": clash,
                      "rules": [s[0] for s in said], "status": int(status),
                      "cells_right": int((claimed & really & (pred == after)).sum()),
                      "cells_wrong": int(wrong.sum()), "cells_clash": int(clash.sum()),
                      "claimed_n": int(claimed.sum())}
        q.put({"ok": True, "cases": out})
    except Exception as e:
        q.put({"ok": False, "reason": f"{type(e).__name__}: {e}"})


def predict(members, ev_path, cases, timeout_s=900):
    if not cases:
        return {}
    res = run_in_child(_predict_worker, (members, ev_path, list(cases)), timeout_s)
    if not res.get("ok"):
        raise RuntimeError(f"re-running the book's rules failed: {res.get('reason')}")
    return res["cases"]


# ---- one level's cases, for the page ----------------------------------------------------------------
def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _bits(mask):
    return _b64(np.packbits(np.asarray(mask, bool).ravel()))


def action_text(a):
    a = int(a)
    if a < 5:
        return f"ACTION{a + 1}"
    y, x = divmod(a - 5, 64)
    return f"click at row {y}, column {x}"


def _s1_case(s1, k, changed):
    if s1 is None:
        return None
    r = s1["recs"].get(k)
    if r is None:
        return {"explained": True, "unit": 1.0, "items": []} if not changed else None
    return {"explained": r["explained"], "unit": r["unit"], "items": [it["text"] for it in r["items"]]}


def level_cases(g, which, ev_path, members, st, flags, per_flag, label, arm, s1=None):
    """The flagged cases of one level, with their boards and predictions."""
    ev = LevelEvidence.load(ev_path)
    chosen = pick(flags_of(ev, st, flags), per_flag)
    got = predict(members, ev_path, sorted(chosen))
    cases = []
    for k in sorted(chosen, key=lambda k: (-int(ev.count[k]), k)):
        p = got[k]
        stored = {"status": int(st["status"][k]), "cells_right": int(st["cells_right"][k]),
                  "cells_wrong": int(st["cells_wrong"][k]), "cells_clash": int(st["cells_clash"][k]),
                  "claimed_n": int(st["claimed_n"][k])}
        mine = {key: p[key] for key in stored}
        if mine != stored:
            raise RuntimeError(f"{g['game']} {which} case {k}: re-run gives {mine}, the stage stored {stored}")
        changed = (ev.before[k] != ev.after[k]) & ev.live
        cases.append({"k": k, "id": f"{g['game']}/L{ev.level + 1}/{which}/{k}", "action": action_text(ev.actions[k]),
                      "group": ev.group[k], "moves": int(ev.count[k]), "variants": int(ev.n_variants[k]),
                      "frozen": bool(ev.frozen[k]), "status": STATUS[p["status"]], "flags": chosen[k],
                      "cells": {"right": p["cells_right"], "wrong": p["cells_wrong"], "clash": p["cells_clash"],
                                "claimed": p["claimed_n"], "changed": int(changed.sum())},
                      "rules": p["rules"], "s1": _s1_case(s1, k, bool(ev.changed[k])),
                      "b": _b64(ev.before[k].astype(np.uint8)),
                      "a": _b64(ev.after[k].astype(np.uint8)), "p": _b64(p["pred"]),
                      "c": _bits(p["claimed"]), "x": _bits(p["clash"])})
    counts = {f: len(v) for f, v in flags_of(ev, st, flags).items()}
    return {"key": which, "label": label, "arm": arm, "level": ev.level + 1, "live": _bits(ev.live),
            "flag_counts": counts, "shown": len(cases), "cases": cases}


def game_cases(g, flags, per_flag):
    """Every level the viewer shows for one game: the training level (the book)
    and, if there is one, the next level's score split (T1)."""
    rules = g["rules"]
    book_ids = g["book"]["members"]
    members = [{"id": i, "src": g["code"][i]["code"], "binding": rules[i]["binding"], "layout_path": g["train_path"]}
               for i in book_ids]
    levels = [level_cases(g, "train", g["train_path"], members, g["status"], flags, per_flag,
                          f"Level {g['k']}: the training level, the book", "book", g["s1_train"])]
    if g["carry"] is not None:
        t1 = {r["id"]: r for r in g["carry"]["rules"] if r.get("in_t1")}
        kept = [int(i) for i in g["t1_status"]["members"]]
        members1 = [{"id": i, "src": g["code"][i]["code"], "binding": t1[i]["binding"], "layout_path": g["fit_path"]}
                    for i in kept]
        levels.append(level_cases(g, "score", g["score_path"], members1, g["t1_status"], flags, per_flag,
                                  f"Level {g['k_next']}: the next level (score split), T1 after re-binding", "T1",
                                  g["s1_score"]))
    shown = {i for lv in levels for c in lv["cases"] for i in c["rules"]} | set(book_ids)
    carry = {r["id"]: r for r in (g["carry"] or {}).get("rules", [])}
    rule_info = {str(i): {"rule": rules[i].get("rule") or "", "group": rules[i]["group"],
                          "origin": rules[i].get("origin"), "binding": rules[i]["binding"],
                          "t1_binding": carry.get(i, {}).get("binding"), "t1": carry.get(i, {}).get("text"),
                          "code": g["code"][i]["code"], "trusted": rules[i].get("v2_trusted"),
                          "admitted": rules[i].get("v2_admitted"), "in_book": i in book_ids}
                 for i in sorted(shown)}
    return {"game": g["game"], "levels": levels, "rules": rule_info}


# ---- the figure for the report ------------------------------------------------------------
STAGES = [("in_called", "In a group the LLM\nwas asked about"),
          ("admitted", "An admitted rule\nexplains part of it"),
          ("book_part", "The book explains part,\nnothing wrong"),
          ("book_whole", "The book gets the\nwhole screen right")]
INK, INK2, MUTED, GRID_C, AXIS, SURFACE, BLUE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#2a78d6"


def funnel_png(funnels, path, dpi=200):
    """Small multiples, one panel per funnel stage, one bar per game; every bar labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    games = [f["game"] for f in funnels]
    stages = ([("s1", "S1 describes it as\nwhole objects")] if all(f["s1"] is not None for f in funnels) else []) \
        + STAGES + [("t1_whole", "Next level: T1 gets\nthe whole screen right")]
    fig, axes = plt.subplots(1, len(stages), figsize=(2.3 * len(stages) + 0.8, 0.42 * len(games) + 1.5),
                             sharey=True, dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, stages):
        vals = []
        for f in funnels:
            if key == "t1_whole":
                t = f["transfer"]
                vals.append(t["t1_whole"] / t["changing"] if t and t["changing"] else 0.0)
            else:
                vals.append(f[key] / f["changing"] if f["changing"] else 0.0)
        y = np.arange(len(games))
        ax.barh(y, vals, height=0.5, color=BLUE)
        for yi, v in zip(y, vals):
            ax.text(v + 0.03, yi, f"{v:.0%}", va="center", ha="left", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.25)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=7, color=MUTED)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID_C, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
    axes[0].set_yticks(np.arange(len(games)))
    axes[0].set_yticklabels([f"{f['game']}  ({f['changing']:,} / {f['transfer']['changing']:,})" if f["transfer"]
                             else f"{f['game']}  ({f['changing']:,})" for f in funnels], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Share of each level's distinct changing cases "
                 "(in brackets: how many, on the training level / the next level)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- the page ----------------------------------------------------------------------------
def _gz64(obj):
    return base64.b64encode(gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), 9)).decode()


def build(run_name, root=None, log=print):
    """Build s6_show/ for one run. Returns the path of index.html."""
    run = A.Run(run_name, root=root)
    config = run.config.get("show", {})
    per_flag = int(config.get("cases_per_flag", 50))
    flags = [f for f in config.get("flags", KNOWN_FLAGS) if f in KNOWN_FLAGS] or list(KNOWN_FLAGS)
    games = games_in(run.dir)
    if not games:
        raise SystemExit(f"{run.dir} has no S3 output (s3_check/<game>_L<k>.book.json) to show")
    out = run.stage_dir("s6_show")
    os.makedirs(os.path.join(out, "cases"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    funnels, blobs, inputs, llm = [], {}, {}, {}
    for game in games:
        g = load_game(run.dir, game)
        train = LevelEvidence.load(g["train_path"])
        funnels.append(funnel(g, train))
        data = game_cases(g, flags, per_flag)
        with open(os.path.join(out, "cases", f"{game}.json"), "w") as f:
            json.dump(data, f)
        blobs[game] = _gz64(data)
        llm[game] = g["llm"]
        inputs.update({f"{game}_book": os.path.join(run.dir, "s3_check", f"{game}_L{g['k']}.book.json"),
                       f"{game}_status": os.path.join(run.dir, "s3_check", f"{game}_L{g['k']}.status.npz")})
        shown = ", ".join("{}: {}".format(lv["key"], lv["shown"]) for lv in data["levels"])
        log(f"[{game}] funnel, and cases for the viewer ({shown})")
    manifest = json.load(open(os.path.join(run.dir, "manifest.json")))
    headers = [run.read_header("s3_check", g) for g in games]
    imported = sorted({h["settings"].get("imported_from") for h in headers if h and h["settings"].get("imported_from")})
    answers = [v for v in llm.values() if v]
    meta = {"run": run_name, "built": datetime.now().isoformat(timespec="seconds"), "commit": A.git_commit(),
            "created": manifest.get("created"), "sessions": manifest.get("sessions", []),
            "config_hash": A.digest(run.config), "imported_from": imported,
            "model": sorted({v.get("model") for v in answers if v.get("model")}),
            "llm": {"answers": sum(v.get("answers", 0) for v in answers),
                    "tokens": sum(v.get("tokens", 0) for v in answers),
                    "ran_out": sum(v.get("ran_out", 0) for v in answers),
                    "hours": round(sum(v.get("seconds", 0) for v in answers) / 3600, 1)},
            "flags": flags, "per_flag": per_flag, "near_cells": NEAR_CELLS}
    gate = cp1_gate(funnels) if any("rulebook/v2" in s for s in imported) else None
    page = {"meta": meta, "gate": gate, "funnel": funnels, "palette": PALETTE_HEX, "games": blobs}
    blob = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")      # never ends the <script> early
    html = open(TEMPLATE).read().replace("/*GLASSBOX_DATA*/null", blob)
    p_html = os.path.join(out, "index.html")
    with open(p_html, "w") as f:
        f.write(html)
    p_png = os.path.join(out, "figures", "funnel.png")
    funnel_png(funnels, p_png)
    run.write_header("s6_show", "all", VERSION, config, inputs, {"page": p_html, "funnel": p_png})
    log(f"wrote {os.path.relpath(p_html, A.ROOT)} ({os.path.getsize(p_html) / 1e6:.1f} MB) and "
        f"{os.path.relpath(p_png, A.ROOT)}")
    return p_html
