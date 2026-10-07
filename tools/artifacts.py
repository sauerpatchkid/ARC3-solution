#!/usr/bin/env python3
"""artifacts.py - freeze and check registered artifacts (custom_agents/wm/registry.py).

    uv run python tools/artifacts.py list
    uv run python tools/artifacts.py add <name> <file or folder>... --games tu93,ft09 [--note "..."] [--kind results]
    uv run python tools/artifacts.py verify [<name>]      # exit status 1 if anything is missing or changed
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from wm import registry, tiers  # noqa: E402


def games(text):
    out = []
    for g in text.split(","):
        out += list(tiers.TIERS[g]) if g in tiers.TIERS else [g]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("paths", nargs="+")
    a.add_argument("--games", required=True, help="source games: names or tier names (dev, seen), comma-separated")
    a.add_argument("--note", default="")
    a.add_argument("--kind", default="artifact")
    v = sub.add_parser("verify")
    v.add_argument("name", nargs="?")
    args = ap.parse_args()
    if args.cmd == "list":
        for e in registry.load()["artifacts"]:
            print(f"{e['name']:<24} {e['kind']:<9} frozen {e['frozen']}  {e['n_files']:>4} files  "
                  f"{e['sha256'][:12]}  from {', '.join(e['source_games'])}")
    elif args.cmd == "add":
        src = games(args.games)
        if args.kind == "artifact":
            outside = [g for g in src if g not in tiers.BUILD_FROM]
            if outside:
                sys.exit(f"cross-game artifacts may be built from dev and seen games only; not {outside}")
        e = registry.add(args.name, args.paths, src, args.note, args.kind)
        print(f"registered {e['name']}: {e['n_files']} files, sha256 {e['sha256'][:16]}...")
    else:
        bad = registry.verify(args.name)
        for name, rel, why in bad:
            print(f"{why.upper():8} {name}: {rel}")
        print("all registered files are present and unchanged" if not bad
              else f"{len(bad)} registered file(s) missing or changed")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
